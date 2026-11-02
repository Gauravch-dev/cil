"""Local web adapter. Original ontology, decisions and search remain authoritative.

Default: replay genuine, bundled Task 7 forecasts. Never manufacture inference,
uncertainty, feature attribution, or retrain when a customer presses a button.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
import csv
import io
import json
import os
import sqlite3
import uuid

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator
from starlette.concurrency import run_in_threadpool
from urllib.parse import unquote

from readings import (FEATURES, MAX_BYTES, ReadingsError, check_csv, forecast_available,
                      predict_uploaded, schema as readings_schema)

from task9_core import (Appliance, Scenario, OntologyKnowledge, ForecastBridge,
                        default_appliances, optimize_decisions)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "web_data"
DATA.mkdir(exist_ok=True)
DB = DATA / "history.sqlite3"
LOCK = Lock()
knowledge = OntologyKnowledge((ROOT / "SmartElectricityOptimizer.owl").read_text(encoding="utf-8"))


class ApplianceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=80)
    power_w: float = Field(gt=0, le=20000, allow_inf_nan=False)
    duration_min: StrictInt = Field(default=60, gt=0, le=720)
    requested: StrictBool = True
    on: StrictBool = False
    essential: StrictBool = False
    flexible: StrictBool = True
    priority: str = "Medium"
    earliest_min: StrictInt = Field(default=0, ge=0, le=720)
    deadline_min: StrictInt = Field(default=360, ge=15, le=720)
    requires_occupancy: StrictBool = False
    ontology_id: str = Field(default="", max_length=80)


class ScenarioInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    occupancy: StrictInt = Field(default=3, ge=0, le=20)
    temperature_c: float = Field(default=35, ge=-50, le=60, allow_inf_nan=False)
    peak_start_hour: StrictInt = Field(default=18, ge=0, le=23)
    peak_end_hour: StrictInt = Field(default=20, ge=0, le=23)
    peak_rate: float = Field(default=9, ge=0, le=10000, allow_inf_nan=False)
    offpeak_rate: float = Field(default=6, ge=0, le=10000, allow_inf_nan=False)
    horizon_hours: StrictInt = Field(default=6, ge=1, le=12)
    max_parallel_jobs: StrictInt = Field(default=1, ge=1, le=6)
    node_limit: StrictInt = Field(default=50000, ge=1, le=50000)


class RunInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str | None = None
    upload_id: str | None = None
    sample_forecast: StrictBool = False
    scenario: ScenarioInput
    appliances: list[ApplianceInput] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def single_source(self):
        if bool(self.case_id) == bool(self.upload_id):
            raise ValueError("Choose either a recorded history or one uploaded readings file.")
        if self.sample_forecast and not self.upload_id:
            raise ValueError("A sample forecast requires a checked sample file.")
        return self


class ForecastSource:
    def __init__(self):
        self.bridge = None
        self.issue = "The current combined forecast requires a matching completed model bundle. Recorded forecasts are available."
        explicit = os.environ.get("CIL_TASK7_FOLDER")
        if explicit:
            # Fail closed: ForecastBridge normally fits on a cache miss. Require its
            # exact cache and block fit even if an incompatible signature is given.
            import unittest.mock
            try:
                from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
                with unittest.mock.patch.object(GradientBoostingRegressor, "fit", side_effect=RuntimeError("Prepare the matching forecast cache offline first.")), unittest.mock.patch.object(RandomForestRegressor, "fit", side_effect=RuntimeError("Prepare the matching forecast cache offline first.")):
                    self.bridge = ForecastBridge(explicit, Path(os.environ.get("CIL_TASK9_OUTPUT", str(ROOT / "task9_outputs"))))
                self.issue = None
            except Exception as error:
                self.issue = "Configured forecast bundle could not be loaded; see the server log. Recorded forecasts remain available."
                print("Configured forecast integration:", error, flush=True)
        self.folder = ROOT / "task7_outputs" / "full_20261004T223051_253137Z"
        self.upload_forecaster = self.bridge if forecast_available(self.bridge) else None
        if self.upload_forecaster is None:
            try:
                from local_forecast import LocalPowerForecast
                self.upload_forecaster = LocalPowerForecast(self.folder)
                from hybrid_forecast import ARTIFACT_ROOT, LocalHybridForecast
                if (ARTIFACT_ROOT / "active.json").is_file():
                    try:
                        self.upload_forecaster = LocalHybridForecast.load_active(self.upload_forecaster)
                        print("Loaded evaluated local hybrid:", self.upload_forecaster.selected_name, flush=True)
                    except Exception as error:
                        print("Local hybrid unavailable; standalone LSTM retained:", error, flush=True)
                self.issue = None
            except Exception as error:
                print("Local uploaded-readings forecast unavailable:", error, flush=True)
        self.manifest = json.loads((self.folder / "manifest.json").read_text())
        self.saved = pd.read_csv(self.folder / "test_predictions.csv")
        self.power = pd.read_csv(self.folder / "power_15min.csv", parse_dates=["Interval_end"])
        self.explanation_folder = ROOT / "task8_outputs" / "run_20261004T224434_997783Z"
        self.diagnostics = pd.read_csv(self.explanation_folder / "diagnostic_cases.csv")
        self.cases = []
        if self.bridge:
            ids = self.bridge._sample(self.bridge.test_ids, 64)
            for index in ids:
                self.cases.append({"id": str(int(index)), "origin": str(self.bridge.timestamps[int(index)-4]), "explanation": True})
        else:
            ids = set(np.linspace(0, len(self.saved)-1, 64, dtype=int).tolist())
            for target in self.diagnostics.Target_time:
                ids.update(self.saved.index[self.saved.Target_interval_end == target].tolist())
            for index in sorted(ids):
                row = self.saved.iloc[index]
                self.cases.append({"id": str(index), "origin": row.Forecast_origin,
                    "explanation": row.Target_interval_end in set(self.diagnostics.Target_time)})

    def forecast(self, case_id):
        if case_id not in {case["id"] for case in self.cases}:
            raise ValueError("Choose an available electricity history.")
        if self.bridge:
            return self.bridge.predict(int(case_id))
        row = self.saved.iloc[int(case_id)]
        if getattr(self.upload_forecaster, "is_hybrid", False):
            with np.load(self.folder / "prepared_arrays.npz") as a:
                target = int(a["test_indices"][int(case_id)])
                inputs = {"history": a["history_values"][target-99:target-3][None], "context": a["context_values"][target][None]}
            return self.upload_forecaster.forecast(inputs, row.Forecast_origin, row.Target_interval_end)
        return {"origin": row.Forecast_origin, "target_end": row.Target_interval_end,
            "target_interval_start": str(pd.Timestamp(row.Target_interval_end)-pd.Timedelta(minutes=15)),
            "forecast_kw": float(row.lstm_compact_kW), "threshold_kw": self.manifest["peak_threshold_kw"],
            "model": "Recorded Task 7 LSTM", "source": "recorded_forecast", "source_run": self.manifest["run_id"],
            "units": "kW, average over the target 15-minute interval"}

    def evidence(self, forecast, case_id):
        origin = pd.Timestamp(forecast["origin"])
        if self.bridge:
            inputs = self.bridge.inputs(np.array([int(case_id)]))
            names = self.bridge.manifest["history_features"]
            channel = next(i for i, name in enumerate(names) if "global_active_power" in name.lower())
            values = inputs["history"][0, :, channel] * self.bridge.scale + self.bridge.mean
            history = [{"time": str(t), "kw": float(v)} for t, v in zip(pd.date_range(end=origin, periods=len(values), freq="15min"), values)]
        else:
            readings = self.power[(self.power.Interval_end <= origin) & (self.power.Interval_end > origin-pd.Timedelta(hours=24))]
            history = [{"time": str(row.Interval_end), "kw": float(row.input_kw) if pd.notna(row.input_kw) else None} for row in readings.itertuples()]
        if forecast.get("forecaster_sha256") and getattr(self.upload_forecaster, "is_hybrid", False):
            with np.load(self.folder / "prepared_arrays.npz") as a:
                target = int(a["test_indices"][int(case_id)])
                inputs = {"history": a["history_values"][target-99:target-3][None], "context": a["context_values"][target][None]}
            return self.upload_forecaster.evidence(inputs, history)
        explanation = {"available": False, "reason": "A verified contribution breakdown is not saved for this history. Appliance reasons and rule evidence remain available."}
        if self.bridge:
            try:
                exp = self.bridge.explain(int(case_id))
                explanation = {"available": True, "method": "SHAP", "reference_kw": exp["base"],
                    "raw_forecast_kw": exp["raw_prediction"], "reconstruction_verified": True,
                    "factors": [{"name": key, "value_kw": float(value)} for key, value in zip(exp["features"], exp["values"])],
                    "activation": self.bridge.activation(int(case_id)).T.tolist(),
                    "note": "Contributions explain the forecast stage, not physical causes. Internal responses are not attention weights."}
            except Exception as error:
                print("Explanation unavailable:", error, flush=True)
        else:
            matched = self.diagnostics[self.diagnostics.Target_time == forecast["target_end"]]
            if len(matched):
                case = matched.iloc[0]
                analysis_manifest = json.loads((self.explanation_folder / "analysis_manifest.json").read_text())
                if analysis_manifest["task7_run"] != self.manifest["run_id"] or not np.isclose(float(case.LSTM_kW), forecast["forecast_kw"], atol=1e-5):
                    return {"history": history, "explanation": {"available": False, "reason": "Saved explanation does not match this forecast source."},
                        "confidence": {"available": False, "label": "Not quantified", "reason": "No calibrated score or prediction interval is supplied."}}
                attrs = pd.read_csv(self.explanation_folder / "local_attributions.csv")
                attrs = attrs[(attrs.Case == case.Case) & (attrs.Model == "LSTM")]
                checks = pd.read_csv(self.explanation_folder / "ig_completeness_checks.csv")
                check = checks[(checks.Model == "LSTM") & (checks.Example_position == case.Example_position)].iloc[0]
                explanation = {"available": True, "method": "Integrated gradients", "reference_kw": 1.3425101041793823,
                    "reconstruction_verified": bool(check.Within_tolerance), "residual_kw": float(check.Absolute_residual_kW),
                    "factors": [{"name": row.Input_group, "value_kw": float(row.Signed_attribution_kW)} for row in attrs.itertuples()],
                    "note": "Saved contributions for this exact recorded forecast, relative to a reference history. They show associations, not causes."}
        return {"history": history, "explanation": explanation,
            "confidence": {"available": False, "label": "Not quantified", "reason": "This forecast source does not provide a calibrated confidence score or prediction interval."}}


source = ForecastSource()


def initialize_database(path):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, body TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS uploads (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, filename TEXT NOT NULL, timezone TEXT NOT NULL, content BLOB NOT NULL)")


initialize_database(DB)

app = FastAPI(title="Electricity Optimizer", docs_url=None, redoc_url=None)


@app.middleware("http")
async def local_boundary(request: Request, call_next):
    # Same-origin writes; no permissive CORS. This is a local unauthenticated app.
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        allowed = {str(request.base_url).rstrip("/"), "http://127.0.0.1:5173", "http://localhost:5173"}
        if origin and origin not in allowed:
            return Response(status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


@app.get("/api/bootstrap")
def bootstrap():
    hybrid = getattr(source.upload_forecaster, "is_hybrid", False)
    return {"mode": "connected_history" if source.bridge else "hybrid_history" if hybrid else "recorded_forecasts", "cases": source.cases,
        "default_case": next((c["id"] for c in source.cases if c["explanation"]), source.cases[0]["id"]),
        "scenario": asdict(Scenario()), "appliances": [asdict(a) for a in default_appliances()],
        "integration_gap": source.issue, "ontology": {"classes": len(knowledge.classes), "class_names": sorted(knowledge.classes), "rules": knowledge.rules},
        "readings": {**readings_schema(), "service_available": forecast_available(source.upload_forecaster)},
        "forecasting": {"hybrid_available": bool(hybrid), "model": source.upload_forecaster.selected_name if hybrid else "Standalone LSTM"},
        "capabilities": {"live_meter": False, "hardware_control": False, "confidence": False,
            "forecast": True, "scheduling": True, "rules": True, "history": True,
            "readings_upload": True, "uploaded_forecast": forecast_available(source.upload_forecaster)}}


@app.get("/api/readings/template")
def readings_template():
    return Response(",".join(["Datetime", *FEATURES])+"\r\n", media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="electricity-readings-template.csv"'})


@app.get("/api/readings/samples/{kind}")
def sample_download(kind: str):
    if kind == "filled":
        path = ROOT/"frontend/public/templates/SYNTHETIC-electricity-readings-8-days.csv"
        media_type = "text/csv"
    elif kind == "guide":
        path = ROOT/"frontend/public/templates/Readings-template-guide.txt"
        media_type = "text/plain"
    else:
        raise HTTPException(404, "Sample format not found.")
    return Response(path.read_bytes(), media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="{path.name}"', "Cache-Control":"no-store"})


def upload_content(upload_id):
    with sqlite3.connect(DB) as conn:
        row = conn.execute("SELECT filename, timezone, content FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Uploaded readings were not found. Please upload the file again.")
    return row


def checked_upload(upload_id):
    filename, zone, content = upload_content(upload_id)
    checked = check_csv(content, filename, zone, model_available=forecast_available(source.upload_forecaster))
    checked.report["id"] = upload_id
    return checked


@app.post("/api/readings")
async def upload_readings(request: Request, reading_timezone: str = Query("Asia/Kolkata", alias="timezone")):
    if request.headers.get("content-type", "").split(";")[0].strip() not in ("text/csv", "application/octet-stream"):
        raise HTTPException(415, "Upload the CSV file using the readings form.")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BYTES:
            raise HTTPException(413, "This file is larger than 10 MB. Upload the most recent eight to 60 days.")
        chunks.append(chunk)
    filename = unquote(request.headers.get("x-file-name", "readings.csv"))
    filename = Path(filename.replace("\\", "/")).name[:160]
    content = b"".join(chunks)
    try:
        checked = await run_in_threadpool(check_csv, content, filename, reading_timezone,
                                          model_available=forecast_available(source.upload_forecaster))
    except ReadingsError as error:
        raise HTTPException(422, str(error)) from error
    upload_id = uuid.uuid4().hex
    with sqlite3.connect(DB) as conn:
        conn.execute("INSERT INTO uploads VALUES (?, ?, ?, ?, ?)",
                     (upload_id, datetime.now(timezone.utc).isoformat(), filename, reading_timezone, content))
    checked.report["id"] = upload_id
    return checked.report


@app.get("/api/readings/{upload_id}")
def get_readings(upload_id: str):
    try:
        return checked_upload(upload_id).report
    except ReadingsError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/analyze")
def analyze(payload: RunInput):
    try:
        with LOCK:
            checked = None
            if payload.upload_id:
                checked = checked_upload(payload.upload_id)
                if payload.sample_forecast:
                    if not checked.report.get("is_sample") or not checked.report.get("sample_ready"):
                        raise HTTPException(409, "Choose a complete, marked synthetic sample for a sample forecast. Actual current estimates require fresh household readings.")
                    if not getattr(source.upload_forecaster, "upload_ready", False):
                        raise HTTPException(503, "The sample forecasting service is unavailable.")
                    forecast, evidence = source.upload_forecaster.predict_readings(checked, sample=True)
                elif not checked.report["data_ready"]:
                    raise HTTPException(409, " ".join(checked.report["blockers"]))
                elif not checked.report["forecaster_available"]:
                    raise HTTPException(503, "Your readings have been checked, but a compatible forecasting service is not connected. No estimate or schedule was generated.")
                else:
                    forecast, evidence = predict_uploaded(checked, source.upload_forecaster)
            else:
                forecast = source.forecast(payload.case_id)
                evidence = source.evidence(forecast, payload.case_id)
            appliances = [Appliance(**a.model_dump()) for a in payload.appliances]
            # Preserve named ontology identities; a renamed appliance cannot detach
            # a known essential identity from its immutable ontology assertion.
            result = optimize_decisions(forecast, appliances, Scenario(**payload.scenario.model_dump()), knowledge)
            result.update(evidence)
        result.update(id=uuid.uuid4().hex, created_at=datetime.now(timezone.utc).isoformat(), case_id=payload.case_id or "upload:"+payload.upload_id,
            source_mode="uploaded_readings" if checked else "connected_history" if source.bridge else "hybrid_history" if forecast.get("forecaster_sha256") else "recorded_forecasts")
        if checked:
            result.update(upload_id=payload.upload_id, uploaded_readings=checked.report,
                          provenance=("Demonstration forecast generated from synthetic sample readings at their original dates. This is not the user's actual current electricity use. " if payload.sample_forecast else "Forecast generated from user-uploaded minute readings. Authenticity and household generalization are not established by timestamps. ")+"Savings are scheduled-job scenario estimates, not measured bills.")
        with sqlite3.connect(DB) as conn:
            conn.execute("INSERT INTO runs VALUES (?, ?, ?)", (result["id"], result["created_at"], json.dumps(result, allow_nan=False)))
        return result
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        print("Analysis error:", error, flush=True)
        raise HTTPException(503, "Analysis is unavailable. Your settings are preserved; please try again.") from error


@app.get("/api/history")
def history():
    with sqlite3.connect(DB) as conn:
        rows = conn.execute("SELECT body FROM runs ORDER BY created_at DESC LIMIT 100").fetchall()
    return [{key: value[key] for key in ("id", "created_at", "forecast", "plan", "case_id", "source_mode")} for (body,) in rows for value in [json.loads(body)]]


@app.get("/api/history/{run_id}")
def get_run(run_id: str):
    with sqlite3.connect(DB) as conn:
        row = conn.execute("SELECT body FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Saved analysis not found.")
    return json.loads(row[0])


@app.post("/api/demo/{run_id}/evidence")
def demo_evidence(run_id: str):
    """Run real frozen-model diagnostics; leave the saved result unchanged."""
    result = get_run(run_id)
    forecaster = source.upload_forecaster
    if getattr(forecaster, "is_hybrid", False) and not result["forecast"].get("forecaster_sha256"):
        forecaster = forecaster.base
    if not getattr(forecaster, "upload_ready", False):
        raise HTTPException(503, "The bundled model runtime is needed for this demonstration. Existing results remain available.")
    from demo_evidence import model_evidence
    try:
        with LOCK:
            if result.get("upload_id"):
                # Explain exact saved inputs even after a current upload becomes
                # stale. This is historical explanation, never a fresh estimate.
                filename, zone, content = upload_content(result["upload_id"])
                report = result["uploaded_readings"]
                checked_at = datetime.fromisoformat(report["checked_at"])
                checked = check_csv(content, filename, zone, model_available=True, now=checked_at)
                if checked.report["sha256"] != report["sha256"]:
                    raise ValueError("Stored readings do not match the saved analysis.")
                inputs = forecaster.inputs(checked, sample=bool(result["forecast"].get("is_sample")))
            else:
                if result["source_mode"] not in ("recorded_forecasts", "hybrid_history"):
                    raise ValueError("This model-specific demo requires the bundled power-history source.")
                index = int(result["case_id"])
                with np.load(forecaster.folder / "prepared_arrays.npz") as archive:
                    target = int(archive["test_indices"][index])
                    inputs = {"history": archive["history_values"][target-4-95:target-4+1][None],
                              "context": archive["context_values"][target][None]}
            return model_evidence(forecaster, inputs, result["forecast"])
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except Exception as error:
        print("Demo evidence error:", error, flush=True)
        raise HTTPException(503, "Model evidence could not be computed. Your forecast and settings are preserved.") from error


@app.get("/api/history/{run_id}/export/{kind}")
def export(run_id: str, kind: str):
    result = get_run(run_id)
    if kind == "json":
        return Response(json.dumps(result, indent=2), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="analysis-{run_id}.json"'})
    if kind not in ("schedule", "recommendations"):
        raise HTTPException(404, "Export format not found.")
    rows = result["plan"]["schedule"] if kind == "schedule" else result["recommendations"]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]) if rows else ["Appliance"])
    writer.writeheader()
    for row in rows:
        # Avoid spreadsheet formula interpretation in user-supplied names.
        writer.writerow({k: "'"+v if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t", "\r")) else v for k, v in row.items()})
    return Response(out.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{kind}-{run_id}.csv"'})


@app.get("/api/health")
def health():
    return {"status": "ok", "source": "connected_history" if source.bridge else "hybrid_history" if getattr(source.upload_forecaster, "is_hybrid", False) else "recorded_forecasts"}


DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
