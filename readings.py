"""Validate local meter CSVs and build causal inputs for the frozen Task 9 bridge.

Uploaded CSV is data only. The three preprocessing functions are extracted from
the trusted repository source, never from an upload, and do not train anything.
"""
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import io
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

FEATURES = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
            "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
UNITS = ["kW", "kVAR", "V", "A", "Wh per minute", "Wh per minute", "Wh per minute"]
MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 100_000
MIN_HOURS = 8 * 24
MAX_SPAN_DAYS = 60
MAX_AGE_MINUTES = 30
PREPROCESSOR = Path(__file__).resolve().parent / "task7_final_peak_gated_multitask_lstm.py"


class ReadingsError(ValueError):
    """Actionable invalid-file error, with no model inference or persistence."""


@dataclass
class CheckedReadings:
    report: dict
    power: pd.DataFrame
    context: pd.DataFrame
    origin: pd.Timestamp | None


@lru_cache(maxsize=1)
def preprocessors():
    text = PREPROCESSOR.read_text(encoding="utf-8")
    tree = ast.parse(text)
    names = {"prepare_multivariate", "calendar_features", "build_context"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError("Required frozen preprocessing functions are unavailable.")
    namespace = {"np": np, "pd": pd, "TARGET_FEATURE": FEATURES[0], "ELECTRICAL_FEATURES": FEATURES}
    # Only locally versioned function definitions, without top-level training code.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PREPROCESSOR), "exec"), namespace)
    return namespace


def schema():
    return {"columns": [{"name": "Datetime", "unit": "Local minute timestamp: YYYY-MM-DD HH:MM:00"},
                        *[{"name": name, "unit": unit} for name, unit in zip(FEATURES, UNITS)]],
            "minimum_days": 8, "maximum_days": MAX_SPAN_DAYS, "maximum_bytes": MAX_BYTES,
            "maximum_age_minutes": MAX_AGE_MINUTES, "interval": "One row per minute",
            "template_url": "/api/readings/template"}


def read_meter_csv(content):
    """Read common Excel CSV encodings/delimiters without guessing measurements."""
    import csv
    try:
        encoding = "utf-16" if content.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        text = content.decode(encoding).lstrip("\ufeff\r\n ")
    except UnicodeDecodeError:
        raise ReadingsError("Save the file as UTF-8 CSV or Unicode CSV, then upload it again.") from None
    if text.lower().startswith(("<!doctype html", "<html")):
        raise ReadingsError("This file contains a webpage, not electricity readings. Download a new CSV from Sample templates, or choose Use filled sample directly on the website.")
    if text.startswith("CURRENT - ELECTRICITY READINGS TEMPLATES"):
        raise ReadingsError("This is the instructions guide, not the readings CSV. Choose Blank CSV template or Filled sample from Sample templates.")
    notes = []
    directive = text.splitlines()[0] if text.splitlines() else ""
    explicit = directive[4:] if directive.lower().startswith("sep=") and directive[4:] in (",", ";", "\t") else None
    if explicit:
        text = text[len(directive):].lstrip("\r\n")
        notes.append("The Excel separator declaration was read.")
    known = {name.casefold(): name for name in ["Datetime", *FEATURES, "Data_source"]}
    def clean(name):
        stripped = name.strip().lstrip("\ufeff")
        return known.get(stripped.casefold(), stripped)
    candidates = []
    for delimiter in ([explicit] if explicit else [",", ";", "\t"]):
        try:
            raw_header = next(csv.reader(io.StringIO(text), delimiter=delimiter, strict=True))
            header = [clean(name) for name in raw_header]
            score = sum(name in ["Datetime", *FEATURES] for name in header)
            candidates.append((score, delimiter, raw_header, header))
        except (StopIteration, csv.Error):
            continue
    if not candidates:
        raise ReadingsError("The CSV header could not be read. Download a fresh CSV from Sample templates.")
    _, delimiter, raw_header, header = max(candidates, key=lambda item: item[0])
    if len(header) != len(set(header)):
        raise ReadingsError("The CSV has duplicate column names. Each required column must appear once.")
    missing = [name for name in ["Datetime", *FEATURES] if name not in header]
    if missing:
        if len(missing) == len(FEATURES)+1:
            if len(header) == 1 and all(name in header[0] for name in ["Datetime", *FEATURES]):
                raise ReadingsError("The whole header is stored in one spreadsheet cell. Split it into separate columns or download a fresh CSV. You can also choose Use filled sample directly on the website.")
            raise ReadingsError("This file does not contain the electricity-reading columns. Use filled sample directly on the website, or download a fresh CSV from Sample templates. The text guide and saved webpages cannot be uploaded as readings.")
        raise ReadingsError("Missing columns: " + ", ".join(missing) + ". Use the CSV template and preserve its column names.")
    try:
        frame = pd.read_csv(io.StringIO(text), sep=delimiter, header=0, names=header, nrows=MAX_ROWS+1, dtype=str)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        raise ReadingsError("This CSV could not be read. Use one reading per row, with a separate column for each measurement.") from None
    if len(frame) == 0:
        raise ReadingsError("This is a blank template with no readings yet. Add your actual meter readings, or choose Use filled sample on the website to explore an example.")
    if len(frame) > MAX_ROWS:
        raise ReadingsError("Supply no more than 100,000 readings, covering at most 60 days.")
    if delimiter != ",":
        notes.append("Semicolon-separated CSV was read." if delimiter == ";" else "Tab-separated CSV was read.")
    if header != raw_header:
        notes.append("Header spacing and capitalization were normalized; measurement meanings and units were not changed.")
    if encoding == "utf-16":
        notes.append("The Unicode/UTF-16 CSV encoding was read.")
    return frame, notes


def check_csv(content: bytes, file_name: str, timezone_name: str, *, model_available=False,
              now: datetime | None = None) -> CheckedReadings:
    if not content or len(content) > MAX_BYTES:
        raise ReadingsError("Choose a non-empty CSV file no larger than 10 MB.")
    if not file_name.lower().endswith(".csv"):
        raise ReadingsError("Choose a .csv file. Excel workbooks must first be exported as CSV.")
    try:
        zone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ReadingsError("Choose a valid timezone for the timestamps in your meter file.") from None
    frame, import_notes = read_meter_csv(content)
    is_sample = "Data_source" in frame.columns and frame["Data_source"].fillna("").str.strip().str.lower().eq("synthetic_sample").any()
    raw_stamps = frame["Datetime"].fillna("").str.strip()
    if not raw_stamps.str.match(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::00(?:\.0+)?)?$").all():
        raise ReadingsError("Use minute timestamps such as 2026-10-31 10:15:00, without timezone suffixes. Select the file’s timezone separately.")
    stamps = pd.to_datetime(raw_stamps, format="ISO8601", errors="coerce")
    if stamps.isna().any():
        raise ReadingsError("Some timestamps are invalid. Check the dates and use YYYY-MM-DD HH:MM:00.")
    if stamps.duplicated().any():
        raise ReadingsError("Duplicate timestamps were found. Keep one household reading per minute.")
    if (stamps.max()-stamps.min()) > pd.Timedelta(days=MAX_SPAN_DAYS):
        raise ReadingsError("The date range is too large. Upload the most recent eight to 60 days.")
    try:
        aware = pd.DatetimeIndex(stamps).tz_localize(zone.key, ambiguous="raise", nonexistent="raise")
    except Exception:
        raise ReadingsError("Some local timestamps are ambiguous or do not exist because of a clock change. Export UTC timestamps and choose UTC.") from None
    now_stamp = pd.Timestamp(now or datetime.now(timezone.utc))
    if now_stamp.tzinfo is None:
        raise ValueError("The validation clock must include a timezone.")
    now_utc = now_stamp.tz_convert("UTC")
    if aware.max().tz_convert("UTC") > now_utc.floor("min"):
        raise ReadingsError("The file contains future readings. Check the meter clock and selected timezone.")
    values = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    valid_values = np.isfinite(values) & values.ge(0)
    invalid = int((~valid_values).sum().sum())
    minute = values.where(valid_values)
    minute.index = pd.DatetimeIndex(stamps)
    was_sorted = minute.index.is_monotonic_increasing
    minute = minute.sort_index()
    funcs = preprocessors()
    power = funcs["prepare_multivariate"](minute, FEATURES, 4)
    now_local = now_utc.tz_convert(zone).tz_localize(None)
    completed = power.index <= now_local.floor("15min")
    fully_observed = power[["observed_"+name for name in FEATURES]].eq(1).all(axis=1)
    available_origins = power.index[completed & fully_observed]
    origin = available_origins[-1] if len(available_origins) else None
    context = pd.DataFrame()
    blockers, warnings, checks = [], list(import_notes), []
    if is_sample:
        message = "This file is a synthetic format example. Upload actual meter readings in the blank template for a current estimate."
        blockers.append(message)
        checks.append({"code":"sample_data", "label":"Synthetic sample file", "status":"fail", "message":message})
    hours = (minute.index[-1]-minute.index[0]).total_seconds()/3600 + 1/60
    missing_minutes = max(0, int((minute.index[-1]-minute.index[0]).total_seconds()/60)+1-len(minute))
    enough = hours >= MIN_HOURS-1e-8
    checks.append({"code": "history", "label": "Enough history", "status": "pass" if enough else "fail",
                   "message": f"{hours/24:.2f} days supplied; at least eight days are required."})
    if not enough:
        blockers.append("Upload at least eight days of minute readings, including the most recent complete 15-minute interval.")
    input_ready = False
    age = None
    fresh = False
    if origin is None:
        blockers.append("No complete 15-minute interval has all seven measurements. Each interval needs 15 valid minute readings.")
        checks.append({"code": "inputs", "label": "Usable measurements", "status": "fail", "message": blockers[-1]})
    else:
        power = power.loc[:origin].copy()
        # Append empty target slots solely for known future calendar features.
        # No future meter values or actual target values are invented or required.
        expanded = power.reindex(pd.date_range(start=power.index[0], end=origin+pd.Timedelta(hours=1), freq="15min"))
        context = funcs["build_context"](expanded, 4)
        history_window = power.tail(96)[["input_"+name for name in FEATURES]]
        input_ready = len(history_window) == 96 and np.isfinite(history_window.to_numpy()).all() and np.isfinite(context.iloc[-1].to_numpy()).all()
        checks.append({"code": "inputs", "label": "Usable measurements", "status": "pass" if input_ready else "fail",
                       "message": "Recent readings and daily/weekly context are complete." if input_ready else "Required recent or daily/weekly measurements are missing. Supply a more complete history."})
        if not input_ready:
            blockers.append("Required readings remain missing after the original limited, past-only gap filling. Supply a more complete meter export.")
        origin_utc = origin.tz_localize(zone.key, ambiguous="raise", nonexistent="raise").tz_convert("UTC")
        age = max(0, (now_utc-origin_utc).total_seconds()/60)
        fresh = age <= MAX_AGE_MINUTES
        checks.append({"code": "freshness", "label": "Recent enough for a current estimate", "status": "pass" if fresh else "fail",
                       "message": f"Latest usable interval ended {age:.0f} minutes ago. The limit is {MAX_AGE_MINUTES} minutes."})
        if not fresh:
            blockers.append("These readings are too old for a current estimate. Export a file ending within the last 30 minutes.")
    if invalid or missing_minutes:
        warnings.append(f"{invalid:,} missing/invalid measurement cells and {missing_minutes:,} missing minute rows were found. Only the original limited past-only gap filling is used.")
    if not was_sorted:
        warnings.append("Readings were sorted into timestamp order.")
    if origin is not None and minute.index[-1] >= origin:
        warnings.append("The final incomplete interval is excluded. The forecast starts from the latest complete interval.")
    if not model_available:
        blockers.append("Your file can be checked, but the forecasting service for new readings is not configured. No current estimate has been generated.")
    checks.append({"code": "service", "label": "Forecasting connection", "status": "pass" if model_available else "fail",
                   "message": "A compatible frozen forecasting service is connected." if model_available else "A compatible forecasting service must be connected before an estimate can be generated."})
    data_ready = bool(enough and input_ready and fresh and not is_sample)
    preview = []
    for stamp, row in minute.head(5).iterrows():
        preview.append({"Datetime": str(stamp), **{name: float(row[name]) if pd.notna(row[name]) else None for name in FEATURES}})
    history = [{"time": str(stamp), "kw": float(row.input_kw) if pd.notna(row.input_kw) else None} for stamp, row in power.tail(96).iterrows()] if origin is not None else []
    return CheckedReadings({"file_name": Path(file_name.replace("\\", "/")).name[:160], "byte_size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(), "timezone": zone.key, "row_count": len(minute),
        "start": str(minute.index[0]), "end": str(minute.index[-1]), "days_covered": round(hours/24, 3),
        "origin": str(origin) if origin is not None else None,
        "target_interval_start": str(origin+pd.Timedelta(minutes=45)) if origin is not None else None,
        "target_end": str(origin+pd.Timedelta(hours=1)) if origin is not None else None,
        "age_minutes": round(age, 1) if age is not None else None, "invalid_cells": invalid,
        "missing_minutes": missing_minutes, "is_sample": bool(is_sample), "data_ready": data_ready, "forecaster_available": model_available,
        "can_analyze": bool(data_ready and model_available),
        "sample_ready": bool(is_sample and enough and input_ready and model_available),
        "status": "ready" if data_ready and model_available else "needs_service" if data_ready else "needs_readings",
        "checked_at": now_utc.isoformat(), "checks": checks, "blockers": blockers, "warnings": warnings,
        "preview": preview, "history": history}, power, context, origin)


def forecast_available(bridge):
    if bridge is None:
        return False
    if getattr(bridge, "upload_ready", False):
        return True
    required = ["electrical_scalers.joblib", "context_scaler.joblib"]
    manifest = bridge.manifest
    return (all((bridge.folder/name).is_file() for name in required)
            and manifest.get("electrical_features") == FEATURES
            and manifest.get("sequence_length") == 96 and manifest.get("horizon_steps") == 4)


def frozen_inputs(checked: CheckedReadings, bridge):
    """Transform only. Never fit/refit a scaler, regressor or network."""
    import joblib
    if not checked.report["data_ready"] or checked.origin is None:
        raise ReadingsError("The readings are not ready for a current estimate.")
    if not forecast_available(bridge):
        raise ReadingsError("A compatible forecasting service for uploaded readings is not connected.")
    if list(checked.context.columns) != bridge.manifest["context_features"]:
        raise ReadingsError("The supplied forecasting service has incompatible input preparation.")
    scalers = joblib.load(bridge.folder/"electrical_scalers.joblib")
    context_scaler = joblib.load(bridge.folder/"context_scaler.joblib")
    power = checked.power.tail(96)
    expected = [*(name+"_scaled" for name in FEATURES), "electrical_observed_fraction", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    if expected != bridge.manifest["history_features"]:
        raise ReadingsError("The supplied forecasting service has incompatible measurement channels.")
    scaled = [scalers[name].transform(power[["input_"+name]].to_numpy()).ravel() for name in FEATURES]
    calendar = preprocessors()["calendar_features"](power.index)
    history = np.column_stack([*scaled, power.electrical_observed_fraction.to_numpy(),
                              calendar[["hour_sin", "hour_cos", "dow_sin", "dow_cos"]].to_numpy()]).astype("float32")
    context = context_scaler.transform(checked.context.tail(1)).astype("float32")
    inputs = {"history": history[None, ...], "context": context}
    if not all(np.isfinite(array).all() for array in inputs.values()):
        raise ReadingsError("Required measurements are unavailable after input preparation.")
    return inputs


def predict_uploaded(checked: CheckedReadings, bridge):
    if getattr(bridge, "upload_ready", False):
        return bridge.predict_readings(checked)
    inputs = frozen_inputs(checked, bridge)
    latent = np.asarray(bridge.encoder([inputs[key] for key in bridge.input_order], training=False))
    features = np.concatenate([latent, inputs["context"]], axis=1)
    raw = float(bridge.regressor.predict(features)[0])
    if not np.isfinite(raw):
        raise ReadingsError("The forecasting service did not return a valid estimate.")
    report = checked.report
    forecast = {"origin": report["origin"], "target_end": report["target_end"],
        "target_interval_start": report["target_interval_start"], "forecast_kw": max(raw, 0.),
        "raw_forecast_kw": raw, "threshold_kw": float(bridge.threshold), "model": bridge.selected_name,
        "source": "uploaded_readings", "timezone": report["timezone"],
        "units": "kW, average over the target 15-minute interval"}
    explanation = {"available": False, "reason": "A forecast contribution breakdown is unavailable for these readings. Household rule evidence remains available."}
    try:
        import shap
        explainer = shap.TreeExplainer(bridge.regressor)
        values = np.asarray(explainer.shap_values(features)).reshape(-1)
        base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        np.testing.assert_allclose(base+values.sum(), raw, rtol=1e-5, atol=1e-5)
        explanation = {"available": True, "method": "SHAP", "reference_kw": base,
            "raw_forecast_kw": raw, "reconstruction_verified": True,
            "factors": [{"name": name, "value_kw": float(value)} for name, value in zip(bridge.feature_names, values)],
            "note": "Contributions explain this forecast relative to its reference, not physical causes."}
    except Exception:
        pass  # Missing explanation never substitutes archived or invented factors.
    return forecast, {"history": report["history"], "explanation": explanation,
        "confidence": {"available": False, "label": "Not quantified", "reason": "The forecasting service does not supply a calibrated prediction interval."}}
