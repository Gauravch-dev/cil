"""Real frozen-model fidelity; fixtures are not current household forecasts."""
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("tensorflow")
import web_api as web
from readings import CheckedReadings, ReadingsError, check_csv, forecast_available, predict_uploaded


@pytest.fixture(scope="module")
def forecaster():
    model = web.source.upload_forecaster
    assert model is not None, "Installed forecasting runtime must connect the real bundled model"
    assert forecast_available(model)
    return model


def test_original_preprocessing_reproduces_saved_inputs_and_forecasts(forecaster):
    power = pd.read_csv(forecaster.folder / "power_15min.csv", index_col="Interval_end", parse_dates=True)
    saved = pd.read_csv(forecaster.folder / "test_predictions.csv")
    with np.load(forecaster.folder / "prepared_arrays.npz") as arrays:
        for position in [0, len(saved)//2, len(saved)-1]:
            row = saved.iloc[position]
            origin = pd.Timestamp(row.Forecast_origin)
            checked = CheckedReadings({"data_ready": True}, power.loc[origin-pd.Timedelta(days=8):origin], pd.DataFrame(), origin)
            inputs = forecaster.inputs(checked)
            target = int(arrays["test_indices"][position])
            np.testing.assert_allclose(inputs["history"][0], arrays["history_values"][target-4-95:target-4+1], atol=1e-6)
            np.testing.assert_allclose(inputs["context"][0], arrays["context_values"][target], atol=1e-6)
            assert float(forecaster.values_kw(inputs)[0]) == pytest.approx(row.lstm_compact_kW, abs=1e-4)
    assert forecaster.fidelity_max_error_kw < 1e-4


def test_real_model_runs_upload_route_without_archived_prediction(tmp_path, monkeypatch, forecaster):
    from fastapi.testclient import TestClient
    from readings import FEATURES
    now = datetime(2026, 11, 2, 10, 7, tzinfo=timezone.utc)
    stamps = pd.date_range(end="2026-11-02 09:59", periods=8*24*60, freq="min")
    frame = pd.DataFrame({"Datetime": stamps.astype(str), **{n: v for n, v in zip(FEATURES, [1.1, .2, 230, 4.8, 1, 2, 3])}})
    database = tmp_path / "history.sqlite3"
    web.initialize_database(database)
    monkeypatch.setattr(web, "DB", database)
    monkeypatch.setattr(web, "check_csv", lambda *a, **kw: check_csv(*a, **kw, now=now))
    def no_replay(*args):
        raise AssertionError("An uploaded forecast cannot replay a saved prediction")
    monkeypatch.setattr(web.source, "forecast", no_replay)
    # Any attempted scaler/model fitting must fail this check.
    monkeypatch.setattr(forecaster.model, "fit", no_replay)
    monkeypatch.setattr(forecaster.power_scaler, "fit", no_replay)
    monkeypatch.setattr(forecaster.context_scaler, "fit", no_replay)
    with TestClient(web.app) as client:
        boot = client.get("/api/bootstrap").json()
        assert boot["readings"]["service_available"] is True
        report = client.post("/api/readings?timezone=UTC", content=frame.to_csv(index=False).encode(),
                             headers={"Content-Type": "text/csv", "X-File-Name": "synthetic-inference-contract.csv"}).json()
        assert report["can_analyze"] is True
        response = client.post("/api/analyze", json={"upload_id": report["id"], "scenario": boot["scenario"], "appliances": boot["appliances"]})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["source_mode"] == "uploaded_readings"
        assert result["forecast"]["source"] == "uploaded_readings"
        assert result["forecast"]["target_end"] == "2026-11-02 11:00:00"
        assert result["forecast"]["model_sha256"] == forecaster.model_sha256
        assert result["forecast"]["forecast_kw"] >= 0
        assert result["confidence"]["available"] is False
        assert result["plan"]["status"] == "optimal"
        assert client.get("/api/history").json()[0]["id"] == result["id"]


def test_stale_and_sample_files_stay_blocked_with_real_service(forecaster):
    sample = (web.ROOT / "frontend/public/templates/SYNTHETIC-electricity-readings-8-days.csv").read_bytes()
    checked = check_csv(sample, "sample.csv", "UTC", model_available=True)
    assert checked.report["is_sample"] is True
    assert checked.report["can_analyze"] is False
    with pytest.raises(ValueError, match="Recent household"):
        predict_uploaded(checked, forecaster)


def test_sample_requires_explicit_demo_and_retains_original_dates(tmp_path, monkeypatch, forecaster):
    from fastapi.testclient import TestClient
    database = tmp_path / "history.sqlite3"
    web.initialize_database(database)
    monkeypatch.setattr(web, "DB", database)
    with TestClient(web.app) as client:
        boot = client.get("/api/bootstrap").json()
        sample = client.get("/api/readings/samples/filled").content
        report = client.post("/api/readings?timezone=UTC", content=sample,
                             headers={"Content-Type": "text/csv", "X-File-Name": "sample.csv"}).json()
        assert report["can_analyze"] is False
        assert report["sample_ready"] is True
        request = {"upload_id": report["id"], "scenario": boot["scenario"], "appliances": boot["appliances"]}
        assert client.post("/api/analyze", json=request).status_code == 409
        assert client.get("/api/history").json() == []
        result = client.post("/api/analyze", json={**request, "sample_forecast": True})
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["forecast"]["is_sample"] is True
        assert body["forecast"]["origin"] == "2026-01-09 00:00:00"
        assert body["forecast"]["target_end"] == "2026-01-09 01:00:00"
        assert "not the user's actual current" in body["provenance"]
        assert client.get(f"/api/history/{body['id']}/export/json").json()["forecast"]["is_sample"] is True
