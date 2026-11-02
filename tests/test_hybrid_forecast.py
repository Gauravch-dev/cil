"""Real newly trained classical head, frozen inference and faithful explanations."""
from pathlib import Path
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("tensorflow")
pytest.importorskip("shap")
import web_api as web
import demo_evidence
from hybrid_forecast import ARTIFACT_ROOT, LocalHybridForecast, SETTINGS, digest, source_fingerprints


@pytest.fixture(scope="module")
def hybrid():
    if not (ARTIFACT_ROOT/"active.json").exists():
        pytest.skip("Prepare the explicit offline hybrid artifact first")
    result = web.source.upload_forecaster
    assert getattr(result, "is_hybrid", False), "Evaluated local hybrid must load successfully"
    return result


@pytest.fixture()
def client(tmp_path, monkeypatch, hybrid):
    database = tmp_path/"hybrid-history.sqlite3"
    web.initialize_database(database)
    monkeypatch.setattr(web, "DB", database)
    with TestClient(web.app) as value:
        yield value


def payload(client):
    boot = client.get("/api/bootstrap").json()
    return {"case_id": boot["cases"][0]["id"], "scenario": boot["scenario"], "appliances": boot["appliances"]}


def test_training_selection_and_provenance_are_real(hybrid):
    m = hybrid.metadata
    assert m["status"] == "evaluated"
    assert m["counts"] == {"train": 20000, "validation": 5000, "test": 18889}
    assert m["selected_name"] == min(m["validation"], key=lambda row: row["RMSE_kW"])["Model"]
    assert m["feature_count"] == 33
    assert m["lstm_retrained"] is False and m["scalers_refitted"] is False
    assert m["source_fingerprints"] == source_fingerprints(hybrid.folder)
    assert digest(hybrid.artifact_folder/m["selected_file"]) == m["selected_sha256"]
    assert {row["Model"] for row in m["validation"]} == set(SETTINGS["candidates"])
    assert all(row["Samples"] == 18889 for row in m["test"] if row["Segment"] == "All test")


def test_loading_and_inference_never_fit_models(hybrid, monkeypatch):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    def forbid(*args, **kwargs):
        raise AssertionError("Frozen runtime cannot fit")
    monkeypatch.setattr(RandomForestRegressor, "fit", forbid)
    monkeypatch.setattr(GradientBoostingRegressor, "fit", forbid)
    loaded = LocalHybridForecast.load_active(hybrid.base)
    with np.load(loaded.artifact_folder/"verification_inputs.npz") as a:
        x = {"history": a["history"], "context": a["context"]}
        np.testing.assert_allclose(loaded.values_kw(x), a["prediction_kw"], atol=1e-6)
        np.testing.assert_allclose(loaded.values_kw(x), np.maximum(loaded.regressor.predict(loaded.features(x)), 0), atol=1e-8)


def test_recorded_history_executes_dl_ml_and_treeshap(client, hybrid, monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("No training or substitute permutation explanation")
    monkeypatch.setattr(hybrid.regressor, "fit", forbid)
    monkeypatch.setattr(hybrid.model, "fit", forbid)
    monkeypatch.setattr(demo_evidence, "grouped_shap", forbid)
    boot = client.get("/api/bootstrap").json()
    assert boot["forecasting"]["hybrid_available"] is True
    result = client.post("/api/analyze", json=payload(client))
    assert result.status_code == 200, result.text
    run = result.json()
    assert run["source_mode"] == "hybrid_history"
    assert run["forecast"]["model"] == hybrid.selected_name
    assert run["forecast"]["forecaster_sha256"] == hybrid.forecaster_sha256
    assert run["explanation"]["method"] == "TreeSHAP"
    before = client.get(f"/api/history/{run['id']}").json()
    response = client.post(f"/api/demo/{run['id']}/evidence")
    assert response.status_code == 200, response.text
    evidence = response.json()
    assert evidence["classical_ml"]["available"] is True
    assert len(evidence["classical_ml"]["latent_values"]) == 16
    shap = evidence["shap"]
    assert shap["method"] == "TreeSHAP" and len(shap["factors"]) == 33
    assert shap["reconstruction_verified"] is True
    assert shap["reference_kw"]+sum(x["value_kw"] for x in shap["factors"]) == pytest.approx(shap["raw_forecast_kw"], abs=1e-5)
    assert client.get(f"/api/history/{run['id']}").json() == before


def test_sample_upload_uses_trained_head_and_keeps_its_dates(client, hybrid):
    sample = client.get("/api/readings/samples/filled").content
    report = client.post("/api/readings?timezone=UTC", content=sample,
                         headers={"Content-Type":"text/csv", "X-File-Name":"sample.csv"}).json()
    p = payload(client)
    p.pop("case_id")
    p.update(upload_id=report["id"])
    assert client.post("/api/analyze", json=p).status_code == 409
    p["sample_forecast"] = True
    result = client.post("/api/analyze", json=p)
    assert result.status_code == 200, result.text
    run = result.json()
    assert run["forecast"]["is_sample"] is True
    assert run["forecast"]["target_end"] == "2026-01-09 01:00:00"
    assert run["forecast"]["forecaster_sha256"] == hybrid.forecaster_sha256
    response = client.post(f"/api/demo/{run['id']}/evidence")
    assert response.status_code == 200, response.text
    assert response.json()["shap"]["method"] == "TreeSHAP"


def test_old_lstm_history_keeps_original_model_and_explanation(client, hybrid, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(web.source, "upload_forecaster", hybrid.base)
        old = client.post("/api/analyze", json=payload(client)).json()
    assert old["source_mode"] == "recorded_forecasts"
    response = client.post(f"/api/demo/{old['id']}/evidence")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["classical_ml"]["available"] is False
    assert result["shap"]["method"] == "Permutation SHAP"
    assert result["forecast_kw"] == pytest.approx(old["forecast"]["forecast_kw"], abs=1e-4)


def test_model_fingerprint_blocks_wrong_head_explanation(client, hybrid, tmp_path):
    import json
    import sqlite3
    run = client.post("/api/analyze", json=payload(client)).json()
    run["forecast"]["forecaster_sha256"] = "wrong-head"
    with sqlite3.connect(web.DB) as conn:
        conn.execute("UPDATE runs SET body=? WHERE id=?", (json.dumps(run), run["id"]))
    response = client.post(f"/api/demo/{run['id']}/evidence")
    assert response.status_code == 409
    assert "different trained prediction head" in response.json()["detail"]
