"""Real grouped SHAP and truthful demo contracts, without modifying user history."""
from pathlib import Path
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("tensorflow")
pytest.importorskip("shap")
import web_api as web
import demo_evidence as evidence


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = tmp_path / "demo.sqlite3"
    web.initialize_database(db)
    monkeypatch.setattr(web, "DB", db)
    active = web.source.upload_forecaster
    monkeypatch.setattr(web.source, "upload_forecaster", getattr(active, "base", active))
    with TestClient(web.app) as value:
        yield value


def make_run(client):
    boot = client.get("/api/bootstrap").json()
    response = client.post("/api/analyze", json={"case_id": boot["cases"][0]["id"],
                           "scenario": boot["scenario"], "appliances": boot["appliances"]})
    assert response.status_code == 200
    return response.json()


def test_demo_reexecutes_model_and_real_shap_reconstructs(client, monkeypatch):
    run = make_run(client)
    model = web.source.upload_forecaster
    assert model is not None
    def no_fit(*args, **kwargs):
        raise AssertionError("Demo must not fit any model or scaler")
    monkeypatch.setattr(model.model, "fit", no_fit)
    monkeypatch.setattr(model.power_scaler, "fit", no_fit)
    response = client.post(f"/api/demo/{run['id']}/evidence")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["prediction_reexecuted"] is True
    assert result["forecast_kw"] == pytest.approx(run["forecast"]["forecast_kw"], abs=1e-4)
    assert result["history_shape"] == [1, 96, 6]
    assert result["context_shape"] == [1, 17]
    assert np.asarray(result["activation"]).shape == (32, 96)
    exp = result["shap"]
    assert exp["available"] and exp["method"] == "Permutation SHAP"
    assert len(exp["factors"]) == 23
    assert exp["reconstruction_verified"] is True
    assert exp["reference_kw"] + sum(x["value_kw"] for x in exp["factors"]) == pytest.approx(result["forecast_kw"], abs=1e-4)
    assert result["classical_ml"]["available"] is False
    assert client.get(f"/api/history/{run['id']}").json() == run
    assert client.post(f"/api/demo/{run['id']}/evidence").json() == result


def test_demo_missing_run_and_unavailable_runtime(client, monkeypatch):
    assert client.post("/api/demo/missing/evidence").status_code == 404
    run = make_run(client)
    monkeypatch.setattr(web.source, "upload_forecaster", None)
    response = client.post(f"/api/demo/{run['id']}/evidence")
    assert response.status_code == 503
    assert "runtime" in response.json()["detail"]


def test_explanation_cannot_be_attached_to_another_prediction():
    model = web.source.upload_forecaster
    with np.load(model.folder / "prepared_arrays.npz") as a:
        t = int(a["test_indices"][0])
        inputs = {"history": a["history_values"][t-99:t-3][None], "context": a["context_values"][t][None]}
    with pytest.raises(ValueError, match="does not reproduce"):
        evidence.model_evidence(model, inputs, {"forecast_kw": 999})
    with pytest.raises(ValueError, match="different model"):
        evidence.model_evidence(model, inputs, {"forecast_kw": 999, "model_sha256": "wrong"})


def test_shap_failure_is_explicit_and_not_replaced_with_saved_factors(client, monkeypatch):
    run = make_run(client)
    evidence.CACHE.clear()
    def unavailable(*args):
        raise RuntimeError("Deliberate unavailable SHAP fixture")
    monkeypatch.setattr(evidence, "grouped_shap", unavailable)
    response = client.post(f"/api/demo/{run['id']}/evidence")
    assert response.status_code == 200
    assert response.json()["prediction_reexecuted"] is True
    exp = response.json()["shap"]
    assert exp["available"] is False and "factors" not in exp
    assert "substituted" in exp["reason"]
