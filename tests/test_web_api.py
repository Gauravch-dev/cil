"""Adapter contracts and policy/cost evidence, with the real saved forecast data.

Test-generated scenarios use a temporary database, never the user's saved history.
"""
import copy
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import web_api as web


@pytest.fixture()
def client(tmp_path, monkeypatch):
    database = tmp_path / "history.sqlite3"
    web.initialize_database(database)
    monkeypatch.setattr(web, "DB", database)
    # These route contracts deliberately exercise unavailable-service behavior.
    # Real bundled-model inference is checked separately in test_local_forecast.
    monkeypatch.setattr(web.source, "upload_forecaster", None)
    with TestClient(web.app) as client:
        yield client


def payload(client, explained=True):
    setup = client.get("/api/bootstrap").json()
    case = next(c for c in setup["cases"] if c["explanation"] == explained)
    return {"case_id": case["id"], "scenario": setup["scenario"], "appliances": setup["appliances"]}


def test_recorded_forecast_matches_source_and_explanation(client):
    request = payload(client)
    response = client.post("/api/analyze", json=request)
    assert response.status_code == 200
    result = response.json()
    source = web.source.saved.iloc[int(request["case_id"])]
    assert result["forecast"]["forecast_kw"] == float(source.lstm_compact_kW)
    assert result["confidence"]["available"] is False
    assert result["source_mode"] == "recorded_forecasts"
    assert result["explanation"]["method"] == "Integrated gradients"
    assert result["explanation"]["reconstruction_verified"] is True
    assert all(point["time"] <= result["forecast"]["origin"] for point in result["history"])
    reference = result["explanation"]["reference_kw"]
    total = sum(f["value_kw"] for f in result["explanation"]["factors"])
    assert abs(reference + total-result["forecast"]["forecast_kw"]) < .02


def test_unavailable_explanation_not_fabricated(client):
    result = client.post("/api/analyze", json=payload(client, False)).json()
    assert result["explanation"]["available"] is False
    assert "factors" not in result["explanation"]


def test_tariff_costs_and_forecast_invariance(client):
    request = payload(client)
    # Choose a genuinely recorded evening origin, rather than creating a forecast.
    request["case_id"] = next(c["id"] for c in web.source.cases if 17 <= int(c["origin"][11:13]) < 20)
    first = client.post("/api/analyze", json=request).json()
    modified = copy.deepcopy(request)
    modified["scenario"].update(peak_rate=30, offpeak_rate=2, occupancy=4, temperature_c=40)
    second = client.post("/api/analyze", json=modified).json()
    assert first["forecast"] == second["forecast"]
    assert first["plan"]["estimated_cost"] != second["plan"]["estimated_cost"]
    for result in [first, second]:
        plan = result["plan"]
        assert plan["status"] == "optimal"
        assert plan["estimated_saving"] == pytest.approx(plan["baseline_cost"]-plan["estimated_cost"])
        scenario = web.Scenario(**result["scenario"])
        import pandas as pd
        from task9_core import rate_at
        for job in plan["schedule"]:
            timestamps = pd.date_range(job["Start"], job["End"], freq="15min", inclusive="left")
            independent = sum(job["Power_W"]/1000*.25*rate_at(t, scenario) for t in timestamps)
            assert job["Estimated_cost"] == pytest.approx(independent)


def test_essential_ontology_cannot_be_overridden(client):
    request = payload(client)
    request["scenario"]["occupancy"] = 0
    request["appliances"][0].update(essential=False, flexible=True, requested=False, on=False)
    result = client.post("/api/analyze", json=request).json()
    fridge = result["recommendations"][0]
    assert fridge["Action"] == "Keep ON"
    assert "R1-essential" in fridge["Rule_trace"]
    assert fridge["Essential"] is True
    assert result["recommendations"][1]["Action"] == "Turn OFF"


def test_infeasible_deadline_has_no_cost_or_savings(client):
    request = payload(client)
    request["appliances"][2].update(duration_min=60, deadline_min=30)
    result = client.post("/api/analyze", json=request).json()
    assert result["plan"]["status"] == "infeasible"
    assert result["plan"]["estimated_saving"] is None
    assert result["plan"]["estimated_cost"] is None
    assert not result["plan"]["schedule"]


@pytest.mark.parametrize("alter", ["invalid_case", "string_boolean", "fractional_minutes", "negative_tariff", "too_many_nodes", "bad_step"])
def test_invalid_inputs_rejected_without_saving(client, alter):
    request = payload(client)
    if alter == "invalid_case": request["case_id"] = "not-a-case"
    if alter == "string_boolean": request["appliances"][0]["essential"] = "false"
    if alter == "fractional_minutes": request["appliances"][2]["duration_min"] = 15.2
    if alter == "negative_tariff": request["scenario"]["peak_rate"] = -1
    if alter == "too_many_nodes": request["scenario"]["node_limit"] = 50001
    if alter == "bad_step": request["appliances"][2]["duration_min"] = 16
    assert client.post("/api/analyze", json=request).status_code == 422
    assert client.get("/api/history").json() == []


def test_history_restore_exports_and_missing_record(client):
    result = client.post("/api/analyze", json=payload(client)).json()
    run_id = result["id"]
    assert client.get("/api/history").json()[0]["id"] == run_id
    assert client.get(f"/api/history/{run_id}").json() == result
    assert client.get(f"/api/history/{run_id}/export/json").json() == result
    csv = client.get(f"/api/history/{run_id}/export/schedule")
    assert csv.status_code == 200 and "Energy_kWh" in csv.text
    assert "attachment" in csv.headers["content-disposition"]
    assert client.get("/api/history/missing").status_code == 404
    assert client.get(f"/api/history/{run_id}/export/unknown").status_code == 404


def test_cross_origin_write_rejected(client):
    assert client.post("/api/analyze", json=payload(client), headers={"Origin":"https://unrelated.example"}).status_code == 403


def test_csv_formula_injection_is_neutralized(client):
    request = payload(client)
    request["appliances"][2]["name"] = "=2+2"
    result = client.post("/api/analyze", json=request).json()
    csv = client.get(f"/api/history/{result['id']}/export/schedule").text
    assert "'=2+2" in csv


@pytest.fixture()
def fresh_upload(monkeypatch):
    """Explicit synthetic upload fixture; no real household or model result."""
    from datetime import datetime, timezone
    import pandas as pd
    from readings import FEATURES, check_csv
    now = datetime(2026, 10, 31, 10, 7, tzinfo=timezone.utc)
    stamps = pd.date_range(end="2026-10-31 09:59", periods=8*24*60, freq="min")
    frame = pd.DataFrame({"Datetime": stamps.astype(str), **{name: value for name, value in zip(FEATURES, [1.1,.2,230,4.8,1,2,3])}})
    monkeypatch.setattr(web, "check_csv", lambda *args, **kwargs: check_csv(*args, **kwargs, now=now))
    return frame.to_csv(index=False).encode()


def post_readings(client, content):
    return client.post("/api/readings?timezone=UTC", content=content,
                       headers={"Content-Type":"text/csv", "X-File-Name":"synthetic-local-api-fixture.csv"})


def test_upload_template_and_checked_file_persist_without_forecast(client, fresh_upload):
    template = client.get("/api/readings/template")
    assert template.status_code == 200
    assert template.text.startswith("Datetime,Global_active_power,")
    assert len(template.text.strip().splitlines()) == 1  # No invented example readings.
    response = post_readings(client, fresh_upload)
    assert response.status_code == 200
    checked = response.json()
    assert checked["data_ready"] and not checked["can_analyze"]
    assert checked["status"] == "needs_service"
    assert client.get(f"/api/readings/{checked['id']}").json()["sha256"] == checked["sha256"]
    assert client.get("/api/history").json() == []
    assert client.get("/api/readings/missing").status_code == 404


def test_current_analysis_never_substitutes_recorded_forecast(client, fresh_upload, monkeypatch):
    checked = post_readings(client, fresh_upload).json()
    request = payload(client)
    del request["case_id"]
    request["upload_id"] = checked["id"]
    def forbid_replay(*args):
        raise AssertionError("A current estimate must not replay historical forecasts")
    monkeypatch.setattr(web.source, "forecast", forbid_replay)
    response = client.post("/api/analyze", json=request)
    assert response.status_code == 503
    assert "No estimate or schedule" in response.json()["detail"]
    assert client.get("/api/history").json() == []


def test_current_file_rechecked_at_analysis_time(client, fresh_upload, monkeypatch):
    checked = post_readings(client, fresh_upload).json()
    from readings import check_csv
    from datetime import datetime, timezone
    monkeypatch.setattr(web, "check_csv", lambda *args, **kwargs: check_csv(*args, **kwargs, now=datetime(2026,10,31,12,tzinfo=timezone.utc)))
    request = payload(client)
    del request["case_id"]
    request["upload_id"] = checked["id"]
    assert client.get(f"/api/readings/{checked['id']}").json()["data_ready"] is False
    response = client.post("/api/analyze", json=request)
    assert response.status_code == 409
    assert "too old" in response.json()["detail"]
    assert client.get("/api/history").json() == []


def test_uploaded_result_keeps_its_own_provenance(client, fresh_upload, monkeypatch):
    """Routing/persistence contract with a deliberately stubbed test forecaster."""
    monkeypatch.setattr(web, "forecast_available", lambda bridge: True)
    def fixture_prediction(checked, bridge):
        r = checked.report
        return {"origin":r["origin"], "target_interval_start":r["target_interval_start"], "target_end":r["target_end"],
                "forecast_kw":1.23,"threshold_kw":2.5,"model":"synthetic API contract fixture","source":"uploaded_readings"}, {
                "history":r["history"],"explanation":{"available":False,"reason":"Synthetic contract fixture"},
                "confidence":{"available":False,"label":"Not quantified","reason":"Synthetic contract fixture"}}
    monkeypatch.setattr(web, "predict_uploaded", fixture_prediction)
    checked = post_readings(client, fresh_upload).json()
    assert checked["can_analyze"] is True
    request = payload(client)
    del request["case_id"]
    request["upload_id"] = checked["id"]
    response = client.post("/api/analyze", json=request)
    assert response.status_code == 200
    result = response.json()
    assert result["source_mode"] == "uploaded_readings"
    assert result["upload_id"] == checked["id"]
    assert result["uploaded_readings"]["sha256"] == checked["sha256"]
    assert result["forecast"]["target_end"] == "2026-10-31 11:00:00"
    assert result["plan"]["status"] == "optimal"
    assert client.get(f"/api/history/{result['id']}").json() == result


def test_upload_rejects_bad_format_and_ambiguous_source(client, fresh_upload):
    assert client.post("/api/readings", content=b"{}", headers={"Content-Type":"application/json"}).status_code == 415
    assert post_readings(client, b"Datetime,Voltage\n2026-10-31 09:59:00,230\n").status_code == 422
    assert post_readings(client, b"x"*(web.MAX_BYTES+1)).status_code == 413
    checked = post_readings(client, fresh_upload).json()
    request = payload(client)
    request["upload_id"] = checked["id"]
    assert client.post("/api/analyze", json=request).status_code == 422
    del request["upload_id"]
    del request["case_id"]
    assert client.post("/api/analyze", json=request).status_code == 422


def test_filled_sample_download_upload_roundtrip(client):
    response=client.get("/api/readings/samples/filled")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="SYNTHETIC-electricity-readings-8-days.csv"' == response.headers["content-disposition"]
    assert response.content.startswith(b"Datetime,Global_active_power,")
    uploaded=post_readings(client,response.content)
    assert uploaded.status_code == 200
    assert uploaded.json()["row_count"] == 11520
    assert uploaded.json()["is_sample"] is True
    assert uploaded.json()["can_analyze"] is False
    guide=client.get("/api/readings/samples/guide")
    assert guide.status_code == 200
    assert guide.headers["content-type"].startswith("text/plain")
    assert client.get("/api/readings/samples/unknown").status_code == 404
