"""Upload validation and forecast-input contracts using explicit synthetic fixtures.

Fixture outputs are never household forecasts or model evaluation results.
"""
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from readings import FEATURES, ReadingsError, check_csv, frozen_inputs, predict_uploaded

NOW = datetime(2026, 10, 31, 10, 7, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def minute_frame():
    stamps = pd.date_range(end="2026-10-31 09:59:00", periods=8*24*60, freq="min")
    values = [1.1, .2, 230., 4.8, 1., 2., 3.]
    return pd.DataFrame({"Datetime": stamps.astype(str), **{name: value for name, value in zip(FEATURES, values)}})


def encoded(frame):
    return frame.to_csv(index=False).encode()


def check(frame, **kwargs):
    return check_csv(encoded(frame), "synthetic-meter-fixture.csv", "UTC", now=NOW, **kwargs)


def test_fresh_file_checks_without_claiming_a_forecast(minute_frame):
    checked = check(minute_frame)
    report = checked.report
    assert report["data_ready"] is True
    assert report["can_analyze"] is False
    assert report["status"] == "needs_service"
    assert report["origin"] == "2026-10-31 10:00:00"
    assert report["target_interval_start"] == "2026-10-31 10:45:00"
    assert report["target_end"] == "2026-10-31 11:00:00"
    assert report["age_minutes"] == 7
    assert len(report["history"]) == 96
    assert len(report["preview"]) == 5
    assert "forecast_kw" not in report
    assert checked.context.iloc[-1].latest_kw == pytest.approx(1.1)
    assert checked.context.iloc[-1].target_slot_previous_week_kw == pytest.approx(1.1)


@pytest.mark.parametrize("problem", ["missing_header", "duplicates", "future", "bad_seconds", "zone_suffix", "large_span", "duplicate_header"])
def test_bad_files_are_actionable(minute_frame, problem):
    frame = minute_frame.copy()
    if problem == "missing_header": frame = frame.drop(columns=["Voltage"])
    if problem == "duplicates": frame.loc[1, "Datetime"] = frame.loc[0, "Datetime"]
    if problem == "future": frame.loc[len(frame)-1, "Datetime"] = "2027-01-01 00:00:00"
    if problem == "bad_seconds": frame.loc[0, "Datetime"] = "2026-10-23 10:00:01"
    if problem == "zone_suffix": frame.loc[0, "Datetime"] += "Z"
    if problem == "large_span": frame.loc[0, "Datetime"] = "2026-01-01 00:00:00"
    content = encoded(frame)
    if problem == "duplicate_header": content = content.replace(b"Voltage", b"Global_intensity", 1)
    with pytest.raises(ReadingsError):
        check_csv(content, "fixture.csv", "UTC", now=NOW)


def test_short_or_old_upload_never_ready(minute_frame):
    short = check(minute_frame.tail(1440), model_available=True)
    assert short.report["can_analyze"] is False
    assert any(c["code"] == "history" and c["status"] == "fail" for c in short.report["checks"])
    old = check_csv(encoded(minute_frame), "fixture.csv", "UTC", now=datetime(2026, 10, 31, 12, tzinfo=timezone.utc), model_available=True)
    assert old.report["data_ready"] is False
    assert any(c["code"] == "freshness" and c["status"] == "fail" for c in old.report["checks"])


def test_incomplete_tail_excluded_and_no_future_target_needed(minute_frame):
    partial = minute_frame.tail(5).copy()
    partial["Datetime"] = pd.date_range("2026-10-31 10:00", periods=5, freq="min").astype(str)
    checked = check(pd.concat([minute_frame, partial]), model_available=True)
    assert checked.report["origin"] == "2026-10-31 10:00:00"
    assert checked.report["can_analyze"] is True
    assert checked.power.index[-1] == pd.Timestamp("2026-10-31 10:00")
    assert checked.power.actual_kw.iloc[-1] == pytest.approx(1.1)
    assert checked.context.index[-1] > checked.power.index[-1]
    # Feature creation uses known calendar information, with no actual future label.
    assert np.isfinite(checked.context.tail(1).to_numpy()).all()


def test_gaps_invalid_values_and_unsorted_rows(minute_frame):
    frame = minute_frame.copy()
    # Four hours missing in the recent input window exceeds original gap-fill limit.
    frame.loc[len(frame)-360:len(frame)-120, "Voltage"] = np.nan
    checked = check(frame, model_available=True)
    assert checked.report["data_ready"] is False
    assert checked.report["invalid_cells"] > 0
    unsorted = check(minute_frame.iloc[::-1])
    assert unsorted.report["data_ready"] is True
    assert any("sorted" in note for note in unsorted.report["warnings"])


def test_timezone_does_not_relabel_stale_data(minute_frame):
    checked = check_csv(encoded(minute_frame), "fixture.csv", "Asia/Kolkata", now=NOW, model_available=True)
    assert checked.report["age_minutes"] == 337
    assert checked.report["can_analyze"] is False
    with pytest.raises(ReadingsError, match="timezone"):
        check_csv(encoded(minute_frame), "fixture.csv", "unknown/zone", now=NOW)


def test_encoding_and_empty_template_rejected_as_data(minute_frame):
    with pytest.raises(ReadingsError, match="UTF-8"):
        check_csv(b"\xff\x81", "bad.csv", "UTC", now=NOW)
    with pytest.raises(ReadingsError, match="readings"):
        check_csv(encoded(minute_frame.head(0)), "empty.csv", "UTC", now=NOW)


def test_frozen_transform_and_forecast_contract(minute_frame, tmp_path, monkeypatch):
    """A deliberately synthetic bridge checks shape/order and zero fitting."""
    checked = check(minute_frame, model_available=True)
    class TransformOnly:
        def transform(self, value):
            return np.asarray(value, dtype="float32")
        def fit(self, *args):
            raise AssertionError("Uploaded readings must never fit a scaler")
    scalers = {name: TransformOnly() for name in FEATURES}
    fake_joblib = SimpleNamespace(load=lambda path: scalers if Path(path).name == "electrical_scalers.joblib" else TransformOnly())
    monkeypatch.setitem(sys.modules, "joblib", fake_joblib)
    for name in ["electrical_scalers.joblib", "context_scaler.joblib"]:
        (tmp_path/name).write_bytes(b"explicit-synthetic-test-fixture")
    history_features = [*(name+"_scaled" for name in FEATURES), "electrical_observed_fraction", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    context_features = list(checked.context.columns)
    encoder = Mock(return_value=np.array([[.2, .4]], dtype="float32"))
    regressor = SimpleNamespace(predict=Mock(return_value=np.array([1.23])))
    bridge = SimpleNamespace(folder=tmp_path, manifest={"electrical_features":FEATURES,"context_features":context_features,"history_features":history_features,"sequence_length":96,"horizon_steps":4},
                             input_order=["history", "context"],encoder=encoder,regressor=regressor,threshold=2.5,selected_name="explicit synthetic test bridge")
    inputs = frozen_inputs(checked, bridge)
    assert inputs["history"].shape == (1, 96, 12)
    assert inputs["context"].shape == (1, len(context_features))
    assert inputs["history"][0, -1, 0] == pytest.approx(1.1)
    monkeypatch.setitem(sys.modules, "shap", None)
    forecast, evidence = predict_uploaded(checked, bridge)
    assert forecast["source"] == "uploaded_readings"
    assert forecast["forecast_kw"] == 1.23
    assert forecast["target_end"] == "2026-10-31 11:00:00"
    assert evidence["confidence"]["available"] is False
    assert evidence["explanation"]["available"] is False
    actual_features = regressor.predict.call_args.args[0]
    assert actual_features.shape == (1, 2+len(context_features))
    np.testing.assert_allclose(actual_features[0, :2], [.2, .4])


def test_downloadable_synthetic_sample_cannot_be_a_current_forecast():
    sample = Path(__file__).resolve().parents[1]/"frontend/public/templates/SYNTHETIC-electricity-readings-8-days.csv"
    frame = pd.read_csv(sample)
    assert len(frame) == 11520
    assert frame.Datetime.is_unique
    assert frame.Data_source.eq("synthetic_sample").all()
    # Even at a fresh timestamp with a connected service, sample data is blocked.
    checked = check_csv(sample.read_bytes(), sample.name, "UTC", model_available=True,
                        now=datetime(2026, 1, 9, 0, 7, tzinfo=timezone.utc))
    assert checked.report["is_sample"] is True
    assert checked.report["can_analyze"] is False
    assert checked.report["data_ready"] is False
    assert checked.report["age_minutes"] == 7
    assert any(c["code"] == "sample_data" and c["status"] == "fail" for c in checked.report["checks"])
    assert all(c["status"] == "pass" for c in checked.report["checks"] if c["code"] != "sample_data")


@pytest.mark.parametrize("export_format", ["semicolon", "tab", "utf16", "excel_sep", "header_spacing"])
def test_common_excel_exports_keep_the_same_reading_columns(minute_frame, export_format):
    frame = minute_frame.copy()
    if export_format == "header_spacing":
        frame.columns = ["  "+name.upper()+"  " for name in frame.columns]
    delimiter = ";" if export_format in ("semicolon", "excel_sep") else "\t" if export_format == "tab" else ","
    text = frame.to_csv(index=False, sep=delimiter)
    if export_format == "excel_sep":
        text = "sep=;\r\n"+text
    content = text.encode("utf-16" if export_format == "utf16" else "utf-8-sig")
    checked = check_csv(content, "excel-export.csv", "UTC", now=NOW)
    assert checked.report["row_count"] == 11520
    assert checked.report["data_ready"] is True
    assert checked.report["history"][-1]["kw"] == pytest.approx(1.1)
    assert checked.report["warnings"]


def test_webpage_or_guide_is_not_reported_as_missing_every_column():
    with pytest.raises(ReadingsError, match="webpage"):
        check_csv(b"<!DOCTYPE html><html><body>Download</body></html>", "download.csv", "UTC", now=NOW)
    with pytest.raises(ReadingsError, match="instructions guide"):
        check_csv(b"CURRENT - ELECTRICITY READINGS TEMPLATES\nBlank CSV template", "guide.csv", "UTC", now=NOW)
