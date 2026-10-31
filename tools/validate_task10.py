"""Exercise Task 10 with the real ontology and explicitly synthetic fixtures.

No full Task 7 training or claimed project evaluation results are generated.
"""
import ast
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import nbformat
import numpy as np
import pandas as pd

nb = nbformat.read(root / "task10_final_evaluation.ipynb", as_version=4)
nbformat.validate(nb)
code = [cell.source for cell in nb.cells if cell.cell_type == "code"]
for cell in code:
    ast.parse(cell)
assert all(cell.get("execution_count") is None and not cell.get("outputs") for cell in nb.cells if cell.cell_type == "code")
resources = {}
exec(next(cell for cell in code if "base64.b64decode" in cell), resources)
assert resources["ONTOLOGY_XML"].encode("utf-8") == (root / "SmartElectricityOptimizer.owl").read_bytes()
assert resources["TASK7_SOURCE_TEXT"].encode("utf-8") == (root / "task7_final_peak_gated_multitask_lstm.py").read_bytes()

test_root = Path(tempfile.mkdtemp(prefix=".task10_validation_", dir=root)).resolve()
old_cwd, old_temp = Path.cwd(), tempfile.tempdir
checks = ["Notebook schema/syntax and exact embedded project resources"]
try:
    os.chdir(test_root)
    tempfile.tempdir = str(test_root)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from sklearn.preprocessing import StandardScaler
    import joblib
    import matplotlib.pyplot as plt

    features = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    preprocessing = {"np": np, "pd": pd, "ELECTRICAL_FEATURES": features,
        "TARGET_FEATURE": "Global_active_power", "HISTORY_INPUT_COLUMNS": ["input_"+n for n in features]}
    exec(resources["TASK7_PREPROCESSOR_SOURCE"], preprocessing)
    rng = np.random.default_rng(42)

    def synthetic_minutes(start):
        index = pd.date_range(start, periods=11*24*60, freq="min")
        position = np.arange(len(index))
        active = 1.4 + .6*np.sin(2*np.pi*position/1440) + .1*rng.normal(size=len(index))
        return pd.DataFrame({"Global_active_power": active, "Global_reactive_power": .1+.05*rng.random(len(index)),
            "Voltage": 230+2*rng.normal(size=len(index)), "Global_intensity": active*1000/230,
            "Sub_metering_1": rng.random(len(index))*10, "Sub_metering_2": rng.random(len(index))*10,
            "Sub_metering_3": rng.random(len(index))*10}, index=index)

    minute = synthetic_minutes("2020-01-01")
    power = preprocessing["prepare_multivariate"](minute, features, 4)
    context = preprocessing["build_context"](power, 4)
    train, val, test = np.arange(700, 780), np.arange(790, 820), np.arange(830, 926)
    scalers, scaled_history = {}, []
    for feature in features:
        scaler = StandardScaler().fit(power[["actual_"+feature]].iloc[:780].to_numpy())
        scalers[feature] = scaler
        scaled_history.append(scaler.transform(power[["input_"+feature]].to_numpy()).ravel())
    calendar = preprocessing["calendar_features"](power.index)
    history = np.column_stack([*scaled_history, power["electrical_observed_fraction"].to_numpy(),
        calendar[["hour_sin", "hour_cos", "dow_sin", "dow_cos"]].to_numpy()]).astype("float32")
    context_scaler = StandardScaler().fit(context.iloc[train])
    context_values = context_scaler.transform(context).astype("float32")
    target_scaler = scalers["Global_active_power"]
    y_scaled = target_scaler.transform(power[["actual_kw"]].to_numpy()).astype("float32")
    history_names = [*(f+"_scaled" for f in features), "electrical_observed_fraction", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    builders = {"tf": tf, "keras": keras, "layers": layers, "regularizers": regularizers,
        "SEQUENCE_LENGTH": 96, "HISTORY_FEATURES": history_names, "CONTEXT_FEATURES": list(context.columns), "AUXILIARY_PEAK_LOSS_WEIGHT": .2}
    for node in ast.parse(resources["TASK7_SOURCE_TEXT"]).body:
        if isinstance(node, ast.FunctionDef) and node.name in ["build_peak_gate_model", "build_regression_model"]:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "synthetic_builder", "exec"), builders)
    print("Creating synthetic multivariate fixtures; not training full Task 7...", flush=True)
    network = builders["build_peak_gate_model"](units=(4, 3))
    basic = builders["build_regression_model"](kind="mlp", name="basic_mlp")
    folder = test_root / "synthetic_task7_fixture"
    folder.mkdir()
    network.save(folder / "task7_FINAL_lstm.keras")
    basic.save(folder / "task7_basic_mlp.keras")
    joblib.dump(target_scaler, folder / "power_scaler.joblib")
    joblib.dump(scalers, folder / "electrical_scalers.joblib")
    joblib.dump(context_scaler, folder / "context_scaler.joblib")
    np.savez_compressed(folder / "prepared_arrays.npz", history_values=history, context_values=context_values,
        target_scaled=y_scaled, train_indices=train, validation_indices=val, test_indices=test, timestamps_ns=power.index.asi8)
    xx = {"history": history[test[:, None]-4-96+1+np.arange(96)[None, :]], "context": context_values[test]}
    mean, scale = float(target_scaler.mean_[0]), float(target_scaler.scale_[0])
    p_lstm = np.maximum(np.asarray(network(xx, training=False)["forecast_scaled"]).reshape(-1)*scale+mean, 0)
    p_basic = np.maximum(np.asarray(basic(xx, training=False)).reshape(-1)*scale+mean, 0)
    manifest = {"run_id": "SYNTHETIC_VALIDATION_ONLY_NOT_PROJECT_RESULTS", "run_mode": "full", "sequence_length": 96,
        "horizon_steps": 4, "peak_threshold_kw": float(power["actual_kw"].iloc[:780].quantile(.9)),
        "history_features": history_names, "context_features": list(context.columns), "electrical_features": features,
        "selection": {"selected_lstm": "lstm_final_peak_gate"}, "data_audit": {"raw_end": str(minute.index[-1])}}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame({"Target_interval_end": power.index[test], "Forecast_origin": power.index[test-4],
        "Actual_kW": power["actual_kw"].iloc[test].to_numpy(), "lstm_final_peak_gate_kW": p_lstm,
        "basic_mlp_kW": p_basic, "Persistence_kW": power["input_kw"].iloc[test-4].to_numpy()}).to_csv(folder / "test_predictions.csv", index=False)
    later = synthetic_minutes("2020-02-01")
    external_file = test_root / "synthetic_later_readings.csv"
    later.rename_axis("Datetime").reset_index().to_csv(external_file, index=False)
    preview_dir = root / "tools" / "task10_validation_preview"
    preview_dir.mkdir(exist_ok=True)
    figure_count = [0]
    def capture_plot(*args, **kwargs):
        for number in plt.get_fignums():
            figure_count[0] += 1
            plt.figure(number).savefig(preview_dir / f"synthetic_{figure_count[0]:02d}.png", dpi=90, bbox_inches="tight")

    env = {"__name__": "__main__", "sys": sys}
    with patch("IPython.display.display", lambda *args, **kwargs: None), patch("IPython.display.clear_output", lambda **kwargs: None), patch.object(plt, "show", capture_plot):
        for i, cell in enumerate(code):
            if i == 0:
                continue  # Dependencies are already installed. Never pip install from a test.
            print("Executing Task 10 validation cell", i, flush=True)
            exec(compile(cell, f"task10_validation_cell_{i}", "exec"), env)
            if "LIVE_TASK9_BRIDGE =" in cell:
                env.update(TASK7_FOLDER=str(folder), TRAIN_EXAMPLES=40, VALIDATION_EXAMPLES=10,
                    SCENARIO_HISTORY_EXAMPLES=2, SHAP_SUMMARY_EXAMPLES=6, BOOTSTRAP_REPLICATES=20,
                    UNSEEN_MINUTE_CSV=str(external_file), PREPARE_MODELS_IF_MISSING=True, BUILD_TASK7_IF_MISSING=False)
        assert env["audit"]["models_unchanged_during_evaluation"]
        assert env["scenario_checks"]["Passed"].all(), env["scenario_checks"].loc[~env["scenario_checks"]["Passed"]].to_dict("records")
        assert env["fairness_checks"]["Passed"].all()
        assert env["xai"]["checks"]["Passed"].all()
        assert len(env["external_frame"]) > 0 and env["freshness"]["external_csv_supplied"]
        assert env["evaluation_dashboard"]["state"]["last_result"] is not None
        assert env["setup_mode"] == "prepared_missing_task9_ml_before_freeze"
        directory = env["OUTPUT_DIR_10"]
        for filename in ["evaluation_report.md", "model_card.md", "deployment_guide.md", "presentation_notes.md",
                         "evaluation_audit.json", "end_to_end_scenario_checks.csv", "independent_csv_performance.csv"]:
            assert (directory / filename).is_file()
        external_metrics_count = len(env["external_frame"])
        system_check_count = len(env["scenario_checks"])
        bridge = env["bridge"]
        # Cached setup is strictly read-only when preparation is disabled.
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        env.update(PREPARE_MODELS_IF_MISSING=False, BUILD_TASK7_IF_MISSING=False)
        with patch.object(GradientBoostingRegressor, "fit", side_effect=AssertionError("Unexpected fitting during frozen setup")), patch.object(RandomForestRegressor, "fit", side_effect=AssertionError("Unexpected fitting during frozen setup")):
            reused, mode = env["load_evaluation_system"]()
            assert mode == "reused_saved_task9_models"
            # Pickle/object hashes can change on deserialization because of array
            # aliasing/layout. Freeze checks compare the same live object before/
            # after evaluation; cache reloads must preserve files, weights and
            # predictions rather than that internal serialization hash.
            reloaded_fingerprint = env["freeze_fingerprint"](reused, env["knowledge"])
            for name in env["frozen_before"]:
                if name != "ml_model_hash":
                    assert reloaded_fingerprint[name] == env["frozen_before"][name], name
            original_predictions = bridge.regressor.predict(bridge.features(bridge.test_ids))
            reloaded_predictions = reused.regressor.predict(reused.features(reused.test_ids))
            np.testing.assert_allclose(original_predictions, reloaded_predictions, rtol=0, atol=1e-12)
        # No external input means no claim of new independent measured data.
        no_external = dict(env["freshness"], external_csv_supplied=False)
        no_external.pop("external_provenance", None)
        text = env["write_evaluation_documents"](directory, bridge, env["metrics"], env["scenario_checks"],
            env["fairness_checks"], env["xai"], no_external, env["intervals_note"], mode, env["frozen_before"])
        assert "Independent later-time-range CSV supplied: False" in text
        # Bad external schema and overlap must fail, not invent observations.
        bad = test_root / "bad.csv"
        pd.DataFrame({"Datetime": ["2020-01-01"]}).to_csv(bad, index=False)
        try:
            env["external_csv_predictions"](bridge, bad, env["TASK7_PREPROCESSORS"])
            raise AssertionError("Bad CSV schema was accepted")
        except ValueError:
            pass
        overlap = test_root / "overlapping.csv"
        minute.rename_axis("Datetime").reset_index().to_csv(overlap, index=False)
        try:
            env["external_csv_predictions"](bridge, overlap, env["TASK7_PREPROCESSORS"])
            raise AssertionError("Previously used time range was accepted as independent future data")
        except ValueError:
            pass
        constant = pd.DataFrame({"Target_interval_end": pd.date_range("2020-01-01", periods=4, freq="15min"),
            "Actual_kW": [1.]*4, "Hybrid_kW": [1.]*4, "Task7_LSTM_kW": [1.]*4, "Persistence_kW": [1.]*4})
        grouped, detection = env["performance_tables"](constant, 2.)
        intervals, message = env["day_block_intervals"](constant, 10)
        assert grouped.loc[grouped["Segment"].eq("High demand"), "Samples"].eq(0).all()
        assert intervals.empty and detection["True_negative"] == 4
    checks.extend(["Every analysis cell and optional dashboard on a synthetic frozen 96-step multivariate system",
        f"{system_check_count} end-to-end/error-handling checks plus policy invariance and SHAP reconstruction passed",
        f"Frozen original preprocessors/scalers scored {external_metrics_count} synthetic later-time-range examples without refitting",
        "Frozen ML/LSTM fingerprints unchanged through evaluation; cached read-only setup blocks fitting",
        "Reports, model card, deployment guide, presentation notes and audit exports generated",
        "Absent external measurements are labelled honestly; bad schema and overlapping-time CSVs rejected",
        "Empty high-demand group, constant targets and insufficient days handled without invented metrics"])
    report = {"status": "passed", "fixture_results_are_project_results": False, "full_task7_retraining_performed": False,
        "checks": checks, "synthetic_chart_count": figure_count[0]}
    (root / "task10_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
finally:
    os.chdir(old_cwd)
    tempfile.tempdir = old_temp
    if test_root.parent != root or not test_root.name.startswith(".task10_validation_"):
        raise RuntimeError("Unexpected Task 10 test cleanup path")
    shutil.rmtree(test_root)
