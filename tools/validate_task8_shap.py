"""Exercise Task 8 on explicitly synthetic fixtures and existing saved networks.

Never trains models or labels fixture outputs as project results. Temporary files
stay within the workspace and are removed after the checks.
"""
import ast
import argparse
import faulthandler
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
root = Path(__file__).resolve().parents[1]
os.chdir(root)
parser = argparse.ArgumentParser()
parser.add_argument("--preview-dir", type=Path)
args = parser.parse_args()
print("Loading notebook and explanation dependencies...", flush=True)
faulthandler.dump_traceback_later(60, repeat=True)
import nbformat

nb = nbformat.read(root / "task8_shap.ipynb", as_version=4)
nbformat.validate(nb)
for cell in nb.cells:
    if cell.cell_type == "code":
        ast.parse(cell.source)
assert all(cell.get("execution_count") is None and not cell.get("outputs")
           for cell in nb.cells if cell.cell_type == "code")
code = [cell.source for cell in nb.cells if cell.cell_type == "code"]
env = {"__name__": "task8_validation", "sys": sys}
exec(compile(code[1], "task8_helpers", "exec"), env)
np, pd, tf, keras = [env[name] for name in ["np", "pd", "tf", "keras"]]
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import StandardScaler
import joblib

env["GLOBAL_EXAMPLES"] = 2
env["BACKGROUND_EXAMPLES"] = 3
env["SHAP_NSAMPLES"] = 32
env["SHAP_MAX_SAMPLES"] = 64
env["display"] = lambda value: None
original_compute = env["compute_shap"]
env["compute_shap"] = lambda view, reference, inputs: original_compute(
    view, reference, inputs, nsamples=32, max_samples=64, batch_size=16)
env["plt"].show = lambda: None

task7_tree = ast.parse((root / "task7_final_peak_gated_multitask_lstm.py").read_text(encoding="utf-8"))
builders = dict(env, SEQUENCE_LENGTH=8, HISTORY_FEATURES=["power", "current", "time"],
                CONTEXT_FEATURES=["lag", "mean", "hour", "day"], layers=layers,
                regularizers=regularizers, AUXILIARY_PEAK_LOSS_WEIGHT=.2)
for node in task7_tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ["build_peak_gate_model", "build_regression_model"]:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "task7_builders", "exec"), builders)

test_root = Path(tempfile.mkdtemp(prefix=".task8_shap_validation_", dir=root)).resolve()
previous_tempdir = tempfile.tempdir
tempfile.tempdir = str(test_root)
result = {"fixture_results_are_project_results": False, "checks": []}
try:
    os.chdir(test_root)
    fixture = test_root / "synthetic_fixture"
    fixture.mkdir()
    rng = np.random.default_rng(42)
    history = rng.normal(size=(10, 8, 3)).astype("float32")
    context = rng.normal(size=(10, 4)).astype("float32")
    inputs = {"history": history[4:], "context": context[4:]}
    background = {"history": history[:4], "context": context[:4]}
    models = {"LSTM": builders["build_peak_gate_model"](units=(4, 3)),
              "Basic neural network": builders["build_regression_model"](kind="mlp", name="basic_mlp")}
    for name, filename in {"LSTM": "task7_FINAL_lstm.keras", "Basic neural network": "task7_basic_mlp.keras"}.items():
        models[name].save(fixture / filename)
    scaler = StandardScaler().fit(np.array([[.5], [1.5]]))
    joblib.dump(scaler, fixture / "power_scaler.joblib")
    mean, scale = float(scaler.mean_[0]), float(scaler.scale_[0])
    times = pd.date_range("2020-01-01", periods=160, freq="15min")
    explain_indices = np.arange(100, 106)
    bg_indices = np.arange(20, 24)
    y = np.linspace(.5, 2, 6)
    np.savez_compressed(fixture / "task8_examples.npz", explain_history=inputs["history"],
        explain_context=inputs["context"], background_history=background["history"],
        background_context=background["context"], background_indices=bg_indices,
        explain_indices=explain_indices, explain_actual_kw=y,
        explain_target_timestamps_ns=times[explain_indices].asi8)
    np.savez_compressed(fixture / "prepared_arrays.npz", train_indices=np.arange(10, 80),
        test_indices=explain_indices, timestamps_ns=times.asi8)
    selected = "lstm_final_peak_gate"
    manifest = {"run_mode": "full", "run_id": "SYNTHETIC_VALIDATION_ONLY_NOT_PROJECT_RESULTS",
        "sequence_length": 8, "horizon_steps": 4, "interval_minutes": 15,
        "history_features": builders["HISTORY_FEATURES"], "context_features": builders["CONTEXT_FEATURES"],
        "selection": {"selected_lstm": selected}, "peak_threshold_kw": 1.6,
        "used_split_counts": {"test": 6}}
    (fixture / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    predictions = {name: env["predict_view"](env["make_forecast_view"](model, (8, 3), (4,), mean, scale), inputs)
                   for name, model in models.items()}
    pd.DataFrame({"Forecast_origin": times[explain_indices - 4], "Target_interval_end": times[explain_indices],
        "Actual_kW": y, f"{selected}_kW": predictions["LSTM"], "basic_mlp_kW": predictions["Basic neural network"],
        "Persistence_kW": np.full(6, 1.)}).to_csv(fixture / "test_predictions.csv", index=False)
    # Nested ZIP accepted; paths escaping extraction directory rejected.
    archive = Path(shutil.make_archive(str(test_root / "fixture"), "zip", root_dir=test_root, base_dir="synthetic_fixture"))
    resolved = env["resolve_artifacts"](str(archive))
    assert (resolved / "task7_FINAL_lstm.keras").is_file()
    bad_archive = test_root / "unsafe.zip"
    with zipfile.ZipFile(bad_archive, "w") as handle:
        handle.writestr("../escape.txt", "must not extract")
    try:
        env["resolve_artifacts"](str(bad_archive))
        raise AssertionError("Unsafe ZIP was accepted")
    except ValueError:
        pass
    env["TASK7_SOURCE"] = str(fixture)
    for i, block in enumerate(code[2:], start=2):
        print(f"Running synthetic validation cell {i}/{len(code)-1}...", flush=True)
        exec(compile(block, f"task8_validation_cell_{i}", "exec"), env)
    charts = list(env["OUTPUT_DIR"].glob("*.png"))
    assert len(charts) >= 10
    assert (env["OUTPUT_DIR"] / "task8_report.md").is_file()
    assert env["analysis_manifest"]["training_performed"] is False
    if args.preview_dir:
        preview_dir = (root / args.preview_dir).resolve()
        if not preview_dir.is_relative_to(root):
            raise ValueError("Validation previews must stay within the project folder")
        preview_dir.mkdir(parents=True, exist_ok=True)
        chosen = ["shap_feature_importance.png", "shap_time_feature_heatmaps.png", "lstm_activation_heatmaps.png"]
        chosen += [next(env["OUTPUT_DIR"].glob("shap_waterfall_*_lstm.png")).name]
        for filename in chosen:
            shutil.copy2(env["OUTPUT_DIR"] / filename, preview_dir / filename)
        (preview_dir / "README.txt").write_text("SYNTHETIC VALIDATION PREVIEWS ONLY. These are not project results.", encoding="utf-8")
    result["checks"].append("All notebook analysis cells, multitask output, ZIP handling, SHAP, activations, reports")
    result["synthetic_chart_count"] = len(charts)
    # Also exercise the actual compact LSTM and MLP previously saved by Task 7.
    old = root / "task7_outputs" / "full_20261004T223051_253137Z"
    if old.is_dir():
        print("Testing SHAP against existing trained single-output LSTM and basic NN...", flush=True)
        with np.load(old / "task8_examples.npz", allow_pickle=False) as archive:
            bx = {key: archive["background_" + key][:3].astype("float32") for key in ["history", "context"]}
            xx = {key: archive["explain_" + key][:2].astype("float32") for key in ["history", "context"]}
        saved_scaler = joblib.load(old / "power_scaler.joblib")
        for filename in ["task7_lstm.keras", "task7_basic_mlp.keras"]:
            network = keras.models.load_model(old / filename, compile=False)
            view = env["make_forecast_view"](network, xx["history"].shape[1:], xx["context"].shape[1:],
                                            float(saved_scaler.mean_[0]), float(saved_scaler.scale_[0]))
            phi, base, check, seconds = original_compute(view, bx, xx, nsamples=32, max_samples=64, batch_size=16)
            assert np.isfinite(phi[0]).all() and np.isfinite(phi[1]).all()
            assert len(check) == 2
            result["checks"].append(f"SHAP runtime on saved {filename}; no retraining")
    result["status"] = "passed"
finally:
    os.chdir(root)
    tempfile.tempdir = previous_tempdir
    # Delete only the temporary test directory created by this invocation.
    if test_root.parent != root.resolve() or not test_root.name.startswith(".task8_shap_validation_"):
        raise RuntimeError("Refusing to remove an unexpected validation path")
    shutil.rmtree(test_root)

(root / "task8_shap_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
faulthandler.cancel_dump_traceback_later()
print(json.dumps(result, indent=2), flush=True)
