"""Technical tests using the real ontology and explicitly synthetic model fixtures.

No full Task 7 training is performed. Fixture results are never project results.
"""
import ast
import io
import itertools
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import nbformat
import numpy as np
import pandas as pd
from task9_core import Appliance, Scenario, OntologyKnowledge, optimize_decisions, plan_jobs, rate_at, peak_at, file_hash

checks = []
notebook = nbformat.read(root / "task9_integrated_optimizer.ipynb", as_version=4)
nbformat.validate(notebook)
code = [c.source for c in notebook.cells if c.cell_type == "code"]
for cell in code:
    ast.parse(cell)
assert all(c.get("execution_count") is None and not c.get("outputs") for c in notebook.cells if c.cell_type == "code")
resources = {}
exec(next(c for c in code if "base64.b64decode" in c), resources)
assert resources["ONTOLOGY_XML"].encode("utf-8") == (root.parent / "SmartElectricityOptimizer.owl").read_bytes()
assert resources["TASK7_SOURCE_TEXT"].encode("utf-8") == (root / "task7_final_peak_gated_multitask_lstm.py").read_bytes()
knowledge = OntologyKnowledge(resources["ONTOLOGY_XML"])
assert len(knowledge.classes) == 12 and len(knowledge.rules) == 2
assert knowledge.asserted_decisions["Refrigerator_1"] == {"KeepON_Action"}
checks.append("Notebook structure, exact embedded ontology/source and actual imported SWRL rules")

origin = pd.Timestamp("2020-01-01 18:00")
forecast = {"origin": str(origin), "target_end": str(origin + pd.Timedelta(hours=1)),
            "forecast_kw": 1., "threshold_kw": 2.5}
fridge = Appliance("Refrigerator", 200, essential=False, flexible=True, ontology_id="Refrigerator_1")
result = optimize_decisions(forecast, [fridge], Scenario(occupancy=0), knowledge)
assert result["recommendations"][0]["Action"] == "Keep ON"
assert set(result["recommendations"][0]["OWL_candidates"]) == {"KeepON_Action", "Delay_Action"}
knowledge.asserted_decisions["Refrigerator_1"] = set()
assert "KeepON_Action" in knowledge.infer(fridge)["candidate_decisions"]
checks.append("Imported SWRL executes independently of asserted decisions; essential overrides flexible conflict")

wash = Appliance("Washing Machine", 500, ontology_id="WashingMachine_Test")
cost_result = optimize_decisions(forecast, [wash], Scenario(), knowledge)
assert cost_result["plan"]["status"] == "optimal"
assert abs(cost_result["plan"]["estimated_saving"] - 1.5) < 1e-8
assert pd.Timestamp(cost_result["plan"]["schedule"][0]["Start"]).hour == 20
flat = Scenario(peak_rate=6, offpeak_rate=6)
low = optimize_decisions(forecast, [wash], flat, knowledge)
high = optimize_decisions(dict(forecast, forecast_kw=3), [wash], flat, knowledge)
assert low["plan"]["schedule"][0]["Start"] != high["plan"]["schedule"][0]["Start"]
assert pd.Timestamp(high["plan"]["schedule"][0]["Start"]) >= origin + pd.Timedelta(hours=1)
checks.append("Forecast changes decisions; exact single-interval avoidance; tariff saving equals independent energy calculation")

ac = Appliance("Air Conditioner", 1800, on=True, priority="High", flexible=False, requires_occupancy=True)
for people, expected in [(3, "Keep ON"), (0, "Turn OFF")]:
    assert optimize_decisions(forecast, [ac], Scenario(occupancy=people, temperature_c=40), knowledge)["recommendations"][0]["Action"] == expected
no_solution = optimize_decisions(forecast, [Appliance("Impossible", 500, duration_min=60, deadline_min=30)], Scenario(), knowledge)
assert no_solution["plan"]["status"] == "infeasible" and no_solution["plan"]["estimated_saving"] is None
limited = optimize_decisions(forecast, [wash], Scenario(node_limit=1), knowledge)
assert limited["plan"]["status"] == "search_limit" and limited["plan"]["estimated_cost"] is None
try:
    optimize_decisions(forecast, [Appliance("Invalid", 500, essential="false")], Scenario(), knowledge)
    raise AssertionError("String boolean was accepted")
except ValueError:
    pass
night = Scenario(peak_start_hour=22, peak_end_hour=6)
assert peak_at(pd.Timestamp("2020-01-02 01:00"), night) and not peak_at(pd.Timestamp("2020-01-02 12:00"), night)
checks.append("Comfort/occupancy priority, invalid inputs, infeasibility, bounded search and midnight tariffs")

# An independent exhaustive oracle verifies A* for small competing jobs.
for capacity, demand_high in itertools.product([1, 2], [False, True]):
    scenario = Scenario(horizon_hours=3, max_parallel_jobs=capacity)
    jobs = [(Appliance("A", 500, duration_min=30, deadline_min=120), True),
            (Appliance("B", 1200, duration_min=60, deadline_min=180), True)]
    oracle = []
    for start_a, start_b in itertools.product(range(7), range(9)):
        loads = [0] * 12
        valid, total = True, 0.
        for (job, _), start in zip(jobs, [start_a, start_b]):
            duration = job.duration_min // 15
            if demand_high and start < 4 and start + duration > 3:
                valid = False
            for slot in range(start, start + duration):
                loads[slot] += 1
                total += job.power_w / 1000 * .25 * rate_at(origin + pd.Timedelta(minutes=15 * slot), scenario)
        if valid and max(loads) <= capacity:
            oracle.append(total)
    planned = plan_jobs(jobs, scenario, origin, origin + pd.Timedelta(hours=1), demand_high)
    assert planned["status"] == "optimal" and abs(planned["estimated_cost"] - min(oracle)) < 1e-8
checks.append("A* costs match independent exhaustive search across concurrency/forecast scenarios")

test_root = Path(tempfile.mkdtemp(prefix=".task9_validation_", dir=root)).resolve()
old_cwd, old_temp = Path.cwd(), tempfile.tempdir
try:
    os.chdir(test_root)
    tempfile.tempdir = str(test_root)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from sklearn.preprocessing import StandardScaler
    import joblib
    import matplotlib.pyplot as plt
    print("Building small synthetic model fixtures; no full Task 7 training...", flush=True)
    rng = np.random.default_rng(42)
    builders = {"tf": tf, "keras": keras, "layers": layers, "regularizers": regularizers,
        "SEQUENCE_LENGTH": 96, "HISTORY_FEATURES": ["Global_active_power_scaled"] + [f"history_{i}" for i in range(11)],
        "CONTEXT_FEATURES": [f"context_{i}" for i in range(6)], "AUXILIARY_PEAK_LOSS_WEIGHT": .2}
    tree = ast.parse(resources["TASK7_SOURCE_TEXT"])
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ["build_peak_gate_model", "build_regression_model"]:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "task7_fixture_builder", "exec"), builders)
    network = builders["build_peak_gate_model"](units=(4, 3))
    basic = builders["build_regression_model"](kind="mlp", name="basic_mlp")
    folder = test_root / "synthetic_task7_fixture"
    folder.mkdir()
    network.save(folder / "task7_FINAL_lstm.keras")
    basic.save(folder / "task7_basic_mlp.keras")
    scaler = StandardScaler().fit(np.array([[1.5], [2.5]]))
    joblib.dump(scaler, folder / "power_scaler.joblib")
    n = 360
    history, context = rng.normal(size=(n, 12)).astype("float32"), rng.normal(size=(n, 6)).astype("float32")
    y_scaled = (.3 * context[:, 0] + .1 * context[:, 1]).astype("float32")
    train, val, test = np.arange(128, 220), np.arange(228, 248), np.arange(268, 284)
    times = pd.date_range("2020-01-01", periods=n, freq="15min")
    np.savez_compressed(folder / "prepared_arrays.npz", history_values=history, context_values=context,
        target_scaled=y_scaled, train_indices=train, validation_indices=val, test_indices=test, timestamps_ns=times.asi8)
    positions = test[:, None] - 4 - 96 + 1 + np.arange(96)[None, :]
    xx = {"history": history[positions], "context": context[test]}
    prediction = np.asarray(network(xx, training=False)["forecast_scaled"]).reshape(-1) * .5 + 2
    basic_prediction = np.asarray(basic(xx, training=False)).reshape(-1) * .5 + 2
    manifest = {"run_mode": "full", "run_id": "SYNTHETIC_VALIDATION_ONLY_NOT_PROJECT_RESULTS", "horizon_steps": 4,
        "sequence_length": 96, "peak_threshold_kw": 2.1, "history_features": builders["HISTORY_FEATURES"],
        "context_features": builders["CONTEXT_FEATURES"], "selection": {"selected_lstm": "lstm_final_peak_gate"}}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame({"Target_interval_end": times[test], "Actual_kW": y_scaled[test] * .5 + 2,
        "lstm_final_peak_gate_kW": np.maximum(prediction, 0), "basic_mlp_kW": np.maximum(basic_prediction, 0),
        "Persistence_kW": history[test-4, 0] * .5 + 2}).to_csv(folder / "test_predictions.csv", index=False)
    source_hash_before = file_hash(folder / "task7_FINAL_lstm.keras")
    previews = root / "tools" / "task9_validation_preview"
    previews.mkdir(exist_ok=True)
    image_counter = [0]
    def capture_plot(*args, **kwargs):
        for number in plt.get_fignums():
            image_counter[0] += 1
            plt.figure(number).savefig(previews / f"synthetic_{image_counter[0]:02d}.png", dpi=100, bbox_inches="tight")
    env = {"__name__": "__main__"}
    with patch("IPython.display.display", lambda *args, **kwargs: None), patch("IPython.display.clear_output", lambda **kwargs: None), patch.object(plt, "show", capture_plot):
        for i, cell in enumerate(code):
            if i == 0:
                continue  # Dependencies are already installed; never pip-install from this test.
            print("Executing Task 9 notebook test cell", i, flush=True)
            exec(compile(cell, f"task9_test_cell_{i}", "exec"), env)
            if "TASK7_FOLDER =" in cell:
                env.update(TASK7_FOLDER=str(folder), BUILD_TASK7_IF_MISSING=False, TRAIN_EXAMPLES=40, VALIDATION_EXAMPLES=10)
        dashboard = env["dashboard"]
        assert dashboard["state"]["last_result"] is not None, dashboard["widget"].children[-2].children[1].value
        dashboard["controls"]["occupancy"].value = 0
        dashboard["controls"]["explain"].value = False
        dashboard["run"]()
        rows = dashboard["state"]["last_result"]["recommendations"]
        assert next(r for r in rows if r["Appliance"] == "Air Conditioner")["Action"] == "Turn OFF"
        assert next(r for r in rows if r["Appliance"] == "Refrigerator")["Action"] == "Keep ON"
        last_path = dashboard["state"]["last_path"]
        dashboard["controls"]["appliances"][2]["controls"]["duration_min"].value = 17
        dashboard["run"]()
        assert dashboard["state"]["last_path"] == last_path
        bridge = env["bridge"]
        target = int(bridge.test_ids[0])
        before = bridge.predict(target)["forecast_kw"]
        bridge.y[target] += 1000
        assert bridge.predict(target)["forecast_kw"] == before
        bridge.y[target] -= 1000
        original = bridge.history[target-3:].copy()
        bridge.history[target-3:] += 999
        assert bridge.predict(target)["forecast_kw"] == before
        bridge.history[target-3:] = original
        # Repeating setup must reuse the cached model, not fit again.
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        with patch.object(GradientBoostingRegressor, "fit", side_effect=AssertionError("Unexpected refit")), patch.object(RandomForestRegressor, "fit", side_effect=AssertionError("Unexpected refit")):
            again = env["ForecastBridge"](folder, env["TASK9_OUTPUT_DIR"], 40, 10)
            assert again.predict(target)["forecast_kw"] == before
        assert file_hash(folder / "task7_FINAL_lstm.keras") == source_hash_before
    assert image_counter[0] >= 4
    checks.extend(["Every Task 9 analysis cell and dashboard callback on a synthetic 96-step multitask fixture",
        "Validation-selected hybrid ML, full-test comparisons, TreeSHAP reconstruction and temporal activations",
        "Dashboard occupancy/input changes, error messages and saved JSON/CSV reports",
        "Inference ignores held-out target values and future history rows; frozen LSTM file unchanged",
        "Repeated setup reuses cached models without ML fitting"])
    report = {"status": "passed", "fixture_results_are_project_results": False,
        "full_task7_retraining_performed": False, "checks": checks, "synthetic_chart_count": image_counter[0]}
    (root / "task9_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
finally:
    os.chdir(old_cwd)
    tempfile.tempdir = old_temp
    if test_root.parent != root or not test_root.name.startswith(".task9_validation_"):
        raise RuntimeError("Unexpected test cleanup path")
    shutil.rmtree(test_root)
