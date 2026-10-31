# %% [markdown]
# # Task 10 — Final Validation, Explainability and Responsible AI
# ## Smart Electricity Consumption Optimizer
#
# This separate notebook evaluates the Task 9 integrated system against the
# supplied Task 10 requirements. It checks forecasting performance, complete
# decision behavior, explanations, privacy/fairness/safety limits and prototype
# deployment, and produces a report plus structured presentation notes.
#
# **Run all cells.** No ontology or ZIP upload is required. Existing Task 9 models
# are reused when available. **A fresh Colab runtime without saved models will,
# by default, rebuild Task 9 once before evaluation; this can take several minutes
# and a GPU helps.** Set `PREPARE_MODELS_IF_MISSING=False` to forbid preparation.
# Evaluation starts only after preparation ends and the models are fingerprinted.
# No model fitting/tuning is allowed during the evaluation phase.
#
# The earlier held-out test data is unseen to training/model selection but has
# already been evaluated in Task 9. Reusing it is reported explicitly, not presented
# as an independent new test. Additional household scenarios are designed tests.
# An optional later-time-range minute CSV supports genuinely new supplied readings.
# No new independent measured electricity data is claimed when no CSV is supplied.

# %%
import importlib.util
import subprocess
import sys
requirements = {"tensorflow": "tensorflow==2.20.0", "shap": "shap>=0.48,<0.51", "ipywidgets": "ipywidgets>=8,<9",
    "numpy": "numpy>=1.26,<3", "pandas": "pandas>=2.2,<3", "sklearn": "scikit-learn>=1.5,<2",
    "matplotlib": "matplotlib>=3.8,<4", "joblib": "joblib>=1.4,<2"}
missing = [spec for name, spec in requirements.items() if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
if "google.colab" in sys.modules:
    from google.colab import output
    output.enable_custom_widget_manager()

# %% [markdown]
# ## 1. Configuration and data provenance
# Use the same Task 7 folder and Task 9 model-cache root to evaluate the exact
# previously trained system. A saved notebook's visible results do not contain
# its trained model weights. If setup rebuilds the models, the report labels that
# fact; it does not claim to restore the exact historical run from its screenshots.

# %%
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from IPython.display import display, HTML

LIVE_TASK9_BRIDGE = globals().get("bridge")  # Reused if these cells follow Task 9 in the same runtime.
TASK7_FOLDER = ""
TASK9_OUTPUT_DIR = Path("task9_outputs")  # Existing cache root, optionally a saved Drive folder.
PREPARE_MODELS_IF_MISSING = True
BUILD_TASK7_IF_MISSING = PREPARE_MODELS_IF_MISSING
TRAIN_EXAMPLES = 20000
VALIDATION_EXAMPLES = 5000
UNSEEN_MINUTE_CSV = ""  # Optional path; see the required schema below.
SCENARIO_HISTORY_EXAMPLES = 4
SHAP_SUMMARY_EXAMPLES = 24
BOOTSTRAP_REPLICATES = 200
SHOW_DASHBOARD = True
OUTPUT_DIR_10 = Path("task10_outputs") / datetime.now(timezone.utc).strftime("evaluation_%Y%m%dT%H%M%S_%fZ")
OUTPUT_DIR_10.mkdir(parents=True, exist_ok=False)

# %% [markdown]
# ## 2. Embedded system and evaluation code
# Collapsed code cells contain the supplied ontology, exact Task 7 source, Task 9
# implementation and evaluation helpers. The ontology/source hashes are checked.
# The ontology adapter remains the explicitly documented supported SWRL subset.

# %%
# EMBEDDED_RESOURCES

# %%
# TASK9_DEFINITIONS

# %%
# TASK9_PREPARATION_FUNCTION

# %%
# TASK10_DEFINITIONS

# %% [markdown]
# ## 3. Prepare the system, then freeze it
# Preparation may reuse cached models or, if permitted and necessary, rebuild
# them using the unchanged Task 9 training procedure. The following evaluation
# uses fixed model weights, a fixed ML head and training-derived threshold.

# %%
def load_evaluation_system():
    from unittest.mock import patch
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    if LIVE_TASK9_BRIDGE is not None and all(hasattr(LIVE_TASK9_BRIDGE, n) for n in ["network", "regressor", "test_ids", "signature", "folder"]):
        if not TASK7_FOLDER or Path(TASK7_FOLDER).resolve() == LIVE_TASK9_BRIDGE.folder.resolve():
            return LIVE_TASK9_BRIDGE, "reused_live_task9_models"
    roots = [Path.cwd()]
    for key in ["TASK7_COMPLETED_OUTPUT_DIR", "ARTIFACT_DIR"]:
        if globals().get(key):
            roots.append(Path(globals()[key]))
    folder = find_task7_folder(TASK7_FOLDER, roots)
    rebuilt_task7 = folder is None
    if folder is None:
        if not PREPARE_MODELS_IF_MISSING:
            raise FileNotFoundError("No completed Task 7 model folder is available; preparation is disabled.")
        folder = prepare_task7_results()
    cache_before = {str(path.resolve()) for path in (TASK9_OUTPUT_DIR / "model_cache").glob("*/hybrid_forecaster.joblib")}
    if PREPARE_MODELS_IF_MISSING:
        current = ForecastBridge(folder, TASK9_OUTPUT_DIR, TRAIN_EXAMPLES, VALIDATION_EXAMPLES, False)
    else:
        message = "No matching frozen Task 9 ML cache. Supply the original cache or enable model preparation before evaluation."
        with patch.object(GradientBoostingRegressor, "fit", side_effect=RuntimeError(message)), patch.object(RandomForestRegressor, "fit", side_effect=RuntimeError(message)):
            current = ForecastBridge(folder, TASK9_OUTPUT_DIR, TRAIN_EXAMPLES, VALIDATION_EXAMPLES, False)
    cache_file = str((current.cache / "hybrid_forecaster.joblib").resolve())
    mode = "rebuilt_task7_and_task9_before_freeze" if rebuilt_task7 else "prepared_missing_task9_ml_before_freeze" if cache_file not in cache_before else "reused_saved_task9_models"
    return current, mode

bridge, setup_mode = load_evaluation_system()
knowledge = OntologyKnowledge(ONTOLOGY_XML)
frozen_before = freeze_fingerprint(bridge, knowledge)
print("Setup:", setup_mode, "| Task 7 run:", bridge.manifest["run_id"], "| Model:", bridge.selected_name)
print("EVALUATION STARTS HERE: model weights and ML predictions are now frozen.")
print("Output folder:", OUTPUT_DIR_10.resolve())

# %% [markdown]
# ## 4. Forecast performance on inputs excluded from fitting
# All saved test examples are disjoint from training and validation; temporal
# boundaries and target alignment are checked. This is the same holdout already
# examined in Task 9. MAE/RMSE are lower-is-better; R² is higher-is-better.
# High-demand subgroup R² can be negative and is not directly comparable with
# the overall R² because the target variance differs.
#
# Day-block bootstrap intervals describe uncertainty of aggregate error statistics,
# not the uncertainty of a particular prediction. One household and day-to-day
# dependence limit these exploratory intervals.

# %%
evaluation_frame = collect_frozen_predictions(bridge)
evaluation_frame.to_csv(OUTPUT_DIR_10 / "reused_holdout_predictions.csv", index=False)
metrics, peak_detection = performance_tables(evaluation_frame, bridge.threshold)
groups = temporal_error_profiles(evaluation_frame, bridge.threshold)
intervals, intervals_note = day_block_intervals(evaluation_frame, BOOTSTRAP_REPLICATES)
metrics.to_csv(OUTPUT_DIR_10 / "forecast_performance.csv", index=False)
groups.to_csv(OUTPUT_DIR_10 / "temporal_demand_error_profiles.csv", index=False)
intervals.to_csv(OUTPUT_DIR_10 / "error_statistic_block_intervals.csv", index=False)
(OUTPUT_DIR_10 / "peak_detection.json").write_text(json.dumps(peak_detection, indent=2), encoding="utf-8")
display(metrics.round(4))
display(intervals.round(4))
print(intervals_note)
print("Thresholded hybrid peak detection:", peak_detection)
freshness = {"external_csv_supplied": bool(UNSEEN_MINUTE_CSV), "test_disjoint_from_fitting": True,
    "existing_test_already_evaluated_in_task9": True, "existing_test_claimed_independent_new_test": False,
    "household_scenarios": "Additional designed combinations, not measured new-household observations",
    "test_examples": len(evaluation_frame), "model_preparation_mode": setup_mode}

# %% [markdown]
# ## 5. Optional independent later-time-range observations
# Leave the path empty if none are available. The report then explicitly records
# that no independent new measured dataset was supplied. The CSV must contain
# **Datetime** (ISO local minute timestamps without timezone suffix) and:
# `Global_active_power`, `Global_reactive_power`, `Voltage`, `Global_intensity`,
# `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`.
#
# Supply at least eight days of minute readings, including warm-up for the 7-day
# context lag. Only complete observed target intervals with forecast origins later
# than the original dataset end are scored. Original training-only scalers and
# Task 7 causal transformations are applied without fitting or interpolation from
# future readings. Date separation alone cannot establish data authenticity or
# guarantee independent households. Missing columns or insufficient history fail
# clearly rather than generating invented readings.

# %%
external_metrics = None
if UNSEEN_MINUTE_CSV:
    external_frame, external_provenance = external_csv_predictions(bridge, UNSEEN_MINUTE_CSV, TASK7_PREPROCESSORS)
    external_metrics, external_detection = performance_tables(external_frame, bridge.threshold)
    external_frame.to_csv(OUTPUT_DIR_10 / "independent_csv_predictions.csv", index=False)
    external_metrics.to_csv(OUTPUT_DIR_10 / "independent_csv_performance.csv", index=False)
    freshness["external_provenance"] = external_provenance
    display(external_metrics.round(4))
else:
    print("No independent external measurements supplied. Existing holdout results are labelled as reused.")
(OUTPUT_DIR_10 / "data_freshness.json").write_text(json.dumps(freshness, indent=2), encoding="utf-8")

# %%
# VISUAL_DEFINITIONS

# %%
plot_evaluation(evaluation_frame, metrics, groups, peak_detection, OUTPUT_DIR_10)
display(groups.round(4))
print("Hour/weekend/demand groups are performance diagnostics, not demographic fairness evidence.")

# %% [markdown]
# ## 6. End-to-end scenario testing
# Each scenario uses a real model forecast and user-specified household settings.
# Checks cover ontology protection, occupied/empty hot homes, overnight and flat
# tariffs, concurrency, short windows, running jobs, user restrictions, impossible
# deadlines and search limits. Invalid-input cases must be rejected clearly.
# Checks independently reconstruct costs and verify schedule constraints.
# Successfully reporting infeasibility is a correct outcome when constraints
# cannot be met. Latency includes inference, reasoning and scheduling; excludes
# SHAP and rendering and does not imply production latency guarantees.

# %%
scenario_checks, scenario_results = run_scenario_evaluation(bridge, knowledge, SCENARIO_HISTORY_EXAMPLES)
scenario_checks.to_csv(OUTPUT_DIR_10 / "end_to_end_scenario_checks.csv", index=False)
(OUTPUT_DIR_10 / "scenario_inputs_and_results.json").write_text(json.dumps(scenario_results, indent=2), encoding="utf-8")
display(scenario_checks)
latencies = scenario_checks["Latency_ms"].dropna()
latency_summary = {"Timed_checks": len(latencies), "Median_ms": float(latencies.median()),
    "P95_ms": float(latencies.quantile(.95)), "First_timed_call_ms": float(latencies.iloc[0]),
    "Definition": "Includes inference + rules + scheduling; excludes explanation/rendering; first call may include warm-up"}
(OUTPUT_DIR_10 / "latency_summary.json").write_text(json.dumps(latency_summary, indent=2), encoding="utf-8")
print("Scenario checks passed:", int(scenario_checks["Passed"].sum()), "/", len(scenario_checks))
print("Latency:", latency_summary)
plot_scenario_checks(scenario_checks, OUTPUT_DIR_10)

# %% [markdown]
# ## 7. Explain predictions and operational decisions
# TreeSHAP explains the ML head using learned LSTM patterns and context inputs.
# It does not provide exact sensor-level attributions for the whole composed
# system. The tree reference is based on fitted training-path statistics.
# Reconstruction checks concern the raw ML output; clipping to non-negative
# power is shown separately. Passing a reconstruction check does not prove causal
# validity or general reliability. LSTM activation maps are internal responses,
# not attention weights or causal feature importance.
#
# Summary examples are evenly spaced. Small-error, typical-error and high-demand/
# large-error cases are deliberately selected diagnostics; they are not an unbiased
# performance sample. Rule traces separately explain why an appliance action was
# selected and how ontology candidates were resolved.

# %%
xai = shap_diagnostics(bridge, evaluation_frame, SHAP_SUMMARY_EXAMPLES)
xai["checks"].to_csv(OUTPUT_DIR_10 / "shap_reconstruction_checks.csv", index=False)
xai["importance"].to_csv(OUTPUT_DIR_10 / "shap_feature_importance.csv", index=False)
np.savez_compressed(OUTPUT_DIR_10 / "ml_shap_values.npz", values=xai["values"], features=np.array(bridge.feature_names),
    base_value=xai["base"], test_frame_positions=xai["positions"])
diagnostics = plot_explanations(bridge, evaluation_frame, xai, OUTPUT_DIR_10)
diagnostics.to_csv(OUTPUT_DIR_10 / "diagnostic_forecasts.csv", index=False)
display(diagnostics.round(4))
display(xai["checks"].round(6))
rule_examples = []
for label, position in xai["cases"].items():
    forecast = bridge.predict(int(evaluation_frame.iloc[position]["Target_index"]))
    decision = optimize_decisions(forecast, default_appliances(), Scenario(), knowledge)
    for row in decision["recommendations"]:
        rule_examples.append({"Case": label, "Appliance": row["Appliance"], "Action": row["Action"],
            "Rule_trace": " → ".join(row["Rule_trace"]), "Reason": row["Reason"],
            "Suggested_start": row.get("Suggested_start"), "Schedule_status": decision["plan"]["status"]})
rule_examples = pd.DataFrame(rule_examples)
rule_examples.to_csv(OUTPUT_DIR_10 / "explained_decision_paths.csv", index=False)
display(rule_examples)

# %% [markdown]
# ## 8. Ethics, fairness, privacy, safety and bias
# The source dataset contains no demographic labels. We therefore do not invent
# gender/income/ethnicity fairness scores. Temporal error groups describe error
# distribution, while controlled checks verify limited policy invariances.
# Neither establishes population fairness. Essential-device protection is tested
# under renaming and different occupancy/temperature inputs.
#
# Practical limits include one-household training, fixed comfort thresholds,
# incorrect peak forecasts, scenario tariff assumptions, privacy risks and the
# advisory nature of this prototype. Model interpretation is not a causal proof.
# Data permissions/terms and legal obligations depend on deployment context; no
# jurisdiction-specific compliance certification is claimed.

# %%
fairness_checks = fairness_safety_checks(bridge, knowledge)
responsible_ai = responsible_ai_table()
fairness_checks.to_csv(OUTPUT_DIR_10 / "controlled_policy_invariance.csv", index=False)
responsible_ai.to_csv(OUTPUT_DIR_10 / "responsible_ai.csv", index=False)
display(fairness_checks)
display(responsible_ai)

# %% [markdown]
# ## 9. Working prototype demonstration
# The optional dashboard reuses the exact frozen Task 9 bridge. It allows a
# demonstrator to change household settings and inspect recommendations. A
# Colab notebook is a prototype deployment; saving/sharing it is not a continuously
# hosted production service. No hardware is controlled. Keep private inputs and
# identifiers out of public exports. The same dashboard is available in Task 9.

# %%
# DASHBOARD_DEFINITIONS

# %%
if SHOW_DASHBOARD:
    evaluation_dashboard = build_optimizer_dashboard(bridge, knowledge, OUTPUT_DIR_10 / "demonstration", metrics)
    display(evaluation_dashboard["widget"])
    evaluation_dashboard["run"]()
else:
    print("Dashboard display skipped; evaluation and reports remain available.")

# %% [markdown]
# ## 10. Final audit, report and presentation material
# Fingerprints must match before/after evaluation. The report preserves failed
# checks and weak results rather than hiding them. It includes the architecture,
# methodology, actual metrics, explanation scope, responsible-AI discussion and
# limitations. Presentation notes organize these into eight talking points.

# %%
frozen_after = freeze_fingerprint(bridge, knowledge)
if frozen_before != frozen_after:
    raise RuntimeError("A frozen model/ontology fingerprint changed during evaluation. Investigate before reporting results.")
report_text = write_evaluation_documents(OUTPUT_DIR_10, bridge, metrics, scenario_checks, fairness_checks, xai,
    freshness, intervals_note, setup_mode, frozen_before)
audit = {"evaluation_completed": True, "models_unchanged_during_evaluation": True, "setup_mode": setup_mode,
    "scenario_checks_passed": int(scenario_checks["Passed"].sum()), "scenario_checks_total": len(scenario_checks),
    "shap_checks_passed": int(xai["checks"]["Passed"].sum()), "shap_checks_total": len(xai["checks"]),
    "policy_invariance_checks_passed": int(fairness_checks["Passed"].sum()), "policy_invariance_checks_total": len(fairness_checks),
    "external_csv_supplied": bool(UNSEEN_MINUTE_CSV), "reused_holdout": True, "task7_run": bridge.manifest["run_id"],
    "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}}
(OUTPUT_DIR_10 / "evaluation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(report_text)
print("Model fingerprints unchanged:", frozen_before == frozen_after)
print("All Task 10 tables, charts, report, model card, deployment guide and presentation notes:", OUTPUT_DIR_10.resolve())
print("Save this executed notebook and inspect any failed scenario/explanation checks before submission.")
