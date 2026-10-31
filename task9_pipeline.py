# %% [markdown]
# # Task 9 — Smart Electricity Consumption Optimizer
# ## One notebook with an interactive dashboard
#
# **Run all cells, then use the dashboard at the bottom.** Choose a historical
# electricity case, enter household settings, and click **Run optimizer**.
# The result includes the forecast, recommendations, schedule, estimated job
# savings, ontology/rule trace, SHAP explanation and LSTM activation heatmap.
#
# This is a separate, self-contained notebook. Your supplied Protégé ontology
# and unchanged Task 7 source are embedded, so no output ZIP transfer is needed.
# If this runtime has a completed updated Task 7 folder, its models are reused.
# **In a fresh runtime without saved models, the default settings download the
# real UCI data and run Task 7 once first.** That can take several minutes; a
# Colab GPU helps. The small classical ML stage then trains once and is cached.
# Opening a different Colab notebook does not automatically share its runtime.
# You can optionally set `TASK7_FOLDER` to a saved folder instead of retraining.
#
# Electricity history is real recorded UCI data, not a live meter feed. Household
# occupancy, temperature, tariff and appliance settings are editable scenario
# inputs. No real appliance is switched. Savings are estimates for requested
# schedulable jobs, not measured reductions in a household bill.

# %%
import importlib.util
import subprocess
import sys
requirements = {"tensorflow": "tensorflow==2.20.0", "shap": "shap>=0.48,<0.51",
    "ipywidgets": "ipywidgets>=8,<9", "numpy": "numpy>=1.26,<3", "pandas": "pandas>=2.2,<3",
    "sklearn": "scikit-learn>=1.5,<2", "matplotlib": "matplotlib>=3.8,<4", "joblib": "joblib>=1.4,<2"}
missing = [spec for name, spec in requirements.items() if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
if "google.colab" in sys.modules:
    from google.colab import output
    output.enable_custom_widget_manager()

# %% [markdown]
# ## How the previous tasks connect
# | Component | Earlier tasks | Role here |
# |---|---|---|
# | Ontology and inference | 1–2 | Imported OWL/SWRL plus operational decision rules |
# | Data and feature preparation | 4–5 | Updated Task 7's shared causal preprocessing |
# | Deep learning | 7 | Frozen LSTM extracts temporal patterns |
# | Classical ML | 6 | Regressor predicts demand from the learned patterns |
# | Scheduling | 3 | A* finds a permitted low-cost job schedule |
# | Explanation | 8 | SHAP for the ML stage and LSTM activation visualization |
#
# ## 1. Settings
# Leave `TASK7_FOLDER` empty for automatic discovery. If no saved Task 7 results
# are found, `BUILD_TASK7_IF_MISSING=True` creates them using your current Task 7
# implementation. Set it to False if you only want to use existing model files.
#
# Training/validation caps limit the *new ML head's* fitting cost. Test metrics
# still use the full saved test set. No test targets are used to select models
# or make operational recommendations.

# %%
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path
import tempfile
import json
from IPython.display import display, HTML

TASK7_FOLDER = ""  # Optional: a completed task7_FINAL_outputs/full_... directory.
BUILD_TASK7_IF_MISSING = True
TRAIN_EXAMPLES = 20000
VALIDATION_EXAMPLES = 5000
FORCE_REFIT_ML = False
TASK9_OUTPUT_DIR = Path("task9_outputs")
TASK9_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Embedded project resources
# The next collapsed cell contains byte-for-byte compressed copies of your
# ontology and Task 7 source. SHA-256 checks verify them before use. Nothing in
# the ontology is treated as an instruction to execute Python.

# %%
# EMBEDDED_RESOURCES

# %% [markdown]
# ## 3. Knowledge representation, inference and scheduling
# The adapter reads the actual classes, named individuals, subclass links and
# two enabled SWRL rules from your OWL file. It supports this file's unary
# ClassAtom → hasDecision IndividualPropertyAtom rules and named subclasses.
# It is not a general OWL DL consistency reasoner; unsupported SWRL forms fail
# explicitly. Asserted example decisions are recorded separately from inferred
# decisions, so an existing assertion is not presented as newly inferred.
#
# SWRL candidates feed ordered operational rules for essential appliances,
# occupancy, comfort, priority, tariffs and the forecast. Essential protection
# overrides conflicting delay candidates. The A* scheduler uses an admissible
# remaining-cost bound and checks runtime, deadlines and parallel-job limits.
# It reports infeasibility or search limits rather than inventing a schedule.

# %%
# CORE_DEFINITIONS

# %% [markdown]
# ## 4. Connect the learned models
# Frozen LSTM summaries represent past patterns. A classical ML regressor then
# predicts power using those summaries plus the Task 7 context features.
# Gradient Boosting and Random Forest reuse the algorithm families from Task 6;
# their new weights must be learned on this new representation. Selection uses
# validation RMSE only. Training examples are sampled evenly from the training
# period. The LSTM itself is not trained by this new ML head.

# %%
# FORECAST_DEFINITIONS

# %%
knowledge = OntologyKnowledge(ONTOLOGY_XML)
print("Ontology classes:", len(knowledge.classes))
print("Imported enabled SWRL rules:", [rule["id"] for rule in knowledge.rules])
print("Original asserted examples:", knowledge.asserted_decisions)
print("Ontology SHA-256:", knowledge.digest)

def prepare_task7_results():
    roots = [Path.cwd()]
    for key in ["TASK7_COMPLETED_OUTPUT_DIR", "ARTIFACT_DIR", "OUTPUT_DIR"]:
        value = globals().get(key)
        if value:
            roots.append(Path(value))
    folder = find_task7_folder(TASK7_FOLDER, roots)
    if folder is not None:
        print("Reusing saved Task 7 models:", folder)
        return folder
    if not BUILD_TASK7_IF_MISSING:
        raise FileNotFoundError("No completed updated Task 7 folder is available in this runtime. Set TASK7_FOLDER, or enable BUILD_TASK7_IF_MISSING.")
    print("Fresh runtime: building Task 7 models once from the real UCI dataset. This includes model training.", flush=True)
    print("SHAP and the dashboard run after model preparation. No ZIP transfer is required.", flush=True)
    # Keras temporary model files stay within this writable project directory.
    old_tempdir = tempfile.tempdir
    tempdir = (TASK9_OUTPUT_DIR / "runtime_temp").resolve()
    tempdir.mkdir(parents=True, exist_ok=True)
    namespace = {"__name__": "task7_bootstrap"}
    try:
        tempfile.tempdir = str(tempdir)
        exec(compile(TASK7_SOURCE_TEXT, "embedded_task7_unchanged.py", "exec"), namespace)
        folder = Path(namespace["OUTPUT_DIR"]).resolve()
    finally:
        tempfile.tempdir = old_tempdir
    return find_task7_folder(str(folder))

task7_folder = prepare_task7_results()
bridge = ForecastBridge(task7_folder, TASK9_OUTPUT_DIR, TRAIN_EXAMPLES, VALIDATION_EXAMPLES, FORCE_REFIT_ML)
print("Task 9 selected ML stage:", bridge.selected_name)
display(bridge.validation_results.round(4))
performance = bridge.evaluate()
display(performance.round(4))
print("The hybrid is not assumed to beat the original LSTM or basic neural network; compare the generated metrics.")

# %% [markdown]
# ## 5. End-to-end consistency checks
# These are controlled decision scenarios, not new measured electricity results.
# They check imported rules, protected appliances, occupancy, forecast feedback,
# scheduling feasibility and error handling. Task 10's broader unseen-data,
# fairness and deployment evaluation remains separate.

# %%
from dataclasses import replace
check_forecast = bridge.predict(int(bridge.test_ids[0]))
scenarios = []
fridge = Appliance("Refrigerator", 200, on=True, essential=False, flexible=True, ontology_id="Refrigerator_1")
essential_check = optimize_decisions(check_forecast, [fridge], Scenario(occupancy=0), knowledge)
assert essential_check["recommendations"][0]["Action"] == "Keep ON"
scenarios.append({"Check": "Ontology protects refrigerator even if editable essential flag is false", "Passed": True})
ac = Appliance("Air Conditioner", 1800, on=True, flexible=False, priority="High", requires_occupancy=True)
occupied_ac = optimize_decisions(check_forecast, [ac], Scenario(temperature_c=40, occupancy=3), knowledge)
empty_ac = optimize_decisions(check_forecast, [ac], Scenario(temperature_c=40, occupancy=0), knowledge)
assert occupied_ac["recommendations"][0]["Action"] == "Keep ON"
assert empty_ac["recommendations"][0]["Action"] == "Turn OFF"
scenarios.append({"Check": "Comfort protects occupied AC; empty-home AC can turn off", "Passed": True})
high_forecast = dict(check_forecast, forecast_kw=bridge.threshold + 1)
wash = Appliance("Washing Machine", 500, ontology_id="WashingMachine_Test")
high_result = optimize_decisions(high_forecast, [wash], Scenario(), knowledge)
assert high_result["plan"]["status"] == "optimal"
for slot in high_result["plan"]["schedule"]:
    assert not (pd.Timestamp(slot["Start"]) < pd.Timestamp(high_forecast["target_end"]) and
                pd.Timestamp(slot["End"]) > pd.Timestamp(high_forecast["target_end"]) - pd.Timedelta(minutes=15))
scenarios.append({"Check": "Forecast high demand prevents flexible job overlap with the one known target interval", "Passed": True})
impossible = optimize_decisions(high_forecast, [replace(wash, deadline_min=30, duration_min=60)], Scenario(), knowledge)
assert impossible["plan"]["status"] == "infeasible" and impossible["plan"]["estimated_saving"] is None
scenarios.append({"Check": "Impossible deadline is reported without claimed savings", "Passed": True})
checks = pd.DataFrame(scenarios)
checks.to_csv(TASK9_OUTPUT_DIR / "decision_consistency_checks.csv", index=False)
display(checks)

# %% [markdown]
# ## 6. Dashboard
# Select a historical case and edit the household scenario. In each appliance's
# expandable settings, change requested jobs, power ratings,
# durations, priorities, earliest starts and deadlines using the controls.
# Use multiples of 15 minutes for scheduling values.
#
# The forecast means average kW over one 15-minute interval ending one hour
# after the selected origin. High-demand avoidance applies only to that interval;
# no full-day demand profile is fabricated. Later scheduling choices use the
# supplied tariffs and job constraints. Equal peak-start/end hours mean no peak
# period. The concurrency control limits schedulable jobs, not total home load.
#
# SHAP TreeExplainer explains the classical ML head in its learned-feature space.
# Its reconstruction is checked against the raw prediction before non-negative
# clipping. LSTM activation heatmaps inspect internal responses. This is not an
# exact sensor-level explanation of the entire composed pipeline or a causal
# explanation. These uses extend the SHAP/activation approach from Task 8.
# [SHAP TreeExplainer reference](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html).

# %%
# DASHBOARD_DEFINITIONS

# %%
dashboard = build_optimizer_dashboard(bridge, knowledge, TASK9_OUTPUT_DIR / "scenarios", performance)
display(dashboard["widget"])
dashboard["run"]()  # Initial result; subsequent clicks perform inference and scheduling, not training.

# %% [markdown]
# ## Deliverables and interpretation
# Every successful dashboard run writes a JSON report plus recommendation and
# schedule CSVs under `task9_outputs/scenarios`. The hybrid model, validation
# comparison and full-test comparison are cached separately. Save the executed
# notebook for submission. No model accuracy improvement or measured bill saving
# is claimed without evidence. Recommendations preserve essential/comfort rules;
# scheduling is optimal only when the search status says `optimal`, under the
# declared scenario constraints. Existing Tasks 1–8 files remain separate.
