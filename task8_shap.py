# %% [markdown]
# # Task 8 — Explainable Electricity Forecasting with SHAP
# **SHAP explanations, LSTM activation heatmaps, and comparison with the basic neural network.**
#
# This notebook uses the saved outputs of `task7_final_peak_gated_multitask_lstm.py`.
# It never trains, tunes, or changes Task 7. The forecast is average power (kW)
# in the 15-minute interval ending one hour after the forecast origin.
#
# **Easiest in Colab:** use `task7_and_task8_shap.ipynb`. Run Task 7 there, then
# continue to Task 8 in the same notebook. Its saved files are used directly;
# no ZIP download or upload is needed. If Task 7 already ran in an active Colab
# notebook, these Task 8 cells can also be appended to that notebook.
# A separate Colab notebook normally has a separate runtime and cannot see the
# other notebook's temporary files. Use a shared saved folder or the combined
# notebook. `TASK7_SOURCE` can optionally identify an existing folder or ZIP.
# Locally, a single completed run in `task7_FINAL_outputs` is found automatically.
# If several runs exist, choose one explicitly. Earlier Task 7 outputs are not
# silently substituted for the updated implementation.
#
# Full-test metrics use every saved test prediction. SHAP uses a small, explicitly
# reported test sample to limit computation. All reference examples come from
# training data. No LIME or separate Integrated Gradients analysis is used.

# %%
import importlib.util
import subprocess
import sys

requirements = {
    "tensorflow": "tensorflow>=2.19,<2.21", "shap": "shap>=0.48,<0.51",
    "numpy": "numpy>=1.26,<3", "pandas": "pandas>=2.2,<3",
    "matplotlib": "matplotlib>=3.8,<4", "sklearn": "scikit-learn>=1.5,<2",
    "joblib": "joblib>=1.4,<2",
}
missing = [spec for module, spec in requirements.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

# %% [markdown]
# ## 1. Settings and helper functions
# The defaults explain up to 24 evenly spaced saved test examples plus three
# diagnostic cases, using 32 training references. Increase the sample counts for
# a broader analysis; doing so increases explanation time, not training time.
#
# SHAP GradientExplainer uses **expected gradients** to approximate SHAP values.
# It is a SHAP method related to Integrated Gradients, not an exact enumeration
# of feature coalitions. We explain the final non-negative forecast in kW,
# including the peak gate when present. The peak score is not an attention layer.
# Reference: [SHAP GradientExplainer documentation](https://shap.readthedocs.io/en/stable/generated/shap.GradientExplainer.html).

# %%
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import hashlib
import json
import re
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from IPython.display import display

TASK7_SOURCE = globals().get("TASK7_COMPLETED_OUTPUT_DIR", "")  # Automatic handoff in the combined notebook.
SEED = 42
GLOBAL_EXAMPLES = 24
BACKGROUND_EXAMPLES = 32
SHAP_NSAMPLES = 256
SHAP_MAX_SAMPLES = 1024
SHAP_BATCH_SIZE = 64
ADDITIVITY_ABS_TOL_KW = 0.03
ADDITIVITY_REL_TOL = 0.05
keras.utils.set_random_seed(SEED)
try:
    tf.config.threading.set_intra_op_parallelism_threads(min(4, os.cpu_count() or 2))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_artifacts(source):
    if source:
        candidate = Path(source).expanduser()
    elif Path("task7_artifacts.zip").is_file():
        candidate = Path("task7_artifacts.zip")
    else:
        runs = sorted(p.parent for p in Path("task7_FINAL_outputs").glob("*/manifest.json")
                      if (p.parent / "task7_FINAL_lstm.keras").is_file())
        if len(runs) != 1:
            raise FileNotFoundError(
                f"Found {len(runs)} completed updated Task 7 folders. Set TASK7_SOURCE to your "
                "completed run folder or ZIP, or upload its ZIP as task7_artifacts.zip. "
                "This notebook will not train Task 7 or use old models automatically.")
        candidate = runs[0]
    if not candidate.exists():
        raise FileNotFoundError(f"Task 7 source does not exist: {candidate}")
    if candidate.is_file():
        if not zipfile.is_zipfile(candidate):
            raise ValueError("TASK7_SOURCE must be a Task 7 ZIP or an extracted folder.")
        cache = Path("task8_shap_inputs") / sha256(candidate)[:16]
        cache.mkdir(parents=True, exist_ok=True)
        root = cache.resolve()
        with zipfile.ZipFile(candidate) as archive:
            for entry in archive.infolist():
                # Check Windows and POSIX separators, absolute paths and links.
                normalized = entry.filename.replace("\\", "/")
                if (normalized.startswith("/") or ":" in normalized
                        or not (root / normalized).resolve().is_relative_to(root)
                        or ((entry.external_attr >> 16) & 0o170000) == 0o120000):
                    raise ValueError(f"Unsafe ZIP entry: {entry.filename}")
            archive.extractall(root)
        manifests = list(root.rglob("manifest.json"))
        if len(manifests) != 1:
            raise ValueError("ZIP must contain exactly one completed Task 7 run.")
        candidate = manifests[0].parent
    required = ["manifest.json", "task7_FINAL_lstm.keras", "task7_basic_mlp.keras",
                "power_scaler.joblib", "task8_examples.npz", "test_predictions.csv"]
    absent = [name for name in required if not (candidate / name).is_file()]
    if absent:
        raise FileNotFoundError(f"Incomplete updated Task 7 run; missing: {absent}")
    return candidate.resolve()


def forecast_tensor(outputs):
    if isinstance(outputs, dict):
        return outputs["forecast_scaled"]
    if isinstance(outputs, (tuple, list)):
        raise ValueError("Ambiguous model outputs; expected forecast tensor or named output dictionary.")
    return outputs


def make_forecast_view(network, history_shape, context_shape, mean, scale):
    # A read-only view of existing weights. Fix inference mode to disable dropout.
    history = keras.Input(history_shape, name="explain_history")
    context = keras.Input(context_shape, name="explain_context")
    output = forecast_tensor(network({"history": history, "context": context}, training=False))
    output = keras.layers.Rescaling(scale=scale, offset=mean)(output)
    output = keras.layers.ReLU()(output)  # Same clipping as Task 7 prediction_outputs.
    return keras.Model([history, context], output, name=network.name + "_forecast_kw")


def predict_view(view, inputs, batch_size=128):
    return np.concatenate([np.asarray(view([inputs["history"][i:i + batch_size],
                                           inputs["context"][i:i + batch_size]], training=False)).reshape(-1)
                           for i in range(0, len(inputs["history"]), batch_size)])


def normalize_shap_values(values, inputs):
    # SHAP >=0.45 commonly returns a trailing singleton output dimension.
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError("Expected separate SHAP arrays for history and context.")
    normalized = []
    for value, original in zip(values, inputs):
        value = np.asarray(value)
        if value.shape == original.shape + (1,):
            value = value[..., 0]
        if value.shape != original.shape or not np.isfinite(value).all():
            raise ValueError(f"Unexpected SHAP shape/content: {value.shape}; expected {original.shape}")
        normalized.append(value)
    return normalized


def compute_shap(view, reference, inputs, seed=SEED, nsamples=SHAP_NSAMPLES,
                 max_samples=SHAP_MAX_SAMPLES, batch_size=SHAP_BATCH_SIZE):
    bg = [reference["history"], reference["context"]]
    xx = [inputs["history"], inputs["context"]]
    explainer = shap.GradientExplainer(view, bg, batch_size=batch_size, local_smoothing=0)
    base_value = float(predict_view(view, reference).mean())
    prediction = predict_view(view, inputs)
    started = time.perf_counter()
    phi = normalize_shap_values(explainer.shap_values(xx, nsamples=nsamples, rseed=seed), xx)
    samples_used = np.full(len(prediction), nsamples, dtype=int)
    tolerance = ADDITIVITY_ABS_TOL_KW + ADDITIVITY_REL_TOL * np.abs(prediction - base_value)
    budget = nsamples
    while True:
        summed = phi[0].sum(axis=(1, 2)) + phi[1].sum(axis=1)
        residual = prediction - base_value - summed
        flagged = np.flatnonzero(np.abs(residual) > tolerance)
        if not len(flagged) or budget >= max_samples:
            break
        budget = min(budget * 2, max_samples)
        subset = [array[flagged] for array in xx]
        refined = normalize_shap_values(explainer.shap_values(subset, nsamples=budget, rseed=seed), subset)
        for target, update in zip(phi, refined):
            target[flagged] = update
        samples_used[flagged] = budget
    checks = pd.DataFrame({"Forecast_kW": prediction, "Reference_mean_kW": base_value,
                           "SHAP_sum_kW": summed, "Residual_kW": residual,
                           "Tolerance_kW": tolerance, "Within_tolerance": np.abs(residual) <= tolerance,
                           "Monte_Carlo_samples": samples_used})
    return phi, base_value, checks, time.perf_counter() - started


def regression_scores(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if not len(y):
        return {"Samples": 0, "MAE_kW": np.nan, "RMSE_kW": np.nan, "R2": np.nan}
    return {"Samples": len(y), "MAE_kW": float(mean_absolute_error(y, p)),
            "RMSE_kW": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p)) if len(y) > 1 and np.var(y) > 0 else np.nan}


def choose_cases(y, prediction, threshold):
    order = np.argsort(np.abs(y - prediction), kind="stable")
    high = np.flatnonzero(y >= threshold)
    if len(high):
        peak_index = int(high[np.argmax((y - prediction)[high])])
        peak_label = "Largest high-demand underprediction" if y[peak_index] > prediction[peak_index] else "High-demand example"
    else:
        peak_index, peak_label = int(order[-1]), "Largest error (no high-demand example in pool)"
    return {"Smallest error": int(order[0]), "Median error": int(order[len(order) // 2]), peak_label: peak_index}


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def save_figure(name):
    plt.gcf().savefig(OUTPUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close("all")

# %% [markdown]
# ## 2. Load and verify the saved run
# We check timestamps, test targets, training-reference provenance and prediction
# agreement before explaining anything. A compact LSTM remains valid if Task 7
# selected it instead of the multitask candidate; its identity is printed below.

# %%
ARTIFACT_DIR = resolve_artifacts(TASK7_SOURCE)
manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
if manifest["run_mode"] != "full":
    raise ValueError("Use a completed full Task 7 run, not smoke-test results.")
SEQUENCE_LENGTH = int(manifest["sequence_length"])
HORIZON = int(manifest["horizon_steps"])
INTERVAL_MINUTES = int(manifest["interval_minutes"])
history_features, context_features = manifest["history_features"], manifest["context_features"]
selected_name = manifest["selection"]["selected_lstm"]
peak_threshold = float(manifest["peak_threshold_kw"])
model_files = {"LSTM": "task7_FINAL_lstm.keras", "Basic neural network": "task7_basic_mlp.keras"}
source_hashes = {filename: sha256(ARTIFACT_DIR / filename) for filename in
                 [*model_files.values(), "manifest.json", "power_scaler.joblib", "task8_examples.npz", "test_predictions.csv"]}
models = {name: keras.models.load_model(ARTIFACT_DIR / filename, compile=False)
          for name, filename in model_files.items()}
scaler = joblib.load(ARTIFACT_DIR / "power_scaler.joblib")
POWER_MEAN, POWER_STD = float(scaler.mean_[0]), float(scaler.scale_[0])
with np.load(ARTIFACT_DIR / "task8_examples.npz", allow_pickle=False) as data:
    examples = {key: data[key] for key in data.files}
X = {key: np.asarray(examples[f"explain_{key}"], dtype=np.float32) for key in ["history", "context"]}
reference_all = {key: np.asarray(examples[f"background_{key}"], dtype=np.float32) for key in X}
actual = np.asarray(examples["explain_actual_kw"]).reshape(-1)
target_times = pd.DatetimeIndex(pd.to_datetime(examples["explain_target_timestamps_ns"]))
assert len(actual) > 0 and X["history"].shape == (len(actual), SEQUENCE_LENGTH, len(history_features))
assert X["context"].shape == (len(actual), len(context_features))
assert len(reference_all["history"]) == len(reference_all["context"]) > 0
assert reference_all["history"].shape[1:] == X["history"].shape[1:]
assert reference_all["context"].shape[1:] == X["context"].shape[1:]
assert all(np.isfinite(a).all() for a in [*X.values(), *reference_all.values(), actual])
assert len(examples["background_indices"]) == len(reference_all["history"])
assert len(examples["explain_indices"]) == len(actual)
assert np.max(examples["background_indices"]) < np.min(examples["explain_indices"])
# The saved prepared arrays establish exact membership in the original split.
arrays_path = ARTIFACT_DIR / "prepared_arrays.npz"
if not arrays_path.is_file():
    raise FileNotFoundError("Keep prepared_arrays.npz in the Task 7 ZIP to verify reference/test membership.")
with np.load(arrays_path, allow_pickle=False) as arrays:
    assert np.isin(examples["background_indices"], arrays["train_indices"]).all()
    assert np.isin(examples["explain_indices"], arrays["test_indices"]).all()
    np.testing.assert_array_equal(examples["explain_target_timestamps_ns"], arrays["timestamps_ns"][examples["explain_indices"]])
saved = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv", parse_dates=["Forecast_origin", "Target_interval_end"])
assert saved["Target_interval_end"].is_unique and saved["Target_interval_end"].is_monotonic_increasing
assert len(saved) == manifest["used_split_counts"]["test"]
assert (saved["Target_interval_end"] - saved["Forecast_origin"]).eq(pd.Timedelta(minutes=HORIZON * INTERVAL_MINUTES)).all()
matched = saved.set_index("Target_interval_end").loc[target_times]
np.testing.assert_allclose(actual, matched["Actual_kW"], atol=1e-6)
views = {name: make_forecast_view(model, X["history"].shape[1:], X["context"].shape[1:], POWER_MEAN, POWER_STD)
         for name, model in models.items()}
columns = {"LSTM": f"{selected_name}_kW", "Basic neural network": "basic_mlp_kW"}
for baseline in ["Persistence", "Daily seasonal", "Weekly seasonal"]:
    if f"{baseline}_kW" in saved:
        columns[baseline] = f"{baseline}_kW"
assert np.isfinite(saved[["Actual_kW", *columns.values()]].to_numpy()).all()
predictions = {name: predict_view(view, X) for name, view in views.items()}
for name, prediction in predictions.items():
    np.testing.assert_allclose(prediction, matched[columns[name]], atol=2e-5, rtol=2e-5)
OUTPUT_DIR = Path("task8_shap_outputs") / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
print("Task 7 run:", manifest["run_id"], "| Selected LSTM:", selected_name)
print(f"Verified {len(actual)} reloaded predictions per neural model; no training performed.")
print(f"High-demand threshold from training: {peak_threshold:.3f} kW")
print("Output folder:", OUTPUT_DIR.resolve())

# %% [markdown]
# ## 3. Compare performance on the complete test set
# MAE is average absolute error. RMSE gives greater weight to large errors.
# Lower is better for both; higher R² is better. High-demand rows are selected
# using the threshold learned in Task 7's training period. Segment R² can be
# negative or undefined and should not be compared directly with overall R².
# These tables, rather than the small SHAP sample, establish predictive accuracy.

# %%
metric_rows = []
for segment, mask in {"All test": np.ones(len(saved), dtype=bool),
                      "High demand": saved["Actual_kW"].to_numpy() >= peak_threshold}.items():
    for name, column in columns.items():
        metric_rows.append({"Model": name, "Segment": segment,
                            **regression_scores(saved.loc[mask, "Actual_kW"], saved.loc[mask, column])})
performance = pd.DataFrame(metric_rows)
performance.to_csv(OUTPUT_DIR / "performance_comparison.csv", index=False)
display(performance.round(4))
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
overall = performance[performance["Segment"].eq("All test")].set_index("Model")
for ax, metric in zip(axes, ["MAE_kW", "RMSE_kW", "R2"]):
    bars = ax.bar(overall.index, overall[metric], color=["#176b87", "#31a6a0"] + ["#a5afbc"] * (len(overall) - 2))
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_title(metric.replace("_", " "))
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.axhline(0, color="black", linewidth=.6)
    ax.margins(y=.2)
fig.suptitle("Full-test forecasting performance")
fig.tight_layout()
save_figure("performance_comparison")
view = saved.set_index("Target_interval_end")
start = view.index[0]
grid = pd.date_range(start, start + pd.Timedelta(days=2), freq=f"{INTERVAL_MINUTES}min")
view = view.reindex(grid)  # Leave gaps visible rather than bridging missing readings.
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(view.index, view["Actual_kW"], color="black", label="Actual", linewidth=1.5)
for name in models:
    ax.plot(view.index, view[columns[name]], label=name, alpha=.8)
ax.axhline(peak_threshold, linestyle=":", color="gray", label="High-demand threshold")
ax.set(title="First 48 hours of test forecasts", ylabel="Average power (kW)", xlabel="Target interval end")
ax.legend(ncol=2)
fig.autofmt_xdate()
fig.tight_layout()
save_figure("actual_vs_predicted")

# %% [markdown]
# ## 4. Select shared examples and training references
# Both neural models use exactly the same reference examples, test examples and
# explanation settings. Summary plots use only the evenly spaced sample.
# Diagnostic cases deliberately include successes and errors; they do not enter
# summary averages unless already selected by the even-spacing rule.

# %%
global_positions = np.unique(np.linspace(0, len(actual) - 1, min(GLOBAL_EXAMPLES, len(actual)), dtype=int))
cases = choose_cases(actual, predictions["LSTM"], peak_threshold)
explain_positions = np.unique(np.r_[global_positions, list(cases.values())])
position_to_row = {int(position): row for row, position in enumerate(explain_positions)}
global_rows = np.array([position_to_row[int(position)] for position in global_positions])
bg_positions = np.unique(np.linspace(0, len(reference_all["history"]) - 1,
                                     min(BACKGROUND_EXAMPLES, len(reference_all["history"])), dtype=int))
reference = {key: value[bg_positions] for key, value in reference_all.items()}
explain_inputs = {key: value[explain_positions] for key, value in X.items()}
case_table = pd.DataFrame([{"Case": label, "Example_position": position, "Target_time": target_times[position],
                            "Actual_kW": actual[position],
                            **{name + "_kW": prediction[position] for name, prediction in predictions.items()}}
                           for label, position in cases.items()])
case_table.to_csv(OUTPUT_DIR / "diagnostic_cases.csv", index=False)
display(case_table.round(4))
print(f"SHAP: {len(explain_positions)} examples per model, {len(bg_positions)} training references.")

# %% [markdown]
# ## 5. Calculate SHAP and check explanation sums
# For each example: **reference mean forecast + all SHAP contributions ≈ forecast**.
# Values are in kW. Positive contributions raise the forecast relative to the
# reference mean; negative contributions lower it. Monte Carlo approximation means
# the sum is not exact. Flagged examples are retried with more samples, then any
# remaining discrepancy is saved and shown, never redistributed among features.
# Passing this check verifies approximate reconstruction, not causal correctness.

# %%
explanations, base_values, checks_by_model, elapsed_by_model = {}, {}, {}, {}
for name, view in views.items():
    print(f"Calculating SHAP for {name}...", flush=True)
    phi, base, checks, elapsed = compute_shap(view, reference, explain_inputs)
    checks.insert(0, "Example_position", explain_positions)
    checks.insert(1, "Target_time", target_times[explain_positions])
    checks.insert(0, "Model", name)
    checks.to_csv(OUTPUT_DIR / f"{slug(name)}_shap_checks.csv", index=False)
    np.savez_compressed(OUTPUT_DIR / f"{slug(name)}_shap_values.npz", history=phi[0], context=phi[1],
                        base_value_kw=base, example_positions=explain_positions,
                        task7_target_indices=examples["explain_indices"][explain_positions],
                        reference_target_indices=examples["background_indices"][bg_positions])
    explanations[name], base_values[name], checks_by_model[name], elapsed_by_model[name] = phi, base, checks, elapsed
    print(f"{name}: {checks['Within_tolerance'].sum()}/{len(checks)} sums within tolerance; {elapsed:.1f} seconds.")
checks_all = pd.concat(checks_by_model.values(), ignore_index=True)
checks_all.to_csv(OUTPUT_DIR / "shap_additivity_checks.csv", index=False)
display(checks_all.groupby("Model").agg(Examples=("Within_tolerance", "size"),
                                       Checks_passed=("Within_tolerance", "sum"),
                                       Max_absolute_residual_kW=("Residual_kW", lambda x: x.abs().max())))

# %% [markdown]
# ## 6. Which features and past times contribute?
# These are summaries of the sampled forecasts, not the whole population.
# For each history channel we first sum its signed contributions over time,
# then average the absolute totals across examples. This preserves the additive
# grouping used in local waterfalls, but opposite effects at different times
# can cancel. The time-feature heatmaps separately show mean absolute individual
# contributions without that cancellation. Correlated features can share credit.

# %%
group_names = ["History: " + name for name in history_features] + ["Context: " + name for name in context_features]
grouped = {name: np.concatenate([phi[0].sum(axis=1), phi[1]], axis=1) for name, phi in explanations.items()}
importance = pd.DataFrame({name: np.abs(values[global_rows]).mean(axis=0) for name, values in grouped.items()}, index=group_names)
importance.index.name = "Feature_group"
importance.to_csv(OUTPUT_DIR / "shap_feature_importance.csv")
top_groups = importance.mean(axis=1).nlargest(15).index[::-1]
ax = importance.loc[top_groups].plot.barh(figsize=(11, 7), color=["#176b87", "#31a6a0"])
ax.set(xlabel="Mean absolute grouped SHAP contribution (kW)", ylabel="", title="Same sampled forecasts: LSTM vs basic neural network")
plt.tight_layout()
save_figure("shap_feature_importance")
age_hours = np.arange(SEQUENCE_LENGTH - 1, -1, -1) * INTERVAL_MINUTES / 60
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
common_max = max(float(np.abs(phi[0][global_rows]).mean(axis=0).max()) for phi in explanations.values())
ticks = np.unique(np.linspace(0, SEQUENCE_LENGTH - 1, 7, dtype=int))
for ax, (name, phi) in zip(axes, explanations.items()):
    values = np.abs(phi[0][global_rows]).mean(axis=0).T
    img = ax.imshow(values, aspect="auto", cmap="magma", vmin=0, vmax=common_max or 1)
    ax.set_yticks(np.arange(len(history_features)), history_features, fontsize=8)
    ax.set_title(name)
    fig.colorbar(img, ax=ax, label="Mean |SHAP| (kW)")
axes[-1].set_xticks(ticks, [f"{age_hours[t]:g}" for t in ticks])
axes[-1].set_xlabel("Hours before forecast origin (0 = latest available interval)")
fig.tight_layout()
save_figure("shap_time_feature_heatmaps")
fig, ax = plt.subplots(figsize=(12, 4))
time_importance = {}
for name, phi in explanations.items():
    values = np.abs(phi[0][global_rows]).sum(axis=2).mean(axis=0)
    time_importance[name] = values
    ax.plot(-age_hours, values, label=name)
ax.set(xlabel="Hours relative to forecast origin", ylabel="Mean sum of |SHAP| (kW)", title="Contribution magnitude across history; not attention weights")
ax.legend()
fig.tight_layout()
save_figure("shap_time_importance")
pd.DataFrame({"Hours_before_origin": age_hours, **time_importance}).to_csv(OUTPUT_DIR / "shap_time_importance.csv", index=False)

# %% [markdown]
# ## 7. Explain individual forecasts
# Waterfalls group each history channel across time and keep each context feature
# separate. They reconstruct the forecast approximately; titles report the true
# forecast and any residual. The plotted SHAP endpoint can differ from it when
# Monte Carlo error remains. These explanations describe model behavior, not
# physical causes or proven appliance-saving opportunities.

# %%
for label, position in cases.items():
    row = position_to_row[position]
    for name in models:
        explanation = shap.Explanation(values=grouped[name][row], base_values=base_values[name], feature_names=group_names)
        shap.plots.waterfall(explanation, max_display=12, show=False)
        check = checks_by_model[name].iloc[row]
        status = "within tolerance" if bool(check["Within_tolerance"]) else "FLAGGED approximation"
        plt.gcf().set_size_inches(12, 7)
        plt.title(f"{name}: {label}\nActual {actual[position]:.3f} kW; forecast {predictions[name][position]:.3f} kW\n"
                  f"Sum residual {check['Residual_kW']:+.3f} kW ({status})", fontsize=10)
        plt.tight_layout()
        save_figure(f"shap_waterfall_{slug(label)}_{slug(name)}")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    scale = max(float(np.abs(phi[0][row]).max()) for phi in explanations.values()) or 1
    for ax, (name, phi) in zip(axes, explanations.items()):
        img = ax.imshow(phi[0][row].T, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
        ax.set_yticks(np.arange(len(history_features)), history_features, fontsize=8)
        ax.set_title(f"{label} — {name}")
        fig.colorbar(img, ax=ax, label="Signed SHAP (kW)")
    axes[-1].set_xticks(ticks, [f"{age_hours[t]:g}" for t in ticks])
    axes[-1].set_xlabel("Hours before forecast origin")
    fig.tight_layout()
    save_figure(f"shap_local_time_heatmap_{slug(label)}")

# %% [markdown]
# ## 8. LSTM activation heatmaps
# Each row is a unit in the first LSTM layer; each column is a history time step.
# Color is the unit's hidden-state value, not a SHAP contribution or attention
# weight. A strong activation does not by itself prove importance to the forecast.
# The basic neural network's dense activations are shown separately: it does not
# have a recurrent hidden state evolving across the 96 steps. Unit numbers have
# no matching meaning between the two networks, so magnitudes are not ranked
# against one another as a measure of interpretability.

# %%
case_positions = list(cases.values())
case_inputs = {key: value[case_positions] for key, value in X.items()}
for name, layer_name in [("LSTM", "lstm_temporal_features"), ("Basic neural network", "dense_features")]:
    network = models[name]
    activation_view = keras.Model(network.inputs, network.get_layer(layer_name).output)
    input_names = [tensor.name.split(":")[0] for tensor in network.inputs]
    activations = np.asarray(activation_view([case_inputs[key] for key in input_names], training=False))
    np.savez_compressed(OUTPUT_DIR / f"{slug(name)}_activations.npz", values=activations,
                        example_positions=case_positions, layer_name=layer_name)
    if name == "LSTM":
        assert activations.shape[1] == SEQUENCE_LENGTH
        fig, axes = plt.subplots(len(cases), 1, figsize=(13, 3 * len(cases)), squeeze=False)
        limit = float(np.abs(activations).max()) or 1
        for ax, (label, _), value in zip(axes[:, 0], cases.items(), activations):
            img = ax.imshow(value.T, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.set(title=label, ylabel="LSTM unit")
            ax.set_xticks(ticks, [f"{age_hours[t]:g}" for t in ticks])
            fig.colorbar(img, ax=ax, label="Hidden activation")
        axes[-1, 0].set_xlabel("Hours before forecast origin")
    else:
        fig, ax = plt.subplots(figsize=(13, 3))
        img = ax.imshow(activations, aspect="auto", cmap="viridis")
        ax.set_yticks(np.arange(len(cases)), list(cases), fontsize=8)
        ax.set(xlabel="Dense unit", title="Basic neural network: dense-layer activations")
        fig.colorbar(img, ax=ax, label="Activation")
    fig.tight_layout()
    save_figure(f"{slug(name)}_activation_heatmaps")

# %% [markdown]
# ## 9. Compare interpretability and record limitations
# We compare explanation cost, reconstruction checks and prominent input groups.
# A smaller SHAP residual or fewer parameters does not establish that one model
# is inherently more understandable. SHAP is reference-dependent, correlated
# features complicate credit allocation, and interpolated reference/input points
# may not correspond to real household states. Sampling also introduces variation.
# Neither SHAP nor activations establish causality. No attention mechanism is
# claimed. Task 7's existing test performance is reported without retuning.

# %%
interpretability_rows = []
for name in models:
    check = checks_by_model[name]
    interpretability_rows.append({"Model": name, "Parameters": models[name].count_params(),
        "Explained_examples": len(explain_positions), "Summary_examples": len(global_positions),
        "Training_references": len(bg_positions), "SHAP_seconds": elapsed_by_model[name],
        "Additivity_checks_passed": int(check["Within_tolerance"].sum()),
        "Flagged_explanations": int((~check["Within_tolerance"]).sum()),
        "Mean_absolute_residual_kW": float(check["Residual_kW"].abs().mean()),
        "Top_sampled_group": importance[name].idxmax(),
        "Internal_visualization": "Temporal LSTM hidden states" if name == "LSTM" else "Dense activations"})
interpretability = pd.DataFrame(interpretability_rows)
interpretability.to_csv(OUTPUT_DIR / "interpretability_comparison.csv", index=False)
display(interpretability)
lstm_rmse, basic_rmse = overall.loc["LSTM", "RMSE_kW"], overall.loc["Basic neural network", "RMSE_kW"]
gain = 100 * (1 - lstm_rmse / basic_rmse) if basic_rmse else np.nan
verdict = "lower" if lstm_rmse < basic_rmse else "higher" if lstm_rmse > basic_rmse else "equal"
flagged_count = int((~checks_all["Within_tolerance"]).sum())
report = f"""# Task 8 — SHAP interpretation and model comparison

Task 7 run: {manifest['run_id']}. Selected LSTM: {selected_name}.
No model training, tuning or weight changes were performed.

## Performance
Full-test LSTM RMSE is {lstm_rmse:.4f} kW; basic neural network RMSE is {basic_rmse:.4f} kW.
The LSTM has {verdict} RMSE; relative improvement versus the basic model is {gain:.2f}%.
See performance_comparison.csv for all-test and high-demand MAE, RMSE, R2 and counts.

## Interpretation
SHAP GradientExplainer explains the clipped power forecast in kW, using the same
{len(bg_positions)} training reference examples for both networks. Summary plots
cover {len(global_positions)} evenly spaced examples from the saved test pool.
Diagnostic cases are chosen separately and do not bias summary averages.
LSTM activations show temporal internal responses; dense activations show the
basic network's responses. Activations are not contribution scores or attention.

{flagged_count} of {len(checks_all)} SHAP reconstructions remain outside the stated
tolerance. Any flagged local explanation requires caution; residuals are not
hidden or assigned to a feature. Reconstruction checks alone do not validate an
explanation's causal meaning, stability or behavior away from these examples.

Both models support feature/time attribution. The LSTM additionally supplies
temporal hidden-state visualizations. Neither model is declared inherently more
interpretable based solely on accuracy, activation intensity or reconstruction.
Prominent sampled groups and explanation costs appear in interpretability_comparison.csv.

## Limits
Expected-gradients SHAP estimates depend on the background and Monte Carlo draws.
Correlated inputs and interpolated, potentially unrealistic household states limit
interpretation. The sample is small, temporal, and from one household dataset;
these are model explanations, not causal appliance-control recommendations.
Results explain the selected Task 7 model, which need not be the peak-gated candidate.
"""
(OUTPUT_DIR / "task8_report.md").write_text(report, encoding="utf-8")
analysis_manifest = {"task7_run_id": manifest["run_id"], "selected_lstm": selected_name,
    "source_folder": str(ARTIFACT_DIR), "source_sha256": source_hashes,
    "method": "SHAP GradientExplainer (expected gradients)", "explained_output": "clipped power forecast in kW",
    "seed": SEED, "background_positions": bg_positions.tolist(), "summary_positions": global_positions.tolist(),
    "explain_positions": explain_positions.tolist(), "diagnostic_cases": cases,
    "nsamples_initial": SHAP_NSAMPLES, "nsamples_max": SHAP_MAX_SAMPLES, "batch_size": SHAP_BATCH_SIZE,
    "additivity_absolute_tolerance_kw": ADDITIVITY_ABS_TOL_KW, "additivity_relative_tolerance": ADDITIVITY_REL_TOL,
    "flagged_explanations": flagged_count, "training_performed": False,
    "versions": {"python": sys.version.split()[0], "tensorflow": tf.__version__, "keras": keras.__version__,
                 "shap": shap.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
(OUTPUT_DIR / "analysis_manifest.json").write_text(json.dumps(analysis_manifest, indent=2), encoding="utf-8")
for filename, digest in source_hashes.items():
    assert sha256(ARTIFACT_DIR / filename) == digest, f"Source artifact unexpectedly changed: {filename}"
archive_path = shutil.make_archive(str(OUTPUT_DIR), "zip", root_dir=OUTPUT_DIR)
print(report)
print("Task 8 charts, explanation arrays, checks, tables and report:", OUTPUT_DIR.resolve())
print("Downloadable archive:", Path(archive_path).resolve())

# %% [markdown]
# ## Submission checklist
# Keep the executed notebook and generated ZIP. Report the actual selected model,
# full-test results, SHAP reference choice, sample size and any flagged sums.
# Include SHAP feature/time plots, at least one individual explanation, LSTM
# activations, and the basic-model comparison. Empty outputs mean the notebook
# has not yet been run against your saved Task 7 artifacts.
