"""Task 10 publication-style evaluation plots; saved alongside the report."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML


def save_eval_figure(output_dir, name):
    plt.gcf().savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close("all")


def plot_evaluation(frame, metrics, groups, detection, output_dir):
    overall = metrics[metrics["Segment"].eq("All test")].set_index("Model")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, column in zip(axes, ["MAE_kW", "RMSE_kW", "R2"]):
        bars = ax.bar(overall.index, overall[column], color=["#176b87", "#31a6a0", "#7da2bc", "#a7b0bc"][:len(overall)])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
        ax.set_title(column.replace("_", " "))
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.axhline(0, color="black", linewidth=.6)
        ax.margins(y=.2)
    fig.suptitle("Frozen system: previously evaluated holdout, reused transparently")
    fig.tight_layout()
    save_eval_figure(output_dir, "forecast_comparison")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    hourly = groups[groups["Grouping"].eq("Hour")]
    axes[0].bar(hourly["Group"].astype(int), hourly["MAE_kW"], color="#176b87")
    axes[0].set(title="Time-of-day error profile", xlabel="Target interval end hour", ylabel="MAE (kW)")
    weekend = groups[groups["Grouping"].eq("Weekend")]
    axes[1].bar(weekend["Group"].astype(str), weekend["MAE_kW"], color="#31a6a0")
    axes[1].set(title="Weekday/weekend error profile", ylabel="MAE (kW)")
    confusion = np.array([[detection["True_negative"], detection["False_positive"]],
                          [detection["False_negative"], detection["True_positive"]]])
    image = axes[2].imshow(confusion, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(confusion[i, j]), ha="center", va="center",
                         color="white" if confusion[i, j] > confusion.max()/2 else "black")
    axes[2].set_xticks([0, 1], ["Normal", "High"])
    axes[2].set_yticks([0, 1], ["Normal", "High"])
    axes[2].set(title="Thresholded hybrid forecast", xlabel="Predicted demand", ylabel="Actual demand")
    fig.colorbar(image, ax=axes[2], label="Examples")
    fig.tight_layout()
    save_eval_figure(output_dir, "error_profiles_and_peak_detection")
    fig, ax = plt.subplots(figsize=(10, 4))
    residual = frame["Actual_kW"]-frame["Hybrid_kW"]
    ax.hist(residual, bins=45, color="#176b87", alpha=.85)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(title="Hybrid forecast errors: positive = underprediction", xlabel="Actual − predicted (kW)", ylabel="Count")
    fig.tight_layout()
    save_eval_figure(output_dir, "forecast_error_distribution")


def plot_scenario_checks(checks, output_dir):
    valid_times = checks["Latency_ms"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = [int(checks["Passed"].sum()), int((~checks["Passed"]).sum())]
    axes[0].bar(["Passed", "Failed"], counts, color=["#31a6a0", "#ce5757"])
    axes[0].set(title="Designed scenario and invalid-input checks", ylabel="Check count")
    if len(valid_times):
        axes[1].hist(valid_times, bins=min(20, max(3, len(valid_times)//2)), color="#176b87")
    axes[1].set(title="Inference + reasoning + scheduling latency", xlabel="Milliseconds (SHAP/rendering excluded)", ylabel="Count")
    fig.tight_layout()
    save_eval_figure(output_dir, "system_checks_and_latency")


def plot_explanations(bridge, frame, xai, output_dir):
    import shap
    importance = xai["importance"].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(importance["Feature"], importance["Mean_abs_SHAP_kW"], color="#176b87")
    ax.set(xlabel="Mean absolute SHAP contribution (kW)", title=f"ML-stage importance over {xai['summary_examples']} evenly spaced test examples")
    fig.tight_layout()
    save_eval_figure(output_dir, "shap_global_importance")
    case_rows = []
    for label, position in xai["cases"].items():
        row = xai["lookup"][position]
        actual = frame.iloc[position]
        obj = shap.Explanation(values=xai["values"][row], base_values=xai["base"], feature_names=bridge.feature_names)
        shap.plots.waterfall(obj, max_display=12, show=False)
        plt.gcf().set_size_inches(12, 7)
        plt.title(f"{label}: actual {actual['Actual_kW']:.3f} kW; final forecast {actual['Hybrid_kW']:.3f} kW\n"
                  f"SHAP explains raw ML output; reconstruction residual {xai['checks'].iloc[row]['Residual_kW']:+.6f} kW", fontsize=10)
        plt.tight_layout()
        safe_label = label.lower().replace(" ", "_").replace("-", "_")
        save_eval_figure(output_dir, "shap_" + safe_label)
        activation = bridge.activation(int(actual["Target_index"]))
        fig, ax = plt.subplots(figsize=(12, 4))
        image = ax.imshow(activation.T, aspect="auto", cmap="RdBu_r")
        ticks = np.linspace(0, activation.shape[0]-1, 7, dtype=int)
        ax.set_xticks(ticks, [f"{(activation.shape[0]-1-t)*.25:g}" for t in ticks])
        ax.set(title=label + ": LSTM activations (not attention weights)", xlabel="Hours before forecast origin", ylabel="LSTM unit")
        fig.colorbar(image, ax=ax, label="Hidden-state activation")
        fig.tight_layout()
        save_eval_figure(output_dir, "activations_" + safe_label)
        case_rows.append({"Case": label, "Target_index": int(actual["Target_index"]), "Forecast_origin": str(actual["Forecast_origin"]),
                          "Target_interval_end": str(actual["Target_interval_end"]), "Actual_kW": actual["Actual_kW"],
                          "Predicted_kW": actual["Hybrid_kW"], "Absolute_error_kW": abs(actual["Actual_kW"]-actual["Hybrid_kW"])})
    return pd.DataFrame(case_rows)
