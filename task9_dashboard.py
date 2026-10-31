"""Interactive in-notebook controls and Task 9 result views."""
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, clear_output, display

if "Appliance" not in globals():
    from task9_core import Appliance, Scenario, default_appliances, optimize_decisions


def render_optimizer_result(result, bridge, explanation=None):
    forecast, plan = result["forecast"], result["plan"]
    high = forecast["forecast_kw"] >= forecast["threshold_kw"]
    saving = "Unavailable" if plan["estimated_saving"] is None else f"₹{plan['estimated_saving']:.2f}"
    cards = [("FORECAST AVERAGE POWER", f"{forecast['forecast_kw']:.3f} kW"),
             ("FORECAST DEMAND", "High" if high else "Normal"),
             ("SCHEDULE STATUS", plan["status"].replace("_", " ")),
             ("ESTIMATED JOB SAVING", saving)]
    markup = '<div style="display:flex;flex-wrap:wrap;gap:12px;font-family:Arial,sans-serif">'
    for label, value in cards:
        markup += f'<div style="background:#edf5fa;border-left:4px solid #176b87;padding:15px;min-width:185px;border-radius:8px"><small>{escape(label)}</small><br><strong style="font-size:23px">{escape(value)}</strong></div>'
    markup += '</div>'
    display(HTML(markup))
    display(HTML(f"<p><b>Origin:</b> {escape(forecast['origin'])} &nbsp; <b>Predicted interval:</b> "
                 f"{escape(forecast['target_interval_start'])} to {escape(forecast['target_end'])}<br>"
                 f"<b>Model:</b> {escape(forecast['model'])}. High demand starts at {forecast['threshold_kw']:.3f} kW.</p>"))
    display(HTML("<p>Historical electricity case with editable household scenario settings. "
                 "Savings cover the requested schedulable jobs versus a feasible earliest-start schedule; they are estimates, not a measured household bill.</p>"))
    rows = [{"Appliance": r["Appliance"], "Recommendation": r["Action"],
             "Suggested start": r.get("Suggested_start") or "Protected / inactive",
             "Reason": r["Reason"]} for r in result["recommendations"]]
    display(pd.DataFrame(rows))
    display(HTML(f"<p><b>Search:</b> {escape(plan['reason'])} Explored {plan['nodes']} A* states. "
                 "The concurrency setting counts scheduled jobs, not the entire household.</p>"))
    input_values = bridge.inputs(np.array([forecast["target_index"]]))
    history_names = bridge.manifest["history_features"]
    power_channel = next(i for i, name in enumerate(history_names) if "global_active_power" in name.lower())
    history_kw = input_values["history"][0, :, power_channel] * bridge.scale + bridge.mean
    origin, target = pd.Timestamp(forecast["origin"]), pd.Timestamp(forecast["target_end"])
    history_times = pd.date_range(end=origin, periods=len(history_kw), freq="15min")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history_times, history_kw, color="#176b87", label="Historical model inputs")
    axes[0].scatter([target], [forecast["forecast_kw"]], color="#d86526", s=65, label="One-hour-ahead target")
    axes[0].axvspan(target-pd.Timedelta(minutes=15), target, alpha=.15, color="#d86526")
    axes[0].set(title="Recent history and forecast target", ylabel="Average power (kW)")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=25)
    if plan["estimated_cost"] is not None:
        axes[1].bar(["Feasible earliest start", "A* schedule"], [plan["baseline_cost"], plan["estimated_cost"]], color=["#94a3b8", "#31a6a0"])
        axes[1].set(title="Estimated cost of scheduled jobs", ylabel="Scenario cost (₹)")
    else:
        axes[1].text(.5, .5, "No feasible cost estimate\nReview scheduling constraints", ha="center", va="center", transform=axes[1].transAxes)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    if plan["schedule"]:
        schedule = pd.DataFrame(plan["schedule"])
        display(schedule)
        fig, ax = plt.subplots(figsize=(12, 3.5))
        for i, row in schedule.iterrows():
            start = (pd.Timestamp(row["Start"]) - origin).total_seconds() / 3600
            duration = (pd.Timestamp(row["End"]) - pd.Timestamp(row["Start"])).total_seconds() / 3600
            ax.barh(i, duration, left=start, color="#31a6a0", height=.55)
        ax.set_yticks(range(len(schedule)), schedule["Appliance"])
        tick_values = np.arange(result["scenario"]["horizon_hours"] + 1)
        ax.set_xticks(tick_values, [(origin + pd.Timedelta(hours=int(h))).strftime("%H:%M") for h in tick_values])
        if high:
            ax.axvspan(.75, 1., alpha=.15, color="red", label="Single forecast high-demand interval")
            ax.legend(fontsize=8)
        ax.set(xlabel=f"Planning clock from {origin.date()}", title="Requested-job schedule")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    with_details = "<details><summary><b>Why were these decisions made? — ontology and operational rule trace</b></summary>"
    for row in result["recommendations"]:
        with_details += f"<p><b>{escape(row['Appliance'])}</b>: {escape(' → '.join(row['Rule_trace']))}<br>"
        with_details += f"OWL candidate decisions: {escape(', '.join(row['OWL_candidates']) or 'None')}. "
        with_details += f"Final decision: {escape(row['Action'])}. {escape(row['Reason'])}</p>"
    display(HTML(with_details + "</details>"))
    if explanation is not None:
        import shap
        display(HTML("<h4>SHAP: why the ML prediction stage produced this forecast</h4>"
                     "<p>Features include learned LSTM patterns and context inputs. These SHAP values explain the ML stage, "
                     "not separate causal effects of raw sensors. LSTM activations are shown below.</p>"))
        obj = shap.Explanation(values=explanation["values"], base_values=explanation["base"],
                               feature_names=explanation["features"])
        shap.plots.waterfall(obj, max_display=12, show=False)
        fig = plt.gcf()
        fig.set_size_inches(12, 7)
        plt.title(f"Raw ML forecast: {explanation['raw_prediction']:.3f} kW; final clipped forecast: {forecast['forecast_kw']:.3f} kW", fontsize=10)
        fig.tight_layout()
        plt.show()
        plt.close(fig)
        activation = bridge.activation(forecast["target_index"])
        fig, ax = plt.subplots(figsize=(12, 4))
        image = ax.imshow(activation.T, aspect="auto", cmap="RdBu_r")
        ticks = np.linspace(0, activation.shape[0]-1, 7, dtype=int)
        ax.set_xticks(ticks, [f"{(activation.shape[0]-1-t)*.25:g}" for t in ticks])
        ax.set(title="LSTM temporal activations — internal response, not attention weights", xlabel="Hours before forecast origin", ylabel="LSTM unit")
        fig.colorbar(image, ax=ax, label="Hidden-state activation")
        fig.tight_layout()
        plt.show()
        plt.close(fig)


def build_optimizer_dashboard(bridge, knowledge, output_dir, performance=None):
    from dataclasses import asdict
    # Defined in task9_core, available as imports locally or as earlier notebook definitions.
    ids = bridge._sample(bridge.test_ids, 64)
    evening = bridge.test_ids[bridge.timestamps[bridge.test_ids - 4].hour == 18]
    if len(evening):
        ids = np.unique(np.r_[ids, evening[0]])
    case = widgets.Dropdown(options=[(str(bridge.timestamps[int(i)-4]), int(i)) for i in ids],
                            value=int(evening[0] if len(evening) else ids[0]), description="History case:", layout=widgets.Layout(width="470px"))
    occupancy = widgets.IntSlider(value=3, min=0, max=10, description="People home:")
    temperature = widgets.FloatSlider(value=35, min=0, max=50, step=.5, description="Temp (°C):")
    peak_rate = widgets.BoundedFloatText(value=9, min=0, max=1000, description="Peak ₹/kWh:")
    offpeak_rate = widgets.BoundedFloatText(value=6, min=0, max=1000, description="Other ₹/kWh:")
    peak_start = widgets.IntSlider(value=18, min=0, max=23, description="Peak starts:")
    peak_end = widgets.IntSlider(value=20, min=0, max=23, description="Peak ends:")
    horizon = widgets.Dropdown(options=[4, 6, 8, 12], value=6, description="Plan hours:")
    concurrency = widgets.IntSlider(value=1, min=1, max=3, description="Parallel jobs:")
    appliance_forms, appliance_sections = [], []
    for appliance in default_appliances():
        initial = asdict(appliance)
        locked = "EssentialAppliance" in knowledge.infer(appliance)["types"]
        options = {"requested": widgets.Checkbox(value=appliance.requested, description="Requested"),
            "on": widgets.Checkbox(value=appliance.on, description="Already running"),
            "essential": widgets.Checkbox(value=appliance.essential or locked, description="Essential", disabled=locked),
            "flexible": widgets.Checkbox(value=appliance.flexible, description="Allow rescheduling"),
            "requires_occupancy": widgets.Checkbox(value=appliance.requires_occupancy, description="Needs people home"),
            "priority": widgets.Dropdown(value=appliance.priority, options=["Low", "Medium", "High"], description="Priority:"),
            "power_w": widgets.BoundedFloatText(value=appliance.power_w, min=1, max=20000, description="Power (W):"),
            "duration_min": widgets.BoundedIntText(value=appliance.duration_min, min=15, max=720, step=15, description="Runtime (min):"),
            "earliest_min": widgets.BoundedIntText(value=appliance.earliest_min, min=0, max=720, step=15, description="Earliest (min):"),
            "deadline_min": widgets.BoundedIntText(value=appliance.deadline_min, min=15, max=720, step=15, description="Finish by (min):")}
        for control in options.values():
            control.layout.width = "245px"
            if hasattr(control, "style"):
                control.style.description_width = "105px"
        grid = widgets.GridBox(list(options.values()), layout=widgets.Layout(
            grid_template_columns="repeat(3, minmax(245px, 1fr))", grid_gap="6px", width="100%", overflow="auto"))
        appliance_sections.append(grid)
        appliance_forms.append({"initial": initial, "controls": options})
    advanced = widgets.Accordion(children=appliance_sections)
    for i, appliance in enumerate(default_appliances()):
        advanced.set_title(i, appliance.name)
    appliance_help = widgets.HTML("<p>Expand an appliance to change its settings. Earliest start and finish deadline are minutes "
        "after the selected origin, in 15-minute steps. Defaults are scenario examples. Essential appliances remain protected.</p>")
    explain = widgets.Checkbox(value=True, description="Show SHAP and LSTM activations", indent=False)
    button = widgets.Button(description="Run optimizer", button_style="success", icon="play", layout=widgets.Layout(width="180px"))
    status, result_output = widgets.HTML(), widgets.Output()
    state = {"last_result": None, "last_path": None}

    def run(_=None):
        button.disabled = True
        status.value = "<p>Calculating forecast and schedule…</p>"
        try:
            appliance_values = [Appliance(**dict(form["initial"], **{key: control.value for key, control in form["controls"].items()}))
                                for form in appliance_forms]
            scenario = Scenario(occupancy=occupancy.value, temperature_c=temperature.value,
                peak_start_hour=peak_start.value, peak_end_hour=peak_end.value, peak_rate=peak_rate.value,
                offpeak_rate=offpeak_rate.value, horizon_hours=horizon.value, max_parallel_jobs=concurrency.value)
            forecast = bridge.predict(case.value)
            result = optimize_decisions(forecast, appliance_values, scenario, knowledge)
            explanation = bridge.explain(case.value) if explain.value else None
            folder = Path(output_dir) / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
            folder.mkdir(parents=True, exist_ok=False)
            report = dict(result, task7_run_id=bridge.manifest.get("run_id"), hybrid_signature=bridge.signature)
            if explanation is not None:
                report["ml_shap"] = {"base_kw": explanation["base"], "raw_forecast_kw": explanation["raw_prediction"],
                                     "features": explanation["features"], "values_kw": explanation["values"].tolist()}
            (folder / "optimizer_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            pd.DataFrame(result["recommendations"]).to_csv(folder / "recommendations.csv", index=False)
            pd.DataFrame(result["plan"]["schedule"]).to_csv(folder / "schedule.csv", index=False)
            with result_output:
                clear_output(wait=True)
                render_optimizer_result(result, bridge, explanation)
                if performance is not None:
                    display(HTML("<details><summary><b>Forecasting performance on the full saved test period</b></summary></details>"))
                    display(performance.round(4))
                print("Saved this scenario:", folder.resolve())
            state["last_result"], state["last_path"] = result, folder
            status.value = "<p style='color:#126b46'><b>Optimizer complete.</b> Change settings and run again to compare scenarios.</p>"
        except Exception as error:
            status.value = f"<p style='color:#b42318'><b>Inputs need attention:</b> {escape(str(error))}</p>"
            with result_output:
                clear_output(wait=True)
                print("No new result was produced. Review the message above.")
        finally:
            button.disabled = False

    button.on_click(run)
    header = widgets.HTML("<h2 style='color:#176b87'>Smart Electricity Consumption Optimizer</h2>"
        "<p><b>Task 9:</b> LSTM patterns → ML forecast → ontology/rules → A* schedule → explanation.</p>"
        "<p>Choose a recorded electricity case and enter a household scenario. Tariffs and appliance settings below are editable examples. "
        "The forecast follows the selected electricity history; household settings affect decisions and estimated costs. "
        "This dashboard provides recommendations; it does not switch hardware.</p>")
    controls = widgets.VBox([header, case,
        widgets.HBox([occupancy, temperature]), widgets.HBox([peak_rate, offpeak_rate]),
        widgets.HBox([peak_start, peak_end]), widgets.HBox([horizon, concurrency]), appliance_help, advanced, explain,
        widgets.HBox([button, status]), result_output], layout=widgets.Layout(width="100%"))
    return {"widget": controls, "run": run, "state": state,
            "controls": {"case": case, "occupancy": occupancy, "temperature": temperature,
                         "appliances": appliance_forms, "explain": explain, "peak_rate": peak_rate,
                         "offpeak_rate": offpeak_rate, "horizon": horizon, "concurrency": concurrency}}
