"""Task 10 evaluation helpers. Evaluation functions never fit or tune models."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

if "Appliance" not in globals():
    from task9_core import Appliance, Scenario, default_appliances, optimize_decisions, rate_at, file_hash


def freeze_fingerprint(bridge, knowledge):
    import joblib
    weights = hashlib.sha256()
    for value in bridge.network.get_weights():
        value = np.asarray(value)
        weights.update(str((value.shape, value.dtype)).encode())
        weights.update(value.tobytes())
    return {"lstm_weights_sha256": weights.hexdigest(), "ml_model_hash": joblib.hash(bridge.regressor),
            "task7_model_file_sha256": file_hash(bridge.folder / "task7_FINAL_lstm.keras"),
            "hybrid_cache_file_sha256": file_hash(bridge.cache / "hybrid_forecaster.joblib"),
            "ontology_sha256": knowledge.digest, "hybrid_signature": bridge.signature}


def collect_frozen_predictions(bridge):
    """Same held-out inputs as Task 9; unseen to fitting, already evaluated."""
    ids = np.asarray(bridge.test_ids, dtype=int)
    train = np.asarray(bridge.arrays["train_indices"])
    validation = np.asarray(bridge.arrays["validation_indices"])
    if np.intersect1d(ids, train).size or np.intersect1d(ids, validation).size:
        raise ValueError("Test indices overlap training or validation.")
    if not (train.max() < validation.min() - 4 and validation.max() < ids.min() - 4):
        raise ValueError("Temporal partition/gap check failed.")
    z = bridge.features(ids)
    prediction = np.maximum(bridge.regressor.predict(z), 0.)
    saved = pd.read_csv(bridge.folder / "test_predictions.csv", parse_dates=["Target_interval_end"])
    aligned = saved.set_index("Target_interval_end").loc[bridge.timestamps[ids]]
    np.testing.assert_allclose(bridge.y[ids], aligned["Actual_kW"], atol=1e-5)
    frame = pd.DataFrame({"Target_index": ids, "Forecast_origin": bridge.timestamps[ids-4],
        "Target_interval_end": bridge.timestamps[ids], "Actual_kW": bridge.y[ids],
        "Hybrid_kW": prediction, "Task7_LSTM_kW": aligned[bridge.manifest["selection"]["selected_lstm"] + "_kW"].to_numpy(),
        "Basic_NN_kW": aligned["basic_mlp_kW"].to_numpy(), "Persistence_kW": aligned["Persistence_kW"].to_numpy()})
    if not np.isfinite(frame[["Actual_kW", "Hybrid_kW", "Task7_LSTM_kW", "Basic_NN_kW", "Persistence_kW"]]).all().all():
        raise ValueError("Non-finite evaluation values.")
    return frame


def external_csv_predictions(bridge, csv_path, task7_preprocessors):
    """Apply frozen Task 7 transformations to later, supplied minute readings.

    Never fit scalers. Complete observed targets are required for scored rows.
    Origins must be later than the original raw data span, not merely nontraining.
    """
    import joblib
    frame = pd.read_csv(csv_path)
    features = list(bridge.manifest["electrical_features"])
    if set(["Datetime", *features]) - set(frame.columns):
        raise ValueError("External CSV needs Datetime and all seven original electrical columns.")
    stamps = pd.to_datetime(frame["Datetime"], errors="coerce")
    if stamps.isna().any() or stamps.dt.tz is not None or stamps.duplicated().any():
        raise ValueError("Use unique valid local-time Datetime values without a timezone suffix.")
    if (stamps.dt.second.ne(0) | stamps.dt.microsecond.ne(0)).any():
        raise ValueError("External observations must lie on minute boundaries.")
    minute = frame[features].apply(pd.to_numeric, errors="coerce")
    minute = minute.where(np.isfinite(minute) & minute.ge(0))
    minute.index = pd.DatetimeIndex(stamps)
    minute = minute.sort_index()
    functions = task7_preprocessors
    power = functions["prepare_multivariate"](minute, features, 4)
    context = functions["build_context"](power, 4)
    if list(context.columns) != list(bridge.manifest["context_features"]):
        raise ValueError("External feature names/order do not match the frozen model.")
    ids = functions["valid_target_indices"](power, context, 96, 4)
    old_end = pd.Timestamp(bridge.manifest["data_audit"]["raw_end"])
    ids = ids[power.index[ids - 4] > old_end]
    if not len(ids):
        raise ValueError("No scoreable new-time-range examples. Supply at least eight days of minute readings, including warm-up, with forecast origins after the original dataset ends.")
    scalers = joblib.load(bridge.folder / "electrical_scalers.joblib")
    context_scaler = joblib.load(bridge.folder / "context_scaler.joblib")
    scaled = [scalers[name].transform(power[["input_"+name]].to_numpy()).ravel() for name in features]
    calendar = functions["calendar_features"](power.index)
    history = np.column_stack([*scaled, power["electrical_observed_fraction"].to_numpy(),
        calendar[["hour_sin", "hour_cos", "dow_sin", "dow_cos"]].to_numpy()]).astype("float32")
    expected = [*(name + "_scaled" for name in features), "electrical_observed_fraction", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    if expected != bridge.manifest["history_features"]:
        raise ValueError("External history channels do not match the frozen model.")
    contexts = context_scaler.transform(context).astype("float32")
    rows = []
    for offset in range(0, len(ids), 256):
        part = ids[offset:offset+256]
        positions = part[:, None] - 4 - 96 + 1 + np.arange(96)[None, :]
        inputs = {"history": history[positions], "context": contexts[part]}
        if not all(np.isfinite(array).all() for array in inputs.values()):
            raise ValueError("Unavailable external inputs survived eligibility checks.")
        latent = np.asarray(bridge.encoder([inputs[k] for k in bridge.input_order], training=False))
        z = np.concatenate([latent, inputs["context"]], axis=1)
        hybrid = np.maximum(bridge.regressor.predict(z), 0)
        raw_lstm = bridge.network(inputs, training=False)
        if isinstance(raw_lstm, dict):
            raw_lstm = raw_lstm["forecast_scaled"]
        lstm = np.maximum(np.asarray(raw_lstm).reshape(-1)*bridge.scale+bridge.mean, 0)
        rows.append(pd.DataFrame({"Forecast_origin": power.index[part-4], "Target_interval_end": power.index[part],
            "Actual_kW": power["actual_kw"].iloc[part].to_numpy(), "Hybrid_kW": hybrid,
            "Task7_LSTM_kW": lstm, "Persistence_kW": power["input_kw"].iloc[part-4].to_numpy()}))
    return pd.concat(rows, ignore_index=True), {"csv_sha256": file_hash(csv_path), "source": "user-supplied later-time-range minute data",
        "scored_examples": len(ids), "origin_min": str(power.index[ids[0]-4]), "origin_max": str(power.index[ids[-1]-4]),
        "independence_limit": "Time-range separation is checked. Authenticity and absence of copied/relabelled measurements cannot be established automatically."}


def performance_tables(frame, threshold):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix
    names = {"Hybrid_kW": "Integrated LSTM + ML", "Task7_LSTM_kW": "Task 7 LSTM",
             "Basic_NN_kW": "Basic neural network", "Persistence_kW": "Persistence"}
    rows = []
    for segment, mask in {"All test": np.ones(len(frame), dtype=bool), "High demand": frame["Actual_kW"].ge(threshold).to_numpy()}.items():
        y = frame.loc[mask, "Actual_kW"].to_numpy()
        for column, label in names.items():
            if column not in frame:
                continue
            p = frame.loc[mask, column].to_numpy()
            rows.append({"Model": label, "Segment": segment, "Samples": len(y),
                "MAE_kW": float(mean_absolute_error(y, p)) if len(y) else np.nan,
                "RMSE_kW": float(np.sqrt(mean_squared_error(y, p))) if len(y) else np.nan,
                "R2": float(r2_score(y, p)) if len(y) > 1 and np.var(y) > 0 else np.nan})
    y_high, p_high = frame["Actual_kW"].ge(threshold), frame["Hybrid_kW"].ge(threshold)
    tn, fp, fn, tp = confusion_matrix(y_high, p_high, labels=[False, True]).ravel()
    detection = {"Threshold_kW": threshold, "Precision": float(precision_score(y_high, p_high, zero_division=0)),
        "Recall": float(recall_score(y_high, p_high, zero_division=0)), "F1": float(f1_score(y_high, p_high, zero_division=0)),
        "True_negative": int(tn), "False_positive": int(fp), "False_negative": int(fn), "True_positive": int(tp),
        "Definition": "Thresholded hybrid power forecast, not a calibrated probability"}
    return pd.DataFrame(rows), detection


def temporal_error_profiles(frame, threshold):
    work = frame.copy()
    work["Abs_error_kW"] = (work["Actual_kW"] - work["Hybrid_kW"]).abs()
    work["Squared_error"] = (work["Actual_kW"] - work["Hybrid_kW"]) ** 2
    work["Hour"] = pd.to_datetime(work["Target_interval_end"]).dt.hour
    work["Weekend"] = pd.to_datetime(work["Target_interval_end"]).dt.dayofweek.ge(5).map({True: "Weekend", False: "Weekday"})
    work["Demand"] = np.where(work["Actual_kW"].ge(threshold), "High demand", "Other demand")
    tables = []
    for grouping in ["Hour", "Weekend", "Demand"]:
        grouped = work.groupby(grouping).agg(Samples=("Abs_error_kW", "size"), MAE_kW=("Abs_error_kW", "mean"), MSE=("Squared_error", "mean")).reset_index()
        grouped["RMSE_kW"] = np.sqrt(grouped.pop("MSE"))
        grouped = grouped.rename(columns={grouping: "Group"})
        grouped.insert(0, "Grouping", grouping)
        tables.append(grouped)
    return pd.concat(tables, ignore_index=True)


def day_block_intervals(frame, repeats=200, seed=42):
    """Uncertainty of aggregate error statistics, not per-forecast intervals."""
    work = frame.copy()
    work["Day"] = pd.to_datetime(work["Target_interval_end"]).dt.normalize()
    work["AE"] = (work["Actual_kW"] - work["Hybrid_kW"]).abs()
    work["SE"] = (work["Actual_kW"] - work["Hybrid_kW"])**2
    daily = work.groupby("Day").agg(N=("AE", "size"), AE=("AE", "sum"), SE=("SE", "sum"))
    if len(daily) < 2:
        return pd.DataFrame(), "Fewer than two days; block intervals were not calculated."
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(daily), size=(repeats, len(daily)))
    numbers = daily["N"].to_numpy()[draws].sum(axis=1)
    mae = daily["AE"].to_numpy()[draws].sum(axis=1) / numbers
    rmse = np.sqrt(daily["SE"].to_numpy()[draws].sum(axis=1) / numbers)
    result = pd.DataFrame([{"Metric": label, "Percentile_2.5": float(np.quantile(value, .025)),
        "Percentile_97.5": float(np.quantile(value, .975)), "Day_blocks": len(daily), "Replicates": repeats}
        for label, value in [("MAE_kW", mae), ("RMSE_kW", rmse)]])
    return result, "Exploratory 95% day-block bootstrap intervals for aggregate errors; not prediction intervals. Day-to-day dependence and one-household sampling remain limitations."


def check_system_invariants(result, knowledge):
    origin = pd.Timestamp(result["forecast"]["origin"])
    target = pd.Timestamp(result["forecast"]["target_end"])
    high = result["forecast"]["forecast_kw"] >= result["forecast"]["threshold_kw"]
    scenario = Scenario(**result["scenario"])
    appliances = {a["name"]: Appliance(**a) for a in result["appliances"]}
    failures = []
    recommendations = {r["Appliance"]: r for r in result["recommendations"]}
    for name, appliance in appliances.items():
        inferred = knowledge.infer(appliance)
        row = recommendations[name]
        essential = "EssentialAppliance" in inferred["types"]
        if essential and row["Action"] != "Keep ON":
            failures.append(f"Essential appliance not protected: {name}")
        if not essential and scenario.occupancy == 0 and appliance.requires_occupancy:
            if row["Action"] not in ["Turn OFF", "Do not start"]:
                failures.append(f"Empty-home occupancy constraint violated: {name}")
        if not essential and scenario.occupancy > 0 and name == "Air Conditioner" and scenario.temperature_c >= 38:
            if row["Action"] != "Keep ON":
                failures.append("Occupied high-temperature AC comfort constraint violated")
    loads = np.zeros(scenario.horizon_hours*4, dtype=int)
    total = 0.
    for slot in result["plan"]["schedule"]:
        appliance = appliances[slot["Appliance"]]
        row = recommendations[appliance.name]
        begin, end = pd.Timestamp(slot["Start"]), pd.Timestamp(slot["End"])
        start = (begin-origin).total_seconds()/60
        finish = (end-origin).total_seconds()/60
        if start % 15 or finish % 15 or start < appliance.earliest_min or finish > min(appliance.deadline_min, scenario.horizon_hours*60) or finish-start != appliance.duration_min:
            failures.append(f"Runtime/window violation: {appliance.name}")
            continue
        if row["Kind"] != "job":
            failures.append(f"Protected/inactive appliance rescheduled: {appliance.name}")
        if not row["Flexible"] and start != appliance.earliest_min:
            failures.append(f"Non-flexible appliance delayed: {appliance.name}")
        if row["Flexible"] and high and begin < target and end > target-pd.Timedelta(minutes=15):
            failures.append(f"Flexible job overlaps single forecast high-demand interval: {appliance.name}")
        for i in range(int(start//15), int(finish//15)):
            loads[i] += 1
            total += appliance.power_w / 1000 * .25 * rate_at(origin + pd.Timedelta(minutes=15*i), scenario)
    if len(loads) and loads.max() > scenario.max_parallel_jobs:
        failures.append("Parallel-job constraint violated")
    plan = result["plan"]
    if plan["status"] in ["optimal", "feasible_search_limit"]:
        if abs(total - plan["estimated_cost"]) > 1e-7 or abs(plan["baseline_cost"]-total-plan["estimated_saving"]) > 1e-7:
            failures.append("Cost or saving calculation inconsistent")
        expected_jobs = {row["Appliance"] for row in result["recommendations"] if row["Kind"] == "job"}
        if {row["Appliance"] for row in plan["schedule"]} != expected_jobs:
            failures.append("Feasible schedule is missing requested jobs")
    elif plan["estimated_saving"] is not None or plan["estimated_cost"] is not None:
        failures.append("Unsuccessful search claimed a cost/saving")
    return failures


def scenario_suite():
    base = default_appliances()
    cases = [("Occupied household", Scenario(), base),
        ("Empty home, hot weather", Scenario(occupancy=0, temperature_c=43), base),
        ("Occupied home, very hot", Scenario(occupancy=5, temperature_c=43), base),
        ("Overnight peak tariff", Scenario(peak_start_hour=22, peak_end_hour=6), base),
        ("No peak period", Scenario(peak_start_hour=0, peak_end_hour=0), base),
        ("Two jobs permitted in parallel", Scenario(max_parallel_jobs=2), base),
        ("Short planning horizon", Scenario(horizon_hours=1), base),
        ("Flat price", Scenario(peak_rate=6, offpeak_rate=6), base)]
    fridge = Appliance("Refrigerator", 200, essential=False, flexible=True, ontology_id="Refrigerator_1")
    cases.append(("Conflicting flexible/essential ontology classes", Scenario(occupancy=0), [fridge]))
    wash = Appliance("Washing Machine", 500, ontology_id="WashingMachine_Test")
    cases.extend([("Impossible job deadline", Scenario(), [replace(wash, duration_min=60, deadline_min=30)]),
        ("Job already running", Scenario(), [replace(wash, on=True)]),
        ("User disallows shifting", Scenario(), [replace(wash, flexible=False)]),
        ("High-priority requested job", Scenario(), [replace(wash, priority="High")]),
        ("No requested inactive jobs", Scenario(), [replace(wash, requested=False)]),
        ("Very low search budget", Scenario(node_limit=1), [wash])])
    return cases


def run_scenario_evaluation(bridge, knowledge, example_count=4):
    ids = bridge._sample(bridge.test_ids, example_count)
    rows, results = [], []
    for case_id in ids:
        for name, scenario, appliances in scenario_suite():
            started = time.perf_counter()
            try:
                forecast = bridge.predict(int(case_id))
                result = optimize_decisions(forecast, appliances, scenario, knowledge)
                elapsed = time.perf_counter()-started
                failures = check_system_invariants(result, knowledge)
                # Certain cases have explicit expected outcomes, not only invariants.
                if name == "Impossible job deadline" and result["plan"]["status"] != "infeasible":
                    failures.append("Impossible deadline was not reported as infeasible")
                if name == "Very low search budget" and result["plan"]["status"] != "search_limit":
                    failures.append("Search-budget exhaustion was not reported")
                if name == "Job already running" and result["recommendations"][0]["Action"] != "Run as scheduled":
                    failures.append("Running job was interrupted")
                rows.append({"Scenario": name, "Target_index": int(case_id), "Passed": not failures,
                    "Expected_error_rejected": False, "Schedule_status": result["plan"]["status"],
                    "Latency_ms": elapsed*1000, "Violations": "; ".join(failures)})
                results.append({"scenario_name": name, "result": result})
            except Exception as error:
                rows.append({"Scenario": name, "Target_index": int(case_id), "Passed": False,
                    "Expected_error_rejected": False, "Schedule_status": "unexpected_error",
                    "Latency_ms": (time.perf_counter()-started)*1000, "Violations": repr(error)})
    forecast = bridge.predict(int(ids[0]))
    invalid = [("Negative appliance rating", [Appliance("Invalid", -100)], Scenario()),
        ("String essential flag", [Appliance("Invalid", 100, essential="false")], Scenario()),
        ("Negative tariff", [Appliance("Job", 100)], Scenario(peak_rate=-1)),
        ("Zero job concurrency", [Appliance("Job", 100)], Scenario(max_parallel_jobs=0)),
        ("Runtime not in 15-minute steps", [Appliance("Job", 100, duration_min=17)], Scenario())]
    for name, appliances, scenario in invalid:
        rejected = False
        try:
            optimize_decisions(forecast, appliances, scenario, knowledge)
        except (ValueError, TypeError):
            rejected = True
        rows.append({"Scenario": name, "Target_index": int(ids[0]), "Passed": rejected,
            "Expected_error_rejected": rejected, "Schedule_status": "expected_rejection" if rejected else "invalid_input_accepted",
            "Latency_ms": np.nan, "Violations": "" if rejected else "Invalid input accepted"})
    return pd.DataFrame(rows), results


def fairness_safety_checks(bridge, knowledge):
    forecast = bridge.predict(int(bridge.test_ids[0]))
    rows = []
    for occupancy, temp in [(0, 15), (1, 45), (8, 25)]:
        a = Appliance("Essential device A", 200, essential=True, flexible=True)
        b = replace(a, name="Essential device B")
        first = optimize_decisions(forecast, [a], Scenario(occupancy=occupancy, temperature_c=temp), knowledge)
        second = optimize_decisions(forecast, [b], Scenario(occupancy=occupancy, temperature_c=temp), knowledge)
        passed = first["recommendations"][0]["Action"] == second["recommendations"][0]["Action"] == "Keep ON"
        rows.append({"Check": "Renaming identical essential device does not change protection", "Occupancy": occupancy,
            "Temperature_C": temp, "Passed": passed, "Scope": "Controlled policy invariance; not demographic fairness"})
    return pd.DataFrame(rows)


def shap_diagnostics(bridge, frame, summary_examples=24):
    import shap
    errors = (frame["Actual_kW"]-frame["Hybrid_kW"]).abs().to_numpy()
    order = np.argsort(errors, kind="stable")
    high_rows = np.flatnonzero(frame["Actual_kW"].to_numpy() >= bridge.threshold)
    under = (frame["Actual_kW"]-frame["Hybrid_kW"]).to_numpy()
    peak = int(high_rows[np.argmax(under[high_rows])]) if len(high_rows) else int(order[-1])
    cases = {"Small error": int(order[0]), "Typical error": int(order[len(order)//2]), "High-demand diagnostic" if len(high_rows) else "Large error": peak}
    summary_rows = np.unique(np.linspace(0, len(frame)-1, min(summary_examples, len(frame)), dtype=int))
    positions = np.unique(np.r_[summary_rows, list(cases.values())])
    ids = frame["Target_index"].to_numpy(dtype=int)[positions]
    z = bridge.features(ids)
    explainer = shap.TreeExplainer(bridge.regressor)
    values = np.asarray(explainer.shap_values(z))
    if values.shape == z.shape + (1,):
        values = values[..., 0]
    if values.shape != z.shape or not np.isfinite(values).all():
        raise ValueError("Unexpected or non-finite SHAP values.")
    base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    raw = np.asarray(bridge.regressor.predict(z))
    residual = raw - base - values.sum(axis=1)
    tolerance = 1e-5 + 1e-5*np.abs(raw)
    checks = pd.DataFrame({"Frame_position": positions, "Target_index": ids, "Raw_forecast_kW": raw,
        "Clipped_forecast_kW": np.maximum(raw, 0), "SHAP_base_kW": base, "SHAP_sum_kW": values.sum(axis=1),
        "Residual_kW": residual, "Tolerance_kW": tolerance, "Passed": np.abs(residual) <= tolerance})
    lookup = {int(pos): row for row, pos in enumerate(positions)}
    summary_index = [lookup[int(pos)] for pos in summary_rows]
    importance = pd.DataFrame({"Feature": bridge.feature_names, "Mean_abs_SHAP_kW": np.abs(values[summary_index]).mean(axis=0)}).sort_values("Mean_abs_SHAP_kW", ascending=False)
    return {"positions": positions, "cases": cases, "lookup": lookup, "values": values, "data": z,
        "base": base, "checks": checks, "importance": importance, "summary_examples": len(summary_rows)}


def responsible_ai_table():
    return pd.DataFrame([
        ("Generalization and bias", "One UCI household and a historical period do not represent all homes or regions.", "Report time/demand error groups; validate on other households before wider use.", "No population or demographic fairness claim."),
        ("Comfort and accessibility", "A fixed 38°C AC rule may not fit individual health or comfort needs.", "Protect essential devices; require user review and future personalizable thresholds.", "Not a medical or personalized comfort guarantee."),
        ("False peak alerts and missed peaks", "Forecast errors can delay jobs unnecessarily or miss high demand.", "Report threshold confusion counts and keep essential/comfort overrides.", "No calibrated confidence probability or guaranteed savings."),
        ("Privacy", "Fine-grained electricity data may reveal household routines.", "Minimize retention; remove identifiers; control sharing of the runtime, notebook and exports.", "Colab is cloud-hosted; this is not a local-only privacy guarantee."),
        ("User control and safety", "Automatic switching could disrupt important appliances or running cycles.", "Prototype gives recommendations only; preserve essential/running jobs and report infeasibility.", "No hardware-control or production-safety certification."),
        ("Tariffs and savings", "Example ₹ tariffs, ratings and durations may not match an actual bill.", "Label scenario inputs and estimate only requested-job savings against the stated feasible baseline.", "Do not claim measured whole-house bill reductions."),
        ("Explainability", "Correlated latent/context features can share SHAP credit; activations are not causal importance.", "Name the explained ML stage, verify reconstruction, and show rule traces separately.", "Additivity is not proof of causal validity or complete reliability."),
        ("Legal and data governance", "Permission, data-source terms and applicable data obligations depend on deployment context.", "Document source/provenance and obtain appropriate permission before using identifiable household data.", "No jurisdiction-specific compliance determination was performed.")],
        columns=["Topic", "Risk", "Mitigation", "Residual_limit"])


def write_evaluation_documents(output_dir, bridge, metrics, scenarios, fairness, xai, freshness, intervals_note, setup_mode, fingerprint):
    output_dir = Path(output_dir)
    overall = metrics[metrics["Segment"].eq("All test")].set_index("Model")
    hybrid = overall.loc["Integrated LSTM + ML"]
    lstm = overall.loc["Task 7 LSTM"]
    passed, total = int(scenarios["Passed"].sum()), len(scenarios)
    shap_passed, shap_total = int(xai["checks"]["Passed"].sum()), len(xai["checks"])
    performance_direction = "lower" if hybrid["RMSE_kW"] < lstm["RMSE_kW"] else "higher or equal"
    report = f"""# Task 10 — final evaluation

## Objective and system architecture
Real recorded electricity history → Task 7 causal preparation → frozen LSTM patterns
→ validation-selected classical ML prediction → imported ontology/SWRL and operational
rules → constrained A* job scheduling → dashboard and explanations.

## Method and data provenance
Task 7 run: {bridge.manifest['run_id']}. ML stage: {bridge.selected_name}.
Setup mode: {setup_mode}. Models were fingerprinted before evaluation and checked
afterward. Model preparation, if required by a fresh runtime, preceded the freeze;
no evaluation results were used to fit/tune models.

Existing test observations are disjoint from training and validation, but this
same held-out set was already evaluated in Task 9. They are unseen to fitting,
not a new independent test sample. Additional household scenarios are controlled
designed combinations, not measured new households or proof of population fairness.
Independent later-time-range CSV supplied: {freshness['external_csv_supplied']}.
See data_freshness.json and any external evaluation files for exact coverage.

## Forecast results
Integrated model: MAE {hybrid['MAE_kW']:.4f} kW, RMSE {hybrid['RMSE_kW']:.4f} kW,
R² {hybrid['R2']:.4f}, evaluated on {int(hybrid['Samples'])} test forecasts.
Task 7 LSTM RMSE: {lstm['RMSE_kW']:.4f} kW. The integrated model has
{performance_direction} RMSE on this period. Integration is not automatically
an accuracy improvement. Demand/hour/weekday profiles and high-demand metrics
are saved separately; subgroup R² is not directly comparable with overall R².
{intervals_note}

## End-to-end correctness and runtime
{passed}/{total} scenario/error-handling checks passed. Outcomes include feasible,
infeasible and search-limit cases: successfully reporting infeasibility is not
counted as a system failure when expected. Latencies exclude SHAP and plotting,
include inference and rule/scheduling execution, and are specific to this hardware.
Inspect all failed rows rather than hiding them. Controlled policy-invariance
checks passed: {int(fairness['Passed'].sum())}/{len(fairness)}.

## Explainability
TreeSHAP reconstructs the raw ML-stage prediction in kW: {shap_passed}/{shap_total}
checks passed. Final negative-value clipping is reported separately. The features
include learned LSTM patterns and context; this is not an exact sensor-level
explanation of the whole composed system. Diagnostic cases include a small-error,
typical-error and high-demand/large-error example; they are deliberately selected,
not an unbiased accuracy sample. LSTM activations inspect internal responses;
rule traces explain operational actions. Neither method establishes causality.

## Responsible AI and deployment
See responsible_ai.csv/model_card.md for privacy, essential-appliance protection,
comfort limitations, user control, tariff assumptions, representativeness and data
governance. No demographic labels are available; temporal groups and controlled
device-renaming checks do not establish demographic fairness. ₹ settings and
savings are scenario assumptions; they are not actual French-household bills.
The deployment is a Colab notebook prototype, not a continuously hosted service
or hardware controller. The supplied OWL/SWRL adapter is a documented supported
subset, not a general OWL DL consistency reasoner.

## Remaining limitations
The previously examined test set limits claims of independent validation. More
households and fresh measured future inputs are needed for broad generalization.
Task 10 does not guarantee accuracy, production safety, regulatory compliance or
realized savings. Failed scenario/XAI checks, if any, remain visible in exports.
"""
    (output_dir / "evaluation_report.md").write_text(report, encoding="utf-8")
    card = f"""# Model/system card
Intended use: academic household demand prediction and advisory job scheduling.
Not intended for autonomous appliance switching, medical decisions or guaranteed bills.
Data: UCI Individual Household Electric Power Consumption; one household, 2006–2010.
Inputs: 96 past 15-minute multivariate steps plus frozen context features; separate
household settings enter rules/scheduling, not the learned forecast.
Target: mean kW in the 15-minute interval ending one hour after forecast origin.
Model: frozen Task 7 LSTM + {bridge.selected_name}; Task 9 selects ML on validation only.
Evaluation: reused chronological test period, additional designed scenarios,
optional externally supplied later-time-range data. External CSV supplied: {freshness['external_csv_supplied']}.
Uncertainty: error-statistic block intervals are exploratory; no calibrated prediction intervals.
Fairness: demographic data absent; no demographic fairness certification.
Human oversight: users review/reject recommendations; essential and running jobs protected.
Data handling: runtime/exported files need access controls and retention decisions;
Colab is cloud-hosted. No external household-data API is invoked during evaluation.
Explanations: ML-head TreeSHAP, LSTM activations, explicit rule and scheduling traces.
"""
    (output_dir / "model_card.md").write_text(card, encoding="utf-8")
    notes = f"""# Structured presentation notes
1. **Problem and scope:** anticipate demand and suggest appliance schedules; advisory prototype.
2. **Architecture:** causal data prep → LSTM representation → classical ML → ontology/rules → A* → dashboard.
3. **Method:** chronological partitions, training-only transformations, validation selection, frozen evaluation.
4. **Results:** MAE {hybrid['MAE_kW']:.4f} kW; RMSE {hybrid['RMSE_kW']:.4f} kW; R² {hybrid['R2']:.4f}; compare LSTM and persistence honestly.
5. **Working demonstration:** change a historical case and household settings; show schedule, cost assumptions and rule trace.
6. **Explanation example:** display an error case, TreeSHAP reconstruction and LSTM activations; distinguish attribution from causality.
7. **Validation/responsible AI:** {passed}/{total} designed checks; essential/comfort protection, privacy, representativeness and unavailable demographic fairness evidence.
8. **Limits and future work:** reused holdout, new-household/future measurements, personalization and external validation before wider deployment.
"""
    (output_dir / "presentation_notes.md").write_text(notes, encoding="utf-8")
    guide = """# Deployment and demonstration guide
Open the standalone Task 10 notebook in Colab and run its cells. Completed Task 7
folders and the Task 9 hybrid cache are reused when accessible. An explicit saved
folder can be configured. A separate Colab runtime does not automatically share
temporary files; if permitted, initial preparation rebuilds Task 9 once and labels
the run as rebuilt. Preparation finishes before models are frozen for evaluation.
Use the optional final dashboard for an interactive demonstration. Changing its
settings performs inference/scheduling, not training. Export and retain the executed
notebook and evaluation folder. Sharing a notebook is not a continuously available
production deployment. Keep private raw household data and identifiers out of
public submissions. No hardware is controlled.
"""
    (output_dir / "deployment_guide.md").write_text(guide, encoding="utf-8")
    (output_dir / "frozen_model_fingerprint.json").write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    return report
