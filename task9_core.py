"""Task 9: ontology → LSTM features → ML forecast → rules → A* scheduling.

The ontology adapter executes this project's named-subclass and unary class-to-
decision SWRL subset. It is not a general OWL DL consistency reasoner.
"""
import hashlib
import heapq
import json
import math
from dataclasses import asdict, dataclass, replace
from itertools import count
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def local_name(iri):
    return str(iri).rsplit("}", 1)[-1].rsplit("#", 1)[-1].rsplit("/", 1)[-1]


class OntologyKnowledge:
    NS = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
          "rdfs": "http://www.w3.org/2000/01/rdf-schema#", "owl": "http://www.w3.org/2002/07/owl#",
          "swrl": "http://www.w3.org/2003/11/swrl#", "swrla": "http://swrl.stanford.edu/ontologies/3.3/swrla.owl#"}

    def __init__(self, xml_text):
        self.digest = hashlib.sha256(xml_text.encode("utf-8")).hexdigest()
        root = ET.fromstring(xml_text)
        resource = "{" + self.NS["rdf"] + "}resource"
        about = "{" + self.NS["rdf"] + "}about"
        self.parents, self.individual_types, self.asserted_decisions = {}, {}, {}
        self.classes = set()
        for element in root.findall("owl:Class", self.NS):
            name = local_name(element.attrib[about])
            self.classes.add(name)
            self.parents[name] = {local_name(n.attrib[resource]) for n in element.findall("rdfs:subClassOf", self.NS)
                                  if resource in n.attrib}
        for element in root.findall("owl:NamedIndividual", self.NS):
            name = local_name(element.attrib[about])
            self.individual_types[name] = {local_name(n.attrib[resource]) for n in element.findall("rdf:type", self.NS)}
            self.asserted_decisions[name] = {local_name(n.attrib[resource]) for n in element
                                             if local_name(n.tag) == "hasDecision" and resource in n.attrib}
        self.rules = []
        for element in root:
            types = {n.attrib.get(resource) for n in element.findall("rdf:type", self.NS)}
            if self.NS["swrl"] + "Imp" not in types and element.tag != "{" + self.NS["swrl"] + "}Imp":
                continue
            enabled = element.find("swrla:isRuleEnabled", self.NS)
            if enabled is not None and (enabled.text or "").strip().lower() == "false":
                continue
            label_node = element.find("rdfs:label", self.NS)
            label = label_node.text if label_node is not None else f"SWRL_{len(self.rules)+1}"
            body = self._atoms(element.find("swrl:body", self.NS))
            head = self._atoms(element.find("swrl:head", self.NS))
            if not body or not head or any(a["kind"] != "ClassAtom" for a in body):
                raise ValueError(f"Unsupported SWRL body in {label}; this adapter will not silently ignore it.")
            variables = {a["arg1"] for a in body}
            if len(variables) != 1 or any(a["kind"] != "IndividualPropertyAtom" or a["predicate"] != "hasDecision"
                                           or a["arg1"] not in variables for a in head):
                raise ValueError(f"Unsupported SWRL head in {label}.")
            self.rules.append({"id": label, "classes": [a["predicate"] for a in body],
                               "decisions": [a["arg2"] for a in head]})
        if not self.rules:
            raise ValueError("The ontology contains no supported enabled SWRL rules.")

    def _atoms(self, container):
        if container is None or not len(container):
            return []
        resource = "{" + self.NS["rdf"] + "}resource"
        current, result = container[0], []
        while current is not None:
            first = current.find("rdf:first", self.NS)
            if first is None or not len(first):
                raise ValueError("Unsupported SWRL RDF list encoding.")
            atom = first[0]
            kind = atom.find("rdf:type", self.NS)
            kind = local_name(kind.attrib[resource]) if kind is not None else local_name(atom.tag)
            def value(name):
                node = atom.find("swrl:" + name, self.NS)
                return local_name(node.attrib[resource]) if node is not None and resource in node.attrib else None
            result.append({"kind": kind, "predicate": value("classPredicate") or value("propertyPredicate"),
                           "arg1": value("argument1"), "arg2": value("argument2")})
            rest = current.find("rdf:rest", self.NS)
            if rest is None:
                raise ValueError("SWRL list has no terminator.")
            if rest.attrib.get(resource) == self.NS["rdf"] + "nil":
                break
            if not len(rest):
                raise ValueError("Only nested SWRL RDF lists are supported by this adapter.")
            current = rest[0]
        return result

    def infer(self, appliance):
        types = set(self.individual_types.get(appliance.ontology_id, set())) | {"Appliance"}
        if appliance.essential:
            types.add("EssentialAppliance")
        if appliance.flexible:
            types.add("FlexibleAppliance")
        if appliance.priority == "High":
            types.add("HighPriorityAppliance")
        changed = True
        while changed:
            expanded = types | set().union(*(self.parents.get(t, set()) for t in types))
            changed = expanded != types
            types = expanded
        decisions, fired = set(), []
        for rule in self.rules:
            if set(rule["classes"]) <= types:
                decisions.update(rule["decisions"])
                fired.append("OWL:" + rule["id"])
        return {"types": sorted(types), "candidate_decisions": sorted(decisions), "fired": fired,
                "asserted_decisions": sorted(self.asserted_decisions.get(appliance.ontology_id, set()))}


@dataclass(frozen=True)
class Appliance:
    name: str
    power_w: float
    duration_min: int = 60
    requested: bool = True
    on: bool = False
    essential: bool = False
    flexible: bool = True
    priority: str = "Medium"
    earliest_min: int = 0
    deadline_min: int = 360
    requires_occupancy: bool = False
    ontology_id: str = ""


@dataclass(frozen=True)
class Scenario:
    occupancy: int = 3
    temperature_c: float = 35.0
    peak_start_hour: int = 18
    peak_end_hour: int = 20
    peak_rate: float = 9.0
    offpeak_rate: float = 6.0
    horizon_hours: int = 6
    max_parallel_jobs: int = 1
    node_limit: int = 50000


def default_appliances():
    return [Appliance("Refrigerator", 200, on=True, essential=True, flexible=False,
                      priority="High", ontology_id="Refrigerator_1"),
            Appliance("Air Conditioner", 1800, on=True, flexible=False, priority="High", requires_occupancy=True),
            Appliance("Washing Machine", 500, ontology_id="WashingMachine_Test"),
            Appliance("Dishwasher", 1200)]


def validate_inputs(appliances, scenario):
    if not 0 <= scenario.occupancy <= 20 or not np.isfinite(scenario.temperature_c):
        raise ValueError("Occupancy must be 0–20 and temperature must be finite.")
    if scenario.horizon_hours not in range(1, 13) or scenario.max_parallel_jobs not in range(1, 7):
        raise ValueError("Use a 1–12 hour horizon and 1–6 concurrent schedulable jobs.")
    if not 0 <= scenario.peak_start_hour < 24 or not 0 <= scenario.peak_end_hour < 24:
        raise ValueError("Tariff hours must be 0–23.")
    if any(not np.isfinite(v) or v < 0 for v in [scenario.peak_rate, scenario.offpeak_rate]):
        raise ValueError("Tariffs must be finite and non-negative.")
    if scenario.node_limit < 1 or not 1 <= len(appliances) <= 8:
        raise ValueError("Use 1–8 appliances and a positive search limit.")
    if len({a.name for a in appliances}) != len(appliances):
        raise ValueError("Appliance names must be unique.")
    for a in appliances:
        if any(type(getattr(a, field)) is not bool for field in ["requested", "on", "essential", "flexible", "requires_occupancy"]):
            raise ValueError("Appliance flags must be JSON true/false values.")
        if any(type(getattr(a, field)) is not int for field in ["duration_min", "earliest_min", "deadline_min"]):
            raise ValueError("Runtime, earliest start and deadline must be integer minutes.")
        if not a.name.strip() or not np.isfinite(a.power_w) or not 0 < a.power_w <= 20000:
            raise ValueError("Each appliance needs a name and a power rating in (0, 20000] W.")
        if a.priority not in ["Low", "Medium", "High"]:
            raise ValueError(f"Invalid priority for {a.name}.")
        if a.duration_min <= 0 or a.duration_min % 15 or a.earliest_min < 0 or a.earliest_min % 15 or a.deadline_min % 15:
            raise ValueError("Runtime, earliest start and deadline must use 15-minute steps.")
        if a.deadline_min <= a.earliest_min:
            raise ValueError(f"Deadline must follow earliest start for {a.name}.")


def peak_at(timestamp, scenario):
    hour = timestamp.hour + timestamp.minute / 60
    start, end = scenario.peak_start_hour, scenario.peak_end_hour
    if start == end:
        return False
    return start <= hour < end if start < end else hour >= start or hour < end


def rate_at(timestamp, scenario):
    return scenario.peak_rate if peak_at(timestamp, scenario) else scenario.offpeak_rate


def reason_appliance(appliance, scenario, origin, forecast_kw, threshold, knowledge):
    inferred = knowledge.infer(appliance)
    essential = "EssentialAppliance" in inferred["types"]
    flexible = appliance.flexible and "FlexibleAppliance" in inferred["types"]
    high_priority = "HighPriorityAppliance" in inferred["types"]
    trace = list(inferred["fired"])
    # Imported SWRL heads propose decisions. Ordered operational rules resolve
    # conflicts and add the tariff, occupancy, comfort and forecast conditions.
    if essential:
        action, reason, kind, rule = "Keep ON", "Essential appliance; protected regardless of tariff or forecast.", "protected", "R1-essential"
    elif scenario.occupancy == 0 and appliance.requires_occupancy:
        action, reason, kind, rule = "Turn OFF" if appliance.on else "Do not start", "Unoccupied home; this appliance requires occupancy.", "inactive", "R5-empty-home"
    elif appliance.name == "Air Conditioner" and scenario.occupancy > 0 and scenario.temperature_c >= 38:
        action, reason, kind, rule = "Keep ON", "Occupied home at or above 38°C; comfort overrides cost saving.", "protected", "R2-comfort"
    elif not appliance.requested and not appliance.on:
        action, reason, kind, rule = "Leave OFF", "Appliance was not requested.", "inactive", "R6-not-requested"
    elif high_priority or appliance.on:
        action, reason, kind, rule = "Run as scheduled", "High-priority or already running appliance; no automatic interruption or delay.", "protected", "R3-priority"
    elif flexible:
        high = forecast_kw >= threshold
        action, reason, kind, rule = "Find schedule", (
            "Flexible job; " + ("avoid the single forecast high-demand interval; " if high else "") +
            ("current tariff is peak; " if peak_at(origin, scenario) else "") + "search permitted start times for lowest estimated job cost."), "job", "R4-forecast-tariff"
    else:
        action, reason, kind, rule = "Run at earliest allowed time", "Non-flexible requested job keeps its earliest start.", "job", "R6-fixed-job"
    trace.append(rule)
    return {"Appliance": appliance.name, "Action": action, "Reason": reason, "Kind": kind,
            "Essential": essential, "Flexible": flexible, "Rule_trace": trace,
            "OWL_candidates": inferred["candidate_decisions"], "OWL_asserted_decisions": inferred["asserted_decisions"],
            "Ontology_types": inferred["types"]}


def plan_jobs(jobs, scenario, origin, target_end, forecast_high):
    """A* assignment search; admissible remaining-cost lower bound.

    Capacity counts only these schedulable jobs, not whole-house load. Forecast
    avoidance applies only to the known 15-minute target interval.
    """
    slots = scenario.horizon_hours * 4
    rates = np.array([rate_at(origin + pd.Timedelta(minutes=15 * i), scenario) for i in range(slots)])
    options = []
    for job, flexible in jobs:
        duration = job.duration_min // 15
        latest = min(slots, job.deadline_min // 15) - duration
        starts = list(range(job.earliest_min // 15, latest + 1))
        if not flexible:
            starts = starts[:1]
        choices = []
        for start in starts:
            begin, end = origin + pd.Timedelta(minutes=15 * start), origin + pd.Timedelta(minutes=15 * (start + duration))
            overlaps_forecast = begin < target_end and end > target_end - pd.Timedelta(minutes=15)
            if flexible and forecast_high and overlaps_forecast:
                continue
            cost = float((job.power_w / 1000) * .25 * rates[start:start + duration].sum())
            choices.append((start, duration, cost))
        if not choices:
            return {"status": "infeasible", "reason": f"No allowed start for {job.name}; check deadline and forecast avoidance.",
                    "schedule": [], "estimated_cost": None, "baseline_cost": None, "estimated_saving": None, "nodes": 0}
        options.append(choices)
    if not jobs:
        return {"status": "optimal", "reason": "No schedulable jobs requested.", "schedule": [],
                "estimated_cost": 0., "baseline_cost": 0., "estimated_saving": 0., "nodes": 0}
    # A chronological DFS supplies a feasible earliest-start comparison under
    # exactly the same constraints. It is not claimed to be the actual old bill.
    baseline_nodes = 0
    baseline_limit = False
    def first_feasible(index, loads, starts):
        nonlocal baseline_nodes, baseline_limit
        if baseline_nodes >= scenario.node_limit:
            baseline_limit = True
            return None
        baseline_nodes += 1
        if index == len(jobs):
            return starts
        for start, duration, cost in options[index]:
            if max(loads[start:start + duration], default=0) >= scenario.max_parallel_jobs:
                continue
            updated = list(loads)
            for i in range(start, start + duration):
                updated[i] += 1
            result = first_feasible(index + 1, updated, starts + [(start, duration, cost)])
            if result is not None:
                return result
        return None
    baseline = first_feasible(0, [0] * slots, [])
    if baseline is None:
        return {"status": "search_limit" if baseline_limit else "infeasible",
                "reason": "No feasible schedule found before the search limit." if baseline_limit else "Requested jobs cannot all meet their windows and concurrency limit.",
                "schedule": [], "estimated_cost": None, "baseline_cost": None, "estimated_saving": None, "nodes": 0}
    minimum = [min(x[2] for x in choices) for choices in options]
    lower = [sum(minimum[i:]) for i in range(len(jobs) + 1)]
    queue, counter = [], count()
    heapq.heappush(queue, (lower[0], 0., next(counter), 0, (0,) * slots, []))
    best = {(0, (0,) * slots): 0.}
    chosen, status, explored = baseline, "feasible_search_limit", 0
    while queue and explored < scenario.node_limit:
        _, cost_so_far, _, index, loads, path = heapq.heappop(queue)
        if cost_so_far > best.get((index, loads), float("inf")) + 1e-10:
            continue
        explored += 1
        if index == len(jobs):
            chosen, status = path, "optimal"
            break
        for start, duration, cost in options[index]:
            if max(loads[start:start + duration], default=0) >= scenario.max_parallel_jobs:
                continue
            updated = list(loads)
            for i in range(start, start + duration):
                updated[i] += 1
            new_cost, state = cost_so_far + cost, (index + 1, tuple(updated))
            if new_cost + 1e-10 < best.get(state, float("inf")):
                best[state] = new_cost
                heapq.heappush(queue, (new_cost + lower[index + 1], new_cost, next(counter),
                                       index + 1, state[1], path + [(start, duration, cost)]))
    rows = []
    for (job, _), (start, duration, cost) in zip(jobs, chosen):
        rows.append({"Appliance": job.name, "Start": str(origin + pd.Timedelta(minutes=15 * start)),
                     "End": str(origin + pd.Timedelta(minutes=15 * (start + duration))),
                     "Power_W": job.power_w, "Energy_kWh": job.power_w / 1000 * duration * .25,
                     "Estimated_cost": cost})
    total, baseline_cost = sum(p[2] for p in chosen), sum(p[2] for p in baseline)
    return {"status": status, "reason": "Minimum cost under declared constraints." if status == "optimal" else "Feasible DFS schedule retained; A* reached its node limit, so optimality is unproven.",
            "schedule": rows, "estimated_cost": total, "baseline_cost": baseline_cost,
            "estimated_saving": baseline_cost - total, "nodes": explored}


def optimize_decisions(forecast, appliances, scenario, knowledge):
    validate_inputs(appliances, scenario)
    origin, target = pd.Timestamp(forecast["origin"]), pd.Timestamp(forecast["target_end"])
    value, threshold = float(forecast["forecast_kw"]), float(forecast["threshold_kw"])
    if not np.isfinite([value, threshold]).all() or value < 0 or threshold <= 0:
        raise ValueError("Forecast and training threshold must be finite valid power values.")
    if target - origin != pd.Timedelta(hours=1):
        raise ValueError("Expected a one-hour-ahead forecast target.")
    rows = [reason_appliance(a, scenario, origin, value, threshold, knowledge) for a in appliances]
    jobs = [(a, row["Flexible"]) for a, row in zip(appliances, rows) if row["Kind"] == "job"]
    plan = plan_jobs(jobs, scenario, origin, target, value >= threshold)
    for row in rows:
        if row["Kind"] != "job":
            continue
        scheduled = next((s for s in plan["schedule"] if s["Appliance"] == row["Appliance"]), None)
        if scheduled is None:
            row["Action"] = "No feasible schedule — review inputs"
            row["Suggested_start"] = None
        else:
            row["Suggested_start"] = scheduled["Start"]
            row["Action"] = "Delay" if pd.Timestamp(scheduled["Start"]) > origin else "Run now"
    return {"forecast": forecast, "scenario": asdict(scenario), "appliances": [asdict(a) for a in appliances],
            "recommendations": rows, "plan": plan, "ontology_sha256": knowledge.digest,
            "provenance": "Electricity history is recorded data; household settings are user-supplied scenarios. Savings concern requested schedulable jobs only, not measured whole-house bills."}


class ForecastBridge:
    """Frozen Task 7 LSTM features + validation-selected Task 6 algorithm family."""
    def __init__(self, artifact_dir, output_root, train_cap=20000, validation_cap=5000, force_fit=False):
        import joblib
        from tensorflow import keras
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        self.folder, self.output_root = Path(artifact_dir), Path(output_root)
        self.manifest = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["run_mode"] != "full" or self.manifest["horizon_steps"] != 4 or self.manifest["sequence_length"] != 96:
            raise ValueError("Use a full updated Task 7 run with 96 steps and a four-step horizon.")
        self.network = keras.models.load_model(self.folder / "task7_FINAL_lstm.keras", compile=False)
        self.scaler = joblib.load(self.folder / "power_scaler.joblib")
        self.mean, self.scale = float(self.scaler.mean_[0]), float(self.scaler.scale_[0])
        with np.load(self.folder / "prepared_arrays.npz", allow_pickle=False) as archive:
            self.arrays = {key: archive[key] for key in archive.files}
        self.history = self.arrays["history_values"]
        self.context = self.arrays["context_values"]
        self.timestamps = pd.DatetimeIndex(pd.to_datetime(self.arrays["timestamps_ns"]))
        self.y = self.arrays["target_scaled"].reshape(-1) * self.scale + self.mean
        self.train_ids = self._sample(self.arrays["train_indices"], train_cap)
        self.validation_ids = self._sample(self.arrays["validation_indices"], validation_cap)
        self.test_ids = self.arrays["test_indices"]
        if not len(self.train_ids) or not len(self.validation_ids) or not len(self.test_ids):
            raise ValueError("All temporal partitions must be non-empty.")
        assert self.train_ids.max() < self.validation_ids.min() - 4
        assert self.validation_ids.max() < self.test_ids.min() - 4
        self.encoder = keras.Model(self.network.inputs, self.network.get_layer("lstm_summary").output)
        self.input_order = [tensor.name.split(":")[0] for tensor in self.network.inputs]
        self.feature_names = [f"LSTM_pattern_{i+1:02d}" for i in range(int(self.encoder.output_shape[-1]))] + ["Context: " + n for n in self.manifest["context_features"]]
        self.signature = hashlib.sha256(json.dumps({"model": file_hash(self.folder / "task7_FINAL_lstm.keras"),
            "arrays": file_hash(self.folder / "prepared_arrays.npz"), "manifest": file_hash(self.folder / "manifest.json"),
            "scaler": file_hash(self.folder / "power_scaler.joblib"),
            "train_cap": train_cap, "validation_cap": validation_cap, "schema": 1}, sort_keys=True).encode()).hexdigest()
        self.cache = self.output_root / "model_cache" / self.signature[:16]
        self.cache.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache / "hybrid_forecaster.joblib"
        if cache_file.is_file() and not force_fit:
            saved = joblib.load(cache_file)
            if saved["signature"] != self.signature:
                raise ValueError("Hybrid-model cache does not match this Task 7 run.")
            self.regressor, self.selected_name = saved["model"], saved["selected_name"]
            self.validation_results = pd.DataFrame(saved["validation_results"])
            print("Reusing cached ML prediction stage:", self.selected_name)
        else:
            print(f"Extracting frozen LSTM patterns: {len(self.train_ids)} training / {len(self.validation_ids)} validation examples...")
            train_z, val_z = self.features(self.train_ids), self.features(self.validation_ids)
            candidates = {"LSTM + Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=.05, random_state=42),
                          "LSTM + Random Forest": RandomForestRegressor(n_estimators=60, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=2)}
            records = []
            for name, model in candidates.items():
                print("Fitting ML stage:", name, flush=True)
                model.fit(train_z, self.y[self.train_ids])
                prediction = np.maximum(model.predict(val_z), 0)
                records.append({"Model": name, **self.metrics(self.y[self.validation_ids], prediction)})
            self.validation_results = pd.DataFrame(records).sort_values("RMSE_kW").reset_index(drop=True)
            self.selected_name = self.validation_results.iloc[0]["Model"]
            self.regressor = candidates[self.selected_name]
            joblib.dump({"signature": self.signature, "model": self.regressor, "selected_name": self.selected_name,
                         "validation_results": self.validation_results.to_dict("records")}, cache_file)
        self.validation_results.to_csv(self.cache / "validation_comparison.csv", index=False)
        self.threshold = float(self.manifest["peak_threshold_kw"])

    @staticmethod
    def _sample(ids, cap):
        ids = np.asarray(ids, dtype=np.int64)
        return ids[np.unique(np.linspace(0, len(ids) - 1, min(len(ids), cap), dtype=int))]

    @staticmethod
    def metrics(actual, predicted):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        return {"MAE_kW": float(mean_absolute_error(actual, predicted)),
                "RMSE_kW": float(np.sqrt(mean_squared_error(actual, predicted))), "R2": float(r2_score(actual, predicted))}

    def inputs(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        positions = ids[:, None] - 4 - 96 + 1 + np.arange(96)[None, :]
        if positions.min() < 0:
            raise ValueError("Not enough history for this forecast.")
        result = {"history": np.asarray(self.history[positions], dtype=np.float32),
                  "context": np.asarray(self.context[ids], dtype=np.float32)}
        if not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError("A requested forecast contains unavailable input features.")
        return result

    def features(self, ids, batch_size=256):
        batches = []
        for offset in range(0, len(ids), batch_size):
            x = self.inputs(ids[offset:offset+batch_size])
            latent = np.asarray(self.encoder([x[key] for key in self.input_order], training=False))
            batches.append(np.concatenate([latent, x["context"]], axis=1))
        return np.concatenate(batches)

    def predict(self, target_index):
        target_index = int(target_index)
        if target_index not in self.test_ids:
            raise ValueError("Dashboard examples must come from the held-out test period.")
        z = self.features(np.array([target_index]))
        raw = float(self.regressor.predict(z)[0])
        return {"origin": str(self.timestamps[target_index - 4]), "target_end": str(self.timestamps[target_index]),
                "target_interval_start": str(self.timestamps[target_index] - pd.Timedelta(minutes=15)),
                "forecast_kw": max(raw, 0.), "raw_forecast_kw": raw, "threshold_kw": self.threshold,
                "model": self.selected_name, "target_index": target_index,
                "units": "kW, average over the target 15-minute interval"}

    def evaluate(self):
        from tensorflow import keras
        ids = self.test_ids
        hybrid = np.maximum(self.regressor.predict(self.features(ids)), 0)
        saved = pd.read_csv(self.folder / "test_predictions.csv", parse_dates=["Target_interval_end"])
        matched = saved.set_index("Target_interval_end").loc[self.timestamps[ids]]
        np.testing.assert_allclose(self.y[ids], matched["Actual_kW"], atol=1e-5)
        predictions = {self.selected_name: hybrid,
            "Task 7 LSTM": matched[self.manifest["selection"]["selected_lstm"] + "_kW"].to_numpy(),
            "Basic neural network": matched["basic_mlp_kW"].to_numpy(), "Persistence": matched["Persistence_kW"].to_numpy()}
        records = []
        for segment, mask in {"All test": np.ones(len(ids), dtype=bool), "High demand": self.y[ids] >= self.threshold}.items():
            for name, values in predictions.items():
                metrics = self.metrics(self.y[ids][mask], values[mask]) if mask.sum() >= 2 else {"MAE_kW": None, "RMSE_kW": None, "R2": None}
                records.append({"Model": name, "Segment": segment, "Samples": int(mask.sum()), **metrics})
        result = pd.DataFrame(records)
        result.to_csv(self.cache / "full_test_comparison.csv", index=False)
        return result

    def explain(self, target_index):
        import shap
        z = self.features(np.array([target_index]))
        explainer = shap.TreeExplainer(self.regressor)
        values = np.asarray(explainer.shap_values(z)).reshape(-1)
        base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        raw = float(self.regressor.predict(z)[0])
        np.testing.assert_allclose(base + values.sum(), raw, rtol=1e-5, atol=1e-5)
        return {"values": values, "base": base, "raw_prediction": raw, "data": z[0], "features": self.feature_names}

    def activation(self, target_index):
        from tensorflow import keras
        x = self.inputs(np.array([target_index]))
        view = keras.Model(self.network.inputs, self.network.get_layer("lstm_temporal_features").output)
        return np.asarray(view([x[k] for k in self.input_order], training=False))[0]


def find_task7_folder(explicit="", roots=None):
    required = ["manifest.json", "task7_FINAL_lstm.keras", "task7_basic_mlp.keras", "power_scaler.joblib",
                "prepared_arrays.npz", "test_predictions.csv"]
    def complete(folder):
        try:
            return all((folder / name).is_file() for name in required) and json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["run_mode"] == "full"
        except (ValueError, KeyError):
            return False
    if explicit:
        folder = Path(explicit).expanduser()
        if not complete(folder):
            raise ValueError(f"Incomplete full Task 7 output folder: {folder}")
        return folder.resolve()
    folders = set()
    for root in roots or [Path.cwd()]:
        root = Path(root)
        if complete(root):
            folders.add(root.resolve())
        for manifest in (root / "task7_FINAL_outputs").glob("*/manifest.json"):
            if complete(manifest.parent):
                folders.add(manifest.parent.resolve())
    return max(folders, key=lambda folder: (folder / "manifest.json").stat().st_mtime) if folders else None
