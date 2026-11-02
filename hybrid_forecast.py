"""Load-only LSTM + classical regressor using the completed bundled Task 7 run.

Training is explicit and offline in tools/train_local_hybrid.py. This adapter
never fits on startup or inference and never relabels older saved LSTM results.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / "task9_outputs" / "local_hybrid"
SETTINGS = {"schema": 1, "train_cap": 20000, "validation_cap": 5000, "seed": 42,
            "candidates": {"LSTM + Random Forest": {"n_estimators": 60, "max_depth": 12, "min_samples_leaf": 4, "random_state": 42, "n_jobs": 2},
                           "LSTM + Gradient Boosting": {"n_estimators": 100, "max_depth": 3, "learning_rate": .05, "random_state": 42}}}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_fingerprints(folder):
    names = ["task7_lstm.keras", "power_scaler.joblib", "context_scaler.joblib", "manifest.json", "prepared_arrays.npz"]
    return {**{n: digest(Path(folder)/n) for n in names}, "preprocessing_source": digest(ROOT/"task7_pipeline.py")}


def signature_for(folder):
    source = source_fingerprints(folder)
    signature = hashlib.sha256(json.dumps({"source": source, "settings": SETTINGS}, sort_keys=True).encode()).hexdigest()
    return signature, source


def make_encoder(base):
    from tensorflow import keras
    return keras.Model(base.model.inputs, base.model.get_layer("lstm_summary").output)


def extract_features(encoder, inputs):
    latent = np.asarray(encoder(inputs, training=False), dtype="float32")
    features = np.concatenate([latent, inputs["context"]], axis=1)
    if not np.isfinite(features).all():
        raise ValueError("Hybrid inputs contain unavailable features.")
    return features


class LocalHybridForecast:
    is_hybrid = True
    upload_ready = True

    def __init__(self, base, artifact_folder):
        import joblib
        import sklearn
        self.base = base
        self.folder = base.folder
        self.artifact_folder = Path(artifact_folder)
        self.metadata = json.loads((self.artifact_folder/"training_manifest.json").read_text())
        if self.metadata.get("status") != "evaluated":
            raise ValueError("The hybrid training/evaluation run is incomplete.")
        signature, source = signature_for(self.folder)
        if self.metadata["signature"] != signature or self.metadata["source_fingerprints"] != source:
            raise ValueError("Hybrid cache does not match the frozen LSTM, preprocessing and settings.")
        if self.metadata["versions"]["scikit_learn"] != sklearn.__version__:
            raise ValueError("Use the recorded scikit-learn version for this trained artifact.")
        selected_file = self.artifact_folder / self.metadata["selected_file"]
        if digest(selected_file) != self.metadata["selected_sha256"]:
            raise ValueError("Hybrid regressor fingerprint mismatch.")
        self.regressor = joblib.load(selected_file)
        self.selected_name = self.metadata["selected_name"]
        self.manifest = base.manifest
        self.model = base.model
        self.model_sha256 = base.model_sha256
        self.forecaster_sha256 = hashlib.sha256((signature+self.metadata["selected_sha256"]).encode()).hexdigest()
        self.encoder = make_encoder(base)
        self.power_scaler, self.context_scaler = base.power_scaler, base.context_scaler
        self.feature_names = [f"LSTM_pattern_{i+1:02d}" for i in range(16)] + ["Context: "+x for x in self.manifest["context_features"]]
        with np.load(self.artifact_folder/"verification_inputs.npz") as a:
            inputs = {"history": a["history"], "context": a["context"]}
            np.testing.assert_allclose(self.values_kw(inputs), a["prediction_kw"], rtol=1e-6, atol=1e-6)

    @classmethod
    def load_active(cls, base):
        active = json.loads((ARTIFACT_ROOT/"active.json").read_text())
        run = active["run"]
        if len(run) != 16 or any(c not in "0123456789abcdef" for c in run):
            raise ValueError("Invalid local hybrid artifact reference.")
        return cls(base, ARTIFACT_ROOT/run)

    def inputs(self, checked, *, sample=False):
        return self.base.inputs(checked, sample=sample)

    def features(self, inputs):
        return extract_features(self.encoder, inputs)

    def raw_values_kw(self, inputs):
        values = np.asarray(self.regressor.predict(self.features(inputs)), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Hybrid prediction is not finite.")
        return values

    def values_kw(self, inputs):
        return np.maximum(self.raw_values_kw(inputs), 0.)

    def explain_inputs(self, inputs):
        import shap
        features = self.features(inputs)
        explainer = shap.TreeExplainer(self.regressor)
        values = np.asarray(explainer.shap_values(features)).reshape(-1)
        base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        raw = float(self.regressor.predict(features)[0])
        residual = abs(base+float(values.sum())-raw)
        verified = bool(residual <= 1e-5+1e-5*abs(raw))
        if not verified or not np.isfinite(values).all():
            raise ValueError("Hybrid TreeSHAP reconstruction did not pass.")
        return {"available": True, "method": "TreeSHAP", "reference_kw": base,
                "raw_forecast_kw": raw, "reconstruction_verified": verified, "residual_kw": residual,
                "group_count": len(self.feature_names),
                "factors": [{"name": n, "value_kw": float(v)} for n, v in zip(self.feature_names, values)],
                "note": "TreeSHAP explains the executed classical regressor using 16 learned LSTM patterns and 17 context features. It does not explain physical appliance causes or directly attribute the raw 96-step sequence. Contributions reconstruct the raw output before any non-negative clipping."}

    def forecast(self, inputs, origin, target_end, *, sample=False, timezone=None):
        raw = float(self.raw_values_kw(inputs)[0])
        return {"origin": str(origin), "target_end": str(target_end),
                "target_interval_start": str(pd.Timestamp(target_end)-pd.Timedelta(minutes=15)),
                "forecast_kw": max(raw, 0.), "raw_forecast_kw": raw,
                "threshold_kw": float(self.manifest["peak_threshold_kw"]), "model": self.selected_name,
                "source": "local_hybrid", "source_run": self.metadata["run_id"],
                "model_sha256": self.model_sha256, "forecaster_sha256": self.forecaster_sha256,
                "timezone": timezone, "is_sample": bool(sample),
                "input_scope": "Frozen LSTM usage patterns plus recent, daily, weekly and calendar context; classical regressor trained on the original household data. Household preferences guide the subsequent rules and schedule.",
                "units": "kW, average over the target 15-minute interval"}

    def evidence(self, inputs, history):
        explanation = {"available": False, "reason": "TreeSHAP is unavailable for this forecast; no contributions were substituted."}
        try:
            explanation = self.explain_inputs(inputs)
        except Exception as error:
            print("Hybrid explanation unavailable:", error, flush=True)
        return {"history": history, "explanation": explanation,
                "confidence": {"available": False, "label": "Not quantified", "reason": "No calibrated prediction interval is supplied; accuracy for your household is not established."}}

    def predict_readings(self, checked, *, sample=False):
        inputs = self.inputs(checked, sample=sample)
        report = checked.report
        forecast = self.forecast(inputs, report["origin"], report["target_end"], sample=sample, timezone=report["timezone"])
        forecast["source"] = "uploaded_readings"
        return forecast, self.evidence(inputs, report["history"])
