"""Frozen inference for the completed, bundled Task 7 power-history model.

Uses that run's original preprocessing, scalers and weights. No training, future
measurements, archived prediction substitution or newer hybrid model is used.
"""
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HISTORY = ["power_scaled", "observed", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]


class LocalPowerForecast:
    def __init__(self, folder):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import joblib
        import tensorflow as tf

        self.tf = tf
        try:
            tf.config.threading.set_intra_op_parallelism_threads(4)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except RuntimeError:
            pass
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / "manifest.json").read_text())
        if (self.manifest["history_features"] != HISTORY
                or self.manifest["sequence_length"] != 96
                or self.manifest["horizon_steps"] != 4):
            raise ValueError("Bundled model has an incompatible input contract.")
        # Load only locally versioned functions, never pipeline top-level code.
        path = ROOT / "task7_pipeline.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {"prepare_power", "calendar_features", "build_context"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes} != names:
            raise ValueError("Original power-history preprocessing is unavailable.")
        self.funcs = {"np": np, "pd": pd}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), self.funcs)
        self.power_scaler = joblib.load(self.folder / "power_scaler.joblib")
        self.context_scaler = joblib.load(self.folder / "context_scaler.joblib")
        model_path = self.folder / "task7_lstm.keras"
        self.model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
        if tuple(self.model.get_layer("history").output.shape[1:]) != (96, 6):
            raise ValueError("Unexpected power-history model dimensions.")
        if tuple(self.model.get_layer("context").output.shape[1:]) != (17,):
            raise ValueError("Unexpected power-context model dimensions.")
        self.model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        # Readiness includes real artifact fidelity, not merely file existence.
        self.verify_saved_predictions()
        self.upload_ready = True

    def values_kw(self, inputs):
        scaled = np.asarray(self.model(inputs, training=False))
        values = self.power_scaler.inverse_transform(scaled).ravel()
        if not np.isfinite(values).all():
            raise ValueError("The forecasting model returned a non-finite estimate.")
        return np.maximum(values, 0.)

    def verify_saved_predictions(self):
        saved = pd.read_csv(self.folder / "test_predictions.csv")
        positions = np.unique(np.linspace(0, len(saved)-1, 5, dtype=int))
        with np.load(self.folder / "prepared_arrays.npz") as arrays:
            targets = arrays["test_indices"][positions]
            history, context = arrays["history_values"], arrays["context_values"]
            inputs = {"history": np.stack([history[t-4-95:t-4+1] for t in targets]),
                      "context": context[targets]}
        actual = self.values_kw(inputs)
        expected = saved.iloc[positions].lstm_compact_kW.to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4,
                                   err_msg="Bundled model does not reproduce its saved forecasts")
        self.fidelity_max_error_kw = float(np.max(np.abs(actual-expected)))

    def inputs(self, checked, *, sample=False):
        ready = checked.report.get("sample_ready") if sample else checked.report["data_ready"]
        if not ready or checked.origin is None:
            raise ValueError("Recent household readings are required for a current estimate.")
        # Active-power resampling in check_csv is identical to prepare_power.
        # Rebuild context with the original 17-channel recipe, not the newer one.
        power = checked.power
        expanded = power.reindex(pd.date_range(power.index[0], checked.origin+pd.Timedelta(hours=1), freq="15min"))
        context = self.funcs["build_context"](expanded, 4).tail(1)
        if list(context.columns) != self.manifest["context_features"]:
            raise ValueError("Power-history context feature order does not match the trained model.")
        window = power.tail(96)
        calendar = self.funcs["calendar_features"](window.index)
        history = np.column_stack([
            self.power_scaler.transform(window[["input_kw"]].to_numpy()).ravel(),
            window.observed.to_numpy(dtype=float),
            calendar[["hour_sin", "hour_cos", "dow_sin", "dow_cos"]].to_numpy(),
        ]).astype("float32")
        inputs = {"history": history[None],
                  "context": self.context_scaler.transform(context).astype("float32")}
        if history.shape != (96, 6) or not all(np.isfinite(v).all() for v in inputs.values()):
            raise ValueError("The power-history model needs complete recent and daily/weekly readings.")
        return inputs

    def predict_readings(self, checked, *, sample=False):
        inputs = self.inputs(checked, sample=sample)
        value = float(self.values_kw(inputs)[0])
        report = checked.report
        forecast = {"origin": report["origin"], "target_end": report["target_end"],
                    "target_interval_start": report["target_interval_start"], "forecast_kw": value,
                    "threshold_kw": float(self.manifest["peak_threshold_kw"]),
                    "model": "Bundled Task 7 power-history LSTM", "source": "uploaded_readings",
                    "source_run": self.manifest["run_id"], "model_sha256": self.model_sha256,
                    "timezone": report["timezone"], "units": "kW, average over the target 15-minute interval",
                    "is_sample": bool(sample),
                    "input_scope": "Active power, observation flags and calendar/daily/weekly context. Other electrical columns are checked but are not inputs to this model."}
        return forecast, {"history": report["history"],
            "confidence": {"available": False, "label": "Not quantified",
                           "reason": "No calibrated confidence score or prediction interval is supplied. Accuracy for your household has not been established."},
            "explanation": {"available": False,
                            "reason": "This estimate uses your recent power pattern, daily/weekly history and calendar information. A verified contribution breakdown has not been computed for this upload. Appliance rule evidence remains available."}}
