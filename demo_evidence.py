"""Inspectable frozen-model evidence for the guided demo, with real grouped SHAP."""
from collections import OrderedDict
import hashlib

import numpy as np
import pandas as pd

CACHE = OrderedDict()


def grouped_shap(forecaster, inputs):
    """Permutation SHAP over 23 groups against one recorded training reference.

    A history channel is one group across all 96 timesteps; context values are
    individual groups. Perturbations can be unrealistic combinations, not causes.
    """
    import shap
    with np.load(forecaster.folder / "task8_examples.npz") as archive:
        history = archive["background_history"]
        context = archive["background_context"]
        average = history[:, :, 0].mean(axis=1)
        index = int(np.argmin(np.abs(average-np.median(average))))
        reference = {"history": history[index], "context": context[index]}
        reference_index = int(archive["background_indices"][index])
    actual_h, actual_c = inputs["history"][0], inputs["context"][0]
    names = ["History: "+n for n in forecaster.manifest["history_features"]] + forecaster.manifest["context_features"]
    count = len(names)
    def masked_prediction(mask):
        mask = np.asarray(mask, dtype="float32")
        h = reference["history"][None] + mask[:, None, :6]*(actual_h-reference["history"])[None]
        c = reference["context"][None] + mask[:, 6:]*(actual_c-reference["context"])[None]
        return forecaster.values_kw({"history": h, "context": c})
    explainer = shap.Explainer(masked_prediction, np.zeros((1, count), dtype="float32"),
                               algorithm="permutation", feature_names=names, seed=42)
    explanation = explainer(np.ones((1, count), dtype="float32"), max_evals=(2*count+1)*8,
                            batch_size=64, silent=True)
    values = np.asarray(explanation.values).reshape(-1)
    base = float(np.asarray(explanation.base_values).reshape(-1)[0])
    prediction = float(forecaster.values_kw(inputs)[0])
    residual = abs(base+float(values.sum())-prediction)
    verified = bool(residual <= 1e-4+1e-4*abs(prediction))
    if not np.isfinite(values).all() or not np.isfinite([base, residual]).all():
        raise ValueError("SHAP returned non-finite evidence.")
    return {"available": True, "method": "Permutation SHAP", "reference_kw": base,
            "raw_forecast_kw": prediction, "reconstruction_verified": verified,
            "residual_kw": residual, "reference_training_index": reference_index,
            "permutations": 8, "group_count": count,
            "factors": [{"name": name, "value_kw": float(v)} for name, v in zip(names, values)],
            "note": "Actual model-agnostic SHAP for this frozen LSTM. Each history channel is grouped across 96 steps; each context feature is a group. Eight sampled permutations and one training reference estimate associations, not physical causes. Mixed feature combinations may not describe a real household."}


def model_evidence(forecaster, inputs, forecast):
    if not getattr(forecaster, "upload_ready", False):
        raise ValueError("The bundled forecasting runtime is unavailable.")
    if forecast.get("model_sha256") and forecast["model_sha256"] != forecaster.model_sha256:
        raise ValueError("This saved forecast belongs to a different model. Run a new demonstration.")
    if forecast.get("forecaster_sha256") and forecast["forecaster_sha256"] != getattr(forecaster, "forecaster_sha256", None):
        raise ValueError("This saved forecast belongs to a different trained prediction head. Run a new demonstration.")
    predicted = float(forecaster.values_kw(inputs)[0])
    if not np.isclose(predicted, forecast["forecast_kw"], rtol=1e-4, atol=1e-4):
        raise ValueError("The loaded model does not reproduce this saved forecast. No explanation was substituted.")
    identity = getattr(forecaster, "forecaster_sha256", forecaster.model_sha256)
    key = hashlib.sha256(identity.encode()+inputs["history"].tobytes()+inputs["context"].tobytes()).hexdigest()
    if key in CACHE:
        CACHE.move_to_end(key)
        return CACHE[key]
    import tensorflow as tf
    layer = tf.keras.Model(forecaster.model.inputs, forecaster.model.get_layer("lstm_temporal_features").output)
    activation = np.asarray(layer(inputs, training=False))[0].T
    data = {"forecast_kw": predicted, "model_sha256": forecaster.model_sha256,
            "prediction_reexecuted": True, "history_shape": list(inputs["history"].shape),
            "context_shape": list(inputs["context"].shape),
            "history_channels": forecaster.manifest["history_features"],
            "context_features": [{"name": n, "scaled_value": float(v)} for n, v in zip(forecaster.manifest["context_features"], inputs["context"][0])],
            "architecture": [{"layer": "LSTM", "units": 32}, {"layer": "LSTM", "units": 16}, {"layer": "Dense", "units": 32}, {"layer": "Forecast", "units": 1}],
            "activation": activation.tolist(),
            "training": {"sequence_steps": 96, "channels": 6, "context_count": 17,
                         "train_examples": forecaster.manifest["used_split_counts"]["train"],
                         "validation_examples": forecaster.manifest["used_split_counts"]["validation"],
                         "test_examples": forecaster.manifest["used_split_counts"]["test"]},
            "comparison": pd.read_csv(forecaster.folder / "test_model_comparison.csv").to_dict("records"),
            "classical_ml": {"available": False, "reason": "This saved analysis used the standalone LSTM. A classical prediction head is not retroactively attached to historical results. Run a new demonstration to use the currently configured forecasting pipeline."}}
    if getattr(forecaster, "is_hybrid", False):
        data.update(forecaster_sha256=forecaster.forecaster_sha256,
                    comparison=[r for r in forecaster.metadata["test"] if r["Segment"] == "All test"],
                    architecture=[{"layer": "LSTM", "units": 32}, {"layer": "LSTM", "units": 16}, {"layer": "Context", "units": 17}, {"layer": forecaster.selected_name, "units": 1}],
                    classical_ml={"available": True, "model": forecaster.selected_name,
                                  "feature_count": 33, "latent_features": 16, "context_features": 17,
                                  "latent_values": [{"name": n, "value": float(v)} for n,v in zip(forecaster.feature_names[:16], forecaster.features(inputs)[0,:16])],
                                  "validation": forecaster.metadata["validation"],
                                  "counts": forecaster.metadata["counts"], "run_id": forecaster.metadata["run_id"],
                                  "reason": "Executed newly trained classical head on the frozen older power-history LSTM. Selected by validation RMSE, then evaluated on the full previously used original test period. Not the missing newer multivariate experiment."})
    try:
        data["shap"] = forecaster.explain_inputs(inputs) if getattr(forecaster, "is_hybrid", False) else grouped_shap(forecaster, inputs)
    except Exception as error:
        print("Demo SHAP unavailable:", error, flush=True)
        data["shap"] = {"available": False, "reason": "SHAP could not be computed for this run. No saved or invented contributions have been substituted."}
    if data["shap"]["available"]:
        CACHE[key] = data
        while len(CACHE) > 24:
            CACHE.popitem(last=False)
    return data
