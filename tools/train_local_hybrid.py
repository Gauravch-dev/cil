"""Explicit offline preparation of the missing Task 9 classical prediction head.

Reuses the completed older power-history LSTM, original partitions and the Task 9
RF/GB configurations. This is a new compatible run, not the missing newer model.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_forecast import LocalPowerForecast
from hybrid_forecast import ARTIFACT_ROOT, SETTINGS, digest, extract_features, make_encoder, signature_for, source_fingerprints


def metrics(actual, predicted):
    return {"MAE_kW": float(mean_absolute_error(actual, predicted)), "RMSE_kW": float(np.sqrt(mean_squared_error(actual, predicted))), "R2": float(r2_score(actual, predicted))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Explicitly retrain this preparation run")
    args = parser.parse_args()
    folder = ROOT/"task7_outputs/full_20261004T223051_253137Z"
    signature, hashes = signature_for(folder)
    out = ARTIFACT_ROOT/signature[:16]
    out.mkdir(parents=True, exist_ok=True)
    if (out/"training_manifest.json").is_file() and not args.force:
        print("Existing evaluated artifact retained. Use --force only for intentional retraining.", flush=True)
        return
    started = time.perf_counter()
    base = LocalPowerForecast(folder)
    encoder = make_encoder(base)
    with np.load(folder/"prepared_arrays.npz") as a:
        history, context = a["history_values"], a["context_values"]
        target = a["target_scaled"].ravel()*base.power_scaler.scale_[0]+base.power_scaler.mean_[0]
        timestamps = a["timestamps_ns"]
        def sample(ids, cap):
            return ids[np.unique(np.linspace(0, len(ids)-1, min(len(ids), cap), dtype=int))]
        ids = {"train": sample(a["train_indices"], SETTINGS["train_cap"]),
               "validation": sample(a["validation_indices"], SETTINGS["validation_cap"]), "test": a["test_indices"]}
    assert ids["train"].max() < ids["validation"].min()-4
    assert ids["validation"].max() < ids["test"].min()-4
    feature_file = out/"frozen_features.npz"
    features = {}
    if feature_file.exists() and not args.force:
        with np.load(feature_file) as a:
            if str(a["signature"]) != signature:
                raise ValueError("Cached feature signature mismatch")
            for split in ids:
                np.testing.assert_array_equal(ids[split], a[split+"_ids"])
                features[split] = a[split]
        print("Reusing signature-matched frozen feature extraction.", flush=True)
    else:
        for split, rows in ids.items():
            batches = []
            print(f"Extracting {split}: {len(rows):,} original examples", flush=True)
            for offset in range(0, len(rows), 256):
                positions = rows[offset:offset+256]
                steps = positions[:, None]-99+np.arange(96)[None]
                inputs = {"history": history[steps], "context": context[positions]}
                batches.append(extract_features(encoder, inputs))
                if offset % 4096 == 0:
                    print(f"  {min(offset+256,len(rows)):,}/{len(rows):,}", flush=True)
            features[split] = np.concatenate(batches)
        np.savez_compressed(feature_file, signature=signature, **features, **{k+"_ids": v for k, v in ids.items()})
    candidates = {"LSTM + Random Forest": RandomForestRegressor(**SETTINGS["candidates"]["LSTM + Random Forest"]),
                  "LSTM + Gradient Boosting": GradientBoostingRegressor(**SETTINGS["candidates"]["LSTM + Gradient Boosting"])}
    validation = []
    for name, model in candidates.items():
        print("Training:", name, flush=True)
        model.fit(features["train"], target[ids["train"]])
        score = {"Model": name, **metrics(target[ids["validation"]], np.maximum(model.predict(features["validation"]), 0.))}
        validation.append(score)
        print(json.dumps(score), flush=True)
    validation = sorted(validation, key=lambda r: r["RMSE_kW"])
    selected_name = validation[0]["Model"]
    selected = candidates[selected_name]
    # Freeze validation selection BEFORE touching test predictions/results.
    (out/"selection.json").write_text(json.dumps({"selected_name": selected_name, "criterion": "Lowest validation RMSE; test is excluded from selection", "validation": validation}, indent=2))
    selected_path = out/"hybrid_forecaster.joblib"
    joblib.dump(selected, selected_path, compress=3)
    selected_hash = digest(selected_path)
    print("Selected on validation:", selected_name, flush=True)
    saved = pd.read_csv(folder/"test_predictions.csv")
    actual = target[ids["test"]]
    np.testing.assert_allclose(actual, saved.Actual_kW.to_numpy(), atol=1e-5)
    predictions = {name: np.maximum(model.predict(features["test"]), 0.) for name, model in candidates.items()}
    predictions.update({"Standalone LSTM": saved.lstm_compact_kW.to_numpy(), "Basic MLP": saved.basic_mlp_kW.to_numpy(), "Persistence": saved.Persistence_kW.to_numpy()})
    threshold = base.manifest["peak_threshold_kw"]
    test_results = []
    for segment, mask in {"All test": np.ones(len(actual), dtype=bool), "High demand": actual>=threshold}.items():
        for name, prediction in predictions.items():
            test_results.append({"Model": name, "Segment": segment, "Samples": int(mask.sum()), **metrics(actual[mask], prediction[mask]), "Selected_on_validation": name==selected_name})
    pd.DataFrame(validation).to_csv(out/"validation_comparison.csv", index=False)
    pd.DataFrame(test_results).to_csv(out/"test_comparison.csv", index=False)
    pd.DataFrame({"Forecast_origin": pd.to_datetime(timestamps[ids["test"]-4]), "Target_interval_end": pd.to_datetime(timestamps[ids["test"]]), "Actual_kW": actual, **predictions}).to_csv(out/"test_predictions.csv", index=False)
    check_positions = np.unique(np.linspace(0, len(ids["test"])-1, 5, dtype=int))
    rows = ids["test"][check_positions]
    inputs = {"history": history[rows[:,None]-99+np.arange(96)[None]], "context": context[rows]}
    np.savez_compressed(out/"verification_inputs.npz", **inputs, prediction_kw=predictions[selected_name][check_positions])
    if source_fingerprints(folder) != hashes or digest(selected_path) != selected_hash:
        raise ValueError("Frozen source/model integrity changed unexpectedly")
    manifest = {"status": "evaluated", "schema": 1, "run_id": "local_hybrid_"+signature[:16], "created_at": datetime.now(timezone.utc).isoformat(),
                "signature": signature, "source_fingerprints": hashes, "source_run": base.manifest["run_id"], "settings": SETTINGS,
                "selected_name": selected_name, "selected_file": selected_path.name, "selected_sha256": selected_hash,
                "selection_criterion": "Lowest validation RMSE; no test-based tuning", "validation": validation, "test": test_results,
                "counts": {k: len(v) for k,v in ids.items()}, "feature_count": int(features["train"].shape[1]),
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__},
                "lstm_retrained": False, "scalers_refitted": False, "source_integrity_verified": True,
                "training_seconds": time.perf_counter()-started,
                "limitations": "New classical prediction heads on the completed older six-channel LSTM. Not the missing newer multivariate experiment. Previously evaluated original test period reused; no independent new household measurements or calibrated confidence."}
    (out/"training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Publish the local load-only pointer only after evaluation and integrity pass.
    pointer = ARTIFACT_ROOT/"active.json"
    temporary = pointer.with_suffix(".pending")
    temporary.write_text(json.dumps({"run": signature[:16]}))
    temporary.replace(pointer)
    print(json.dumps({"artifact_folder": str(out), "selected": selected_name, "seconds": manifest["training_seconds"], "test": test_results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
