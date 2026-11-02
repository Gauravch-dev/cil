# Trained local LSTM + classical ML pipeline

Prepared on 3 November 2026 from the completed older Task 7 run.

The missing classical prediction stage is now trained, evaluated and connected. The existing LSTM and scalers were reused unchanged. This is a new compatible run; it does not recreate the missing newer multivariate/peak-gated experiment.

## Training procedure

- Frozen LSTM: 96 steps × six history channels; 16 learned summary values.
- Regressor input: 16 LSTM patterns + 17 standardized context features = 33 features.
- Training: 20,000 evenly spaced eligible original training examples.
- Validation: 5,000 original validation examples. Selection uses lowest validation RMSE.
- Test: all 18,889 original eligible test examples, including 959 high-demand targets.
- Random Forest and Gradient Boosting use the original Task 9 candidate configurations.
- LSTM weights and scalers were not fitted again. No uploads or household settings entered training.
- The original test period was evaluated in prior work; this is not a new independent household dataset.

## Validation selection

| Candidate | MAE (kW) | RMSE (kW) | R² |
| --- | ---: | ---: | ---: |
| LSTM + Random Forest | 0.4736 | 0.7233 | 0.4777 |
| LSTM + Gradient Boosting | 0.4836 | 0.7260 | 0.4738 |

Selected: **LSTM + Random Forest**. Selection was saved before test predictions and metrics were calculated.

## Test results

| Model | Segment | MAE (kW) | RMSE (kW) | R² |
| --- | --- | ---: | ---: | ---: |
| LSTM + Random Forest | All test | 0.4316 | 0.6301 | 0.3766 |
| LSTM + Gradient Boosting | All test | 0.4307 | 0.6268 | 0.3832 |
| Standalone LSTM | All test | 0.4269 | 0.6182 | 0.3999 |
| Basic MLP | All test | 0.4407 | 0.6235 | 0.3896 |
| Persistence | All test | 0.5143 | 0.8055 | -0.0188 |
| LSTM + Random Forest | High demand | 1.4450 | 1.6686 | -5.4053 |
| LSTM + Gradient Boosting | High demand | 1.4838 | 1.6972 | -5.6268 |
| Standalone LSTM | High demand | 1.5016 | 1.7145 | -5.7625 |
| Basic MLP | High demand | 1.5736 | 1.7610 | -6.1343 |
| Persistence | High demand | 1.6169 | 1.8836 | -7.1622 |

**Interpretation:** the selected hybrid has slightly worse overall RMSE than the standalone LSTM (0.6301 vs 0.6182 kW), but better high-demand RMSE (1.6686 vs 1.7145 kW). It improves over persistence overall. Strongly negative high-demand R² remains a limitation. Gradient Boosting happened to have lower test RMSE than Random Forest; the selected head was not changed after inspecting test scores.

## Use and reproduce

Install requirements-forecast.txt, then explicitly prepare the model:

```powershell
.\.venv-web\Scripts\python.exe -B tools\train_local_hybrid.py
```

Restart the local server afterwards. Runtime only loads an evaluated, fingerprint-matched cache and verifies five saved inference examples. It never fits on startup, uploads or demo clicks. Existing evaluated artifacts are retained unless --force is explicitly supplied.

The current pointer is task9_outputs/local_hybrid/active.json. The selected regressor, manifests, comparison/prediction CSVs and verification inputs are saved under its run directory. Frozen extraction arrays are a preparation cache, not required for serving.

The evaluated run's selected model, active pointer, verification inputs and evaluation evidence are checked into Git so another checkout can reproduce the demonstration without fitting. Frozen extraction caches and future generated runs remain ignored. A separate Trained-hybrid-artifacts.zip also preserves the serving files. The recorded scikit-learn version is pinned in requirements-forecast.txt and must match; otherwise explicitly prepare a new local run. Original sources and model/scaler fingerprints must match as well.

The original preprocessing file was fingerprinted with Windows CRLF endings;
.gitattributes preserves those checkout bytes, and the original manifest's bytes,
across platforms. Archive exports or manually reformatted sources can differ;
use a regular Git checkout or explicitly prepare a matching new run with --force.

## Website integration and SHAP

- New recorded-history, actual-upload and sample analyses use the active hybrid.
- Old saved standalone LSTM results keep their original values; they are never silently upgraded.
- Guided demo shows the actual LSTM-to-regressor feature handoff, validation choice and test comparison.
- TreeSHAP explains the executed regressor using its 33 learned/context inputs. Contributions reconstruct the raw output before any non-negative clipping.
- Old standalone analyses retain their own permutation SHAP path; model/head fingerprints prevent mixing explanations.
- Rules, ontology, A* scheduling, source dates and sample/current-data boundaries are unchanged.
- Confidence is unquantified, and new-household accuracy/actual bill savings remain unestablished.

## Artifact identity

- Run: `local_hybrid_c36dc5c8927301cc`
- Signature: `c36dc5c8927301cc8fb2c640016d2ce86efd01db66c1c3a063223584849d6ab1`
- Regressor SHA-256: `8336084855c06a0232d28404e46dc6780a9c4779e829f97a18abcc57e4ee4dc3`
- scikit-learn: `1.9.1`
- Preparation time: 219.1 seconds on this machine.
- Source integrity verified: `True`.
