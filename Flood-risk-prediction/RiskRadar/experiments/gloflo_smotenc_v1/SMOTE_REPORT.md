# Training-only SMOTE-NC comparison

SMOTENC is the SMOTE variant for mixed numeric and categorical predictors. It interpolates numeric values and selects valid category values. This experiment does not manufacture fractional soil or climate categories.

## Protocol

Target: High annual peak discharge relative to training-source 95th percentile; not verified flood occurrence. Training-only discharge cutoff: 1129.84.
Train: 1900–1999, 37,184 rows (1,859 positive). Validation: 2000–2005, 3,387 rows (162 positive). Test: 2006–2015, 5,339 rows (320 positive). Same source, features, partitions and labels as imbalance_v1.
Same 2006–2015 test was already examined in imbalance_v1. This is a follow-up comparison, not fresh independent confirmation. No test metrics are used by this selection stage.

Train-only numeric medians and standardisation determine nearest neighbours. After sampling, numeric scaling is reversed and the original train-fitted model preprocessing is applied. All synthetic rows are training positives; validation/test are never sampled. Category values are checked against the original training vocabulary.
Eighteen candidates: minority/majority ratios 0.25, 0.5, 1.0; k=3 or 5; the same three XGBoost configurations as the previous comparison. Positive class weight is 1 to isolate SMOTE from weighting. Thresholds and candidate selection use validation F1 only (ties AP, then precision).

## Frozen validation selection

Best SMOTENC: ratio 0.5, k=3, config {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.08, 'min_child_weight': 5, 'reg_lambda': 5, 'reg_alpha': 1}, threshold 0.797929; validation F1 0.7086.
Selected training rows: 52987, positives 17662, synthetic positives 15803.
Validation winner when compared with previous frozen methods: balanced_oversampling_weight2.

## Chronological test comparison

| Strategy | Precision | Recall | F1 | F2 | PR AP | ROC-AUC | Accuracy | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| untreated | 0.7647 | 0.6906 | 0.7258 | 0.7043 | 0.8017 | 0.9693 | 0.9687 | 68 | 99 |
| previous_balanced | 0.7468 | 0.7188 | 0.7325 | 0.7242 | 0.7823 | 0.9721 | 0.9685 | 78 | 90 |
| smotenc | 0.8008 | 0.6281 | 0.7040 | 0.6564 | 0.7897 | 0.9694 | 0.9683 | 50 | 119 |

## Interpretation and limits

SMOTENC raises precision from 0.7647 to 0.8008, but recall falls from 0.6906 to 0.6281 and F1 falls from 0.7258 to 0.7040. It does not improve the selected F1 objective. The previous balanced model remains the validation-selected winner; SMOTENC is saved as an experimental comparison, not deployed.
Station-cluster bootstrap 95% interval for SMOTENC minus untreated F1: [-0.059030891258493826, 0.012016160471172318].
Brier scores (lower is better): untreated 0.0250, SMOTENC 0.0400. Oversampling changes score scale; these scores are not calibrated flood probabilities. The 0.5 operating point is saved separately.
The high-discharge label is not verified flood occurrence. Modern static catchment/climate attributes do not establish an operational weather forecast. Synthetic feature combinations remain statistical interpolations, not measured catchments or physically validated hydrological events. Repeated station attributes cross time; unseen-station performance is reported separately.

| SMOTENC subgroup | Rows | Positives | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|
| seen_stations | 4845 | 285 | 0.8051 | 0.6667 | 0.7294 | 0.8098 |
| unseen_stations | 494 | 35 | 0.7333 | 0.3143 | 0.4400 | 0.6137 |

## Saved artifacts

Protocol, 18 validation candidates, frozen selection, fitted inference pipeline, native UBJ model, sampling preprocessor, selected resampled training table with synthetic flags, feature importance, test predictions and metrics are saved here. Prior experiment artifacts and the website model are unchanged.

## Reproduction

Run from the RiskRadar directory in a new experiment version. Existing frozen protocol/selection/results refuse overwrite.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe scripts/evaluate_gloflo_smote.py prepare
.\.venv\Scripts\python.exe scripts/evaluate_gloflo_smote.py select
.\.venv\Scripts\python.exe scripts/evaluate_gloflo_smote.py evaluate
```

Method: [imbalanced-learn SMOTENC documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html).
