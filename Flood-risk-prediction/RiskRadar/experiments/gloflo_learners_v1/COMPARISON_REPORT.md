# Learning algorithms and sampling comparison

The same 2006–2015 test and validation periods were examined in prior experiments. This is an exploratory follow-up, not fresh independent confirmation. Broad validation search can overfit validation; no test results are used in selection.

## Fixed target and data

High annual peak discharge relative to training-source 95th percentile; not verified flood occurrence. The training-only 95th percentile is 1129.84 in source units.
Train: 1900–1999, 37,184 rows / 1,859 positive. Validation: 2000–2005, 3,387 rows / 162 positive. Test: 2006–2015, 5,339 rows / 320 positive. All source rows, labels, features and dates match the previous imbalance studies.

## Training and selection

Numeric medians/standardisation and category vocabulary fitted on original training rows only. All learners use the same scaled numeric plus one-hot representation.
Five standard learners × seven treatments × two configurations = 70 fits. Balanced Random Forest, balanced bagging and EasyEnsemble each add two configurations: 76 candidate fits, with 38 frozen learner/treatment pipelines.
Treatments: untreated, positive class weight sqrt(training ratio), random oversampling to 1:1, random undersampling to 1:4, SMOTENC to 1:2 with k=3, SMOTENC then majority-only Tomek cleaning, and SMOTENC then majority-only ENN cleaning. Cleaning distances use scaled numeric and one-hot categorical columns. Minority examples are preserved by both cleaners.
No resampling, cleaning or weight fitting is applied to validation/test. HistGradientBoosting disables random internal early stopping; no learner fits on validation. The three balanced ensembles perform their own training-only internal sampling.
Max validation positive-class F1; ties average precision then precision. Every operating threshold is chosen on validation only. No test selection/refit.

New-study validation winner: **random_forest__random_oversampling**, validation F1 0.7972, threshold 0.964886.
Validation winner including prior frozen studies: **new_study:random_forest__random_oversampling**.

## Best configuration per learner (chosen on validation)

| Learner / treatment | Validation F1 | Test precision | Recall | F1 | F2 | PR AP | ROC-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_forest__random_oversampling | 0.7972 | 0.8894 | 0.6531 | 0.7532 | 0.6898 | 0.8431 | 0.9707 | 26 | 111 |
| hist_gradient_boosting__untreated | 0.7835 | 0.9083 | 0.6813 | 0.7786 | 0.7171 | 0.8528 | 0.9746 | 22 | 102 |
| extra_trees__untreated | 0.7639 | 0.8773 | 0.6031 | 0.7148 | 0.6433 | 0.8328 | 0.9683 | 27 | 127 |
| xgboost__class_weight | 0.7346 | 0.7879 | 0.7312 | 0.7585 | 0.7419 | 0.8199 | 0.9735 | 63 | 86 |
| balanced_random_forest__internal_balancing | 0.7342 | 0.8140 | 0.6562 | 0.7266 | 0.6827 | 0.8105 | 0.9716 | 48 | 110 |
| balanced_bagging__internal_balancing | 0.5955 | 0.7137 | 0.5375 | 0.6132 | 0.5654 | 0.6926 | 0.9584 | 69 | 148 |
| logistic_regression__random_oversampling | 0.3636 | 0.3871 | 0.2625 | 0.3128 | 0.2806 | 0.2703 | 0.8629 | 133 | 236 |
| easy_ensemble__internal_balancing | 0.3302 | 0.3230 | 0.4906 | 0.3896 | 0.4445 | 0.2774 | 0.8634 | 329 | 163 |

## Previous baselines and uncertainty

| Previous frozen baseline | Precision | Recall | F1 | F2 | PR AP | ROC-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| previous_untreated | 0.7647 | 0.6906 | 0.7258 | 0.7043 | 0.8017 | 0.9693 | 68 | 99 |
| previous_balanced | 0.7468 | 0.7188 | 0.7325 | 0.7242 | 0.7823 | 0.9721 | 78 | 90 |

Paired station-cluster bootstrap 95% intervals for selected-new-model F1 minus baseline:
- previous_untreated: [-0.0208, 0.0753]
- previous_balanced: [-0.0220, 0.0599]
- same_learner_untreated: [-0.0576, 0.0114]

Intervals spanning zero do not establish reliable improvement. The winner is frozen using validation, never chosen by whichever test metric looks best.

## What changed

The validation-selected model has test F1 0.7532, compared with 0.7325 for the previous balanced XGBoost. Its precision increases from 0.7468 to 0.8894, but recall decreases from 0.7188 to 0.6531. This is a precision/recall trade-off, not improvement everywhere.
The untreated Random Forest has test F1 0.7744, above the selected oversampled version at 0.7532. Oversampling won validation but did not improve Random Forest test F1. Untreated HistGradientBoosting has the highest observed test F1 among these cells (0.7786); it is not substituted as the winner after inspecting test results. The results provide no consistent evidence that more elaborate sampling is better.

## All learner / treatment cells

| Cell | Validation F1 | Test precision | Recall | F1 | AP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_forest__random_oversampling | 0.7972 | 0.8894 | 0.6531 | 0.7532 | 0.8431 | 26 | 111 |
| random_forest__untreated | 0.7852 | 0.8971 | 0.6813 | 0.7744 | 0.8573 | 25 | 102 |
| hist_gradient_boosting__untreated | 0.7835 | 0.9083 | 0.6813 | 0.7786 | 0.8528 | 22 | 102 |
| hist_gradient_boosting__class_weight | 0.7808 | 0.8943 | 0.6875 | 0.7774 | 0.8574 | 26 | 100 |
| random_forest__smotenc_tomek | 0.7640 | 0.8169 | 0.7250 | 0.7682 | 0.8454 | 52 | 88 |
| extra_trees__untreated | 0.7639 | 0.8773 | 0.6031 | 0.7148 | 0.8328 | 27 | 127 |
| random_forest__smotenc | 0.7584 | 0.8076 | 0.7344 | 0.7692 | 0.8453 | 56 | 85 |
| random_forest__class_weight | 0.7571 | 0.8364 | 0.7188 | 0.7731 | 0.8418 | 45 | 90 |
| random_forest__random_undersampling | 0.7500 | 0.8352 | 0.6813 | 0.7504 | 0.8142 | 43 | 102 |
| extra_trees__random_undersampling | 0.7476 | 0.7962 | 0.6594 | 0.7214 | 0.7815 | 54 | 109 |
| hist_gradient_boosting__random_oversampling | 0.7468 | 0.8459 | 0.7031 | 0.7679 | 0.8445 | 41 | 95 |
| extra_trees__class_weight | 0.7395 | 0.7173 | 0.7375 | 0.7273 | 0.8134 | 93 | 84 |
| extra_trees__smotenc_tomek | 0.7358 | 0.8638 | 0.5750 | 0.6904 | 0.7873 | 29 | 136 |
| extra_trees__smotenc | 0.7347 | 0.8632 | 0.5719 | 0.6880 | 0.7839 | 29 | 137 |
| xgboost__class_weight | 0.7346 | 0.7879 | 0.7312 | 0.7585 | 0.8199 | 63 | 86 |
| balanced_random_forest__internal_balancing | 0.7342 | 0.8140 | 0.6562 | 0.7266 | 0.8105 | 48 | 110 |
| hist_gradient_boosting__smotenc | 0.7333 | 0.9391 | 0.5781 | 0.7157 | 0.8338 | 12 | 135 |
| xgboost__untreated | 0.7329 | 0.7774 | 0.6875 | 0.7297 | 0.7937 | 63 | 100 |
| hist_gradient_boosting__random_undersampling | 0.7314 | 0.8276 | 0.6000 | 0.6957 | 0.7914 | 40 | 128 |
| extra_trees__random_oversampling | 0.7296 | 0.7984 | 0.6312 | 0.7051 | 0.7969 | 51 | 118 |
| xgboost__random_oversampling | 0.7267 | 0.7774 | 0.6656 | 0.7172 | 0.8002 | 61 | 107 |
| hist_gradient_boosting__smotenc_tomek | 0.7134 | 0.7804 | 0.7219 | 0.7500 | 0.8325 | 65 | 89 |
| extra_trees__smotenc_enn | 0.7103 | 0.7800 | 0.6094 | 0.6842 | 0.7557 | 55 | 125 |
| random_forest__smotenc_enn | 0.7080 | 0.7590 | 0.7281 | 0.7432 | 0.7532 | 74 | 87 |
| xgboost__smotenc | 0.6987 | 0.8016 | 0.6438 | 0.7140 | 0.7886 | 51 | 114 |
| hist_gradient_boosting__smotenc_enn | 0.6974 | 0.7252 | 0.7094 | 0.7172 | 0.7880 | 86 | 93 |
| xgboost__smotenc_tomek | 0.6969 | 0.7130 | 0.7375 | 0.7250 | 0.7944 | 95 | 84 |
| xgboost__smotenc_enn | 0.6907 | 0.6765 | 0.6469 | 0.6613 | 0.7428 | 99 | 113 |
| xgboost__random_undersampling | 0.6823 | 0.8073 | 0.5500 | 0.6543 | 0.7435 | 42 | 144 |
| balanced_bagging__internal_balancing | 0.5955 | 0.7137 | 0.5375 | 0.6132 | 0.6926 | 69 | 148 |
| logistic_regression__random_oversampling | 0.3636 | 0.3871 | 0.2625 | 0.3128 | 0.2703 | 133 | 236 |
| logistic_regression__smotenc_enn | 0.3614 | 0.3482 | 0.2687 | 0.3034 | 0.2651 | 161 | 234 |
| logistic_regression__smotenc_tomek | 0.3590 | 0.3063 | 0.3187 | 0.3124 | 0.2644 | 231 | 218 |
| logistic_regression__smotenc | 0.3590 | 0.3072 | 0.3187 | 0.3129 | 0.2638 | 230 | 218 |
| logistic_regression__class_weight | 0.3557 | 0.4000 | 0.2500 | 0.3077 | 0.2795 | 120 | 240 |
| logistic_regression__untreated | 0.3551 | 0.3816 | 0.2719 | 0.3175 | 0.2897 | 141 | 233 |
| logistic_regression__random_undersampling | 0.3476 | 0.3406 | 0.2938 | 0.3154 | 0.2773 | 182 | 226 |
| easy_ensemble__internal_balancing | 0.3302 | 0.3230 | 0.4906 | 0.3896 | 0.2774 | 329 | 163 |

## Selected model transfer and calibration

| Stations | Rows | Positives | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|
| seen_stations | 4845 | 285 | 0.8894 | 0.7333 | 0.8038 | 0.8739 |
| unseen_stations | 494 | 35 | 0.0000 | 0.0000 | 0.0000 | 0.6883 |

**Transfer limitation:** the frozen winner detects 0 of 35 positive cases at unseen stations (recall 0.0000). Aggregate metrics must not be treated as evidence of reliable performance at new stations. The model is retained as an experiment and is not promoted to the website.
Selected Brier score: 0.0316; previous untreated: 0.0250. Lower Brier is better, but does not by itself establish probability calibration. Oversampled/weighted scores must not be labelled calibrated flood probabilities.
Target: high annual peak discharge, not verified flood occurrence. Static contemporary attributes and repeated stations do not establish a lead-time weather forecast or new-location performance. Synthetic examples are statistical interpolations, not measured or physically validated catchments.
Static modern catchment/climate proxies are not historically issued forecasts; station overlap is reported and unseen-station transfer is not assumed.
All 76 candidate validation scores, 38 pipelines, source/split hashes, sampling audit, test predictions and metrics are preserved. Previous studies and the website model are unchanged.

## Reproduction

Dependencies: requirements-training.txt. Existing frozen files refuse overwrite. Use a new prespecified experiment version for a new study.

```powershell
.\.venv\Scripts\python.exe scripts/compare_gloflo_learners.py prepare
.\.venv\Scripts\python.exe scripts/compare_gloflo_learners.py select
.\.venv\Scripts\python.exe scripts/compare_gloflo_learners.py evaluate
.\.venv\Scripts\python.exe scripts/compare_gloflo_learners.py verify
```

Methods: [imbalanced-learn ensembles](https://imbalanced-learn.org/stable/ensemble.html), [neighbour cleaning](https://imbalanced-learn.org/stable/under_sampling.html).
