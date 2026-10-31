**Scope:** This is a separate retraining study, not an evaluation of the existing deployed XGBoost. The requested existing-model evaluation is in `../../evaluation/existing_model/EVALUATION.md`. These experimental results must not be attributed to the website model.

# Station-separated evaluation of annual high-discharge susceptibility

This study retrains classifiers using the existing manual-input feature representation. It evaluates transfer to station/input groups excluded from fitting and model selection. The deployed website model is unchanged.

## Registered method

- Source: GloFlo, 45,932 catchment-year records. Source SHA256: `20a844f84f78729cf4361f3f13daa3a79626f67c2e96aeb0f1a3b872bfd1a45a`.
- Target: annual peak discharge above the 95th percentile of training records only.
- Training-derived target threshold: 1197.641000 source units.
- The 60/20/20 allocation uses independent station/input groups and one fixed seed. Stations sharing exact coordinates or identical manual input profiles stay together. No labels are used to choose the split.
- The split is geographical/grouped, not chronological. It evaluates previously excluded station groups within this archive; it is not a future-date forecast.
- Median imputation, scaling and category vocabularies are fitted on training rows only. Test-only categories are ignored by one-hot encoding.
- Positive training weights use the training negative/positive ratio. Validation and test rows retain their natural distributions. No test resampling is used.
- Candidate settings are registered in protocol.json. Model and score threshold selection use validation positive-class F1 only, with the registered tie breakers. No validation refit is performed.
- Selection was frozen at 2026-10-25T07:34:41.637784+00:00 before the test opening. The test opening receipt records selection SHA256 `60523fd13af51784fe1ae27a536209081765cb9b83df408834ffc7435869e99f`.
- The full source has been analysed in earlier studies. These groups are held out from the new fits; this is not a pristine external dataset or a proof of zero bias.

## Partition sizes

| Partition | Rows | Stations | Groups | Positive cases |
|---|---:|---:|---:|---:|
| train | 27,521 | 5,822 | 5,769 | 1,376 |
| validation | 9,299 | 1,947 | 1,924 | 406 |
| test | 9,112 | 1,941 | 1,924 | 354 |

Station and group overlap is zero between every pair of partitions.

## Candidate comparison

| Method | Validation F1 | Test accuracy | Positive precision | Positive recall | Positive F1 | Test AP |
|---|---:|---:|---:|---:|---:|---:|
| logistic_plain | 0.2993 | 94.37% | 20.22% | 15.25% | 17.39% | 0.1903 |
| extra_trees_plain | 0.4667 | 92.36% | 27.79% | 60.45% | 38.08% | 0.3669 |
| hist_gradient_plain **selected** | 0.5128 | 95.25% | 38.48% | 37.29% | 37.88% | 0.3795 |
| xgb_regularized_plain | 0.4935 | 95.01% | 33.33% | 28.53% | 30.75% | 0.3453 |
| xgb_original_shape_plain | 0.4425 | 93.43% | 31.91% | 61.02% | 41.90% | 0.3756 |
| logistic_weighted | 0.2929 | 87.03% | 17.14% | 61.02% | 26.77% | 0.1953 |
| extra_trees_weighted | 0.4264 | 94.91% | 36.90% | 43.79% | 40.05% | 0.3613 |
| hist_gradient_weighted | 0.4967 | 94.24% | 32.08% | 43.22% | 36.82% | 0.3715 |
| xgb_regularized_weighted | 0.4788 | 94.50% | 31.67% | 35.88% | 33.64% | 0.3252 |
| xgb_original_shape_weighted | 0.4614 | 94.58% | 35.04% | 46.33% | 39.90% | 0.3910 |
| rainfall_only_logistic | 0.1660 | 80.02% | 10.16% | 52.82% | 17.04% | 0.0965 |
| always_other | — | 96.12% | 0.00% | 0.00% | 0.00% | 0.0388 |
| training_prevalence_constant | — | 96.12% | 0.00% | 0.00% | 0.00% | 0.0388 |
| annual_rainfall_p95_rule | — | 92.44% | 11.67% | 14.41% | 12.90% | 0.0501 |

The selected model is fixed by validation. Test results for other registered candidates are disclosed for comparison and are not used to replace it.

## Selected model evaluation

Selected model: **hist_gradient_plain**, with validation-chosen threshold **0.286972**.

| Metric | Value | 95% group bootstrap interval |
|---|---:|---|
| accuracy | 95.25% | 94.30%–96.18% |
| precision | 38.48% | 28.34%–49.87% |
| recall | 37.29% | 25.88%–47.76% |
| f1 | 37.88% | 27.89%–47.06% |

Balanced accuracy: 67.44%; macro precision: 67.98%; macro recall: 67.44%; macro F1: 67.70%.
ROC-AUC: 0.9277; average precision: 0.3795; F2: 0.3752; Brier score: 0.0299.

| Actual / Predicted | Other discharge | High discharge |
|---|---:|---:|
| Other discharge | 8,547 | 211 |
| High discharge | 222 | 132 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Other discharge | 97.47% | 97.59% | 97.53% | 8,758 |
| High discharge | 38.48% | 37.29% | 37.88% | 354 |

Giving each test station equal total weight yields accuracy 95.71%, precision 40.90%, recall 39.18% and F1 40.02%. This sensitivity analysis reduces the influence of stations with longer records.

## Interpretation

The earlier 96.89% accuracy / 65.64% precision / 79.08% recall / 71.74% F1 values were artifact reconstruction with overlapping stations and a source-wide target threshold. They are not directly comparable to this grouped experiment, which uses freshly fitted models, different held-out records and a training-only target threshold.
Accuracy and class-specific scores are reported together without forcing equality or selecting an averaging method to conceal minority-class errors. Intervals resample whole station/input groups rather than correlated yearly rows. The discharge-percentile label remains a susceptibility proxy; this study does not add observed flood/no-flood labels.

## Reproduction and evidence

Run from RiskRadar: `.venv\Scripts\python.exe grouped_evaluation.py --stage all` for a new registered run. Existing frozen runs refuse overwrite or repeat test access. Use `--output-dir` to specify a different run only for a separately justified, predeclared experiment; do not search seeds after viewing results.
Files: protocol.json, split_manifest.csv, data/train.csv, data/validation.csv, data/test.csv, selection.json, test_opened.json, validation_predictions.csv, test_predictions.csv, final_results.json and evaluated_model_bundle.joblib.
Method references: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html and https://scikit-learn.org/stable/common_pitfalls.html#data-leakage.
