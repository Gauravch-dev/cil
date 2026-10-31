# High discharge imbalance comparison

This is a new chronological comparison of imbalance treatments using GloFlo. The positive class means high annual peak discharge, not verified flood occurrence. The raw source was already inspected during forensics; this is not an external blind validation.

## Target and chronology

The label is annual peak discharge above the **training-only** 95th percentile: 1129.840000 in source units. This differs intentionally from the original reconstructed full-source percentile of 1,145.198. It prevents fitting the target cutoff to future values.
| Split | Years | Rows | High discharge | Other | Positive share |
|---|---|---:|---:|---:|---:|
| train | 1900–1999 | 37184 | 1859 | 35325 | 5.00% |
| validation | 2000–2005 | 3387 | 162 | 3225 | 4.78% |
| test | 2006–2015 | 5339 | 320 | 5019 | 5.99% |

Twenty-two records with unusable source years are excluded before fitting. Periods, labels and feature lists were frozen before this new model selection. No new final-test scores or labels were used for tuning.

## Training treatments

All models share the same 13 engineered inputs and three model configurations. Numeric medians and category encodings are fitted on training data only. Hyperparameters and decision thresholds are selected by validation class-1 F1, then AP and precision. Validation/test retain their original distributions.
- Untreated XGBoost: no class weights or resampling; its threshold is also validation-tuned for a fair comparator.
- Class weighting: sqrt(training imbalance ratio), half the ratio, or the full ratio.
- Random oversampling: repeat real minority training rows to minority/majority ratios 0.25, 0.5 or 1.
- Random undersampling: reduce majority training rows to the same ratios.
- Balanced oversampling plus weight 2: a combined-treatment control inspired by the original artifact; it is not an exact SMOTE reproduction.
No synthetic interpolation of encoded soil/climate categories is used. No resampling or class weighting is applied to validation or test data.

**Frozen validation winner: balanced_oversampling_weight2**, selected from 33 fitted candidates before final evaluation.

| Strategy | Validation F1 | Validation AP | Threshold |
|---|---:|---:|---:|
| untreated | 0.7333 | 0.7817 | 0.3400 |
| class_weight | 0.7304 | 0.7689 | 0.7329 |
| random_oversampling | 0.7168 | 0.7475 | 0.8490 |
| random_undersampling | 0.6875 | 0.6921 | 0.7623 |
| balanced_oversampling_weight2 | 0.7337 | 0.7414 | 0.9108 |

## Chronological test results

| Strategy | Precision | Recall | F1 | F2 | AP | ROC-AUC | Accuracy | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| untreated | 0.7647 | 0.6906 | 0.7258 | 0.7043 | 0.8017 | 0.9693 | 0.9687 | 68 | 99 |
| class_weight | 0.8596 | 0.6125 | 0.7153 | 0.6499 | 0.8154 | 0.9717 | 0.9708 | 32 | 124 |
| random_oversampling | 0.7278 | 0.7188 | 0.7233 | 0.7206 | 0.7942 | 0.9715 | 0.9670 | 86 | 90 |
| random_undersampling | 0.8125 | 0.5687 | 0.6691 | 0.6051 | 0.7442 | 0.9661 | 0.9663 | 42 | 138 |
| balanced_oversampling_weight2 | 0.7468 | 0.7188 | 0.7325 | 0.7242 | 0.7823 | 0.9721 | 0.9685 | 78 | 90 |
| training_prior | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0599 | 0.5000 | 0.9401 | 0 | 320 |

## Interpretation

At validation-selected thresholds, the selected model changes class-1 F1 from 0.7258 to 0.7325, precision from 0.7647 to 0.7468, and recall from 0.6906 to 0.7188. False positives change from 68 to 78; false negatives from 99 to 90.
Ranking by average precision decreases from 0.8017 to 0.7823. Brier score increases from 0.0250 to 0.0711, indicating worse probability calibration. Oversampling changes the score scale; these scores are not calibrated flood probabilities.
At the default 0.5 threshold, the combined treatment would produce 492 false positives, compared with 78 at its validation-selected threshold. Threshold selection is part of the intervention, separate from balancing; the matched untreated comparator also has a validation-tuned threshold.
The 95% station-cluster bootstrap interval for the paired F1 difference is [-0.027740661295774415, 0.0411781241051768]. An interval spanning zero does not establish reliable improvement. The winner is not changed based on final results.
Accuracy alone is misleading for a rare class. The table explicitly reports positive-class F1 and recall; it does not substitute the majority-class F1, which explained the old 0.984 headline.
Original reconstructed scores (96.89% accuracy, class-1 F1 0.717) use another split and full-source cutoff. They are not a valid before/after comparator for these chronological results. The untreated model above is the matched baseline.

## Limits and artifacts

GloFlo was previously inspected for source/model reconstruction. Test is held out for fitting/selection in this new protocol, not pristine external data.
Static modern catchment/climate proxies are not historically issued forecasts; station overlap is reported and unseen-station transfer is not assumed.
Annual peak-flow records and modern static attributes do not establish a causal weather forecast or an operational flood-warning model. New-station subgroup metrics are saved separately. Treating repeated records as independent locations would overstate generalisation.
All selected model pipelines, category vocabularies, imputers, UBJ models, feature importance, partition hashes, prediction records and parameters are preserved in this experiment. The old API/model are unchanged because their display semantics should not silently be changed to a high-discharge label.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_gloflo_imbalance.py prepare
.\.venv\Scripts\python.exe scripts/evaluate_gloflo_imbalance.py select
.\.venv\Scripts\python.exe scripts/evaluate_gloflo_imbalance.py evaluate
```
Frozen selection and final evaluation cannot be overwritten by these commands. Use a new prespecified experiment version for another study.

Source: GloFlo https://zenodo.org/records/7158028. Original model reconstruction: ../../evaluation/upstream_forensics/PROVENANCE_FINDINGS.md.
