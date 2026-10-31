# Original model dataset and target reconstruction

## Finding

The original data source is strongly corroborated as GloFlo, `kuntla_flooddatabase.csv`, the same 45,932-row, 93-column CSV already supplied by the user. The original README explicitly names it. More importantly, all four saved categorical encoder vocabularies match exactly. Replaying its 13-feature inference pipeline with median imputation recreates all 2,538 original high-risk map rows in their original order. Maximum score error is 2.49800873e-07, consistent with floating-point/export rounding.

## Missing target step reconstructed

The strongest supported hypothesis is:

```python
cutoff = df['Annual peak discharge'].quantile(0.95)  # 1145.198 in source units
y = (df['Annual peak discharge'] > cutoff).astype(int)
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

This yields 2,297 high-discharge and 43,635 other peak-flow records. The inferred split has 36,745 training rows and 9,187 held-out rows. Replaying the existing model, without training, reproduces every advertised headline value after rounding:

| Quantity | Reconstructed | README claim |
|---|---:|---:|
| Overall accuracy | 96.88691% | 96.9% |
| Class 1 recall | 0.790849673 | 0.791 |
| Class 0 F1 | 0.983527243 | 0.984 |
| Class 1 precision | 0.656419530 | Not reported |
| Class 1 F1 | 0.717391304 | Not reported |

The advertised F1 therefore matches class 0, whereas the advertised recall matches class 1. This strongly suggests metrics were taken from different rows of a classification report. It does not establish misconduct or intent.

Reconstructed log-loss is 0.094019945215; the model stores best validation log-loss 0.094020101089, a difference of 1.56e-07. The saved first-tree Hessian cover is also compatible with balancing the inferred 34,907 training negatives and 1,838 positives to 34,907 per class with scale_pos_weight=2. This is additional corroboration, not an exact reconstruction of the synthetic SMOTE rows.

## Public source history

The public GitHub history has six commits, one main branch and no tags. The root commit already contains the model, encoders and map outputs. Its initial file tree contains no notebook, raw training CSV or target-generation code. Later commits add documentation/development configuration. The public Hugging Face Space also contains deployment files and artifacts, without the original notebook. API snapshots are retained alongside this report.

The visible `flood_fast.py` is a demo helper. It can generate random examples, but it does not explain the saved artifact: it exports empty encoders and uses 100 estimators with different parameters. The saved artifact requests up to 300 estimators, uses early stopping, retains 214 trees with best_iteration=193, and has real categorical encoders. The deployed artifacts should not be called synthetic solely because this separate helper has a synthetic fallback.

## What this establishes and what it does not

The raw dataset connection is exceptionally strong because its saved map is reproduced. The threshold and split are a highly corroborated reconstruction, not source code recovered from the author’s private Colab. The fitted imputation values and exact SMOTE interpolation rows are not independently recovered.

The target is **high annual peak discharge relative to a global percentile**, not an observed operational flood/no-flood label. The full-source percentile uses the entire data distribution. The inferred split is random by record; 9,029 of 9,187 held-out rows are at stations also present in training. Repeated static catchment features therefore span both partitions. These reconstructed scores are not a new chronological or gauge-independent evaluation and cannot be compared directly with the INDOFLOODS Flood/Severe Flood task.

The reconstruction was selected by forensic agreement with model artifacts and published numbers. It is not a blind confirmation experiment and must not be promoted as newly verified real-world forecasting accuracy. No training or application model replacement was performed.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts/reconstruct_original_dataset.py
```

`reconstructed_original_dataset.csv` retains source identifiers, measured peak discharge, all 13 engineered inputs, an explicitly named derived high-discharge class and the inferred partition. The raw source and original model remain unchanged.

## Sources

- Original repository and README: https://github.com/nitinc264/Flood-prediction-Model/tree/2d823f3692e891ac92de4435bbe2c006131c30b3
- GloFlo source: https://zenodo.org/records/7158028
- Public deployed files: https://huggingface.co/spaces/nitin454545/riskradar/tree/main
- Local raw source SHA256: 20a844f84f78729cf4361f3f13daa3a79626f67c2e96aeb0f1a3b872bfd1a45a
- Original model SHA256: aad88a12f4645f3703c996641a15ce40603f6d5a608918fad4f37496d2f89395
