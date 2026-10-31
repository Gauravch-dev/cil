# Task 8: SHAP and activation visualizations

For the simplest Colab workflow, open **task7_and_task8_shap.ipynb**.
The standalone Task 8 notebook is **task8_shap.ipynb**, with editable source **task8_shap.py**.
The older `task8.ipynb` uses a different explanation method and is preserved separately.

## Run in Colab

1. Open `task7_and_task8_shap.ipynb` in Colab.
2. Run Task 7, then continue to Task 8 in the same notebook and runtime.
3. Task 8 automatically reads Task 7's saved files. No ZIP transfer is needed.
4. Save the executed notebook and any Task 8 charts you want to submit.

Task 7's executable code is unchanged in the combined notebook. Task 8 does not
retrain its models. If Task 7 already completed in your existing active Colab
notebook, append the standalone Task 8 cells there; they detect the saved run folder.
Opening another Colab notebook normally creates a separate runtime, so it cannot
automatically access the first notebook's temporary files. If a runtime is reset
and temporary files are lost, saved results must be restored or training rerun.

Alternatively, set `TASK7_SOURCE` in the settings cell to a ZIP path or an extracted
Task 7 run folder. Paths in Colab start with `/content/`; your Windows desktop path
does not exist inside Colab. For separate-runtime workflows, an optional ZIP includes
`prepared_arrays.npz`, the models, scaler, manifest, predictions and explanation examples.

The notebook requires the **updated** Task 7 model filename `task7_FINAL_lstm.keras`.
It uses the selected model recorded in the manifest, even if selection chose the
compact LSTM instead of the peak-gated candidate. It does not substitute old results.

## Included analysis

- Full-test and high-demand MAE, RMSE and R² for LSTM, basic NN, and available baselines.
- Actual versus predicted power.
- SHAP GradientExplainer on the final power forecast, using training references.
- Shared feature contribution summaries and time-step heatmaps for both neural models.
- Individual SHAP waterfall explanations and signed time-feature contributions.
- LSTM temporal activation heatmaps and basic NN dense activation plots.
- Approximate-additivity checks with retries and explicit residual flags.
- Performance and interpretability comparison, limitations, source hashes and output ZIP.

The default explanations use 24 evenly spaced test-pool examples, up to three
additional diagnostic cases, and 32 training references. Full-test metrics still use
every saved test prediction. Change these settings to adjust explanation cost.
There is no retraining. SHAP expected gradients estimates contributions; it does not
prove physical causation and is sensitive to reference data and correlated inputs.

## Execution status

The main notebook is intentionally unexecuted until your updated Task 7 artifacts
are provided. Technical validation is recorded separately in
`task8_shap_validation.json`; synthetic fixture results are never embedded as project
results. Existing Task 7 and Task 8 files are preserved.

## Rebuild or validate locally

```powershell
.\.venv-task7\Scripts\python.exe tools\build_task8_shap_notebook.py
.\.venv-task7\Scripts\python.exe tools\validate_task8_shap.py
```

References: [SHAP GradientExplainer](https://shap.readthedocs.io/en/stable/generated/shap.GradientExplainer.html)
and [SHAP waterfall plots](https://shap.readthedocs.io/en/stable/generated/shap.plots.waterfall.html).
