# Task 7 — Smart Electricity Consumption Optimizer

Open **task7.ipynb** in Google Colab, choose a GPU runtime if available, and run all cells. The notebook is self-contained. The previous version is preserved as **task7_original.ipynb**.

The default `RUN_MODE = "full"` uses the complete original UCI dataset. It downloads a roughly 20 MB archive automatically, checks its row count, and records its checksum. It does not require the old Task 4 cleaned CSV. The original data source is credited in the notebook.

The notebook uses completed 15-minute intervals, past-only bounded filling, observed targets, a chronological train/validation/test split with a forecast-horizon gap, and training-only scaling. By default it trains a compact stacked LSTM and a basic neural network. Set `EXTENDED_SEARCH = True` to add a larger LSTM candidate; this takes considerably longer on a CPU. It evaluates the retained LSTM, the basic neural network, persistence, and daily/weekly seasonal forecasts on identical targets. It reports actual outcomes even if simpler models win.

Run outputs go into a new folder under `task7_outputs` and a matching ZIP. Download that ZIP before closing a Colab runtime. It includes models, scalers, configuration, predictions with forecast-origin and target timestamps, histories, figures, and prepared examples for Task 8. No saved metrics from the original notebook are presented as results of this revision.

`RUN_MODE = "smoke"` executes only two epochs per model on a small, explicitly labeled sample. It checks the pipeline and saved-model reload, but cannot establish final forecasting quality. Full training is required for submission results. CPU execution is supported but can be slow.

The editable source is `task7_pipeline.py`. To rebuild the notebook after editing that source, run `python tools/build_task7_notebook.py`. The generated notebook needs neither source file nor builder to run in Colab.

Tasks 1–6 are unchanged. Task 8 interpretation is a separate deliverable; this revision supplies the models and data it needs. The prediction is interval-average power in **kW**, not energy in kWh or a guaranteed reduction in cost.

The basic model and compact LSTM completed full-data training for 17 epochs each, restoring their best weights from epoch 10. The larger optional model's run was interrupted to respect the requested time and usage budget; it is excluded from results. The completed models are reused for final evaluation without retraining. Partial execution of the interrupted search is preserved in `task7_partial_run.ipynb`. The notebook's training cell can train the default models afresh in Colab; retained outputs explicitly identify the recovered completed training.
