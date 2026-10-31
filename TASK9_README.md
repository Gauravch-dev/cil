# Task 9 — interactive electricity optimizer

Open **task9_integrated_optimizer.ipynb** in Colab and run the cells. The final
section is an interactive dashboard inside the notebook, not a separate website.
Choose a recorded history case, edit household settings and click **Run optimizer**.

The single notebook embeds your actual Protégé OWL ontology, Task 7 source,
integration code and dashboard. No ZIP transfer is required. Your original
ontology and Task 1–8 notebooks are not edited.

## Model preparation

- Existing complete updated Task 7 folders in the runtime are reused. Automatic
  discovery chooses the latest complete folder by manifest modification time and
  prints the choice; an explicit `TASK7_FOLDER` overrides discovery.
- A fresh Colab runtime cannot access another notebook's temporary model files.
  By default, if no saved models exist, this notebook downloads the real UCI data
  and runs your unchanged Task 7 implementation once. GPU is helpful. Set
  `BUILD_TASK7_IF_MISSING=False` to require existing results instead.
- The Task 9 ML stage learns from frozen LSTM summaries plus context. Gradient
  Boosting and Random Forest are compared on validation data. Training uses at
  most 20,000 training examples by default; validation uses at most 5,000. The
  test comparison uses all saved test examples. Models are cached for reuse.
- Dashboard clicks only perform prediction, explanation and scheduling.

## Dashboard outputs

- Forecast average kW during the 15-minute interval ending one hour after origin.
- High/normal demand against Task 7's training-derived threshold.
- Appliance recommendations, reasons and actual imported SWRL rule traces.
- An A* schedule with runtime, deadline and job-concurrency constraints.
- Estimated schedulable-job cost/saving versus a feasible earliest-start schedule.
- SHAP explanation of the ML head's learned/context features, plus LSTM activations.
- Full-test comparison against Task 7 LSTM, basic neural network and persistence.
- JSON and CSV reports in `task9_outputs/scenarios`.

Tariffs, occupancy, temperature and appliance states/ratings are user-supplied
scenario settings. Defaults are labelled examples. Historical UCI electricity
records are not Indian live-meter readings; ₹ denotes the chosen scenario currency.
The system does not infer appliance identities or physical causal savings from
aggregate measurements. It does not switch hardware.

Only the one forecast target interval is avoided when demand is predicted high;
later job scheduling uses known scenario tariffs and constraints, not an invented
full-day demand forecast. The concurrency setting limits scheduled jobs, not the
whole house. Essential and comfort-protected appliances are excluded from the
shiftable-job savings estimate.

## Ontology scope

The notebook executes the supplied file's two enabled unary SWRL class-to-
`hasDecision` rules and named subclass closure. This is an explicit supported
subset, not a general OWL DL consistency reasoner. Unsupported rule forms fail
with an error. Imported candidate decisions and asserted example decisions are
reported separately. Ordered operational rules resolve conflicts and incorporate
forecast, comfort, occupancy, tariff and priority conditions.

## Files

- `task9_integrated_optimizer.ipynb`: standalone notebook to use.
- `task9_core.py`: ontology, inference, A* and forecasting integration.
- `task9_dashboard.py`: widget controls and visual result rendering.
- `task9_pipeline.py`: notebook workflow template.
- `tools/build_task9_notebook.py`: reproducible notebook builder.
- `tools/validate_task9.py`: technical validation without retraining full Task 7.

The notebook's project results are generated when it is run. Technical validation
uses explicitly labelled synthetic fixtures, separate from project metrics.
