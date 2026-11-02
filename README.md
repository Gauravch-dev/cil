# Current — Smart Electricity Consumption Optimizer

A local electricity planning application with an interactive React interface,
real saved-model inference, household rules, and appliance scheduling. The UI
uses white surfaces, rich blue gradients, and orange highlights, with responsive
layouts, keyboard navigation, and clear explanations.

**The application is not deployed.** Run it on your own computer at
<http://127.0.0.1:8000/>.

## What you can do

- Explore 67 authentic recorded electricity forecasts and their usage histories.
- Upload recent meter readings for a new forecast with the bundled trained model.
- Load the eight-day synthetic sample and generate a forecast at its original dates.
- Set household occupancy, temperature, tariffs, appliance priorities and deadlines.
- Review protected appliances, rule evidence, schedules and estimated job costs.
- Restore saved analyses and download JSON, schedules and recommendations.

For an evaluation presentation, follow the [presenter walkthrough and questions](docs/PRESENTER_GUIDE.md).

A forecast is **average power in kW for one 15-minute interval ending one hour
after the latest complete reading**. It is not a full-day consumption or bill
estimate. Household settings guide rules and schedules; they do not change the
meter-based demand prediction.

## Quick start on Windows

Requirements: Python 3.12, Node.js 20.19+ or 22.12+, npm, and Git. Because this
repository also contains an old tracked Python environment, clone the project
without checking that environment out:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone --no-checkout https://github.com/Gauravch-dev/cil.git
Set-Location cil
git sparse-checkout set --no-cone '/*' '!/.venv-task7/' '!/__pycache__/'
git checkout main
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
py -3.12 -m venv .venv-web
.\.venv-web\Scripts\python.exe -m pip install -r requirements-forecast.txt
.\.venv-web\Scripts\python.exe -B tools\train_local_hybrid.py
Set-Location frontend
npm ci
npm run build
Set-Location ..
.\.venv-web\Scripts\python.exe -B -m uvicorn web_api:app --host 127.0.0.1 --port 8000
```

For an existing checkout, start with creating the environment and installing
dependencies. Open <http://127.0.0.1:8000/>; keep the terminal running. Stop the
server with Ctrl+C. `start-local.ps1` is an alternative launcher if local script
execution is allowed on your computer.

`requirements-forecast.txt` includes the web dependencies and TensorFlow runtime.
For recorded examples and CSV checks only, use the smaller `requirements-web.txt`.
You must still build the frontend. Do not reuse the repository's old environment.

## Household workspace and guided demo

**My household** follows three steps: add real meter readings, set preferences,
then review the appliance plan. Data checks and detailed evidence expand on demand.

**Guided demo** exposes eight connected stages: scenario, data preparation, DL/ML
roles, ontology, rules, A* scheduling, SHAP and final outcome. Enter values or try
comfort, empty-home, tariff-planning and tight-deadline scenarios. Actual model
outputs and rule/search evidence drive the views. Demo settings stay separate
from household preference drafts.

To try the built-in sample, open Guided demo, choose **Filled eight-day sample**,
select **Load sample**, then **Run demonstration**. It retains its original January
2026 dates. SHAP is computed for the exact bundled LSTM using 23 input groups,
eight sampled permutations and one training reference. A reconstruction check,
real activation heatmap and downloadable evidence are included. The active classical
head and its real validation/test results are shown explicitly.

## Documentation

| Guide | Contents |
| --- | --- |
| [Hybrid training report](docs/HYBRID_TRAINING.md) | New fitted classical stage, selection, actual results and reproduction |
| [Evaluator demo guide](docs/DEMO_GUIDE.md) | Eight stages, scenario presets, evidence and SHAP interpretation |
| [User guide](docs/USER_GUIDE.md) | Household settings, sample forecasts, real readings, results and history |
| [Setup and maintenance](README_WEB.md) | Installation, local development, dependencies, model selection and troubleshooting |
| [API reference](docs/API.md) | Endpoints, requests, responses and errors |
| [Integration and architecture audit](WEB_INTEGRATION_AUDIT.md) | Original task coverage, data flow, artifacts and remaining gaps |
| [Verification](docs/VERIFICATION.md) | Checks performed, reproduction commands and limits of the evidence |
| [Changelog](CHANGELOG.md) | Web application changes |

Original offline work remains documented in [Task 7](TASK7_README.md),
[Task 8](TASK8_SHAP_README.md), [Task 9](TASK9_README.md), and
[Task 10](TASK10_README.md).

## Data and model boundaries

- New inference uses the completed bundled **older power-history LSTM**, its
  original preprocessing and frozen scalers. Startup checks five saved forecasts
  before declaring it ready. No training occurs during analysis.
- A compatible LSTM + Random Forest head is now trained and connected. It uses the
  completed older LSTM. The separate newer multivariate bundle remains absent.
- Confidence percentages and prediction intervals are not provided. Exact saved
  explanations exist for three recorded histories; uploaded forecasts do not
  invent contributions. Household rule evidence is available independently.
- Current household forecasts require actual recent meter readings. Timestamps
  and successful validation do not prove data authenticity or household accuracy.
- There is no live utility connection, hardware switching, authentication or public
  hosting. The server binds to loopback. Job savings are scenario estimates.

Uploaded files and history are stored in `web_data/history.sqlite3`, which is
ignored by Git. Fresh environments, dependency directories, caches and generated
frontend builds are also ignored. The checked-in sample is explicitly synthetic.
Third-party UI/font notices are in [frontend/public/licenses](frontend/public/licenses).

## Trained classical stage — 3 November 2026

The compatible older-LSTM hybrid is now trained and connected for new analyses.
It uses 16 frozen LSTM patterns and 17 context features; validation selected Random
Forest over Gradient Boosting. TreeSHAP explains the executed 33-feature regressor.
Old saved standalone forecasts remain unchanged. The newer multivariate experiment
remains distinct and absent. Earlier statements that the classical stage was
unavailable describe the pre-training state. See [the training report](docs/HYBRID_TRAINING.md) for actual
metrics, limits, artifact restoration and the explicit offline preparation command.
