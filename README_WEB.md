# Local setup and maintenance

See the [project README](README.md) for the overview and
[user guide](docs/USER_GUIDE.md) for the application workflow.
Nothing in these instructions deploys the application.

## Requirements and installation

Use Python 3.12 and Node.js 20.19+ or 22.12+. The verified local environment uses
Python 3.12.10 and Node.js 24.21.0. Install a fresh environment rather than using
`.venv-task7`, an old environment already tracked by the original repository.
The README's sparse-clone instructions avoid downloading that environment.

From the repository root:

```powershell
py -3.12 -m venv .venv-web
.\.venv-web\Scripts\python.exe -m pip install -r requirements-forecast.txt
Set-Location frontend
npm ci
npm run build
Set-Location ..
.\.venv-web\Scripts\python.exe -B -m uvicorn web_api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. The first model load can take time. Wait for the
server's startup-complete message before opening the page. Stop with Ctrl+C.
Alternatively, run `start-local.ps1`; use `-Port 8001` to choose another port.
The launcher expects both the environment and frontend build to exist.

If `py` opens the Windows Store, use the absolute path to an installed Python 3.12,
or install with an available `uv`:

```powershell
uv venv --python 3.12 .venv-web
uv pip install --python .venv-web/Scripts/python.exe -r requirements-forecast.txt
```

To run only recorded forecasts and CSV checks, install `requirements-web.txt`
instead. New uploaded/sample inference needs `requirements-forecast.txt`.
Fonts and icons are bundled in the frontend; the built page does not request
third-party fonts. Build output is generated locally and is not committed.

## Local development

Keep the API running on port 8000. In a second terminal:

```powershell
Set-Location frontend
npm run dev
```

Visit <http://127.0.0.1:5173/>. Vite proxies `/api` requests to port 8000.
Run `npm run build` after editing to update the single-server page on port 8000.
No cloud hosting configuration or automatic deployment is added.

## Forecast sources

| Mode | Behavior |
| --- | --- |
| Recorded examples | Reuses authentic saved forecasts, their original dates and matching history. Rules/search run on the submitted household settings. |
| Uploaded household readings | Executes the bundled frozen older LSTM on validated fresh readings. Requires the forecasting runtime. |
| Synthetic sample | Explicit sample inference at the file's original dates; labelled in the result, chart, history and JSON. |
| Optional newer bridge | Uses a matching externally prepared newer model and hybrid cache, when supplied. Not verified here with real matching artifacts. |

The default recorded run is `task7_outputs/full_20261004T223051_253137Z`.
`local_forecast.py` loads that run's `task7_lstm.keras`, `power_scaler.joblib`,
`context_scaler.joblib`, manifest and saved arrays. It applies the original
six history channels and 17 context features extracted from trusted local
`task7_pipeline.py` function definitions. It does not execute the training script.

Five saved forecasts must reproduce within `rtol=1e-4, atol=1e-4` before uploaded
forecasting becomes ready. Inference uses frozen weights/scaler transforms only.
The older model uses active power, observation flags and calendar/daily/weekly
context. Other electrical columns are checked but are not inputs to this model.
No matching contribution breakdown is generated for arbitrary uploads. Confidence
is explicitly unquantified and accuracy on another household is unestablished.

## Optional newer forecasting bridge

Only load trusted artifacts. An operator can supply a completed matching newer
Task 7 bundle and exact Task 9 hybrid cache:

```powershell
$env:CIL_TASK7_FOLDER = 'D:\models\task7_FINAL_outputs\full_run'
$env:CIL_TASK9_OUTPUT = 'D:\models\task9_outputs'
```

The bridge requires its matching model, manifest, prepared arrays, signatures and
cache. Uploaded inference additionally requires `electrical_scalers.joblib`,
`context_scaler.joblib`, 96 history steps, four horizon steps and the exact seven
measurement schema. A renamed older model cannot satisfy the newer contract.

Startup blocks fitting the candidate regressors on cache misses. If the newer
bridge cannot load, recorded examples remain available and the bundled older
upload adapter is attempted. The newer bridge was not verified end-to-end with
real matching artifacts because those are absent from this checkout. Do not
interpret synthetic Task 10 validation fixtures as evidence otherwise.

## Checks

```powershell
.\.venv-web\Scripts\python.exe -m pip install -r requirements-web-dev.txt
.\.venv-web\Scripts\python.exe -B -m pytest tests/test_web_api.py tests/test_readings.py tests/test_local_forecast.py -q --basetemp ../work/pytest-local
Set-Location frontend
npm run build
```

Install forecasting dependencies to run all real-model checks. Without TensorFlow,
`test_local_forecast.py` is skipped; that is not full inference verification.
Tests use temporary databases rather than the user's saved history.
See [verification details](docs/VERIFICATION.md).

## Local storage and source control

`web_data/history.sqlite3` contains uploaded CSV bytes, report metadata and full
saved analyses. It is local and ignored by Git. **Clear selection** only clears the
form; it does not delete stored uploads. Restoring a current-reading analysis
rechecks freshness before another inference. No cleanup or database reset is
performed by the test suite.

Do not commit environments, `node_modules`, generated builds, household exports,
credentials or history. The repository's pre-existing tracked environment and
artifacts were not removed or rewritten by this work. Backend task sources,
notebooks, ontology and original trained artifacts remain unchanged.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Site cannot be reached | Start the API and wait for startup completion; check the port and keep its terminal open. |
| Frontend missing or changes absent | Run `npm ci` once and `npm run build` in `frontend`, then refresh the browser. |
| Port already in use | Use the running site, or start with another port; do not launch multiple servers on the same port. |
| Forecasting unavailable | Install `requirements-forecast.txt`, restart the API and inspect its model-load error. Do not rename model files or bypass fidelity checks. |
| CSV has missing columns | Use the actual CSV template or **Use filled sample**; the TXT guide and saved HTML page are not meter files. |
| File too old for a current estimate | Export real recent readings. The marked sample has a separate **Generate sample forecast** action at its original dates. |
| Sample forecast shows January dates | Expected: its readings are synthetic and dated 1–8 January 2026; dates are never changed to look current. |
| Forecast does not change with occupancy/tariff | Expected: those settings guide rules and scheduling, while the forecast follows meter history. |
| Savings are zero | The search found no lower-cost feasible schedule under the supplied scenario; savings are not invented. |
