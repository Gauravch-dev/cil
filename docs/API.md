# Local API reference

Base URL: `http://127.0.0.1:8000`. The FastAPI adapter serves the frontend after
its API routes. Interactive API docs are disabled. This is a loopback-only,
unauthenticated local service, not a public API deployment.

## Endpoints

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/api/health` | Readiness and recorded/connected-history mode; this does not alone establish upload inference readiness |
| GET | `/api/bootstrap` | Cases, defaults, ontology summary, capabilities and `readings.service_available` |
| GET | `/api/readings/template` | Header-only CSV attachment |
| GET | `/api/readings/samples/filled` | Marked synthetic eight-day CSV attachment |
| GET | `/api/readings/samples/guide` | Column/units TXT attachment |
| POST | `/api/readings?timezone=Asia%2FKolkata` | Validate/persist raw CSV bytes; return report, not a prediction |
| GET | `/api/readings/{id}` | Recheck a stored file against current clock and service readiness |
| POST | `/api/analyze` | Forecast/source lookup plus original rules/search; persist a successful result |
| GET | `/api/history` | Latest 100 saved analysis summaries, newest first |
| GET | `/api/history/{id}` | Full saved analysis |
| GET | `/api/history/{id}/export/json` | Full analysis JSON attachment |
| GET | `/api/history/{id}/export/schedule` | Schedule CSV attachment |
| GET | `/api/history/{id}/export/recommendations` | Recommendations and rule evidence CSV attachment |

## Upload contract

Send the raw CSV as `text/csv` or `application/octet-stream`. Set `X-File-Name`
to its URL-encoded CSV filename. The body is streamed with a 10 MB bound; the
filename is metadata and is not used as a filesystem destination. Required
columns and units are listed in the [user guide](USER_GUIDE.md).

The report includes `id`, filename, SHA-256, timezone, date range, row count,
coverage, `origin`, target interval, age/check timestamp, checks, warnings,
blockers, a five-row preview and recent history. Important flags:

| Flag | Meaning |
| --- | --- |
| `is_sample` | `Data_source=synthetic_sample` is present; this is not verified real meter data |
| `data_ready` | Enough valid fresh non-sample data is available |
| `forecaster_available` | A compatible forecasting service is connected |
| `can_analyze` | Current-data checks and forecast readiness both pass |
| `sample_ready` | A complete marked sample and forecasting connection permit separate demonstration inference |

Freshness and coverage are recomputed before analysis. An old report cannot
bypass current-data checks. Uploading alone never generates or saves a prediction.

## Analyze contract

Choose **exactly one** of `case_id` or `upload_id`. Copy valid `scenario` and
`appliances` objects from `/api/bootstrap`, then edit within the supported bounds.
Unknown request fields and both/neither source selections are rejected.

Runnable recorded-example request in PowerShell:

```powershell
$setup = Invoke-RestMethod 'http://127.0.0.1:8000/api/bootstrap'
$body = @{
    case_id = $setup.default_case
    scenario = $setup.scenario
    appliances = $setup.appliances
} | ConvertTo-Json -Depth 8
Invoke-RestMethod 'http://127.0.0.1:8000/api/analyze' -Method Post -ContentType 'application/json' -Body $body
```

For a checked household upload, omit `case_id` and send its `upload_id`.
For a marked synthetic demonstration also set `sample_forecast: true`; the
server requires `sample_ready`. Leaving that flag false keeps synthetic samples
blocked from the current household path.

Scenario fields: `occupancy`, `temperature_c`, `peak_start_hour`, `peak_end_hour`,
`peak_rate`, `offpeak_rate`, `horizon_hours`, `max_parallel_jobs`, `node_limit`.
Appliance fields: `name`, `power_w`, `duration_min`, `requested`, `on`, `essential`,
`flexible`, `priority`, `earliest_min`, `deadline_min`, `requires_occupancy`,
`ontology_id`. One to eight appliances are accepted. Boolean/integer types are
strict, and the original optimizer enforces timing constraints.

Success returns `id`, creation timestamp, source mode, forecast, scenario,
appliances, recommendations, rule/ontology evidence, plan, history, confidence
and available explanation. Uploaded runs also retain `upload_id`, the checked
report and provenance. Sample results set `forecast.is_sample=true`, keep their
original timestamps and identify the source as synthetic in provenance/exports.

## Errors and boundaries

| Status | Typical meaning |
| --- | --- |
| 403 | Rejected cross-origin write |
| 404 | Unknown upload/history/download/export |
| 409 | Stale/incomplete readings or invalid sample request |
| 413 | CSV body exceeds 10 MB |
| 415 | Unsupported upload content type |
| 422 | Invalid CSV, timezone, request fields or original scenario validation |
| 503 | Forecast/analysis service unavailable |

The UI accepts same-origin writes and the local Vite development origins.
There is no permissive CORS policy. SQLite operations are parameterized, user
filenames are sanitized to metadata and CSV exports neutralize formula prefixes.
A failed uploaded inference does not substitute a recorded forecast or create a
successful analysis. No route trains a model, controls appliances or connects a
utility account. Original full model evaluations remain offline.
