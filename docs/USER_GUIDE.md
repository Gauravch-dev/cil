# Using Current

## Two clear experiences

**My household** is the everyday workspace. It shows one step at a time:
readings, preferences, then the plan. Technical model and ontology details are
kept in Guided demo. Household preference drafts are remembered in this browser;
demo edits do not overwrite them.

**Guided demo** is for demonstrations and evaluation. Choose recorded history,
the marked eight-day sample, or a CSV. Enter household values or select a scenario
preset, then select **Run demonstration**. Inspect these stages with Next/Previous
or the stage navigation:

1. Set the scene: inputs and editable scenario.
2. Prepare the data: authentic history, causal preparation and model features.
3. Make a forecast: active LSTM + Random Forest, architecture, real activation
   heatmap, validation selection and test comparison. Old analyses keep their original model.
4. Understand the home: selected appliance, inferred ontology classes and SWRL candidates.
5. Apply the rules: ordered operational policy and actual fired trace.
6. Find a schedule: original A* output, explored-state count, constraints and costs.
7. Explain the forecast: TreeSHAP for the active hybrid (grouped permutation SHAP for old LSTM analyses),
   signed contributions, reconstruction residual and downloadable model/SHAP evidence.
8. See the outcome: the forecast, appliance plan, savings and exports together.

The analysis executes once, then the walkthrough inspects its real outputs; it
is not an animation pretending each stage ran later. Model evidence re-executes
the frozen model and must match the selected forecast before SHAP is attached.
SHAP is not a confidence score or proof of causality. Expensive computations may
take time on the first request. No training occurs during website requests.

## Set your household values

In My household, choose **Set household preferences**. Household, Tariffs and
Appliances tabs organize settings. Enter occupancy, temperature, planning horizon,
parallel-job limit, rates and appliance power/runtime/priority/timing constraints.
Start/finish limits use 15-minute increments relative to the readings' origin.
Preferences guide rules and scheduling; the demand estimate follows meter history.

## Generate a sample-date forecast

1. Open **Guided demo**.
2. Choose **Filled eight-day sample** as Electricity source, then **Load sample**.
3. Review or edit Household, Tariffs and Appliances preferences.
4. Select **Run demonstration**, then inspect each stage.

The file contains 11,520 explicitly synthetic minute readings in UTC, dated
1–8 January 2026. Its target is 00:45–01:00 on 9 January. It keeps its original
dates and is not actual current household use. The separate sample request does
not bypass freshness requirements for actual readings.

## Forecast from actual recent readings

Open **My household → Add electricity readings**. Download the **Blank CSV template** and
fill it with an actual meter export: one row per minute, at least eight days of
history, ending within the last 30 minutes. Choose the timezone used by the file,
select the CSV and click **Check my readings**. When checks pass and forecasting
is ready, continue to preferences and select **Generate my plan**.

| Required column | Unit / format |
| --- | --- |
| `Datetime` | Local `YYYY-MM-DD HH:MM:00`, no timezone suffix; choose the timezone on the form |
| `Global_active_power` | kW |
| `Global_reactive_power` | kVAR |
| `Voltage` | V |
| `Global_intensity` | A |
| `Sub_metering_1` | Wh per minute |
| `Sub_metering_2` | Wh per minute |
| `Sub_metering_3` | Wh per minute |

Monthly bills, cumulative kWh and invented readings do not match this format.
Leave missing values blank. The limit is 10 MB, 100,000 rows and a maximum span
of 60 days. Comma/semicolon/tab delimiters, UTF-8/UTF-16 BOM files and common Excel
separator declarations are supported; units are never guessed. Duplicate,
malformed, future or ambiguous local timestamps are rejected.

The latest usable interval must contain all seven measurements. Incomplete tails
are excluded. Gaps use only limited past-value filling; longer missing input or
context periods prevent inference. Freshness is rechecked every time you analyze.
File validation cannot establish that the readings are authentic or that the
trained model will be accurate for your household.

## Understand the result

- **Demand estimate:** average power for one 15-minute target interval, one hour
  after the latest complete reading. Read the date and timezone with the value.
- **Demand level:** compares the forecast with the saved training-derived threshold.
  It is not a confidence percentage.
- **Job savings:** compares the recommended schedule with a feasible earliest-start
  plan under the same constraints. It covers schedulable jobs, not the whole bill.
- **Protected appliances:** essentials remain protected. Comfort and occupancy
  rules are applied using the original backend rules and their displayed traces.
- **Appliance plan:** proposed times only. An infeasible or search-limited result
  keeps its actual status; the app does not claim a successful optimum or savings.
- **Why this works:** shows rule checks and available evidence. Three exact recorded
  histories have saved prediction contributions. Arbitrary uploads do not receive
  invented feature attributions.

## Recorded examples, history and exports

**Recorded examples** uses genuine saved older forecasts at their original dates.
Choose a history and analyze to apply your household scenario to its rules/plan.
This is different from executing the model on a new uploaded CSV.

Open **History** to search/filter and restore saved analyses. Restoring a result
keeps its original inputs, source and dates. Changing settings marks old results
as stale until you explicitly update them. Download analysis JSON, schedule CSV
or recommendations from the corresponding analysis. Files and history stay in the
local workspace; clearing a form selection does not delete stored data.

## Trained classical stage — 3 November 2026

The compatible older-LSTM hybrid is now trained and connected for new analyses.
It uses 16 frozen LSTM patterns and 17 context features; validation selected Random
Forest over Gradient Boosting. TreeSHAP explains the executed 33-feature regressor.
Old saved standalone forecasts remain unchanged. The newer multivariate experiment
remains distinct and absent. Earlier statements that the classical stage was
unavailable describe the pre-training state. See [the training report](HYBRID_TRAINING.md) for actual
metrics, limits, artifact restoration and the explicit offline preparation command.
