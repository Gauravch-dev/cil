# Using Current

## Set your household values

Open **Household settings**. Set the number of people at home, temperature,
planning horizon, parallel-job limit and peak/off-peak tariffs. Review each
appliance's power, duration, priority, requested/running state, flexibility,
earliest start and deadline. Start/finish limits use 15-minute increments and
are relative to the selected readings' forecast origin.

Your settings guide protection rules, recommendations, scheduling and estimated
job costs. The demand forecast comes from electricity readings, so changing a
tariff or occupancy value alone does not rewrite that forecast. No appliance is
actually switched by the application.

## Generate a sample-date forecast

1. Select **Sample templates → Use filled sample**.
2. Wait for **Ready for a sample forecast**.
3. Select **Generate sample forecast** below the checks.
4. Read the forecast, chart, appliance plan and rule explanations on the home page.

The file contains eight days of explicitly synthetic minute readings in UTC.
It keeps its original January 2026 dates and uses the household settings already
entered in the app. It is not your actual current household use. The last complete
interval ends at 00:00 on 9 January, so the forecast is for 00:45–01:00 UTC on
9 January. The current-reading freshness check does not apply to this separate,
explicitly requested sample demonstration. Confidence remains unquantified.

## Forecast from actual recent readings

Select **Use my electricity readings**. Download the **Blank CSV template** and
fill it with an actual meter export: one row per minute, at least eight days of
history, ending within the last 30 minutes. Choose the timezone used by the file,
select the CSV and click **Check my readings**. When checks pass and forecasting
is ready, select **Analyze my energy** (or **Update analysis** for an existing run).

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
