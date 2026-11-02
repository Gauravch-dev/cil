# Presenter guide: Current electricity optimizer

This guide describes the locally verified application and the newly evaluated
LSTM + Random Forest pipeline. Use it with [the training report](HYBRID_TRAINING.md)
and [the demo guide](DEMO_GUIDE.md). No public deployment is required.

## Opening explanation

“Our system helps a household decide when to run flexible appliances while
protecting essential and comfort-related needs. It combines an electricity
forecast, structured appliance knowledge, explicit decision rules, a scheduling
search and explanations. Users receive a practical plan; evaluators can inspect
the evidence behind that plan.”

The everyday user follows My household: readings, preferences, plan. The Guided
demo reveals the implementation in eight stages. History restores prior outputs
and provides exports. Household drafts and demo preferences are separate.

## Prepare before the presentation

1. Start the local API using the README and open http://127.0.0.1:8000/.
2. Select Guided demo. Choose recorded history at **24 April 2010, 17:15**
   (the first recorded case). This is historical evidence, not today's meter data.
3. Select **Comfort first**. It resets household and appliance values to the
   defaults, then sets four occupants and 40°C. Check peak tariff 9, off-peak 6,
   peak hours 18–20, six-hour planning horizon and one parallel scheduled job.
4. Leave the default appliances: refrigerator, air conditioner, 500 W/60-minute
   washing job and 1,200 W/60-minute dishwasher job. Inspect deadlines if edited.
5. Select **Run demonstration** and wait for the result/model evidence to load.
   First inference and SHAP can take longer. The entire analysis executes once;
   stepping through the walkthrough inspects its outputs rather than retraining.
6. Confirm **LSTM + Random Forest executed** in Make a forecast → ML & Baselines.
   An old restored standalone result deliberately keeps its original model.

An alternative is **Use filled sample**: a marked synthetic eight-day CSV. It
keeps its January 2026 dates. Do not present it as actual current consumption.
Real current-use forecasts require a compatible recent meter export.

## Explain each stage in 6–8 minutes

| Stage | Show | What to say |
| --- | --- | --- |
| 1. Set the scene | Source, occupants, temperature, tariffs, appliances and constraints | “These are the household's needs and planning limits. Meter readings determine the demand estimate; household values determine decisions and schedules.” |
| 2. Prepare the data | History chart and preprocessing details | “Minute readings become completed 15-minute intervals. The model receives 96 intervals, or 24 hours, plus 17 context features containing recent, daily, weekly and calendar information. Input coverage needs eight days because the context includes previous-week information. Only information available at the forecast origin is used.” |
| 3. Make a forecast | Deep Learning tab, activation heatmap, then ML & Baselines | “The frozen LSTM learns time patterns. Its 16 summary values join 17 context values to form 33 inputs to Random Forest, which produces the final power estimate. Random Forest and Gradient Boosting were compared using validation error.” |
| 4. Understand the home | Select refrigerator, then washing machine | “The ontology gives appliances structured meaning: essential, high priority or flexible. Class inheritance and supported enabled SWRL rules produce candidate actions. Stored asserted examples are shown separately from candidates.” |
| 5. Apply the rules | Refrigerator and AC rule traces | “Policy resolves the final action. Essential refrigeration remains on. An occupied home at 40°C protects AC for comfort. Flexible washing can be delayed. Knowledge candidates do not automatically override operational policy.” |
| 6. Find a schedule | A* schedule, constraints, explored-state count and costs | “A* searches permitted job start times, respecting duration, deadlines and parallel-job capacity while minimizing estimated tariff cost. An optimal status applies to this declared scheduling problem. If search limits intervene or a deadline is impossible, we show that status.” |
| 7. Explain the forecast | SHAP bars, reconstruction and method details | “Orange contributions raise this estimate and blue contributions lower it relative to the model's reference output. TreeSHAP explains the actual Random Forest using its 33 inputs. Reference plus contributions must reconstruct the prediction.” |
| 8. See the outcome | Forecast, protected loads, timed plan and exports | “The user gets a readable recommendation and schedule with estimated job savings. They can inspect the evidence, restore the result later and export it.” |

The ontology adapter implements class inheritance and the class-to-decision SWRL
subset in the existing ontology. It is not a general OWL DL consistency reasoner.
The activation heatmap contains actual LSTM activations; its colors are not SHAP
importance values. SHAP's learned pattern numbers are not named appliances.

## Verified example to narrate

With the preparation settings above, the recorded analysis produced:

- **Forecast: 1.2196 kW**, average power for **24 April 2010, 18:00–18:15**,
  using information through **17:15**. The target ends one hour after origin.
- High-demand threshold: **2.5031 kW**. This particular estimate is below it.
- Refrigerator: **Keep ON**, essential protection (R1-essential).
- Air conditioner: **Keep ON**, occupied hot-home comfort protection (R2-comfort).
- Washing machine: **20:00–21:00**, 0.5 kWh, estimated job cost **3.00**.
- Dishwasher: **21:00–22:00**, 1.2 kWh, estimated job cost **7.20**.
- Plan: **optimal**, with **12 A* states explored** in this run.
- Optimized scheduled-job cost **10.20**, earliest feasible baseline **14.175**,
  estimated difference **3.975**, in the currency units implied by the entered tariffs.
- TreeSHAP: reference approximately **1.082 kW** plus net contributions
  approximately **0.138 kW** gives **1.220 kW**. Reconstruction residual was
  approximately **1.33 × 10^-15 kW**, a floating-point rounding difference.

These numbers belong to this exact historical input, artifact and scenario.
Changed tariffs/deadlines change costs or the schedule; changed readings change
the prediction. Job savings exclude continuously protected loads and do not
represent measured whole-house bill savings. The baseline is an earliest feasible
schedule under the same constraints, not an actual past utility bill.

## Two short interactive checks

**Empty home:** return to Set the scene, choose Empty home, run again, and inspect
the AC's rule trace. Occupancy-dependent AC becomes inactive (R5-empty-home), while
the refrigerator remains protected. With the same electricity source, the demand
estimate remains the same: occupancy is not an input to this aggregate predictor.

**Tight deadline:** choose Tight deadline and run again. The 60-minute washing
job has only a 30-minute deadline. Scheduling must report infeasibility rather
than invent a working plan or a saving. Finish by rerunning Comfort first if you
want the successful plan left on screen. Presets reset the other scenario values.

## Explain the model results honestly

The new classical heads used 20,000 examples from the original training period
and 5,000 from its validation period, in chronological partitions with boundary
gaps. The existing LSTM and scalers were reused without fitting again. Random
Forest's validation RMSE was **0.7233 kW**, versus Gradient Boosting's **0.7260 kW**,
so Random Forest was selected before test evaluation.

The full original test set contained **18,889 examples**, including **959** in
the high-demand segment:

| Method | Overall MAE (kW) | Overall RMSE (kW) | High-demand RMSE (kW) |
| --- | ---: | ---: | ---: |
| Selected LSTM + Random Forest | 0.4316 | 0.6301 | 1.6686 |
| LSTM + Gradient Boosting | 0.4307 | 0.6268 | 1.6972 |
| Standalone LSTM | 0.4269 | 0.6182 | 1.7145 |
| Persistence | 0.5143 | 0.8055 | 1.8836 |

MAE means average absolute prediction error. RMSE gives larger errors extra
weight. Both are in kW, and lower is better. R² is a comparison against a mean
predictor, not an accuracy percentage. Random Forest's overall R² was 0.3766;
its high-demand R² was strongly negative (-5.4053), so high-demand accuracy still
needs improvement despite lower peak RMSE than the compared models.

Say: “The hybrid improves over persistence and reduces high-demand error versus
the standalone LSTM, but the standalone LSTM remains slightly better overall.
We report that tradeoff. We did not change the winner after seeing the test set.”

The original test period had already been evaluated in earlier project work. It
is not fresh independent evidence, and these models were trained on one older
household dataset. Performance on another home or today's conditions is unproven.
Fifty-five integration checks and a successful production frontend build verify
software behavior, not 55 independent demonstrations of model accuracy.

## Likely evaluation questions

**Does it predict the entire next day?** No. It predicts average power for one
15-minute interval ending one hour ahead. A multi-hour appliance planning horizon
is a separate scheduling feature. Full next-day demand needs an explicitly
trained/evaluated multi-step forecast rather than repeating this value 96 times.

**Where do ML, DL and knowledge reasoning each contribute?** DL extracts temporal
patterns; ML turns the patterns/context into a forecast; ontology structures the
appliances and candidate actions; rules protect needs and resolve decisions;
A* chooses job times; SHAP explains the predictor. These stages have distinct jobs.

**Why does changing occupants not change predicted power?** Household preferences
are inputs to the planning/rule stage, not this model. Predicting demand from
occupancy changes would require a model trained on that information.

**Is 1.22 kW energy consumed?** It is average power. Across that 15-minute target
interval, 1.22 kW would correspond to about 0.305 kWh. It is not daily consumption.

**Does SHAP explain why AC should stay on?** No. The operational rule trace
explains that decision. TreeSHAP explains the forecast through learned/context
features; it does not prove appliance causation or assign confidence.

**Why Random Forest when Gradient Boosting has slightly lower test error?** The
winner was chosen on validation data. Switching because of the test result would
use the test set to tune the choice and weaken the evaluation.

**Was the entire network retrained?** No. Two new classical heads were trained
on frozen features from the existing older LSTM. This is a compatible new hybrid,
not reconstruction of the separate missing newer multivariate experiment.

**Does the forecast drive every scheduling slot?** No. Tariffs and job constraints
cover the planning horizon. Forecast-based avoidance covers only the single
known high-demand target interval. Parallel capacity counts scheduled jobs,
not all appliances or total electrical circuit load.

**Does it actually control appliances or guarantee savings?** No. It provides
recommendations and estimated job costs. There is no live meter integration or
physical switching. Actual bills, calibrated confidence and new-home accuracy
require further validation. No confidence percentage is invented.

## Closing statement

“We have connected forecasting, knowledge-based protection, constrained planning
and explanations into a usable local application. The demo exposes actual
intermediate evidence. Our next validation priorities are current household data,
stronger peak forecasting and measured benefits from following the recommended
plan.”
