# Guided demo for evaluation

Open the local website and select **Guided demo**. The ordinary **My household**
workspace remains a separate three-step flow for actual readings and practical
appliance recommendations. Nothing is deployed.

## Start a demonstration

1. Choose a recorded case, load the filled eight-day sample, or upload a compatible CSV.
2. Edit Household, Tariffs and Appliances preferences. Demo edits are separate from household drafts.
3. Select Run demonstration. The original forecast/rule/search flow executes and saves a real analysis.
4. Use Next stage, Previous stage or the stage rail to inspect the outputs.

Preset expectations under the original policy:

| Preset | What to inspect |
| --- | --- |
| Tariff planning | Flexible jobs search cheaper permitted starts; protection and deadlines remain active. Savings can be zero. |
| Comfort first | Occupied 40°C scenario protects AC using R2-comfort. |
| Empty home | Occupancy-dependent AC is inactive using R5-empty-home. Refrigerator remains essential. |
| Tight deadline | A 60-minute washing job with a 30-minute deadline makes the plan infeasible. No successful cost/savings result is invented. |

Presets change scenario assumptions, not meter readings. Source dates and model
predictions are never fabricated to produce a preferred demonstration outcome.

## Inspect the eight stages

1. **Set the scene:** chosen source and editable household/appliance constraints.
2. **Prepare the data:** 96 history intervals, six channels, 17 context features,
   original causal preprocessing, frozen scaling and exact history chart.
3. **Make a forecast:** model re-execution must reproduce the displayed estimate.
   Inspect the architecture and real 32-unit × 96-step activation heatmap.
   The ML/baseline tab shows the active LSTM + Random Forest pipeline, its real
   33-feature handoff, validation selection and test comparison. Old saved LSTM
   analyses retain their original model. The separate newer experiment is absent.
4. **Understand the home:** select an appliance; inspect inferred ontology types,
   actual enabled SWRL candidates and separately asserted examples.
5. **Apply the rules:** inspect ordered operational conditions and the fired
   rule trace. Ontology candidates and final resolved recommendations differ.
6. **Find a schedule:** actual A* state count, status, timelines/table, constraints
   and cost comparison. Capacity applies to scheduled jobs, not whole-house load.
7. **Explain the forecast:** actual TreeSHAP for new hybrid analyses,
   signed contributions, grouped remainder and reconstruction residual. Expand
   groups or show all; download model/SHAP evidence separately from analysis JSON.
8. **See the outcome:** the practical forecast, protected loads, job schedule,
   estimated savings and full saved-analysis exports.

The interface is a walkthrough of one executed analysis, not an animation that
pretends to execute each stage later. No training occurs on a demo click.

## SHAP interpretation

New hybrid analyses use TreeSHAP on 16 learned LSTM patterns and 17 context
features. The reference output plus all 33 contributions must reconstruct the
raw prediction, before any non-negative clipping. Learned patterns are not
individual appliances or direct causes.

Older frozen standalone LSTM analyses are explained using six entire-history-channel groups
and 17 context groups. Eight sampled permutations use one actual median training
reference. These approximate group Shapley contributions. Base plus contributions
are checked against the forecast; residual and pass/fail remain visible. Group
perturbations can create unrealistic feature combinations. They explain model
associations, not appliance causes, calibrated uncertainty or household accuracy.
Existing saved Integrated Gradients are never renamed SHAP. If SHAP fails, the
interface reports unavailability rather than substituting contributions.

## Demonstration boundaries

A marked sample retains 1–8 January 2026 UTC readings and a 9 January target.
Actual current estimates require fresh real meter exports. A scenario result is
not evidence of real-world bill savings or prediction accuracy. Model limitations,
missing newer artifacts, unquantified confidence and absent hardware control
remain explicit. Original Task 10 evaluation is separate from this walkthrough.

## Trained classical stage — 3 November 2026

The compatible older-LSTM hybrid is now trained and connected for new analyses.
It uses 16 frozen LSTM patterns and 17 context features; validation selected Random
Forest over Gradient Boosting. TreeSHAP explains the executed 33-feature regressor.
Old saved standalone forecasts remain unchanged. The newer multivariate experiment
remains distinct and absent. Earlier statements that the classical stage was
unavailable describe the pre-training state. See [the training report](HYBRID_TRAINING.md) for actual
metrics, limits, artifact restoration and the explicit offline preparation command.
