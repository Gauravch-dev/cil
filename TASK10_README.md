# Task 10 — final evaluation

Open **task10_final_evaluation.ipynb** in Colab and run its cells. It is a separate,
self-contained notebook with the ontology, Task 7/9 system code and evaluation
helpers embedded. No ontology or output ZIP transfer is required.

## Existing models versus a fresh Colab runtime

Existing live Task 9 models or matching model/cache folders are reused. Set
`TASK7_FOLDER` and `TASK9_OUTPUT_DIR` to saved folders when evaluating the exact
previous run. Opening another Colab notebook normally creates a separate runtime;
visible outputs in a downloaded notebook do not include its trained weights.

By default, missing models are prepared once before evaluation using the unchanged
Task 9 training procedure. In a fresh session this can take several minutes; a GPU
helps. Set `PREPARE_MODELS_IF_MISSING=False` to require existing frozen models.
Reports label whether the system was reused or rebuilt. Model fingerprints are
checked before/after evaluation. No tuning is performed on evaluation results.

## What is evaluated

- Complete saved test period: MAE, RMSE, R², high-demand errors, peak-detection
  confusion counts and temporal/demand error profiles.
- Exploratory day-block intervals for aggregate error statistics, not prediction
  intervals or population guarantees.
- Additional designed household scenarios, scheduling constraints, independent
  cost checks, invalid-input rejection and inference/rule/scheduling latency.
- ML-stage TreeSHAP, reconstruction checks, selected successful/error examples,
  LSTM activations and decision-rule traces.
- Controlled policy-invariance checks and an explicit privacy, fairness, safety,
  representativeness, user-control and data-governance discussion.
- An optional Task 9 dashboard demonstration using the frozen model.

The original test observations are excluded from fitting/model selection but were
already evaluated in Task 9. The notebook labels them as reused; it does not claim
they are a new independent test. Scenario combinations are designed inputs, not
measured new households. Demographic labels are absent, so no demographic fairness
score or certification is invented.

## Optional genuinely new measured data

Set `UNSEEN_MINUTE_CSV` to a CSV containing `Datetime` plus all seven original
electrical measurements. Use ISO local minute timestamps without timezone suffixes.
At least eight days including warm-up are recommended. Forecast origins must be
later than the original dataset end. Scalers are reused without fitting; future
values are not used to fill historical inputs or missing targets. Complete actual
targets are required for scoring. The code checks schema/time separation but
cannot verify authenticity or exclude copied/relabelled source data.

No external CSV is required to run the notebook. If absent, the report explicitly
states that no independent new measured dataset was supplied.

## Outputs

Each run creates `task10_outputs/evaluation_...` with:

- Forecast/error/scenario/fairness/latency tables and full prediction exports.
- SHAP, LSTM activation, error-profile and comparison PNGs.
- Data-freshness metadata and frozen-model fingerprints.
- `evaluation_report.md`, `model_card.md`, `deployment_guide.md` and
  `presentation_notes.md` with actual generated results.
- `evaluation_audit.json`, including any failed checks and evaluation coverage.

Save the executed notebook and results folder. The Colab dashboard is a prototype
demonstration, not a continuously hosted service or hardware controller. Example
tariffs/savings are scenario assumptions, not measured whole-house bill reductions.

Technical validation is recorded separately in `task10_validation.json`. Synthetic
test fixtures are never presented as project evaluation results.
