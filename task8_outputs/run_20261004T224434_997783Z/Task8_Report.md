# Task 8 results

No models were retrained or tuned in this task.

Performance uses all 18,889 test targets. Input attribution summaries use 32 evenly spaced examples.

| Model | MAE (kW) | RMSE (kW) | R2 | Parameters |
|---|---:|---:|---:|---:|
| LSTM | 0.4269 | 0.6182 | 0.3999 | 9,249 |
| Basic neural network | 0.4407 | 0.6235 | 0.3896 | 40,129 |

The LSTM reduces RMSE by 0.84% versus the basic model. This is a small observed gain; statistical significance was not tested.
Its R2 remains modest, and sudden peaks are difficult to forecast.

## What the explanations show

Temporal LSTM activations and dense-layer activations were visualized for a good forecast, a typical error, and a missed high-demand case.
Integrated Gradients explains changes from one real training reference, not absolute causes of household consumption.
- Largest attribution-magnitude groups for LSTM: History: hour cos, History: hour sin, History: dow sin.
- Largest attribution-magnitude groups for Basic neural network: History: power scaled, History: dow cos, History: hour cos.

These are reference-dependent sample summaries. History channels aggregate 96 values, so their totals are not directly comparable to one scalar context feature.
The direct history-power time chart omits the additional attribution assigned to correlated lag and rolling-summary features.
Both models need post-hoc explanations. Fewer LSTM parameters do not remove the complexity of recurrent processing.

## Verification

All 256 reloaded-model example predictions matched the saved forecasts.
Integrated Gradients completeness checks passed: 66/70; maximum absolute residual 0.016264 kW.
Completeness checks numerical consistency, not causal truth or clinical/safety reliability.

## Files

PNG figures, CSV tables, activation arrays, attribution arrays, and the analysis manifest accompany this report.

Method reference: [TensorFlow Integrated Gradients](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients).