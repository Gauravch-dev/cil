# Smart Electricity Consumption Optimizer

Lab record writeups for Tasks 7 to 10

## Task 7 Deep Learning Model Design and Optimization

### Aim

To design and implement an optimized Long Short-Term Memory network for one-hour-ahead household electricity forecasting using suitable hyperparameters and regularization techniques.

### Theory

Deep learning uses artificial neural networks to learn complex patterns from data. A Long Short-Term Memory network, or LSTM, is a recurrent neural network that processes sequences and retains information from earlier time steps. It is suitable for electricity forecasting because consumption changes over time and follows recurring household patterns.

The Smart Electricity Consumption Optimizer uses the UCI household electricity dataset. Minute readings of active power, reactive power, voltage, current and three submetering signals are converted into 15-minute intervals. Short missing input gaps are forward-filled using past values only; missing target intervals are excluded. Scaling parameters are fitted on training data.

Each example contains the previous 24 hours, represented by 96 time steps, together with past-derived and calendar features. The target is average power in the 15-minute interval ending one hour after the forecast origin. For example, readings available at 6 PM predict average power during 6:45–7 PM, rather than total energy used throughout the hour.

The selected architecture contains stacked LSTM layers with 64 and 32 units, followed by shared dense layers. It learns both a power forecast and a high-demand detection score. A positive adjustment, controlled by this score, is added to the base forecast. High-demand training examples receive greater weight.

Training uses Adam with an initial learning rate of 0.0005, a batch size of 256 and a maximum of 60 epochs. Dropout, L2 regularization, early stopping and learning-rate reduction control overfitting. Data is divided chronologically into approximately 70% training, 15% validation and 15% testing, with gaps at the boundaries. LSTM selection uses validation results only.

On 18,703 test forecasts, the selected LSTM achieved MAE 0.4297 kW, RMSE 0.6162 kW and R² 0.4062. It outperformed persistence, while the basic neural network had slightly lower overall RMSE. High-demand RMSE was 1.6047 kW, compared with 1.7724 kW for the basic network. These results indicate useful but limited forecasting accuracy.

### Conclusion

Thus, a peak-gated multitask LSTM was implemented and evaluated for electricity forecasting. Regularization and validation-based selection supported model training, and the learned temporal representation provided a foundation for the integrated optimizer.

## Task 8 Model Explainability and Visualization

### Aim

To interpret electricity forecasts using SHAP and activation visualizations, and compare the performance and interpretability of the LSTM with a basic neural network.

### Theory

Model interpretability helps us investigate how an ML or DL model produces its predictions. In this project, Explainable AI is used to examine the contributions of historical electricity readings and contextual features to a forecast. An explanation describes model behavior; it does not prove what physically caused electricity consumption.

The trained Task 7 LSTM and basic neural network are reused without retraining. The SHAP Python library, specifically GradientExplainer, estimates contributions using expected gradients. Both networks use the same 32 training reference examples and the same test examples, allowing a consistent comparison. Summary plots cover 24 evenly spaced examples, with additional cases selected for diagnosis.

A positive SHAP value raises a forecast relative to the reference mean prediction, while a negative value lowers it. Feature-importance charts summarize contribution magnitudes, waterfall plots explain individual forecasts, and time-feature heatmaps show which earlier readings contributed. Contributions are expressed in kW for the final non-negative forecast.

Activation heatmaps display how units in the first LSTM layer respond across the 96 historical time steps. Dense-layer activations are also visualized for the basic neural network. Activation strength is different from feature contribution, and these plots are not attention maps because no attention layer is used.

Performance is compared on the complete 18,703-example test set using MAE, RMSE and R². The LSTM obtained RMSE 0.6162 kW, while the basic neural network obtained 0.6152 kW. The comparison therefore does not claim that the more complex LSTM is always more accurate or inherently easier to explain.

The explanation check compares each forecast with the reference mean plus its estimated SHAP contributions. In the executed analysis, 51 of 52 checks were within the stated tolerance and one LSTM explanation remained flagged. Approximation error, correlated inputs and reference selection are reported as limitations. The small explanation sample is not treated as a population-wide evaluation.

### Conclusion

Thus, SHAP and activation visualizations were applied to interpret and compare both neural models. The analysis provided feature and time-based explanations while reporting prediction errors, approximation limits and the difference between model attribution and causality.

## Task 9 Integrated Intelligent Electricity Optimizer

### Aim

To integrate deep learning, machine learning and AI reasoning into an end-to-end system that forecasts demand and recommends appliance schedules while respecting household constraints.

### Theory

An end-to-end intelligent system connects data preparation, learned predictions, decision logic and a user interface. In this project, a frozen LSTM extracts temporal patterns from past electricity readings. A classical ML regressor uses those patterns and context features to predict demand. Gradient Boosting and Random Forest are compared on validation data, and Random Forest is selected in the executed run.

The prediction stage is connected to the knowledge and reasoning layers. The supplied Protégé ontology contains household and appliance concepts and two SWRL rules: essential appliances remain ON, and flexible appliances may be delayed. Imported rule candidates are combined with ordered operational rules for occupancy, temperature, priority, tariffs and forecast demand. Essential protection overrides conflicting delay suggestions.

A* search finds a low-cost schedule for requested jobs using appliance ratings, runtimes, earliest starts, deadlines and a parallel-job limit. Already running and protected appliances are not interrupted. If no permitted schedule exists, the system reports the problem. Forecast-based avoidance applies only to the single predicted 15-minute interval; later scheduling uses supplied tariffs and constraints.

The integrated prototype runs through an interactive ipywidgets dashboard in Google Colab. Users select a recorded electricity case and edit household settings. The output includes the forecast, appliance recommendations, schedule, estimated job savings and decision-rule traces. SHAP TreeExplainer explains the ML prediction stage, while LSTM activation maps display its internal temporal responses.

The integrated LSTM–Random Forest model achieved MAE 0.4225 kW, RMSE 0.6226 kW and R² 0.3938 on 18,703 test forecasts. Its overall RMSE was slightly higher than the standalone LSTM result. Four representative consistency checks passed, covering essential protection, occupancy and comfort, high-demand avoidance and impossible deadlines.

In the saved dashboard example, the forecast was 1.629 kW. The refrigerator remained ON, the AC ran as scheduled, and the washing machine and dishwasher were scheduled at 8–9 PM and 9–10 PM. Estimated job savings were ₹5.10 using the example tariff settings. This is a scenario estimate, not a measured household bill reduction.

### Conclusion

Thus, the earlier project components were connected into a working electricity optimization pipeline. The system combined forecasting, ontology-based reasoning, constrained scheduling and explanations through one dashboard, while preserving essential appliances and clearly identifying estimated outcomes.

## Task 10 Final Validation and Responsible AI Evaluation

### Aim

To validate the integrated optimizer using inputs excluded from model fitting, examine the explanations of its decisions, and discuss ethical, fairness, privacy and bias-related issues.

### Theory

Final validation examines the forecasts, decision rules, scheduler and interface as a complete system. In the executed run, the Task 7 and Task 9 models were prepared in a fresh runtime before evaluation. Their fingerprints remained unchanged throughout evaluation, and test results were not used to retrain or tune them.

Forecasting evaluation reused 18,703 observations excluded from training and model selection. This holdout had already been examined in Task 9 and was not presented as a new independent test. No external measured dataset was supplied. Additional household scenarios were controlled tests, rather than observations from new households.

The integrated LSTM–Random Forest achieved MAE 0.4225 kW, RMSE 0.6226 kW and R² 0.3938. Its RMSE was slightly higher than the standalone LSTM result of 0.6162 kW. Thresholded high-demand detection achieved precision 40.59%, recall 10.18% and F1-score 16.28%, showing that many actual high-demand intervals were missed.

All 65 scenario and error-handling checks passed. These covered essential protection, occupied and empty hot homes, overnight tariffs, deadlines, concurrency, invalid inputs and search limits. Median response time was 41.34 milliseconds and the 95th percentile was 52.36 milliseconds for inference, reasoning and scheduling. These hardware-specific timings excluded SHAP explanation and display time.

SHAP TreeExplainer examined the ML prediction stage, and all 27 reconstruction checks passed. LSTM activation maps and rule traces explained internal responses and operational decisions. All three controlled policy-invariance checks also passed. These checks verified essential-device protection under renaming and different settings; they did not establish demographic fairness or prove causal explanations.

Responsible-AI analysis addressed electricity-data privacy, user control, essential-appliance safety, fixed comfort thresholds and limited representation from one household. Missing demographic labels prevented population fairness claims. Tariffs and savings remained scenario assumptions. The executed notebook generated evaluation outputs, a report, model card, deployment guidance and presentation notes, with the Colab dashboard serving as an advisory prototype demonstration.

### Conclusion

Thus, the integrated optimizer was evaluated for forecasting, decision consistency, explainability and responsible use. Tests confirmed the checked constraint behavior and unchanged model fingerprints, while weak peak detection and limited data representativeness remained important limitations.
