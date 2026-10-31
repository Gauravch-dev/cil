# Smart Electricity Consumption Optimizer

Lab record writeups for Tasks 7 to 10

## Task 7 Deep Learning Model Design and Optimization

### Aim

To design and implement an optimized LSTM model for one-hour-ahead electricity forecasting using suitable hyperparameters and regularization techniques.

### Theory

Deep learning uses neural networks to learn patterns from data. Long Short-Term Memory (LSTM) networks retain information from earlier readings, making them suitable for household electricity forecasting. This project uses seven electrical measurements from the UCI household electricity dataset.

Minute readings are converted into 15-minute averages. Short input gaps are forward-filled using past values; missing targets are excluded. Scaling uses training data only. Each sample contains 96 past intervals, or 24 hours, to predict average power during the 15-minute interval ending one hour later.

The selected peak-gated multitask model contains LSTM layers with 64 and 32 units, followed by dense layers. It predicts power and a high-demand detection score. This score controls a positive adjustment to the base forecast, while high-demand training examples receive greater weight.

Training uses Adam with learning rate 0.0005, batch size 256 and a maximum of 60 epochs. Dropout, L2 regularization, early stopping and learning-rate reduction limit overfitting. Data is split chronologically into 70% training, 15% validation and 15% testing, with boundary gaps. Selection uses validation results only.

On 18,703 test forecasts, the LSTM achieved MAE 0.4297 kW, RMSE 0.6162 kW and R² 0.4062. It outperformed persistence, although the basic neural network had slightly lower overall RMSE. High-demand RMSE was 1.6047 kW.

### Conclusion

Thus, a peak-gated multitask LSTM was implemented and evaluated for electricity forecasting. Regularization supported training, and the learned temporal patterns provided a foundation for the integrated optimizer.

## Task 8 Model Explainability and Visualization

### Aim

To interpret electricity forecasts using SHAP and activation visualizations, and compare the LSTM with a basic neural network.

### Theory

Explainable AI helps investigate how models produce predictions. In this project, it examines how historical readings and contextual features contribute to electricity forecasts. These explanations describe model behavior rather than proving physical causes of consumption.

The trained Task 7 models are reused without retraining. SHAP GradientExplainer estimates contributions using expected gradients. Both models use the same 32 training reference examples and test cases. Summary plots cover 24 evenly spaced examples, with additional diagnostic cases.

Positive SHAP values raise the forecast relative to the reference mean, while negative values lower it. Feature-importance charts summarize contributions, waterfall plots explain individual forecasts, and time-feature heatmaps identify contributing past readings. Contributions are expressed in kW.

LSTM activation heatmaps show how internal units respond across 96 historical time steps. Dense activations are also visualized for the basic neural network. Activation strength differs from feature contribution, and these visualizations are not attention maps.

On the full 18,703-example test set, LSTM RMSE was 0.6162 kW compared with 0.6152 kW for the basic model. The explanation sums were checked against the forecasts: 51 of 52 checks passed the tolerance, with one LSTM explanation flagged. Reference selection, correlated inputs and approximation errors remain limitations.

### Conclusion

Thus, SHAP and activation plots were used to interpret and compare both models. The analysis explained feature and time contributions while reporting accuracy differences and explanation limitations.

## Task 9 Integrated Intelligent Electricity Optimizer

### Aim

To integrate deep learning, machine learning and AI reasoning into a system that forecasts demand and recommends appliance schedules while respecting household constraints.

### Theory

An end-to-end intelligent system connects data preparation, prediction, decision logic and a user interface. Here, a frozen LSTM extracts temporal patterns, which a classical ML regressor combines with context features to forecast demand. Validation comparison selected Random Forest over Gradient Boosting.

The forecast feeds the knowledge and reasoning layers. The Protégé ontology provides appliance concepts and SWRL rules for essential and flexible appliances. Ordered operational rules add occupancy, temperature, priority, tariff and forecast conditions. Essential protection overrides conflicting delay suggestions.

A* search schedules requested jobs using power ratings, runtimes, deadlines and parallel-job limits. Protected and already running appliances are not interrupted. Impossible schedules are reported clearly. Forecast-based avoidance applies only to the predicted 15-minute interval; later choices use supplied tariffs and constraints.

An ipywidgets dashboard in Colab displays forecasts, recommendations, schedules, estimated savings and rule traces. Users select recorded electricity cases and change household settings. SHAP TreeExplainer explains the ML prediction stage, while activation maps show LSTM responses.

The integrated model achieved MAE 0.4225 kW, RMSE 0.6226 kW and R² 0.3938 on 18,703 test forecasts. Its RMSE was slightly higher than the standalone LSTM. Four consistency checks passed. One dashboard scenario delayed the washing machine and dishwasher to cheaper slots, estimating ₹5.10 savings under example tariffs.

### Conclusion

Thus, the project components formed a working forecasting, reasoning and scheduling pipeline with an interactive dashboard. Recommendations respected the tested constraints, while savings remained scenario estimates.

## Task 10 Final Validation and Responsible AI Evaluation

### Aim

To validate the integrated optimizer, explain its decisions, and examine ethical, fairness, privacy and bias-related issues.

### Theory

Final validation evaluates forecasting, decision rules, scheduling and the interface together. Models were prepared before evaluation and then kept fixed. Their fingerprints remained unchanged, and evaluation results were not used for retraining or tuning.

The evaluation reused 18,703 observations excluded from training and model selection. Since this holdout had already been examined in Task 9, it was not claimed as a new independent test. No external measured dataset was supplied; additional household scenarios were controlled tests.

The integrated model achieved MAE 0.4225 kW, RMSE 0.6226 kW and R² 0.3938. High-demand detection recall was only 10.18%, indicating many missed peaks. All 65 scenario and error-handling checks passed, covering essential protection, comfort, tariffs, deadlines, invalid inputs and search limits.

Median response time was 41.34 milliseconds and the 95th percentile was 52.36 milliseconds, excluding SHAP and display time. All 27 SHAP reconstruction checks and three controlled policy-invariance checks passed. LSTM activations and rule traces explained model responses and appliance decisions; these checks did not prove demographic fairness or causality.

Responsible-AI analysis considered privacy, user control, essential-appliance safety, fixed comfort thresholds and bias from training on one household. Savings remained estimates, and demographic fairness could not be established without demographic data. Reports, a model card, deployment guidance and presentation notes documented the findings. The Colab dashboard remained an advisory prototype.

### Conclusion

Thus, the optimizer was evaluated for forecasting, decision consistency, explainability and responsible use. The tested constraints were respected, while weak peak detection and limited data representativeness remained important limitations.
