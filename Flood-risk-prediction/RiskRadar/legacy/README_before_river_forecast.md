# 🌊 RiskRadar — Rainfall & Flood Risk Analysis Using Historical Weather Data

**Causal-history balance improvement:** Added strictly earlier annual-discharge summaries, with a Random Forest history expert and a separate HistGradientBoosting expert for stations without training history. Thirty forward CV fits choose models and precision/FPR-constrained thresholds. On the same 5,339-row test, precision is **81.72%**, recall **74.06%**, F1 **0.7770**, with **53 false positives / 83 misses**. This improves the balance over the previous recall-oriented model (42.45% precision, 93.13% recall, F1 0.5832, 404 false positives). All history comes from training through 1999; no current, validation or test discharge is used as an input. **Limits:** no-history recall remains 40%; the historical-average baseline has higher history-route F1 (0.8112) than learned RF (0.7993). It is a partial improvement, not proof of superior ML or a flood-warning system. [Full report](experiments/gloflo_history_v1/MODEL_REPORT.md). A verified inference bundle is saved as `experiments/gloflo_history_v1/selected_experts.joblib`; the website is unchanged.

**Station-aware expanded-feature experiment:** Added training-selected catchment descriptors and compared CatBoost, HistGradientBoosting and Random Forest using three station-disjoint forward-time folds. Eighteen candidates produce 54 CV fits and nine saved final pipelines. CV selects HistGradientBoosting with 37 inputs. At its frozen recall-oriented threshold, test precision is 42.45%, recall 93.13%, F1 0.5832 and F2 0.7518. It detects 29 of 35 unseen-station positives, but creates 404 false positives overall versus 26 for the previous frozen winner; overall F1 declines, and unseen-station ranking AP also declines. The 50% precision / 75% recall operating goal was infeasible in CV. **This is a documented trade-off, not a completed predictive fix, and it is not deployed.** Dataset/target/chronology remain unchanged; no discharge-derived inputs are predictors. See [the station-aware model report](experiments/gloflo_station_features_v1/MODEL_REPORT.md) and [comparison chart](experiments/gloflo_station_features_v1/model_comparison.png). Earlier validation/test inspection means this remains exploratory follow-up evidence.

**Expanded learner/sampling comparison:** Completed 76 fits across eight learning algorithms and seven treatments, saving 38 pipelines. Validation selects Random Forest with random oversampling (F1 0.7972). On the unchanged chronological test it has precision 88.94%, recall 65.31%, F1 0.7532 and AP 0.8431. This reduces false positives from 78 to 26 versus the previous balanced XGBoost, but increases misses from 90 to 111; uncertainty does not establish reliable F1 improvement. Sampling does not consistently help: untreated Random Forest has test F1 0.7744 and untreated HistGradientBoosting 0.7786, but neither replaces the frozen winner based on test scores. **The selected model detects none of the 35 positive cases at unseen stations at its frozen threshold.** It is an experimental high-discharge classifier, not a validated flood-warning system. The website is unchanged. See [the full learner comparison](experiments/gloflo_learners_v1/COMPARISON_REPORT.md) and [selected model metadata](experiments/gloflo_learners_v1/selected_model_metadata.json). The previously examined validation/test periods make this an exploratory follow-up.

**SMOTE follow-up:** Trained 18 SMOTENC candidates, using mixed-type synthetic sampling on training data only. The selected SMOTENC model adds 15,803 synthetic training positives (minority/majority ratio 0.5, k=3). Test precision improves to 80.08%, but recall falls to 62.81% and F1 to 0.7040, compared with untreated F1 0.7258 and the previous balanced model's 0.7325. SMOTE does not improve the selected objective; the previous balanced model remains the validation-selected winner. The same test period was already examined, so this is a follow-up comparison rather than fresh independent confirmation. The website model is unchanged. See [SMOTE results](experiments/gloflo_smotenc_v1/SMOTE_REPORT.md); training dependencies are in `requirements-training.txt`.

**Class imbalance re-evaluation (24 October 2026):** Compared 33 XGBoost candidates using training-only class weighting, random oversampling, undersampling and a combined treatment. Validation selected balanced oversampling plus positive-class weight 2, with threshold 0.9108. On the chronological 2006–2015 test (5,339 records; 320 positives), positive-class F1 changes from 0.7258 to 0.7325 and recall from 69.06% to 71.88%, while precision falls from 76.47% to 74.68%. This recovers nine more positives at the cost of ten more false alerts. AP decreases, and the paired F1 uncertainty interval includes zero: this is a small observed trade-off, not convincing evidence of general improvement. The target is training-defined high annual peak discharge, not verified flood occurrence. Models and results are frozen separately; the website model is unchanged. See [the imbalance report](experiments/gloflo_imbalance_v1/IMBALANCE_REPORT.md).

**Original dataset traced:** Replaying the unchanged original model on the user's existing GloFlo CSV reproduces all 2,538 saved map rows. A high-discharge label based on the source 95th percentile and a stratified seed-42 split reproduces the advertised rounded accuracy/recall and the class-0 F1. The actual class-1 F1 in this reconstruction is 0.717, not 0.984. This is a strongly supported reconstruction, not an independent flood-occurrence evaluation. See [the provenance findings](evaluation/upstream_forensics/PROVENANCE_FINDINGS.md).

**New training comparison (24 October 2026):** Two XGBoost classifiers were trained from scratch on the supplied INDOFLOODS event labels, predicting **Severe Flood versus Flood**, conditional on an observed flood. These are experimental models with a different target from the current website. Static inputs were selected on validation PR-AUC; adding antecedent rainfall increased held-out recall but also false severe classifications. Results are modest, and neither model is deployed. See [the comparison report](experiments/indofloods_severity_v1/COMPARISON_REPORT.md) for chronological metrics, baselines, exclusions, feature lists and reproduction commands. The fitted pipelines and native models are in `experiments/indofloods_severity_v1/models/`.

**Current dashboard:** Two task-based choices in the top navigation: **Rainfall history** and **Check flood risk**. Rainfall history is the landing page, with year/location filters, yearly totals, monthly heatmaps, daily zoom and a rainfall download. Risk checks offer example conditions, a clear result, rainfall comparisons and a CSV summary. The sidebar, model panels, connection badges and technical source panels have been removed from the user interface. Risk Map is archived under [legacy/](legacy/README.md), with its source CSV preserved. Model methodology and evaluation limitations remain in [MODEL_REPORT.md](MODEL_REPORT.md); setup and data preparation are in [LOCAL_SETUP.md](LOCAL_SETUP.md). The supplied classifier and API logic are unchanged.

**Retained upstream documentation:** The sections below describe the original repository and can include archived features and original screenshots. Its training metrics have not been independently reproduced; the source has no binary flood/no-flood labels. Refer to the current dashboard description and model report above for this checkout's behavior.

**Latest validation:** [Evaluation results](EVALUATION_RESULTS.md) report 73 checks passed and five direct API input-validation failures. Model accuracy remains independently unverified. These findings are documented outside the user interface.

> **An End-to-End Machine Learning System for Flood Risk Prediction using Hydrological, Meteorological, and Geospatial Data.**  
> Built using **XGBoost**, deployed with **FastAPI**, and visualized through an interactive **Streamlit Dashboard**.

---

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=for-the-badge&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-96.9%25-brightgreen?style=for-the-badge)
![F1 Score](https://img.shields.io/badge/F1%20Score-0.984-brightgreen?style=for-the-badge)

---

## 🚀 Live Demo
**[👉 Try RiskRadar Live](https://nitin454545-riskradar.hf.space)**

---

# 📌 Table of Contents

- [📖 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🧠 Model & Methodology](#-model--methodology)
- [📊 Results](#-results)
- [🏗️ Project Structure](#️-project-structure)
- [⚙️ Installation](#️-installation)
- [🚀 Running the Project](#-running-the-project)
- [🔌 API Reference](#-api-reference)
- [🐳 Docker Support](#-docker-support)
- [🛠️ Tech Stack](#️-tech-stack)
- [📈 Future Scope](#-future-scope)
- [📜 License](#-license)
- [👨‍💻 Author](#-author)

---

# 📖 Overview

RiskRadar is a production-grade flood risk prediction system that analyzes hydrological, meteorological, and geospatial parameters to classify locations into flood risk categories.

The project leverages a high-performance **XGBoost Machine Learning model** trained on real-world flood datasets and provides predictions through:

- 🚀 FastAPI REST API
- 📊 Interactive Streamlit Dashboard
- 🐳 Dockerized Deployment Pipeline

Unlike a traditional ML notebook project, RiskRadar focuses on a complete deployment-ready architecture suitable for real-world usage.

---

# ✨ Key Features

✅ Trained on **45,932 global flood locations**  
✅ Achieved **96.9% Accuracy**  
✅ **F1 Score: 0.984**  
✅ Uses **SMOTE** for class imbalance handling  
✅ Real-time prediction dashboard with Plotly visualizations  
✅ FastAPI backend with input validation  
✅ Interactive flood-risk mapping  
✅ Docker-ready deployment  
✅ Production-oriented ML workflow

---

# 🧠 Model & Methodology

## 📂 Dataset

- **Source:** IIT Delhi Static Flood Database
- **Dataset File:** `kuntla_flooddatabase.csv`
- **Raw Features:** 93
- **Final Engineered Features:** 13
- **Target:** Flood Risk Classification

---

## 🔄 Machine Learning Pipeline

```text
Raw Dataset (93 Parameters)
        ↓
Missing Value Handling
        ↓
Feature Engineering
        ↓
Categorical Encoding
        ↓
Train/Test Split
        ↓
SMOTE Balancing
        ↓
Hyperparameter Tuning
        ↓
XGBoost Training
        ↓
Evaluation & Model Export
        ↓
FastAPI + Streamlit Deployment
```

---

## ⚙️ Data Preprocessing

### Missing Value Handling
- Dropped columns with more than 40% missing values
- Median imputation for numerical features

### Feature Engineering
Created custom engineered features such as:
- `precipitation_risk_index`
- `drainage_risk_index`

### Encoding
Applied Label Encoding for:
- Climate Type
- Landcover Type
- Soil Type
- Lithology Type

---

## 🤖 Model Training

### Algorithm Used
- **XGBoost Classifier**

### Hyperparameter Optimization
- `RandomizedSearchCV`
- 50 parameter combinations
- 3-fold cross-validation
- GPU accelerated training (`tree_method=hist`)

### Handling Imbalanced Data
- Applied **SMOTE** on training dataset

---

# 📊 Results

| Metric | Score |
|---|---|
| Accuracy | 96.9% |
| F1 Score | 0.984 |
| Recall | 0.791 |

---

## 🌍 High Risk Locations

- High-risk locations identified: **2,538**
- Total locations analyzed: **45,932**

---

## 🔥 Most Important Risk Factors

- Annual Precipitation
- Seasonal Rainfall Patterns
- Drainage Density
- Basin Relief
- Soil Characteristics
- Land Cover Features
- Temperature Seasonality

---

# 🏗️ Project Structure

```bash
riskradar/
│
├── flood_fast.py
├── api.py
├── app.py
├── Dockerfile
├── requirements.txt
├── flood_model.pkl
├── encoders.pkl
├── feature_columns.pkl
├── high_risk_locations.csv
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/nitinc264/Flood-prediction-Model.git
cd Flood-prediction-Model
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Project

## ▶️ Start FastAPI Backend

```bash
uvicorn api:app --reload --port 8000
```

### API Documentation

```text
http://localhost:8000/docs
```

---

## ▶️ Start Streamlit Dashboard

Open a new terminal and run:

```bash
streamlit run app.py
```

### Dashboard URL

```text
http://localhost:8501
```

---

# 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API Information |
| GET | `/health` | Health Check |
| GET | `/features` | Feature List |
| POST | `/predict` | Flood Risk Prediction |

---

## 📥 Sample Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "annual_precipitation": 1500,
  "precipitation_of_wettest_month": 400,
  "precipitation_seasonality": 60,
  "drainage_density": 3.5,
  "drainage_texture": 12.0,
  "basin_relief": 800,
  "annual_mean_temperature": 22,
  "temperature_seasonality": 600,
  "curve_number_amcii": 75,
  "ruggedness_number": 0.8,
  "infiltration_number": 6,
  "climate_type": 2,
  "landcover_type": 1,
  "soil_type": 3
}'
```

---

## 📤 Sample Response

```json
{
  "flood_risk_score": 0.8341,
  "risk_percentage": "83.4%",
  "risk_level": "HIGH",
  "color": "red",
  "message": "Immediate attention required. High probability of flood event.",
  "features_used": 13
}
```

---

# 🐳 Docker Support

## Build Docker Image

```bash
docker build -t riskradar .
```

---

## Run Docker Container

```bash
docker run -p 8000:8000 riskradar
```

---

# 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | XGBoost, Scikit-learn |
| Data Processing | Pandas, NumPy |
| Imbalanced Learning | SMOTE |
| Backend API | FastAPI, Uvicorn |
| Frontend Dashboard | Streamlit |
| Visualization | Plotly, Folium, Matplotlib |
| Deployment | Docker |
| Training Environment | Google Colab |

---

# 📈 Future Scope

- [x] Hugging Face deployment ✓ — Live at [nitin454545-riskradar.hf.space](https://nitin454545-riskradar.hf.space)
- [ ] Real-time weather API integration
- [ ] Global-scale flood dataset expansion
- [ ] SMS & Email flood alerts
- [ ] Mobile-responsive frontend
- [ ] Live geospatial monitoring
- [ ] Cloud deployment pipeline

---

# 📜 License

This project is licensed under the **MIT License**.

You are free to:
- Use
- Modify
- Distribute
- Improve

with proper attribution.

---

# 👨‍💻 Author

## Nitin Chauhan

🎓 B.Tech — AI & Data Science  
🏫 Dr. D.Y. Patil School of Science and Technology

[![Portfolio](https://img.shields.io/badge/Portfolio-nitinc264.github.io-blue?style=flat&logo=github)](https://nitinc264.github.io/)
[![GitHub](https://img.shields.io/badge/GitHub-nitinc264-black?style=flat&logo=github)](https://github.com/nitinc264)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Nitin%20Chauhan-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/nitin-chauhan-6963612a7/)

---

# ⭐ If you found this project useful, don't forget to star the repository!
