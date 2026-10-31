# Rainfall and Flood Risk Analysis Using Historical Weather Data

RiskRadar combines historical rainfall analysis near Mumbai with a manual flood-susceptibility assessment using rainfall, climate and catchment conditions.

- **Rainfall:** daily, monthly, yearly and monsoon charts, historical comparisons and unusually wet periods.
- **Flood:** editable rainfall and land conditions, example scenarios and a Low, Medium or High susceptibility result from the saved XGBoost classifier.

## Run locally

Requires Python 3.11 or 3.12.

```powershell
git clone https://github.com/AnjanThakur/Flood-risk-prediction.git
cd Flood-risk-prediction/RiskRadar
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. The saved model and the Mumbai rainfall history needed by the dashboard are included. No API key is required.

## Project material

- [Application and detailed documentation](RiskRadar/README.md)
- [Model report](RiskRadar/MODEL_REPORT.md)
- [Existing-model evaluation](RiskRadar/evaluation/existing_model/EVALUATION.md)
- [Final project report](RiskRadar/reports/Final_Rainfall_and_Flood_Risk_Analysis_Report.docx)
- [Presentation slide images](RiskRadar/presentation/Rainfall_Flood_Presentation_Images.zip)

Historical rainfall uses ERA5 via Open-Meteo. Catchment data comes from [GloFlo](https://doi.org/10.5281/zenodo.7158028). The original model/application source is [nitinc264/Flood-prediction-Model](https://github.com/nitinc264/Flood-prediction-Model); the project folder documents subsequent analysis and interface changes.

The assessment describes annual high-discharge susceptibility. Evaluation figures and their reconstructed evaluation method are documented with the existing model.
