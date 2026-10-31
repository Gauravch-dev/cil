# api.py — FastAPI backend for RiskRadar Flood Prediction
# Run with: uvicorn api:app --reload --port 8000

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# ─── Load artifacts ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "flood_model.pkl"))
    le_dict = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))
    feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
    print(f"Model loaded ✓  |  Features: {feature_columns}")
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model artifact not found: {e}\n"
        "Run the Colab notebook first and download the .pkl files."
    )

# ─── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="RiskRadar — Flood Risk Prediction API",
    description="Predicts flood risk from hydrological and climate features using XGBoost.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Input schema ────────────────────────────────────────────────────────────
class FloodInput(BaseModel):
    annual_precipitation: float = Field(..., example=1200.0, description="mm/year")
    precipitation_of_wettest_month: float = Field(..., example=300.0, description="mm")
    precipitation_seasonality: float = Field(..., example=50.0, description="CV %")
    drainage_density: float = Field(..., example=2.5, description="km/km²")
    drainage_texture: float = Field(..., example=10.0)
    basin_relief: float = Field(..., example=500.0, description="meters")
    annual_mean_temperature: float = Field(..., example=20.0, description="°C")
    temperature_seasonality: float = Field(..., example=500.0)
    curve_number_amcii: float = Field(..., example=70.0)
    ruggedness_number: float = Field(..., example=0.5)
    infiltration_number: float = Field(..., example=5.0)
    climate_type: int = Field(..., example=2, description="Encoded climate type (0-5)")
    landcover_type: int = Field(..., example=1, description="Encoded landcover (0-5)")
    soil_type: int = Field(..., example=3, description="Encoded soil type (0-5)")

# ─── Feature builder ─────────────────────────────────────────────────────────
def build_features(data: FloodInput) -> np.ndarray:
    """Build feature array in exact same order as training."""
    precipitation_risk_index = (
        min(data.annual_precipitation, 3000) * 0.4 +
        min(data.precipitation_of_wettest_month, 1000) * 0.3 +
        min(data.precipitation_seasonality, 100) * 0.3
    )
    drainage_risk_index = (
        min(data.drainage_density, 10) * 0.5 +
        min(data.drainage_texture, 50) * 0.3 +
        min(data.basin_relief, 5000) * 0.2
    )

    feature_map = {
        'precipitation_risk_index': precipitation_risk_index,
        'drainage_risk_index': drainage_risk_index,
        'annual_mean_temperature': data.annual_mean_temperature,
        'temperature_seasonality': data.temperature_seasonality,
        'annual_precipitation': data.annual_precipitation,
        'curve_number_amcii': data.curve_number_amcii,
        'drainage_density': data.drainage_density,
        'basin_relief': data.basin_relief,
        'ruggedness_number': data.ruggedness_number,
        'infiltration_number': data.infiltration_number,
        'climate_type': data.climate_type,
        'landcover_type': data.landcover_type,
        'soil_type': data.soil_type,
    }

    # Use only the features the model was trained on, in the right order
    return np.array([[feature_map[f] for f in feature_columns]])

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "RiskRadar Flood Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "XGBoost",
        "features": len(feature_columns)
    }

@app.post("/predict")
def predict(data: FloodInput):
    try:
        features = build_features(data)
        risk_score = float(model.predict_proba(features)[0][1])

        if risk_score > 0.7:
            risk_level = "HIGH"
            color = "red"
            message = "Immediate attention required. High probability of flood event."
        elif risk_score > 0.4:
            risk_level = "MEDIUM"
            color = "orange"
            message = "Elevated risk. Monitor conditions closely."
        else:
            risk_level = "LOW"
            color = "green"
            message = "Conditions appear stable. Continue routine monitoring."

        return {
            "flood_risk_score": round(risk_score, 4),
            "risk_percentage": f"{round(risk_score * 100, 1)}%",
            "risk_level": risk_level,
            "color": color,
            "message": message,
            "features_used": len(feature_columns)
        }

    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Feature mismatch: {e}. Ensure model artifacts match the input schema."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features")
def get_features():
    """Returns the list of features the model expects."""
    return {"feature_columns": feature_columns, "count": len(feature_columns)}