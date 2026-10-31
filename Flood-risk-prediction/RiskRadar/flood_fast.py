"""flood_fast.py
Simple training script for RiskRadar flood model.
Generates a demo model (XGBoost if available, else RandomForest) and saves artifacts:
- flood_model.pkl
- encoders.pkl
- feature_columns.pkl

Usage:
    python flood_fast.py --epochs 10
"""
import os
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_COLUMNS = [
    'precipitation_risk_index',
    'drainage_risk_index',
    'annual_mean_temperature',
    'temperature_seasonality',
    'annual_precipitation',
    'curve_number_amcii',
    'drainage_density',
    'basin_relief',
    'ruggedness_number',
    'infiltration_number',
    'climate_type',
    'landcover_type',
    'soil_type',
]


def generate_dummy(n=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    annual_precipitation = rng.uniform(200, 3000, size=n)
    precipitation_of_wettest_month = rng.uniform(50, 1000, size=n)
    precipitation_seasonality = rng.uniform(5, 100, size=n)
    drainage_density = rng.uniform(0.1, 10, size=n)
    drainage_texture = rng.uniform(1, 50, size=n)
    basin_relief = rng.uniform(10, 5000, size=n)
    annual_mean_temperature = rng.uniform(-5, 35, size=n)
    temperature_seasonality = rng.uniform(10, 1000, size=n)
    curve_number_amcii = rng.uniform(30, 100, size=n)
    ruggedness_number = rng.uniform(0, 2, size=n)
    infiltration_number = rng.uniform(0.1, 20, size=n)
    climate_type = rng.integers(0, 6, size=n)
    landcover_type = rng.integers(0, 6, size=n)
    soil_type = rng.integers(0, 6, size=n)

    precipitation_risk_index = (
        np.minimum(annual_precipitation, 3000) * 0.4 +
        np.minimum(precipitation_of_wettest_month, 1000) * 0.3 +
        np.minimum(precipitation_seasonality, 100) * 0.3
    )
    drainage_risk_index = (
        np.minimum(drainage_density, 10) * 0.5 +
        np.minimum(drainage_texture, 50) * 0.3 +
        np.minimum(basin_relief, 5000) * 0.2
    )

    # crude flood probability function for synthetic labels
    risk_score = (
        0.5 * (precipitation_risk_index / precipitation_risk_index.max()) +
        0.3 * (drainage_risk_index / drainage_risk_index.max()) +
        0.2 * (curve_number_amcii / 100)
    )
    # convert to binary label with some noise
    prob = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
    prob = 0.05 + 0.9 * prob
    y = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        'annual_precipitation': annual_precipitation,
        'precipitation_of_wettest_month': precipitation_of_wettest_month,
        'precipitation_seasonality': precipitation_seasonality,
        'drainage_density': drainage_density,
        'drainage_texture': drainage_texture,
        'basin_relief': basin_relief,
        'annual_mean_temperature': annual_mean_temperature,
        'temperature_seasonality': temperature_seasonality,
        'curve_number_amcii': curve_number_amcii,
        'ruggedness_number': ruggedness_number,
        'infiltration_number': infiltration_number,
        'climate_type': climate_type,
        'landcover_type': landcover_type,
        'soil_type': soil_type,
        'precipitation_risk_index': precipitation_risk_index,
        'drainage_risk_index': drainage_risk_index,
        'flooded': y,
    })
    return df


def train(save_dir=BASE_DIR, epochs=10):
    csv_path = os.path.join(BASE_DIR, 'high_risk_locations.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print('Loaded', csv_path)
    else:
        print('No dataset found, generating synthetic data')
        df = generate_dummy(2000)

    X = df[FEATURE_COLUMNS]
    y = df['flooded'] if 'flooded' in df.columns else (df.iloc[:, 0] > 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if _HAS_XGB:
        print('Using XGBoost')
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100)
    else:
        print('XGBoost not available, using RandomForest')
        model = RandomForestClassifier(n_estimators=100)

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f'AUC on test set: {auc:.4f}')

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, 'flood_model.pkl'))
    joblib.dump({}, os.path.join(save_dir, 'encoders.pkl'))
    joblib.dump(FEATURE_COLUMNS, os.path.join(save_dir, 'feature_columns.pkl'))
    print('Saved artifacts: flood_model.pkl, encoders.pkl, feature_columns.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    train(epochs=args.epochs)
