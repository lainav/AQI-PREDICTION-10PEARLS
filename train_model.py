#train_model.py

import os
import joblib
import hopsworks
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_aqi(pm25):
    """
    Calculate the US AQI for a given PM2.5 concentration using EPA breakpoints.
    """
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            aqi = ((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo
            return round(aqi)
    return None

def train_pipeline():
    # Login and retrieve feature store
    project = hopsworks.login(api_key_value="2djdhFrqCIxvo6ApQ.xERPM8BgPzFrbYJVj0uf2EavQ8PPSbp9nIytq13qINXxpjhNHPwU9XYRZEJ1hBUc")
    fs = project.get_feature_store()

    # Read the feature group
    fg = fs.get_feature_group(name="openmeteo_aq_feature_group", version=1)
    df = fg.read()
    print("Data fetched from Feature Store:")
    print(df.head())

    # Ensure PM2.5 is present
    if 'pm2_5' not in df.columns:
        print("No 'pm2_5' column found in the DataFrame.")
        return

    # Calculate US AQI from PM2.5
    df['aqi'] = df['pm2_5'].apply(calculate_aqi)

    # Define your feature set
    features = [
        'day',
        'month',
        'pm2_5_change_rate',
        'carbon_monoxide',
        'carbon_dioxide',
        'nitrogen_dioxide',
        'sulphur_dioxide'
    ]

    # Drop rows with missing values in features or target
    df = df.dropna(subset=features + ['aqi'])

    X = df[features]
    y = df['aqi']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models to evaluate
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Ridge": Ridge(),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "SVR": SVR(),
        "MLPRegressor": MLPRegressor(random_state=42, max_iter=500),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "Lasso": Lasso()
    }

    best_model = None
    best_model_name = None
    best_score = float("inf")
    best_metrics = {}

    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"[{name}] RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")

        if rmse < best_score:
            best_score = rmse
            best_model_name = name
            best_model = model
            best_metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    print(f"\nBest Model: {best_model_name} with RMSE: {best_score:.3f}")

    # Save the best model to Hopsworks Model Registry
    model_registry = project.get_model_registry()
    model_dir = "openmeteo_aqi_model"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(best_model, f"{model_dir}/model.joblib")

    model_meta = model_registry.python.create_model(
        name="openmeteo_aqi_model",
        metrics=best_metrics,
        model_schema=None,
        description=f"Best model is {best_model_name} with RMSE {best_score:.3f} for US AQI prediction"
    )
    model_meta.save(model_dir)

    print(
        f"Model '{best_model_name}' saved to Model Registry with metrics: {best_metrics}"
    )

if __name__ == "__main__":
    train_pipeline()
