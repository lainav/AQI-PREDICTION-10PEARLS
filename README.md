
# 10Pearls AQI Predictor

The 10Pearls AQI Predictor is a serverless application that forecasts the Air Quality Index (AQI) for Karachi, Pakistan, over the next 72 hours. It combines real-time data ingestion, automated model training, and interactive visualization to deliver continuously updated air quality insights.

---

## Overview

This project demonstrates modern MLOps and cloud automation practices by orchestrating an end-to-end AQI forecasting pipeline.
It fetches air quality data from the Open-Meteo API, processes it through a feature pipeline, trains ML models automatically, and visualizes live predictions through a web-based dashboard.

The system runs 100% serverless using GitHub Actions for CI/CD and Docker for consistent, scalable deployment.

---

## Key Features

🔄 Automated Data Pipeline

  Hourly ingestion of AQI and weather data (PM2.5, temperature, humidity, wind speed) via the Open-Meteo API.

  Automated preprocessing and feature engineering using time-based and derived metrics.

🤖 Machine Learning Forecasting

  Daily model retraining using Random Forest, Ridge Regression, or TensorFlow models.

  Model evaluation via RMSE, MAE, and R² metrics.

  Best-performing model automatically stored in the Hopsworks Model Registry.

📊 Interactive Dashboard

  Streamlit-based dashboard for exploring live AQI forecasts and historical trends.

  Real-time visualization of model outputs and insights into future air quality.

☁️ Serverless CI/CD Automation

  Fully automated GitHub Actions workflows:

  .github/workflows/fetch_features.yaml → Hourly feature updates

  .github/workflows/train_model.yaml → Daily model retraining

  Secrets (e.g., HOPSWORKS_API_KEY) securely stored in GitHub Secrets.

🐳 Containerized Deployment

  Dockerized application for portability, reproducibility, and scalability.

---

## Architecture Overview

1. Data Collection:
  Hourly data fetched from the Open-Meteo API (PM2.5, temperature, humidity, wind speed).

  Stored and versioned in Hopsworks Feature Store.

2. Feature Engineering:
  Feature pipeline transforms raw data into machine-learning-ready features.

  Includes time-based, rolling, and lag features.

3. Model Training:
  Daily retraining fetches historical features and trains models automatically.

  Best model selected based on RMSE and stored in Model Registry.

4. Prediction & Visualization:
  Forecasts AQI for the next 72 hours.

  Predictions and trends displayed on a Streamlit dashboard.

5. CI/CD Automation:
  GitHub Actions triggers pipelines automatically:

    Hourly → fetch_features.py

    Daily → train_model.py

---

## Repository Structure

- `fetch_features.py`: Fetches and processes air quality data, storing features in Hopsworks.  
- `train_model.py`: Trains machine learning models and saves the best model to Hopsworks.  
- `app.py`: Streamlit application for visualizing AQI forecasts and historical data.  
- `Dockerfile`: Defines the Docker image for the Streamlit app.  
- `docker-compose.yml`: Configures the Docker service.  
- `requirements.txt`: Lists Python dependencies.  
- `.github/workflows/fetch_features.yaml`: GitHub Actions workflow for hourly feature updates.  
- `.github/workflows/train_model.yaml`: GitHub Actions workflow for daily model training.

