#fetch_features.py

import os
import requests_cache
import pandas as pd
from retry_requests import retry
import hopsworks
import openmeteo_requests
from datetime import datetime

# Example function to fetch data from Open-Meteo
def fetch_openmeteo_data():
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "hourly": [
            "pm2_5",
            "carbon_monoxide",
            "carbon_dioxide",
            "nitrogen_dioxide",
            "sulphur_dioxide"
        ],
        "start_date": "2015-01-01",
        "end_date": datetime.utcnow().strftime("%Y-%m-%d")
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    pm2_5 = hourly.Variables(0).ValuesAsNumpy()
    carbon_monoxide = hourly.Variables(1).ValuesAsNumpy()
    carbon_dioxide = hourly.Variables(2).ValuesAsNumpy()
    nitrogen_dioxide = hourly.Variables(3).ValuesAsNumpy()
    sulphur_dioxide = hourly.Variables(4).ValuesAsNumpy()

    date_range = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    df = pd.DataFrame({
        "date": date_range,
        "pm2_5": pm2_5,
        "carbon_monoxide": carbon_monoxide,
        "carbon_dioxide": carbon_dioxide,
        "nitrogen_dioxide": nitrogen_dioxide,
        "sulphur_dioxide": sulphur_dioxide
    })
    return df

def process_openmeteo_data(df: pd.DataFrame) -> pd.DataFrame:
    # Convert 'date' to datetime if not already
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    # Create day and month features
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month

    # Compute pm2_5_change_rate
    df['pm2_5_change_rate'] = df['pm2_5'].diff().fillna(0)

    return df

def store_features_in_hopsworks(df: pd.DataFrame):
    # Login to Hopsworks
    # project = hopsworks.login(project="TenPearls82")
    project = hopsworks.login(api_key_value= "djdhFrqCIxvo6ApQ.xERPM8BgPzFrbYJVj0uf2EavQ8PPSbp9nIytq13qINXxpjhNHPwU9XYRZEJ1hBUc")
    fs = project.get_feature_store()

    # Create or get the feature group
    fg = fs.get_or_create_feature_group(
        name="openmeteo_aq_feature_group",
        version=1,
        primary_key=["date"],  # or any unique key
        description="Hourly Open-Meteo air quality data"
    )

    # Insert data
    fg.insert(df, write_options={"wait_for_job": True})

def main():
    # 1. Fetch data
    df = fetch_openmeteo_data()
    # 2. Process data
    df_features = process_openmeteo_data(df)
    # 3. Store in Hopsworks
    if not df_features.empty:
        store_features_in_hopsworks(df_features)
        print("Features stored successfully!")
    else:
        print("No data to store.")

if __name__ == "__main__":
    main()
