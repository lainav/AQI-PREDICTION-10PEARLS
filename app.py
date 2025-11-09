# app.py
import streamlit as st
import hopsworks
import joblib
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go # Added for gauge chart
import numpy as np
from datetime import datetime, timedelta # Added timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
# Basic styling (can be expanded significantly)
# --- Styling ---
# --- Styling ---
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9; /* Lighter background */
            padding: 2rem;
        }
        /* Style the container for the list of tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 18px; /* Adjust space between tab labels */
            padding-bottom: 10px; /* Space below the tab labels */
            border-bottom: 1px solid #e1e1e1 !important; /* Add a subtle line below the tabs */
        }
        /* Style individual tab labels (inactive) */
        .stTabs [data-baseweb="tab"] {
            height: auto; white-space: pre-wrap;
            background-color: transparent !important; border: none !important;
            border-radius: 0px !important; box-shadow: none !important;
            padding: 5px 0px !important; margin: 0px 8px !important;
            color: #666666 !important; font-weight: normal !important;
            border-bottom: 3px solid transparent !important; /* Placeholder for alignment */
        }
        /* Style the currently selected tab label */
        .stTabs [aria-selected="true"] {
            background-color: transparent !important; color: #4CAF50 !important;
            font-weight: bold !important; border-bottom: 3px solid #4CAF50 !important;
            padding-bottom: 2px;
        }

        /* --- Style the Streamlit Metric components --- */
        .stMetric {
            /* Remove background, border, and shadow */
            background-color: transparent !important;
            border: none !important;
            border-radius: 0px !important;
            box-shadow: none !important;
            /* Adjust padding (mostly remove horizontal, keep some vertical) */
            padding: 0.2rem 0 !important;
        }
        /* Optional: Style the metric label */
        .stMetric > label[data-testid="stMetricLabel"] {
            color: #555555 !important; /* Dimmer label color */
            font-size: 0.95rem !important; /* Slightly smaller label */
            font-weight: normal; /* Ensure label is not bold if theme makes it so */
        }
        /* Optional: Style the metric value */
        .stMetric > div[data-testid="stMetricValue"] {
             color: #212121 !important; /* Darker value color */
             font-size: 1.3rem !important; /* Larger value text */
             font-weight: 600; /* Semi-bold value */
             padding-top: 0px !important; /* Reduce space between label and value */
        }
        /* Optional: Control spacing between metrics */
         .stMetric:not(:first-child) {
              margin-top: 0.75rem !important; /* Add space above subsequent metrics */
         }
         /* --- End of Metric Styling --- */

        /* Keep other styles */
        .aqi-metric-container { /* This is for the custom AQI display, leave as is */
             text-align: center;
        }
        .streamlit-expanderHeader {
            font-size: 1.1rem; font-weight: bold; background-color: #f0f2f6;
            border-radius: 5px; padding: 0.5rem 1rem; margin-bottom: 5px;
        }
        .streamlit-expanderContent {
             border-left: 3px solid #4CAF50; padding-left: 1rem; margin-top: 0;
        }

    </style>
""", unsafe_allow_html=True)


# --- AQI Calculation & Categorization ---
def calculate_aqi(pm25):
    """Calculates US AQI for PM2.5."""
    if pm25 is None or pd.isna(pm25) or pm25 < 0:
        return None
    breakpoints = [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo)
    if pm25 > 500.4: # Handle case above highest breakpoint
        return 500
    return None # Should not happen if pm25 >= 0

def get_aqi_category(aqi):
    """Returns the AQI category and a suggested color."""
    if aqi is None:
        return "Unknown", "#808080" # Grey
    elif 0 <= aqi <= 50:
        return "Good", "#00e400" # Green
    elif 51 <= aqi <= 100:
        return "Moderate", "#ffff00" # Yellow
    elif 101 <= aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00" # Orange
    elif 151 <= aqi <= 200:
        return "Unhealthy", "#ff0000" # Red
    elif 201 <= aqi <= 300:
        return "Very Unhealthy", "#8f3f97" # Purple
    elif aqi >= 301:
        return "Hazardous", "#7e0023" # Maroon
    else:
        return "Unknown", "#808080"

# --- Data Loading (Cached) ---
@st.cache_resource(show_spinner="Connecting to Hopsworks and loading model...")
def load_model():
    """Loads the trained model from Hopsworks Model Registry."""
    try:
        # Use environment variables or secrets management in production
        HOPSWORKS_API_KEY = "2EpVtPZvfyir2ZHe.Xq5Zf52NZvrcFMazBANKnavDajjwl759POapcm1FijsZhoDFqhKeY2zu331fo82i" 
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        mr = project.get_model_registry()
        # Ensure you specify the correct name and version
        model_obj = mr.get_model("openmeteo_aqi_model", version=1)
        model_dir = model_obj.download()
        model_path = os.path.join(model_dir, "model.joblib")
        if not os.path.exists(model_path):
             st.error(f"Model file not found at expected path: {model_path}")
             return None
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_data(ttl=600, show_spinner="Fetching historical air quality data...") # Cache for 10 minutes
def get_historical_data():
    """Fetches historical data from Hopsworks Feature Store and calculates AQI."""
    try:
        # Use environment variables or secrets management in production
        HOPSWORKS_API_KEY = "2EpVtPZvfyir2ZHe.Xq5Zf52NZvrcFMazBANKnavDajjwl759POapcm1FijsZhoDFqhKeY2zu331fo82i"
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        # Ensure you specify the correct name and version
        fg = fs.get_feature_group("openmeteo_aq_feature_group", version=1)
        df = fg.read()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date')
        # Compute historical AQI
        df['aqi'] = df['pm2_5'].apply(calculate_aqi)
        # Drop rows where AQI calculation failed (optional, but good practice)
        df = df.dropna(subset=['aqi'])
        df['aqi'] = df['aqi'].astype(int) # Convert AQI to integer
        st.success("✅ Historical data fetched successfully!")
        return df
    except Exception as e:
        st.error(f"❌ Error fetching historical data: {str(e)}")
        return pd.DataFrame()

# --- Forecasting Logic ---
def generate_forecast_inputs(forecast_period_days=3, freq='H', df_hist=None):
    """Generates plausible input features for forecasting based on historical data."""
    if df_hist is None or df_hist.empty:
        st.warning("⚠️ Cannot generate forecast inputs without historical data.")
        return pd.DataFrame()

    # Use recent data (e.g., last 30 days) for more relevant averages/stddevs
    recent_hist = df_hist[df_hist['date'] > (pd.Timestamp.utcnow() - pd.Timedelta(days=30))]
    if recent_hist.empty:
        recent_hist = df_hist # Fallback to all data if no recent data

    # Calculate mean and std dev for simulation - consider seasonality if important
    avg_vals = {
        'carbon_monoxide': recent_hist['carbon_monoxide'].mean(),
        'carbon_dioxide': recent_hist['carbon_dioxide'].mean(),
        'nitrogen_dioxide': recent_hist['nitrogen_dioxide'].mean(),
        'sulphur_dioxide': recent_hist['sulphur_dioxide'].mean(),
        'pm2_5_change_rate': recent_hist['pm2_5_change_rate'].mean(),
    }
    # Use a fraction of std dev for noise, ensure non-negative std dev
    std_vals = {k: max(recent_hist[k].std() * 0.15, 0) for k in avg_vals} # Reduced noise factor

    # Determine forecast start time (next full hour in UTC)
    # Note: The original API call used UTCNow(). Ensure consistency or timezone awareness if needed.
    # Let's assume the feature store data 'date' column is timezone-aware (e.g., UTC)
    # If not, you might need to localize it first.
    last_hist_time = df_hist['date'].max()
    if pd.isna(last_hist_time): # Handle case where historical data might be empty or have NaT dates
         forecast_start = pd.Timestamp.utcnow().tz_localize(None).replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
    else:
         # Start forecast from the hour *after* the last historical data point
         forecast_start = (last_hist_time.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1))
         # Ensure forecast start is not in the past relative to current time (optional, depends on use case)
         forecast_start = max(forecast_start, pd.Timestamp.utcnow().tz_convert(last_hist_time.tz).replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1))


    forecast_end = forecast_start + pd.Timedelta(days=forecast_period_days)
    # Generate date range, ensure it's timezone-aware if start time is
    forecast_dates = pd.date_range(start=forecast_start, end=forecast_end - pd.Timedelta(hours=1), freq=freq, tz=forecast_start.tz)
    n = len(forecast_dates)
    if n == 0:
        st.warning("⚠️ Forecast period resulted in zero timestamps. Check start/end dates.")
        return pd.DataFrame()

    # Generate synthetic feature data
    # Make generated values non-negative where applicable (e.g., concentrations)
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'day': forecast_dates.day,
        'month': forecast_dates.month,
        'carbon_monoxide': np.maximum(0, np.random.normal(avg_vals['carbon_monoxide'], std_vals['carbon_monoxide'], n)),
        'carbon_dioxide': np.maximum(0, np.random.normal(avg_vals['carbon_dioxide'], std_vals['carbon_dioxide'], n)),
        'nitrogen_dioxide': np.maximum(0, np.random.normal(avg_vals['nitrogen_dioxide'], std_vals['nitrogen_dioxide'], n)),
        'sulphur_dioxide': np.maximum(0, np.random.normal(avg_vals['sulphur_dioxide'], std_vals['sulphur_dioxide'], n)),
        # pm2_5_change_rate can be negative
        'pm2_5_change_rate': np.random.normal(avg_vals['pm2_5_change_rate'], std_vals['pm2_5_change_rate'], n),
    })
    return forecast_df.round(6)

# --- Main Application ---
def main():
    # --- Sidebar ---
    with st.sidebar:
        st.title("⚙️ Dashboard Settings")
        st.markdown("---")

        # Load data and model first, handle errors gracefully
        model = load_model()
        df_hist = get_historical_data()

        if model is None or df_hist.empty:
            st.sidebar.error("❌ Cannot proceed without model and historical data. Check logs.")
            # Optionally, stop the app execution if critical data is missing
            st.stop()


        # Get the most recent historical data point
        latest_data = df_hist.iloc[-1] if not df_hist.empty else None

        st.subheader("📍 Location")
        st.info("Karachi, Pakistan (Lat: 24.86, Lon: 67.00)") # As per fetch_features.py

        st.subheader("Theme")
        theme = st.selectbox("Select Plotly Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"])

        st.markdown("---")
        st.subheader("ℹ️ About")
        st.info(
            "This dashboard visualizes historical air quality data and provides AQI forecasts "
            "based on a machine learning model trained on Open-Meteo data for Karachi."
            "\n\n**Data Source:** Open-Meteo Air Quality API"
            "\n**Model:** Loaded from Hopsworks"
        )


    # --- Main Content Area ---
    st.title("🌬️ Karachi Air Quality Dashboard")

    # Define the features the model expects (ensure order matches training)
    features = ['day', 'month', 'pm2_5_change_rate', 'carbon_monoxide', 'carbon_dioxide', 'nitrogen_dioxide', 'sulphur_dioxide']

    # --- Create Tabs ---
    tab1, tab2, tab3 = st.tabs(["📊 Current & Forecast", "⏳ Historical Analysis", "💾 Data Explorer"])

    # --- Tab 1: Current & Forecast ---
    with tab1:
        st.header("Current Conditions & AQI Forecast")

        col1, col2 = st.columns([1, 2]) # Adjust ratio as needed

        with col1:
            st.subheader("🌡️ Latest Reading")
            if latest_data is not None:
                 current_aqi = latest_data['aqi']
                 current_pm25 = latest_data['pm2_5']
                 aqi_category, aqi_color = get_aqi_category(current_aqi)

                 st.markdown(f"""
                 <div class="aqi-metric-container" style="background-color: {aqi_color}; padding: 15px; border-radius: 10px; color: {'black' if aqi_category=='Moderate' else 'white'};">
                     <p style="font-size: 1.1rem; margin-bottom: 5px;">Current AQI</p>
                     <p style="font-size: 3rem; font-weight: bold; margin-bottom: 5px;">{current_aqi}</p>
                     <p style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0;">{aqi_category}</p>
                 </div>
                 """, unsafe_allow_html=True)
                 st.metric(label="PM2.5 (µg/m³)", value=f"{current_pm25:.2f}")
                 st.metric(label="Timestamp", value=latest_data['date'].strftime('%Y-%m-%d %H:%M'))

            else:
                st.warning("No recent data available to display current conditions.")

            st.subheader("📊 AQI Gauge")
            if latest_data is not None:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = current_aqi,
                    title = {'text': "Latest AQI"},
                    gauge = {
                        'axis': {'range': [0, 500]},
                        'bar': {'color': aqi_color},
                        'steps' : [
                            {'range': [0, 50], 'color': "#00e400"},
                            {'range': [50, 100], 'color': "#ffff00"},
                            {'range': [100, 150], 'color': "#ff7e00"},
                            {'range': [150, 200], 'color': "#ff0000"},
                            {'range': [200, 300], 'color': "#8f3f97"},
                            {'range': [300, 500], 'color': "#7e0023"}],
                        'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': current_aqi}
                    }))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), template=theme)
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                 st.info("Gauge requires current AQI data.")


        with col2:
            st.subheader("🔮 AQI Forecast (Next 72 Hours)")
            # Generate inputs for 3 days forecast
            df_forecast_inputs = generate_forecast_inputs(forecast_period_days=3, freq='H', df_hist=df_hist)

            if not df_forecast_inputs.empty:
                # Predict AQI using the loaded model
                predicted_aqi = model.predict(df_forecast_inputs[features])
                # Ensure predictions are non-negative integers
                predicted_aqi = np.maximum(0, predicted_aqi).round().astype(int)
                df_forecast_inputs['aqi_prediction'] = predicted_aqi

                # Create forecast plot
                fig_forecast = px.line(
                    df_forecast_inputs,
                    x='date', y='aqi_prediction',
                    title='Predicted AQI Trend',
                    labels={'date': 'Time', 'aqi_prediction': 'Predicted AQI'},
                    template=theme
                    )
                fig_forecast.update_traces(line=dict(color='royalblue', width=2))
                 # Add AQI category color ranges
                fig_forecast.add_hrect(y0=0, y1=50, line_width=0, fillcolor="green", opacity=0.1, layer="below")
                fig_forecast.add_hrect(y0=51, y1=100, line_width=0, fillcolor="yellow", opacity=0.1, layer="below")
                fig_forecast.add_hrect(y0=101, y1=150, line_width=0, fillcolor="orange", opacity=0.1, layer="below")
                fig_forecast.add_hrect(y0=151, y1=200, line_width=0, fillcolor="red", opacity=0.1, layer="below")
                fig_forecast.add_hrect(y0=201, y1=300, line_width=0, fillcolor="purple", opacity=0.1, layer="below")
                fig_forecast.add_hrect(y0=301, y1=500, line_width=0, fillcolor="maroon", opacity=0.1, layer="below")

                st.plotly_chart(fig_forecast, use_container_width=True)

                with st.expander("View Detailed Forecast Data & Download"):
                    # Select and rename columns for clarity
                    display_cols = ['date', 'aqi_prediction'] + features
                    df_display_forecast = df_forecast_inputs[display_cols].copy()
                    df_display_forecast.rename(columns={'aqi_prediction': 'Predicted AQI'}, inplace=True)

                    st.dataframe(df_display_forecast.style.background_gradient(
                        cmap='viridis', subset=['Predicted AQI'] # Or use RdYlGn_r for red-yellow-green
                        ))

                    csv_forecast = df_display_forecast.to_csv(index=False).encode('utf-8')
                    st.download_button(
                       label="⬇️ Download Forecast CSV",
                       data=csv_forecast,
                       file_name=f"aqi_forecast_karachi_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                       mime="text/csv",
                    )
            else:
                st.warning("Could not generate forecast data.")

        # Add forecast charts for individual pollutants
        if not df_forecast_inputs.empty:
            st.subheader("🧪 Predicted Pollutant Levels (Next 72 Hours)")
            pollutants_to_plot = ['carbon_monoxide', 'carbon_dioxide', 'nitrogen_dioxide', 'sulphur_dioxide']
            fig_pollutants = px.line(
                df_forecast_inputs,
                x='date', y=pollutants_to_plot,
                title='Predicted Pollutant Concentrations',
                labels={'date': 'Time', 'value': 'Concentration', 'variable': 'Pollutant'},
                template=theme
            )
            st.plotly_chart(fig_pollutants, use_container_width=True)

    # --- Tab 2: Historical Analysis ---
    with tab2:
        st.header("Historical Data Analysis")

        if not df_hist.empty:
            # Date Range Selector for Historical Data
            min_date = df_hist['date'].min().date()
            max_date = df_hist['date'].max().date()

            col1_hist, col2_hist = st.columns(2)
            with col1_hist:
                 start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
            with col2_hist:
                 end_date = st.date_input("End date", max_date, min_value=start_date, max_value=max_date)

            # Filter data based on selection (inclusive of end date)
            if start_date <= end_date:
                mask = (df_hist['date'].dt.date >= start_date) & (df_hist['date'].dt.date <= end_date)
                df_filtered_hist = df_hist.loc[mask]

                if not df_filtered_hist.empty:
                    st.subheader(f"Analysis from {start_date} to {end_date}")

                    # Plot Historical AQI
                    st.markdown("##### Historical AQI Trend")
                    fig_hist_aqi = px.line(
                        df_filtered_hist, x='date', y='aqi',
                        title='Hourly Historical AQI', template=theme,
                        labels={'aqi': 'Historical AQI'}
                    )
                    # Add category colors like in forecast
                    fig_hist_aqi.add_hrect(y0=0, y1=50, line_width=0, fillcolor="green", opacity=0.1, layer="below")
                    fig_hist_aqi.add_hrect(y0=51, y1=100, line_width=0, fillcolor="yellow", opacity=0.1, layer="below")
                    fig_hist_aqi.add_hrect(y0=101, y1=150, line_width=0, fillcolor="orange", opacity=0.1, layer="below")
                    fig_hist_aqi.add_hrect(y0=151, y1=200, line_width=0, fillcolor="red", opacity=0.1, layer="below")
                    fig_hist_aqi.add_hrect(y0=201, y1=300, line_width=0, fillcolor="purple", opacity=0.1, layer="below")
                    fig_hist_aqi.add_hrect(y0=301, y1=500, line_width=0, fillcolor="maroon", opacity=0.1, layer="below")

                    st.plotly_chart(fig_hist_aqi, use_container_width=True)


                    # Plot Historical Pollutants
                    st.markdown("##### Historical Pollutant Trends")
                    pollutants = ['pm2_5', 'carbon_monoxide', 'carbon_dioxide', 'nitrogen_dioxide', 'sulphur_dioxide']
                    # Ensure pollutants exist in the dataframe before plotting
                    pollutants_present = [p for p in pollutants if p in df_filtered_hist.columns]
                    if pollutants_present:
                         fig_hist_pollutants = px.line(
                              df_filtered_hist, x='date', y=pollutants_present,
                              title='Hourly Historical Pollutant Levels',
                              labels={'value': 'Concentration', 'variable': 'Pollutant'},
                              template=theme
                         )
                         st.plotly_chart(fig_hist_pollutants, use_container_width=True)
                    else:
                         st.warning("No pollutant data columns found for plotting.")

                    # Distribution Analysis
                    col1_dist, col2_dist = st.columns(2)
                    with col1_dist:
                         st.markdown("##### AQI Category Distribution")
                         df_filtered_hist['aqi_category'] = df_filtered_hist['aqi'].apply(lambda x: get_aqi_category(x)[0])
                         aqi_counts = df_filtered_hist['aqi_category'].value_counts().reset_index()
                         aqi_counts.columns = ['Category', 'Hours']
                          # Define explicit order and colors for categories
                         category_order = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous", "Unknown"]
                         category_colors = {cat: get_aqi_category(idx*75)[1] for idx, cat in enumerate(category_order)} # Map category to color
                         category_colors["Unknown"] = "#808080"
                         category_colors["Unhealthy for Sensitive Groups"] = "#ff7e00" # Ensure correct mapping

                         fig_aqi_dist = px.bar(
                              aqi_counts, x='Category', y='Hours',
                              title='Time Spent in Each AQI Category',
                              color='Category', color_discrete_map=category_colors,
                              category_orders={"Category": category_order}, # Enforce order
                              template=theme
                         )
                         fig_aqi_dist.update_layout(xaxis_title=None) # Remove redundant x-axis label
                         st.plotly_chart(fig_aqi_dist, use_container_width=True)

                    with col2_dist:
                         st.markdown("##### Average Pollutant Levels")
                         avg_pollutants = df_filtered_hist[pollutants_present].mean().reset_index()
                         avg_pollutants.columns = ['Pollutant', 'Average Concentration']
                         fig_avg_poll = px.bar(
                              avg_pollutants, x='Pollutant', y='Average Concentration',
                              title='Average Pollutant Concentration',
                              color='Pollutant', template=theme
                         )
                         st.plotly_chart(fig_avg_poll, use_container_width=True)

                else:
                    st.warning("No historical data available for the selected date range.")
            else:
                st.error("End date must be after or the same as start date.")
        else:
            st.error("Historical data is not available for analysis.")

    # --- Tab 3: Data Explorer ---
    with tab3:
        st.header("Explore Historical Data")
        st.write("Browse the raw historical data used in this dashboard.")

        if not df_hist.empty:
            st.dataframe(df_hist, use_container_width=True)

            csv_hist = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button(
               label="⬇️ Download Full Historical Data CSV",
               data=csv_hist,
               file_name=f"historical_aq_karachi_{min_date}_to_{max_date}.csv",
               mime="text/csv",
               key="download-historical" # Unique key for the button
            )
        else:
            st.warning("No historical data loaded to explore.")


if __name__ == "__main__":
    # Consider adding basic error handling around main() if needed
    main()