import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
import numpy as np

st.set_page_config(page_title="Traffic Prediction", page_icon="🚗", layout="centered")

st.title("🚗 Traffic Demand Predictor")

# Paths
repo_root = Path(__file__).resolve().parent
model_path = repo_root / "models" / "model.pkl"
metrics_path = repo_root / "metrics" / "metrics.json"
test_path = repo_root / "data" / "features" / "test.csv"

# Load model
@st.cache_resource
def load_model():
    if not model_path.exists():
        st.error("❌ Model not found. Run: python run_pipeline.py")
        return None
    return joblib.load(model_path)

model = load_model()
if model is None:
    st.stop()

# Load metrics
metrics = {}
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)

# Load test data for feature names
test_df = pd.read_csv(test_path) if test_path.exists() else None

if test_df is None:
    st.error("❌ Test data not found. Run: python run_pipeline.py")
    st.stop()

# Get feature names (exclude target and components)
feature_names = [col for col in test_df.columns if col not in ['cnt', 'casual', 'registered']]

# Display metrics
if metrics:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("RMSE", f"{metrics['rmse']:.2f}")
    with col2:
        st.metric("R² Score", f"{metrics['r2']:.4f}")

st.divider()
st.subheader("📊 Enter Conditions")

# Create input columns
col1, col2 = st.columns(2)

with col1:
    season = st.select_slider("Season", [1, 2, 3, 4], value=2, help="1=Spring, 2=Summer, 3=Fall, 4=Winter")
    mnth = st.slider("Month", 1, 12, 6)
    temp = st.slider("Temperature (0-1)", 0.0, 1.0, 0.5)
    hum = st.slider("Humidity (0-1)", 0.0, 1.0, 0.5)

with col2:
    yr = st.radio("Year", [0, 1], format_func=lambda x: f"Year {x}")
    weekday = st.select_slider("Weekday", [0, 1, 2, 3, 4, 5, 6], value=2, help="0=Sunday...6=Saturday")
    atemp = st.slider("Feels-like Temp (0-1)", 0.0, 1.0, 0.5)
    windspeed = st.slider("Wind Speed (0-1)", 0.0, 1.0, 0.2)

# Additional options
holiday = st.checkbox("Is Holiday?")
workingday = st.checkbox("Is Working Day?", value=True)
weathersit = st.select_slider("Weather", [1, 2, 3, 4], value=1, help="1=Clear, 2=Mist, 3=Light Rain, 4=Heavy")

# Predict button
if st.button("🔮 Predict Demand", use_container_width=True):
    try:
        # Build input dict with all features
        input_dict = {
            'season': season,
            'yr': float(yr),
            'mnth': mnth,
            'holiday': float(holiday),
            'weekday': weekday,
            'workingday': float(workingday),
            'weathersit': weathersit,
            'temp': temp,
            'atemp': atemp,
            'hum': hum,
            'windspeed': windspeed
        }
        
        # Create input array in correct feature order
        X = np.array([input_dict[col] for col in feature_names]).reshape(1, -1)
        
        # Make prediction
        pred = model.predict(X)[0]
        
        st.success(f"### 🎯 Predicted Demand: **{int(pred):,} bikes**")
        st.info(f"Estimated range: {max(0, int(pred-100)):,} - {int(pred+100):,} bikes")
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.write(f"Features expected: {feature_names}")

st.divider()
st.caption("🔗 [Transportation Repository](https://github.com/aayushagarwaltech-bot/Transportation)")
