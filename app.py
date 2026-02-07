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

# Display metrics
if metrics:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("RMSE", f"{metrics['rmse']:.2f}")
    with col2:
        st.metric("R² Score", f"{metrics['r2']:.4f}")

st.divider()

# Simple input form
st.subheader("📊 Enter Conditions")

col1, col2 = st.columns(2)
with col1:
    season = st.select_slider("Season", [1, 2, 3, 4], value=2, help="1=Spring, 2=Summer, 3=Fall, 4=Winter")
    temp = st.slider("Temperature", 0.0, 1.0, 0.5)
    hum = st.slider("Humidity", 0.0, 1.0, 0.5)

with col2:
    mnth = st.slider("Month", 1, 12, 6)
    windspeed = st.slider("Wind Speed", 0.0, 1.0, 0.2)
    weathersit = st.select_slider("Weather", [1, 2, 3, 4], value=1, help="1=Clear, 2=Mist, 3=Light Rain, 4=Heavy")

# Predict
if st.button("🔮 Predict", use_container_width=True):
    if test_df is not None:
        feature_names = test_df.drop(columns=['cnt']).columns.tolist()
        
        # Build input
        input_dict = {col: 0 for col in feature_names}
        input_dict['season'] = season
        input_dict['mnth'] = mnth
        input_dict['temp'] = temp
        input_dict['hum'] = hum
        input_dict['windspeed'] = windspeed
        input_dict['weathersit'] = weathersit
        
        X = np.array([input_dict[col] for col in feature_names]).reshape(1, -1)
        pred = model.predict(X)[0]
        
        st.success(f"### 🎯 Predicted Demand: **{int(pred)} bikes**")
        st.info(f"Estimated range: {max(0, int(pred-50))} - {int(pred+50)} bikes")

st.divider()
st.caption("🔗 Repository: [Transportation](https://github.com/aayushagarwaltech-bot/Transportation)")
