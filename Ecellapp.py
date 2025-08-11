import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="AI Predictive Maintenance",
    layout="wide",
    page_icon="🔧"
)

# ------------------------------
# Session state setup
# ------------------------------
if "df_all" not in st.session_state:
    st.session_state.df_all = pd.DataFrame()
if "running" not in st.session_state:
    st.session_state.running = False

# ------------------------------
# Sensor Data Simulation
# ------------------------------
def generate_sensor_data(n=1, start_time=0):
    time_steps = np.arange(start_time, start_time + n)
    temperature = 60 + np.random.normal(0, 2, n) + (time_steps * 0.05)
    vibration = 0.3 + np.random.normal(0, 0.02, n) + (time_steps * 0.001)
    pressure = 30 + np.random.normal(0, 1, n) - (time_steps * 0.01)
    return pd.DataFrame({
        "time": time_steps,
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure
    })

# ------------------------------
# Train model
# ------------------------------
@st.cache_resource
def train_model():
    df = generate_sensor_data(200)
    df["RUL"] = 200 - df["time"] - (df["temperature"] - 60) - (df["vibration"] * 50)
    X = df[["temperature", "vibration", "pressure"]]
    y = df["RUL"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# ------------------------------
# Anomaly detection
# ------------------------------
def detect_anomalies(row):
    alerts = []
    if row["temperature"] > 80:
        alerts.append("⚠ High Temperature")
    if row["vibration"] > 0.5:
        alerts.append("⚠ High Vibration")
    if row["pressure"] < 20:
        alerts.append("⚠ Low Pressure")
    return alerts

# ------------------------------
# UI Header
# ------------------------------
st.markdown(
    "<h1 style='text-align:center;color:#1f77b4;'>🔧 AI-Powered Predictive Maintenance Dashboard</h1>",
    unsafe_allow_html=True
)
st.write("Simulated industrial sensor data with AI-predicted **Remaining Useful Life (RUL)**.")

# ------------------------------
# Controls
# ------------------------------
col1, col2 = st.columns(2)
if col1.button("▶ Start Simulation"):
    st.session_state.running = True
if col2.button("⏹ Stop Simulation"):
    st.session_state.running = False

# ------------------------------
# Live Update Section
# ------------------------------
placeholder = st.empty()

if st.session_state.running:
    for _ in range(10):  # updates per click
        start_time = len(st.session_state.df_all)
        df_live = generate_sensor_data(1, start_time=start_time)
        st.session_state.df_all = pd.concat(
            [st.session_state.df_all, df_live], ignore_index=True
        )

        latest_row = st.session_state.df_all.iloc[-1]
        X_live = latest_row[["temperature", "vibration", "pressure"]].values.reshape(1, -1)
        rul_pred = model.predict(X_live)[0]
        anomalies = detect_anomalies(latest_row)

        with placeholder.container():
            top1, top2, top3 = st.columns([2, 2, 1])
            top1.metric("Predicted Remaining Useful Life (RUL)", f"{rul_pred:.1f} hours")

            if anomalies:
                top2.error(" | ".join(anomalies))
            else:
                top2.success("✅ All parameters within safe limits")

            top3.write("Latest Readings:")
            top3.json(latest_row.to_dict())

            fig = px.line(
                st.session_state.df_all,
                x="time",
                y=["temperature", "vibration", "pressure"],
                title="📊 Sensor Data Over Time"
            )
            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Time (s)",
                yaxis_title="Sensor Reading",
                legend_title="Variable"
            )
            st.plotly_chart(fig, use_container_width=True)

        time.sleep(1)  # 1-second delay

# ------------------------------
# Download Button
# ------------------------------
if not st.session_state.df_all.empty:
    csv_data = st.session_state.df_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Sensor Data (CSV)",
        data=csv_data,
        file_name="sensor_data.csv",
        mime="text/csv"
    )


uploaded_file = st.sidebar.file_uploader("C:/Users/YUVA/OneDrive/Documents/Ecell/sensor_data.csv", type=["csv"])
if uploaded_file:
    df_live = pd.read_csv(uploaded_file)
else:
    df_live = generate_sensor_data(1, start_time=len(st.session_state.df_all))


from scipy.fft import fft, fftfreq

def compute_fft(signal, sampling_rate=100):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sampling_rate)
    return xf[:N//2], np.abs(yf[:N//2])

if not st.session_state.df_all.empty:
    xf, yf = compute_fft(st.session_state.df_all["vibration"])
    fig_fft = px.line(x=xf, y=yf, title="Vibration Frequency Spectrum (FFT)")
    st.plotly_chart(fig_fft, use_container_width=True)


import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense


def train_lstm_model(data, lookback=10):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    X = np.array(X)
    y = np.array(y)
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(lookback, X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, verbose=0)
    return model


from streamlit_autorefresh import st_autorefresh

if st.session_state.running:
    st_autorefresh(interval=2000, key="refresh")
    start_time = len(st.session_state.df_all)
    new_data = generate_sensor_data(1, start_time=start_time)
    st.session_state.df_all = pd.concat([st.session_state.df_all, new_data], ignore_index=True)
else:
    st.write("Simulation stopped. Press ▶ Start Simulation to resume.")
