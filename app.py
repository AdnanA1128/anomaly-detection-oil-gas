import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Oil & Gas Anomaly Detection", layout="wide")

# Title
st.title("Oil & Gas Anomaly Detection Dashboard")

# Description
st.write("This system detects abnormal behavior in industrial sensor data using machine learning.")

# Load data
df = pd.read_csv("sensor_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

sensor_columns = ["pressure", "temperature", "flow_rate", "vibration"]

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[sensor_columns])

# Isolation Forest model
model = IsolationForest(contamination=0.035, random_state=42)
df["anomaly"] = model.fit_predict(scaled_data)

# Convert (-1 = anomaly)
df["anomaly"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)

# Metrics
total = len(df)
anomalies_count = df["anomaly"].sum()

col1, col2 = st.columns(2)
col1.metric("Total Data Points", total)
col2.metric("Detected Anomalies", anomalies_count)

# Sensor selection
sensor = st.selectbox("Select Sensor", sensor_columns)

# Plot with timestamp (FIXED)
fig = px.line(df, x="timestamp", y=sensor, title=f"{sensor} Over Time")

# Highlight anomalies
anomalies = df[df["anomaly"] == 1]

fig.add_scatter(
    x=anomalies["timestamp"],
    y=anomalies[sensor],
    mode="markers",
    name="Anomalies"
)

st.plotly_chart(fig, use_container_width=True)

# Show anomaly table
st.subheader("Detected Anomalies")
st.dataframe(anomalies.head(20))
