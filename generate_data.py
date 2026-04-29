import numpy as np
import pandas as pd

np.random.seed(42)

timestamps = pd.date_range(start="2026-01-01", periods=1000, freq="min")

pressure = np.random.normal(75, 4, 1000)
temperature = np.random.normal(180, 8, 1000)
flow_rate = np.random.normal(500, 30, 1000)
vibration = np.random.normal(2.5, 0.4, 1000)

# create anomalies
anomaly_indices = np.random.choice(range(1000), size=35, replace=False)

pressure[anomaly_indices] += 25
temperature[anomaly_indices] += 40
flow_rate[anomaly_indices] -= 150
vibration[anomaly_indices] += 3

df = pd.DataFrame({
    "timestamp": timestamps,
    "pressure": pressure,
    "temperature": temperature,
    "flow_rate": flow_rate,
    "vibration": vibration
})

df["actual_anomaly"] = 0
df.loc[anomaly_indices, "actual_anomaly"] = 1

df.to_csv("sensor_data.csv", index=False)

print("✅ sensor_data.csv created successfully")
print(df.head())
