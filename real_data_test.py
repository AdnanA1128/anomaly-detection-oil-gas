import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load real dataset
df = pd.read_csv("ai4i2020.csv")

# Drop non-numeric columns
df = df.drop(columns=["UDI", "Product ID", "Type"], errors="ignore")

# Drop target/failure labels if present
df = df.drop(columns=["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"], errors="ignore")

# Drop missing values
df = df.dropna()

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
df["anomaly"] = model.fit_predict(scaled_data)

# Convert (-1 → anomaly)
df["anomaly"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)

# Results
print("Total rows:", len(df))
print("Detected anomalies:", df["anomaly"].sum())

print(df.head())
