import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

model = joblib.load("model/isolation_forest.pkl")
scaler = joblib.load("model/scaler.pkl")

today = datetime.now().strftime("%Y%m%d")
file = f"new_data/{today}.csv"
df = pd.read_csv(file)

features = ['MG-LV-MSB_AC_Voltage', 'MG-LV-MSB_Frequency']
df = df.dropna(subset=features + ['Timestamp'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.sort_values('Timestamp')

X = scaler.transform(df[features])
df['score'] = model.decision_function(X)
df['is_anomaly'] = (df['score'] < -0.0454).astype(int)  # ← use tuned threshold

anomalies = df[df['is_anomaly'] == 1]
out_file = f"alerts/detected_anomalies_{today}.csv"
anomalies.to_csv(out_file, index=False)
print(f"Exported {len(anomalies)} anomalies to {out_file}")