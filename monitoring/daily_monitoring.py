import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

model = joblib.load("model/isolation_forest.pkl")
scaler = joblib.load("model/scaler.pkl")

data_dir = "Data/"
output_dir = "alerts/"
os.makedirs(output_dir, exist_ok=True)

features = ['MG-LV-MSB_AC_Voltage', 'MG-LV-MSB_Frequency']
threshold = -0.0454  # Tuned threshold

csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

for filename in csv_files:
    filepath = os.path.join(data_dir, filename)
    try:
        df = pd.read_csv(filepath)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=features + ['Timestamp']).sort_values('Timestamp')

        X = scaler.transform(df[features])
        df['score'] = model.decision_function(X)
        df['is_anomaly'] = (df['score'] < threshold).astype(int)

        score_range = df['score'].max() - df['score'].min()
        if score_range == 0:
            df['confidence'] = 0.0
        else:
            scaler_conf = MinMaxScaler(feature_range=(0, 1))
            df['confidence'] = 1 - scaler_conf.fit_transform(df[['score']])

        anomalies = df[df['is_anomaly'] == 1]

        out_file = os.path.join(output_dir, f"detected_anomalies_{filename}")
        anomalies.to_csv(out_file, index=False)

        print(f"{filename}: {len(anomalies)} anomalies saved to {out_file}")
    except Exception as e:
        print(f"Failed to process {filename}: {e}")