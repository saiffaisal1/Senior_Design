import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Training/historical_clean_data.csv")
features = ['MG-LV-MSB_AC_Voltage', 'MG-LV-MSB_Frequency']
df = df.dropna(subset=features)

scaler = StandardScaler()
X = scaler.fit_transform(df[features])

model = IsolationForest(contamination=0.003, random_state=42)
model.fit(X)

joblib.dump(model, "model/isolation_forest.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("Model and scaler saved.")