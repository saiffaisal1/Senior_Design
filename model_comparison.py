import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np

# === Load and preprocess ===
df = pd.read_csv("Data/Apr_2023.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])
df.sort_values('Timestamp', inplace=True)

features = [
    'MG-LV-MSB_AC_Voltage',
    'Receiving_Point_AC_Voltage',
    'Island_mode_MCCB_AC_Voltage',
    'Island_mode_MCCB_Frequency',
    'MG-LV-MSB_Frequency'
]
data = df[['Timestamp'] + features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

# === Initialize models ===
iso = IsolationForest(contamination=0.003, random_state=42)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.003, novelty=True)
svm = OneClassSVM(nu=0.003, kernel="rbf", gamma="scale")

# === Fit models ===
iso.fit(X_scaled)
lof.fit(X_scaled)
svm.fit(X_scaled)

# === Predict (1=normal, -1=anomaly) ===
iso_pred = iso.predict(X_scaled)
lof_pred = lof.predict(X_scaled)
svm_pred = svm.predict(X_scaled)

# === Ensemble Voting ===
votes = np.stack([iso_pred, lof_pred, svm_pred], axis=1)
data['ensemble_votes'] = np.sum(votes == -1, axis=1)  # Count how many models say "anomaly"
data['is_ensemble_anomaly'] = data['ensemble_votes'] >= 2  # majority vote

# Optional: Store individual model predictions too
data['iso'] = iso_pred
data['lof'] = lof_pred
data['svm'] = svm_pred

# === Show top ensemble anomalies ===
ensemble_anomalies = data[data['is_ensemble_anomaly']]
print(f"Ensemble detected {len(ensemble_anomalies)} anomalies")
print(ensemble_anomalies[['Timestamp', 'ensemble_votes'] + features].head())