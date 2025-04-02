import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import plotly.express as px

clean = pd.read_csv("Data/Apr_2023.csv")
clean['Timestamp'] = pd.to_datetime(clean['Timestamp'], errors='coerce')
clean = clean.dropna(subset=['Timestamp']).sort_values('Timestamp')

features = [
    'MG-LV-MSB_AC_Voltage',
    'Receiving_Point_AC_Voltage',
    'Island_mode_MCCB_AC_Voltage',
    'Island_mode_MCCB_Frequency',
    'MG-LV-MSB_Frequency'
]
clean = clean.dropna(subset=features)

scaler = StandardScaler()
X_train = scaler.fit_transform(clean[features])

iso = IsolationForest(contamination=0.003, random_state=42)
svm = OneClassSVM(nu=0.003, kernel="rbf", gamma="scale")
lof = LocalOutlierFactor(n_neighbors=35, contamination=0.003, novelty=True)

iso.fit(X_train)
svm.fit(X_train)
lof.fit(X_train)

test = pd.read_csv("Data/Apr_2023_with_anomalies.csv")
test['Timestamp'] = pd.to_datetime(test['Timestamp'], errors='coerce')
test = test.dropna(subset=['Timestamp']).sort_values('Timestamp')
test = test.dropna(subset=features)
X_test = scaler.transform(test[features])

iso_pred = iso.predict(X_test)
svm_pred = svm.predict(X_test)
lof_pred = lof.predict(X_test)

votes = np.stack([iso_pred, lof_pred, svm_pred], axis=1)
test['ensemble_votes'] = np.sum(votes == -1, axis=1)
test['is_ensemble_anomaly'] = (test['ensemble_votes'] >= 2).astype(int)

test['iso_score'] = iso.decision_function(X_test)
test['svm_score'] = svm.decision_function(X_test)
try:
    test['lof_score'] = np.clip(lof.decision_function(X_test), -10, 10)
except Exception as e:
    print("LOF scoring error:", e)
    test['lof_score'] = np.nan

y_true = test['label']
y_pred = test['is_ensemble_anomaly']

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Evaluation Results")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

fig = px.scatter(
    test, x='Timestamp', y='MG-LV-MSB_AC_Voltage',
    color=test['is_ensemble_anomaly'].map({1: "Predicted Anomaly", 0: "Normal"}),
    symbol=test['label'].map({1: "True Anomaly", 0: "Not Injected"}),
    title="Anomaly Detection vs Ground Truth (Voltage)",
    hover_data=['iso_score', 'svm_score', 'lof_score', 'ensemble_votes']
)
fig.show()
