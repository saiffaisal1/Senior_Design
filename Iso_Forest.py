import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np

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

iso = IsolationForest(contamination=0.003, random_state=42)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.003, novelty=True)
svm = OneClassSVM(nu=0.003, kernel="rbf", gamma="scale")

iso.fit(X_scaled)
lof.fit(X_scaled)
svm.fit(X_scaled)

iso_pred = iso.predict(X_scaled)
lof_pred = lof.predict(X_scaled)
svm_pred = svm.predict(X_scaled)

votes = np.stack([iso_pred, lof_pred, svm_pred], axis=1)
data['ensemble_votes'] = np.sum(votes == -1, axis=1)
data['is_ensemble_anomaly'] = data['ensemble_votes'] >= 2 

data['iso'] = iso_pred
data['lof'] = lof_pred
data['svm'] = svm_pred

voltage_anomalies = data[data['is_ensemble_anomaly'] & data['MG-LV-MSB_AC_Voltage'].notnull()]
frequency_anomalies = data[data['is_ensemble_anomaly'] & data['MG-LV-MSB_Frequency'].notnull()]

print(f"Ensemble detected {len(data[data['is_ensemble_anomaly']])} anomalies")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=("Voltage Anomalies", "Frequency Anomalies"))

fig.add_trace(go.Scatter(
    x=data['Timestamp'],
    y=data['MG-LV-MSB_AC_Voltage'],
    mode='lines',
    name='Voltage',
    line=dict(color='blue'),
    hoverinfo='skip'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=voltage_anomalies['Timestamp'],
    y=voltage_anomalies['MG-LV-MSB_AC_Voltage'],
    mode='markers',
    name='Voltage Anomalies',
    marker=dict(size=9, color='red', line=dict(width=1, color='black')),
    text=[
        f"Votes: {v}<br>Voltage: {val:.2f}"
        for v, val in zip(voltage_anomalies['ensemble_votes'], voltage_anomalies['MG-LV-MSB_AC_Voltage'])
    ],
    hoverinfo='text'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=data['Timestamp'],
    y=data['MG-LV-MSB_Frequency'],
    mode='lines',
    name='Frequency',
    line=dict(color='green'),
    hoverinfo='skip'
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=frequency_anomalies['Timestamp'],
    y=frequency_anomalies['MG-LV-MSB_Frequency'],
    mode='markers',
    name='Frequency Anomalies',
    marker=dict(size=9, color='red', line=dict(width=1, color='black')),
    text=[
        f"Votes: {v}<br>Freq: {freq:.2f}"
        for v, freq in zip(frequency_anomalies['ensemble_votes'], frequency_anomalies['MG-LV-MSB_Frequency'])
    ],
    hoverinfo='text'
), row=2, col=1)

fig.update_layout(
    height=800,
    title='Hybrid Ensemble Voltage & Frequency Anomaly Detection',
    showlegend=True,
    hovermode='x unified'
)

fig.update_xaxes(title_text="Timestamp", row=1, col=1, showticklabels=True)
fig.update_xaxes(title_text="Timestamp", row=2, col=1, showticklabels=True)
fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)

fig.show()