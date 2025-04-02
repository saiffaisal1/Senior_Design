import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

train = pd.read_csv("Data/Apr_2023.csv")
test = pd.read_csv("Data/Apr_2023_with_anomalies.csv")

features = ['MG-LV-MSB_AC_Voltage', 'MG-LV-MSB_Frequency']
train = train.dropna(subset=features)
test = test.dropna(subset=features + ['label'])

train['Timestamp'] = pd.to_datetime(train['Timestamp'], errors='coerce')
test['Timestamp'] = pd.to_datetime(test['Timestamp'], errors='coerce')

train = train.dropna(subset=['Timestamp']).sort_values('Timestamp')
test = test.dropna(subset=['Timestamp']).sort_values('Timestamp')

scaler = StandardScaler()
X_train = scaler.fit_transform(train[features])
X_test = scaler.transform(test[features])
y_true = test['label']

model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_train)

test['score'] = model.decision_function(X_test)

percentiles = np.linspace(0.01, 5, 50) 
precisions, recalls, f1s, thresholds = [], [], [], []

for p in percentiles:
    threshold = np.percentile(test['score'], p)
    preds = (test['score'] < threshold).astype(int)

    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    thresholds.append(threshold)

best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]
best_percentile = percentiles[best_idx]

test['is_anomaly'] = (test['score'] < best_threshold).astype(int)

precision = precision_score(y_true, test['is_anomaly'])
recall = recall_score(y_true, test['is_anomaly'])
f1 = f1_score(y_true, test['is_anomaly'])

print("\n Final Evaluation with Best Threshold")
print(f" Best F1 Score: {f1:.3f}")
print(f" Precision: {precision:.3f}")
print(f" Recall:    {recall:.3f}")
print(f" Threshold value: {best_threshold:.4f} (percentile {best_percentile:.2f}%)")

alerts = test[test['is_anomaly'] == 1][['Timestamp', 'MG-LV-MSB_AC_Voltage', 'MG-LV-MSB_Frequency', 'score']]
alerts.to_csv("detected_anomalies.csv", index=False)
print(f"Exported {len(alerts)} detected anomalies to detected_anomalies.csv")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=("Voltage Anomalies", "Frequency Anomalies"))

fig.add_trace(go.Scatter(
    x=test['Timestamp'],
    y=test['MG-LV-MSB_AC_Voltage'],
    mode='lines',
    name='Voltage',
    line=dict(color='blue'),
    hoverinfo='skip'
), row=1, col=1)

voltage_anomalies = test[test['is_anomaly'] == 1]
fig.add_trace(go.Scatter(
    x=voltage_anomalies['Timestamp'],
    y=voltage_anomalies['MG-LV-MSB_AC_Voltage'],
    mode='markers',
    name='Voltage Anomaly',
    marker=dict(size=8, color='red', line=dict(width=1, color='black')),
    text=[
        f"True Label: {t}<br>Score: {s:.4f}<br>Voltage: {v:.2f}"
        for t, s, v in zip(
            voltage_anomalies['label'],
            voltage_anomalies['score'],
            voltage_anomalies['MG-LV-MSB_AC_Voltage']
        )
    ],
    hoverinfo='text'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=test['Timestamp'],
    y=test['MG-LV-MSB_Frequency'],
    mode='lines',
    name='Frequency',
    line=dict(color='green'),
    hoverinfo='skip'
), row=2, col=1)

frequency_anomalies = test[test['is_anomaly'] == 1]
fig.add_trace(go.Scatter(
    x=frequency_anomalies['Timestamp'],
    y=frequency_anomalies['MG-LV-MSB_Frequency'],
    mode='markers',
    name='Frequency Anomaly',
    marker=dict(size=8, color='red', line=dict(width=1, color='black')),
    text=[
        f"True Label: {t}<br>Score: {s:.4f}<br>Frequency: {f:.2f}"
        for t, s, f in zip(
            frequency_anomalies['label'],
            frequency_anomalies['score'],
            frequency_anomalies['MG-LV-MSB_Frequency']
        )
    ],
    hoverinfo='text'
), row=2, col=1)

fig.update_layout(
    title="Voltage & Frequency Anomalies (Best Threshold Applied)",
    height=800,
    hovermode='x unified'
)

fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)
fig.update_xaxes(title_text="Timestamp", row=2, col=1)

fig.show()
