import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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

model = IsolationForest(n_estimators=100, contamination=0.003, random_state=42)
model.fit(X_scaled)

data['anomaly_score'] = model.decision_function(X_scaled)
data['is_anomaly'] = model.predict(X_scaled) == -1

voltage_anomalies = data[data['is_anomaly'] & (data['MG-LV-MSB_AC_Voltage'].notnull())]
frequency_anomalies = data[data['is_anomaly'] & (data['MG-LV-MSB_Frequency'].notnull())]

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
        f"Score: {score:.4f}<br>Voltage: {volt:.2f}"
        for score, volt in zip(voltage_anomalies['anomaly_score'], voltage_anomalies['MG-LV-MSB_AC_Voltage'])
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
        f"Score: {score:.4f}<br>Freq: {freq:.2f}"
        for score, freq in zip(frequency_anomalies['anomaly_score'], frequency_anomalies['MG-LV-MSB_Frequency'])
    ],
    hoverinfo='text'
), row=2, col=1)

fig.update_layout(
    height=800,
    title='Voltage and Frequency Anomalies Detected by Isolation Forest',
    showlegend=True,
    hovermode='x unified'
)

fig.update_xaxes(title_text="Timestamp", row=1, col=1, showticklabels=True)
fig.update_xaxes(title_text="Timestamp", row=2, col=1, showticklabels=True)

fig.show()