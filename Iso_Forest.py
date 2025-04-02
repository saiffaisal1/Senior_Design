import pandas as pd
import plotly.graph_objects as go
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

model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(X_scaled)

data['anomaly_score'] = model.decision_function(X_scaled)
data['is_anomaly'] = model.predict(X_scaled) == -1

anomalies = data[data['is_anomaly']]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data['Timestamp'],
    y=data['MG-LV-MSB_AC_Voltage'],
    mode='lines',
    name='MG-LV-MSB Voltage',
    line=dict(color='blue'),
    hoverinfo='skip'
))

fig.add_trace(go.Scatter(
    x=anomalies['Timestamp'],
    y=anomalies['MG-LV-MSB_AC_Voltage'],
    mode='markers',
    name='Anomalies',
    marker=dict(
        size=9,
        color='red',
        symbol='circle',
        line=dict(width=1, color='black')
    ),
    text=[
        f"Score: {score:.4f}<br>Voltage: {volt:.2f}"
        for score, volt in zip(anomalies['anomaly_score'], anomalies['MG-LV-MSB_AC_Voltage'])
    ],
    hoverinfo='text'
))

fig.update_layout(
    title='MG-LV-MSB Voltage with Anomalies Highlighted',
    xaxis_title='Timestamp',
    yaxis_title='Voltage (V)',
    hovermode='x unified',
    height=500
)

fig.show()