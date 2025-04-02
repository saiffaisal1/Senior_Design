import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import os
import re
from glob import glob

parser = argparse.ArgumentParser(description="Visualize voltage and frequency anomalies.")
parser.add_argument("full_data", help="Path to full data CSV path")
parser.add_argument("--output", default=None, help="Optional: output HTML file")
args = parser.parse_args()

match = re.search(r"([A-Za-z]{3}_\d{4})", os.path.basename(args.full_data))
if not match:
    raise ValueError("Could not find Month_YYYY pattern in input filename.")

month_year = match.group(1)

df = pd.read_csv("Data/" + args.full_data)
anomalies = pd.read_csv("alerts/detected_anomalies_" + args.full_data)

df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
anomalies['Timestamp'] = pd.to_datetime(anomalies['Timestamp'], errors='coerce')

df = df.dropna(subset=['Timestamp', 'MG-LV-MSB_AC_Voltage', 'MG-LV-MSB_Frequency'])
anomalies = anomalies.dropna(subset=['Timestamp', 'MG-LV-MSB_AC_Voltage', 'MG-LV-MSB_Frequency', 'confidence'])
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=("Voltage Over Time", "Frequency Over Time"))

fig.add_trace(go.Scatter(
    x=df['Timestamp'],
    y=df['MG-LV-MSB_AC_Voltage'],
    mode='lines',
    name='Voltage',
    line=dict(color='blue'),
    hoverinfo='skip'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=anomalies['Timestamp'],
    y=anomalies['MG-LV-MSB_AC_Voltage'],
    mode='markers',
    name='Voltage Anomaly',
    marker=dict(size=8, color='red', line=dict(width=1, color='black')),
    text=[
        f"{t}<br>Voltage: {v:.2f} V<br>Frequency: {f:.2f} Hz<br>Confidence: {c:.2f}"
        for t, v, f, c in zip(
            anomalies['Timestamp'],
            anomalies['MG-LV-MSB_AC_Voltage'],
            anomalies['MG-LV-MSB_Frequency'],
            anomalies['confidence']
        )
    ],
    hoverinfo='text'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df['Timestamp'],
    y=df['MG-LV-MSB_Frequency'],
    mode='lines',
    name='Frequency',
    line=dict(color='green'),
    hoverinfo='skip'
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=anomalies['Timestamp'],
    y=anomalies['MG-LV-MSB_Frequency'],
    mode='markers',
    name='Frequency Anomaly',
    marker=dict(size=8, color='red', line=dict(width=1, color='black')),
    text=[
        f"{t}<br>Voltage: {v:.2f} V<br>Frequency: {f:.2f} Hz<br>Confidence: {c:.2f}"
        for t, v, f, c in zip(
            anomalies['Timestamp'],
            anomalies['MG-LV-MSB_AC_Voltage'],
            anomalies['MG-LV-MSB_Frequency'],
            anomalies['confidence']
        )
    ],
    hoverinfo='text'
), row=2, col=1)

fig.update_layout(
    title=f"Voltage & Frequency Anomalies — {month_year.replace('_', ' ')}",
    height=800,
    hovermode='x unified'
)

fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
fig.update_xaxes(title_text="Timestamp", row=1, col=1)
fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)
fig.update_xaxes(title_text="Timestamp", row=2, col=1)

if args.output:
    fig.write_html(args.output)
    print(f"Saved dashboard to {args.output}")
else:
    fig.show()
