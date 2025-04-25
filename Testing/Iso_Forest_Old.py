import pandas as pd
import matplotlib.pyplot as plt
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
data['anomaly_score'] = model.fit_predict(X_scaled)

data['is_anomaly'] = data['anomaly_score'] == -1

plt.figure(figsize=(14, 5))
plt.plot(data['Timestamp'], data['MG-LV-MSB_AC_Voltage'], label='Voltage')
plt.scatter(data[data['is_anomaly']]['Timestamp'],
            data[data['is_anomaly']]['MG-LV-MSB_AC_Voltage'],
            color='red', label='Anomaly', s=10)
plt.title("Anomaly Detection on MG-LV-MSB_AC_Voltage")
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.legend()
plt.tight_layout()
plt.show()