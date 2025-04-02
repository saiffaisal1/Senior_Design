import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("Data/Apr_2023.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])
df.sort_values('Timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)


df['label'] = 0


np.random.seed(42)
num_point_anomalies = 40
num_clustered_anomalies = 10  
voltage_col = 'MG-LV-MSB_AC_Voltage'
frequency_col = 'MG-LV-MSB_Frequency'


point_indices = np.random.choice(df.index, size=num_point_anomalies, replace=False)

for idx in point_indices:
    volt_spike = np.random.choice([+40, -40])
    freq_spike = np.random.choice([+1.5, -1.5])

    df.at[idx, voltage_col] += volt_spike
    df.at[idx, frequency_col] += freq_spike
    df.at[idx, 'label'] = 1

for _ in range(num_clustered_anomalies):
    start_idx = np.random.randint(0, len(df) - 5)
    cluster_len = np.random.randint(3, 6)
    for i in range(cluster_len):
        idx = start_idx + i
        volt_spike = np.random.choice([+30, -30])
        freq_spike = np.random.choice([+1, -1])
        df.at[idx, voltage_col] += volt_spike
        df.at[idx, frequency_col] += freq_spike
        df.at[idx, 'label'] = 1

output_file = "Data/Apr_2023_with_anomalies.csv"
df.to_csv(output_file, index=False)
print(f"Injected mild anomalies saved to: {output_file}")
print(f"Total labeled anomalies: {df['label'].sum()}")
