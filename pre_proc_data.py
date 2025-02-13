import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import glob
import os

data_folder = "./Data/"
csv_files = glob.glob(os.path.join(data_folder , "*.csv"))

df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)


features = [
    "Battery_Active_Power", "FC_Active_Power", "GE_Active_Power",
    "MG-LV-MSB_AC_Voltage", "Receiving_Point_AC_Voltage",
    "Island_mode_MCCB_Frequency", "MG-LV-MSB_Frequency"
]
df = df[features].dropna()

print(df)
# 🔹 Normalize data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=features)

print(df_scaled)