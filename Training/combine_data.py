import pandas as pd
import glob
import os

data_path = "Data/"
all_csv_files = glob.glob(os.path.join(data_path, "*_2022.csv")) + glob.glob(os.path.join(data_path, "*_2023.csv"))

df_list = []
for file in all_csv_files:
    df = pd.read_csv(file)
    df['source_file'] = os.path.basename(file)  # optional, for traceability
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)

combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], errors='coerce')
combined_df = combined_df.dropna(subset=['Timestamp'])
combined_df = combined_df.sort_values('Timestamp').reset_index(drop=True)

combined_df.to_csv("historical_clean_data.csv", index=False)
print(f"Combined {len(all_csv_files)} files into historical_clean_data.csv")