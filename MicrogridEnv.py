import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import gymnasium as gym 
from gymnasium import spaces
from stable_baselines3 import PPO
from sklearn.preprocessing import MinMaxScaler
import glob

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

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=features) 

TARGET_VOLTAGE = 230  # V
TARGET_FREQUENCY = 50  # Hz

class MicrogridEnv(gym.Env):
    def __init__(self, data):
        super(MicrogridEnv, self).__init__()

        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=features) 

        if data.empty:
            raise ValueError("Dataset is empty. Please check the CSV file!")
        
        self.data = data
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.max_steps = len(data) - 1
        self.prev_frequency = 0
        self.prev_voltage = 0

        self.action_space = spaces.MultiDiscrete([3, 3, 3]) 

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(features),), dtype=np.float32
        )

    def reset(self, seed=None, options=None):  # type: ignore 
        super().reset(seed=seed)

        self.current_step = 0
        obs = self.data.iloc[self.current_step].values
        return np.array(obs, dtype=np.float32), {}

    def step(self, action): # type: ignore

        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        if done:
            return np.array(self.data.iloc[self.current_step - 1].values, dtype=np.float32), 0, True, truncated, {}

        battery_adjust = action[0] - 1
        fc_adjust = action[1] - 1
        ge_adjust = action[2] - 1

        next_state = self.data.iloc[self.current_step].values

        voltage = next_state[3:5].mean() * 250
        frequency = next_state[5:7].mean() * 60 

        voltage_deviation = abs(voltage - TARGET_VOLTAGE)
        frequency_deviation = abs(frequency - TARGET_FREQUENCY)

        smoothness_penalty = abs(voltage - self.prev_voltage) + abs(frequency - self.prev_frequency)

        tolerance_penalty = 0
        if voltage_deviation > 5:
            tolerance_penalty -= 5
        if frequency_deviation > 1:
            tolerance_penalty -= 5

        reward = -voltage_deviation - frequency_deviation - 0.1 * smoothness_penalty + tolerance_penalty

        self.prev_voltage = voltage
        self.prev_frequency = frequency

        return np.array(next_state, dtype=np.float32), reward, done, truncated,  {} 

env = MicrogridEnv(df_scaled)
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0001,
    n_steps=1024,
    batch_size=64,  
    gae_lambda=0.95,  
    gamma=0.99,  
    ent_coef=0.01, 
    vf_coef=0.5,  
    max_grad_norm=0.5,
    device="auto"
    ) 
model.learn(total_timesteps=100000)

model.save("microgrid_rl_voltage_frequency")
