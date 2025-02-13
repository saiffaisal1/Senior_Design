import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import zipfile
import numpy as np
import pandas as pd
import gymnasium as gym  
from stable_baselines3 import PPO
import torch

model = None 

try:
        model = PPO.load(os.path.join("microgrid_rl_voltage_frequency"))
        print("Model successfully loaded!")
except Exception as e:
    print(f"Error loading model: {e}")

if model is None:
    raise RuntimeError("Model loading failed. Check the extracted files!")


csv_file = "./Data/Sep_2022.csv"
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Dataset file '{csv_file}' not found!")

data = pd.read_csv(csv_file)
print(f"Dataset '{csv_file}' loaded successfully!")

from MicrogridEnv import MicrogridEnv
try:
    env = MicrogridEnv(data)  
    print("Environment initialized!")
except Exception as e:
    raise RuntimeError(f"Error initializing environment: {e}")

def test_rl_model(env, model, num_episodes=10):
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
    
    avg_reward = sum(total_rewards) / num_episodes
    print(f"RL Model - Average Reward over {num_episodes} episodes: {avg_reward}")


print("\nRunning Model Tests...")
test_rl_model(env, model)
