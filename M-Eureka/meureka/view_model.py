import os
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO

ROOT_DIR = Path(os.getcwd())

try:
    model_path = f'{ROOT_DIR}/../videos/ppo_custom_cartpole.zip' 

    if not os.path.exists(model_path):
        print(f"Error: The model file was not found at: {model_path}")
        print("Make sure the model has been trained and saved correctly in that path.")


    model = PPO.load(model_path)
    env = gym.make("CartPole-v1", render_mode="human")
    observation, info = env.reset()

    print("Starting model visualization. Observe how it plays!")
    for _ in range(1000):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    print("Model visualization finished.")
except Exception as e:
    print(f'An unexpected error occurred during visualization: {e}')
