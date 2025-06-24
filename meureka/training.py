from stable_baselines3 import PPO
from custom_reward import custom_reward 
import gymnasium as gym 
from int_gym import CustomCartPoleEnv
import os
from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(os.getcwd())

env = CustomCartPoleEnv()

print("Creating the environment...")
# Create the model with a custom policy using PPO
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    policy_kwargs={"net_arch": [64, 64]}
)

# Train the model
print("Training agent...")
model.learn(total_timesteps=50000)

print("Saving the model to 'ppo_custom_cartpole.zip'...")
model.save(f'{ROOT_DIR}/../videos/ppo_custom_cartpole.zip')

# Test the trained model
print("Testing the trained agent...")
obs, _ = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    
    obs, rewards, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated
    
    # Render the environment
    env.render()
    
    if done:
        obs, _ = env.reset()

# Close the environment
env.close()
print("Execution finished.")
