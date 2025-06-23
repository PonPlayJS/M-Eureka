import gymnasium as gym
import numpy as np
from custom_reward import custom_reward  # Imports a custom reward function from the custom_reward.py file

class CustomCartPoleEnv(gym.Env):
    def __init__(self):
        super(CustomCartPoleEnv, self).__init__()
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        
        # Define observation and action spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def step(self, action):
        # Performs a step in the environment with the provided action
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Uses the custom reward function
        custom_rew = custom_reward(obs)
        
        return np.array(obs, dtype=np.float32), custom_rew, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return np.array(obs, dtype=np.float32), info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()