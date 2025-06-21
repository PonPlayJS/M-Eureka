import gymnasium as gym
import numpy as np
from custom_reward import custom_reward  # Importa una función de recompensa personalizada desde el archivo custom_reward.py

class CustomCartPoleEnv(gym.Env):
    def __init__(self):
        super(CustomCartPoleEnv, self).__init__()
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        
        # Definir los espacios de observación y acción
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def step(self, action):
        # Realiza un paso en el entorno con la acción proporcionada
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Usa la función de recompensa personalizada
        custom_rew = custom_reward(obs)
        
        return np.array(obs, dtype=np.float32), custom_rew, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return np.array(obs, dtype=np.float32), info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()