from stable_baselines3 import PPO
from custom_reward import custom_reward 
import gymnasium as gym 
from int_gym import CustomCartPoleEnv

env = CustomCartPoleEnv()

print("Creando el entorno...")
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    policy_kwargs={"net_arch": [64, 64]}
)

# Entrenar el modelo
print("Entrenando agente...")
model.learn(total_timesteps=50000)

print("Guardando el modelo en 'ppo_custom_cartpole.zip'...")
model.save("/home/joaquin/M-Eureka/videos/ppo_custom_cartpole.zip")

# Probar el modelo entrenado
print("Probando el agente entrenado...")
obs, _ = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    
    obs, rewards, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated
    
    # Renderizar el entorno
    env.render()
    
    if done:
        obs, _ = env.reset()

# Cerrar el entorno
env.close()
print("Ejecución finalizada.")
