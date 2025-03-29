from stable_baselines3 import PPO
from int_gym import CustomCartPoleEnv # Importa el entorno personalizado de CartPole
import numpy as np

# Crear el entorno personalizado
env = CustomCartPoleEnv()
# Crear el modelo con una política personalizada usando PPO
model = PPO(
    "MlpPolicy", # Política basada en una red neuronal de tipo MLP (Perceptrón Multicapa)
    env, # Entorno personalizado en CartPole donde entrenarás el modelo
    verbose=1, 
    policy_kwargs={"net_arch": [64, 64]}  # Define (Lo puedes personalizar cómo quieras) la arquitectura de la red neuronal (2 capas de 64 neuronas cada una)
)
# Entrenar el modelo con un número determinado de pasos (50000)
print("Entrenando agente...")
model.learn(total_timesteps=50000)

# Probar el modelo 
print("Probando el agente entrenado...")
obs, _ = env.reset()  # Resetear el ambiente y obtiene la observación inicial

for _ in range(1000):
    action, _states = model.predict(obs)
    action = int(action)
    obs, rewards, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Renderizar el entorno (mostrar visualmente el progreso del agente)
    env.render()
    
    if done:
        obs, _ = env.reset()  # Reiniciar el entorno si el episodio termina

# Guardar el modelo entrenado en una ruta específica
model.save("PATH_SAVE")

env.close()  # Cerrar el entorno

