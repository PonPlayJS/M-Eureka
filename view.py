import gymnasium as gym
from stable_baselines3 import PPO

# Cargar el modelo desde la ubicación (puedes cambiarla)
model_path = "USER_PATH" # Ruta donde se encuentra el modelo guardado
model = PPO.load(model_path) # Carga el modelo PPO previamente entrenado

# Crear el entorno de CartPole con renderización en modo "human" (renderizar para visualización humana)
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset() # Reinicia el entorno y obtiene la observación inicial

# Ejecutar política aprendida
for _ in range(1000):
    action, _ = model.predict(observation) # Predecir la acción a tomar según la observación actual
    observation, reward, terminated, truncated, info = env.step(action)

     # Si el episodio ha terminado o ha sido truncado, reinicia el entorno
    if terminated or truncated:
        observation, info = env.reset()

# Cerrar el entorno para liberar recursos
env.close()
