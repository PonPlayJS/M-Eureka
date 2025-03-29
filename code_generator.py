import sys
from openai import OpenAI

# Configura el cliente para la API de DeepSeek
try:
    client = OpenAI(
        api_key="DeepSeek_API",
        base_url="https://api.deepseek.com/v1",
        timeout=30.0
    )
except Exception as e:
    print(f"Error al configurar el cliente de DeepSeek: {e}")
    sys.exit(1)

def generate_reward_policy(prompt):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # Modelo DeepSeek-V3
            messages=[
                {"role": "system", "content": "Eres un asistente experto en aprendizaje por refuerzo."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error al generar la política de recompensa: {e}")
        sys.exit(1)

# Ejemplo: Generar una política de recompensa para CartPole (ejemplo)
prompt = """
Genera una función  en python de recompensa personalizada para el entorno CartPole-v1.
La función debe penalizar al agente si el ángulo del poste se aleja demasiado de la vertical,
y recompensarlo si mantiene el poste equilibrado durante más tiempo.
Proporciona SOLO el código Python de la función, sin explicaciones adicionales, ni si quiera los "```"
apa-
+te tiene que tener un formato parecido a este(pero ojala sea mucho mejor):
def custom_reward(observation):
    angle = observation[2]  # El ángulo del poste
    reward = 1.0  # Recompensa base por estar en el paso de tiempo

    # Penalización si el ángulo se aleja de la vertical
    if abs(angle) > 0.2:  # umbral de ángulo
        reward -= 1.0  # penalización

    return reward
"""

custom_reward = generate_reward_policy(prompt)

# Guardar la política de recompensa en un archivo .py
file_name = "custom_reward.py"
with open(file_name, "w") as file:
    file.write(custom_reward)

print(f"Política de recompensa guardada en '{file_name}'")

