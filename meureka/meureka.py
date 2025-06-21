import subprocess
import time
import os
import re
import sys
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from openai import OpenAI

# --- Configuración del cliente OpenAI para DeepSeek ---
try:
    client = OpenAI(
        api_key="API_KEY", # Tu API Key
        base_url="https://api.deepseek.com/v1", # Base URL para DeepSeek
    )
except Exception as e:
    print(f"Error al configurar el cliente de DeepSeek: {e}")
    sys.exit(1)

# --- Rutas ---
ROOT_DIR = Path(os.getcwd())
PROMPT_FILE_PATH = ROOT_DIR / ".." / "prompts" / "test.txt"

# --- Funciones ---
def clean_code_output(code):
    """Elimina marcadores ``` y contenido no deseado del código generado."""
    patterns = [
        r'```python\n',
        r'```\n',
        r'^```.*$',
        r'^#.*$',
    ]
    for pattern in patterns:
        code = re.sub(pattern, '', code, flags=re.MULTILINE)
    return code.strip()

def generate_reward_policy(prompt_content):
    """Genera una política de recompensa usando la API de DeepSeek."""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat", # Modelo DeepSeek-V3
            messages=[
                {"role": "system", "content": "Eres un asistente experto en aprendizaje por refuerzo."},
                {"role": "user", "content": prompt_content}
            ]
        )
        raw_code = response.choices[0].message.content
        return clean_code_output(raw_code)
    except Exception as e:
        print(f"Error al generar la política de recompensa: {e}")
        return None

def open_process(): 
    """Inicia el script de entrenamiento 'traning.py' en un proceso separado."""
    print("Abriendo el proceso de entrenamiento (traning.py)...")
    try:
        script_path = f'{ROOT_DIR}/traning.py' 
        
        if not os.path.exists(script_path):
            print(f"Error: El script de entrenamiento '{script_path}' no se encontró.")
            print("Asegúrate de que 'traning.py' exista en el mismo directorio que este script.")
            return None

        # Iniciar el subproceso
        process = subprocess.Popen(['python', '-u', script_path])
        print(f"Proceso de entrenamiento iniciado con PID: {process.pid}")

        # Esperar a que el proceso termine
        while process.poll() is None:
            print("El proceso de entrenamiento sigue en ejecución... esperando 4 segundos.")
            time.sleep(10)

        print(f"El proceso de entrenamiento ha terminado con código de salida: {process.returncode}")
        return process # devuelve el objeto del proceso
    except FileNotFoundError:
        print(f"Error: No se encontró el ejecutable de 'python' o el script '{script_path}'.")
        print("Asegúrate de que Python esté en tu PATH y el script exista.")
        return None
    except Exception as e:
        print(f'Ha ocurrido un error inesperado al intentar abrir el proceso: {e}')
        return None

def view_train(): 
    """Visualiza el comportamiento de un modelo PPO pre-entrenado en CartPole."""
    try:
        model_path = f'{ROOT_DIR}/../videos/ppo_custom_cartpole.zip' 

        if not os.path.exists(model_path):
            print(f"Error: El archivo del modelo no se encontró en: {model_path}")
            print("Asegúrate de que el modelo haya sido entrenado y guardado correctamente en esa ruta.")
            return None

        model = PPO.load(model_path)
        env = gym.make("CartPole-v1", render_mode="human")
        observation, info = env.reset()

        print("Iniciando visualización del modelo. ¡Observa cómo juega!")
        for _ in range(1000):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

        env.close()
        print("Visualización del modelo terminada.")
    except Exception as e:
        print(f'Ha ocurrido un error inesperado durante la visualización: {e}')
    return None

### Lógica principal del script

if __name__ == "__main__":
    prompt_text_content = ""
    try:
        with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as archive:
            prompt_text_content = archive.read()
            print("Contenido del prompt leído:\n" + prompt_text_content[:200] + "...") # Mostrar solo los primeros 200 caracteres
    except FileNotFoundError:
        print(f"Error: El archivo de prompt no se encontró en '{PROMPT_FILE_PATH}'. Asegúrate de que la ruta sea correcta y el archivo exista.")
        sys.exit(1)
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo de prompt: {e}")
        sys.exit(1)

    if prompt_text_content:
        custom_reward = generate_reward_policy(prompt_text_content)
    else:
        custom_reward = None

    if custom_reward:
        file_name = "custom_reward.py"
        try:
            with open(file_name, "w", encoding='utf-8') as file:
                file.write(custom_reward)
            print(f"\nPolítica de recompensa guardada en '{file_name}' exitosamente.")
        except Exception as e:
            print(f"Error al guardar la política de recompensa en '{file_name}': {e}")
            sys.exit(1)
    else:
        print("\nNo se pudo generar la política de recompensa. No se guardará ningún archivo.")

    print("\n--- Ejecutando proceso de entrenamiento ---")
    proceso_entrenamiento = open_process() 

    if proceso_entrenamiento and proceso_entrenamiento.returncode == 0:
        print("\n--- Iniciando visualización del entrenamiento ---")
        view_train() 
    else:
        print("\nNo se iniciará la visualización, ya que el proceso de entrenamiento no terminó con éxito o no se inició.")

    print("\nFin del script principal.")