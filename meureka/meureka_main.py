import subprocess
import time
import os
import re
import sys
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from openai import OpenAI

# --- OpenAI Client Configuration for DeepSeek ---
try:
    client = OpenAI(
        api_key="sk-9db633f32dce4fa585bde151fe4631c7", # (example api key) DeepSeek Api key
        base_url="https://api.deepseek.com/v1", # Base URL for DeepSeek
    )
except Exception as e:
    print(f"Error configuring DeepSeek client: {e}")
    sys.exit(1)

# --- Paths ---
ROOT_DIR = Path(os.getcwd())
PROMPT_FILE_PATH = ROOT_DIR / ".." / "prompts" / "cartpole" / "test.txt"

# --- Functions ---
def clean_code_output(code):
    """Removes ``` markers and unwanted content from generated code."""
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
    """Generates a reward policy using the DeepSeek API."""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat", 
            messages=[
                {"role": "system", "content": "You are an expert assistant in reinforcement learning."},
                {"role": "user", "content": prompt_content}
            ]
        )
        raw_code = response.choices[0].message.content
        return clean_code_output(raw_code)
    except Exception as e:
        print(f"Error generating reward policy: {e}")
        return None

def open_training_process():
    """Starts the 'training.py' script in a separate process."""
    print("Opening the training process (training.py)...")
    try:
        script_path = f'{ROOT_DIR}/traning.py' 
        
        if not os.path.exists(script_path):
            print(f"Error: The training script '{script_path}' was not found.")
            print("Make sure 'training.py' exists in the same directory as this script.")
            return None

        # Start the subprocess
        process = subprocess.Popen(['python', '-u', script_path])
        print(f"Training process started with PID: {process.pid}")

        # Wait for the process to finish
        while process.poll() is None:
            print("The training process is still running... waiting 10 seconds.")
            time.sleep(10)

        print(f"The training process has finished with exit code: {process.returncode}")
        return process # returns the process object
    except FileNotFoundError:
        print(f"Error: 'python' executable or script '{script_path}' not found.")
        print("Make sure Python is in your PATH and the script exists.")
        return None
    except Exception as e:
        print(f'An unexpected error occurred while trying to open the process: {e}')
        return None
    
def view_trained_model():
    """Visualizes the behavior of a pre-trained PPO model in CartPole."""
    try:
        model_path = f'{ROOT_DIR}/../videos/ppo_custom_cartpole.zip' 

        if not os.path.exists(model_path):
            print(f"Error: The model file was not found at: {model_path}")
            print("Make sure the model has been trained and saved correctly in that path.")
            return None

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
    return None

def execute_training_attempt():
    """
    Performs ONE complete training attempt.
    Generates reward, saves it, and executes the training script.
    Returns the finished process object.
    """
    # 1. Read the prompt (this part does not change)
    prompt_text_content = ""
    try:
        with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as archive:
            prompt_text_content = archive.read()
    except Exception as e:
        print(f"Critical error reading the prompt: {e}")
        return None # Returns None if the prompt cannot even be read

    # 2. Generate and save the reward policy (does not change)
    if prompt_text_content:
        custom_reward = generate_reward_policy(prompt_text_content)
        if custom_reward:
            with open("custom_reward.py", "w", encoding='utf-8') as file:
                file.write(custom_reward)
            print("Reward policy generated/updated for this attempt.")
        else:
            print("Could not generate reward policy for this attempt.")
            return None
    
    # 3. Execute the training process and return it
    #    NOTE: It no longer checks the result or calls view_train() here.
    print("\n--- Executing training process ---")
    process_obj = open_training_process()
    return process_obj

### Main script logic
def main_logic():
    prompt_text_content = ""
    try:
        with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as archive:
            prompt_text_content = archive.read()
            print("Prompt content read:\n" + prompt_text_content[:3000] + "...") # Display only the first 200 characters
    except FileNotFoundError:
        print(f"Error: The prompt file was not found at '{PROMPT_FILE_PATH}'. Make sure the path is correct and the file exists.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the prompt file: {e}")
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
            print(f"\nReward policy saved to '{file_name}' successfully.")
        except Exception as e:
            print(f"Error saving reward policy to '{file_name}': {e}")
            sys.exit(1)
    else:
        print("\nCould not generate reward policy. No file will be saved.")

    print("\n--- Executing training process ---")
    training_process = open_training_process()

    if training_process and training_process.returncode == 0:
        print("\n--- Starting training visualization ---")
        view_trained_model()
    else:
        print("failed")
    print("\nEnd of main script.")
    return None

# Replace your if __name__ == "__main__" block with this improved version.

if __name__ == "__main__":
    
    max_attempts = 3
    current_attempt = 1
    training_successful = False
    
    while not training_successful and current_attempt <= max_attempts:
        
        print(f"\n" + "="*50)
        print(f"--- TRAINING ATTEMPT N° {current_attempt}/{max_attempts} ---")
        print("="*50)

        # Calls the function that only makes one attempt
        finished_process = execute_training_attempt()

        # Checks the result of the attempt
        if finished_process and finished_process.returncode == 0:
            print(f"\nSUCCESS in attempt N° {current_attempt}.")
            training_successful = True
        else:
            print(f"\nFAILURE in attempt N° {current_attempt}.")
            current_attempt += 1
            if current_attempt <= max_attempts:
                print("Waiting 5 seconds before retrying...")
                time.sleep(5)

    # --- FINAL Actions after exiting the loop ---
    print("\n" + "="*50)
    # Only if the final flag is True, we visualize.
    if training_successful:
        print("Training process finished successfully.")
        print("--- Starting visualization of the trained model ---")
        view_trained_model()
    else:
        print(f"The process failed after {max_attempts} attempts. Nothing will be visualized.")
    print("="*50)
