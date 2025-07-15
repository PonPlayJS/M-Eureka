import subprocess
import time
import os
import re
import sys
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from openai import OpenAI
from PyPDF2 import PdfReader

# --- OpenAI Client Configuration for DeepSeek ---
# It's recommended to use environment variables for the API Key for security.
try:
    client = OpenAI(
        api_key=os.getenv("DS_API_KEY"), # Will use the environment variable or the default key.
        base_url="https://api.deepseek.com/v1",
    )
except Exception as e:
    print(f"Error configuring DeepSeek client: {e}")
    sys.exit(1)

# --- Paths ---
ROOT_DIR = Path(os.getcwd())
# Path for the main task prompt.
PROMPT_FILE_PATH = ROOT_DIR / "prompts" / "cartpole" / "test.txt"
# List of paths to the PDFs that will serve as context.
PDF_PATHS = [
    ROOT_DIR / ".." / "pdf" / "1.pdf",
    ROOT_DIR / ".." / "pdf" / "2.pdf",
    ROOT_DIR / ".." / "pdf" / "3.pdf",
]

# --- Functions ---

def read_pdfs_text(pdf_paths):
    """
    Reads a list of PDF files and extracts all their text.
    
    Args:
        pdf_paths (list): A list of Path objects to the PDF files.

    Returns:
        str: A string with all the concatenated text from the PDFs.
    """
    print("Reading PDF files to get context...")
    full_text = ""
    for pdf_path in pdf_paths:
        try:
            if not pdf_path.exists():
                print(f"Warning: PDF file not found at: {pdf_path}")
                continue
            
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += f"\n--- Content from {pdf_path.name}, Page {i+1} ---\n"
                        full_text += page_text
            print(f"  - Successfully read: {pdf_path.name}")
        except Exception as e:
            print(f"Error reading file {pdf_path.name}: {e}")
    
    if not full_text:
        print("Warning: Could not extract text from any PDF.")
        
    return full_text

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
                {"role": "system", "content": "You are an expert reinforcement learning assistant. Generate only the requested Python code."},
                {"role": "user", "content": prompt_content}
            ]
        )
        raw_code = response.choices[0].message.content
        return clean_code_output(raw_code)
    except Exception as e:
        print(f"Error generating reward policy: {e}")
        return None

def run_training_process():
    """Starts the training script in a separate process."""
    print("Starting the training process (training.py)...")
    try:
        # Assumes the correct script name is "training.py"
        script_path = ROOT_DIR / 'meureka' /'training.py'
        
        if not os.path.exists(script_path):
            print(f"Error: The training script '{script_path}' was not found.")
            return None

        process = subprocess.Popen(['python', '-u', str(script_path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Training process started with PID: {process.pid}")

        # Stream stdout and stderr
        while process.poll() is None:
            print("The training process is still running... waiting 10 seconds.")
            time.sleep(10)

        stdout, stderr = process.communicate()
        if stdout:
            print(f"Training script stdout:\n{stdout}")
        if stderr:
            print(f"Training script stderr:\n{stderr}")


        print(f"The training process has finished with exit code: {process.returncode}")
        return process
    except FileNotFoundError:
        print(f"Error: 'python' executable or script '{script_path}' not found.")
        return None
    except Exception as e:
        print(f'An unexpected error occurred while trying to open the process: {e}')
        return None

def view_trained_agent():
    """Visualizes the behavior of a pre-trained PPO model."""
    print("Starting visualization of the trained agent...")
    try:
        model_path = ROOT_DIR / 'videos' / 'ppo_custom_cartpole.zip'

        if not os.path.exists(model_path):
            print(f"Error: The model file was not found at: {model_path}")
            return

        model = PPO.load(model_path)
        env = gym.make("CartPole-v1", render_mode="human")
        observation, _ = env.reset()

        for _ in range(1000):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                observation, _ = env.reset()

        env.close()
        print("Model visualization finished.")
    except Exception as e:
        print(f'An unexpected error occurred during visualization: {e}')

def execute_training_attempt():
    """
    Performs ONE complete training attempt, using context from PDFs.
    """
    # 1. Read the main task prompt.
    try:
        with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as archive:
            base_prompt_text = archive.read()
    except Exception as e:
        print(f"Critical error reading the prompt: {e}")
        return None

    # 2. Read the text from PDFs to use as context.
    pdf_context = read_pdfs_text(PDF_PATHS)

    # 3. Combine the PDF context and the task prompt.
    combined_prompt = f"""
    
    --- REQUESTED TASK ---
    {base_prompt_text}
    --- END OF TASK 
    
    --- PDF DOCUMENT CONTEXT ---
    {pdf_context}
    --- END OF CONTEXT ---

    """

    print("\n Prompt: ...")
    print(base_prompt_text[:1] + "...")

    # 4. Generate and save the reward policy.
    custom_reward = generate_reward_policy(combined_prompt)
    if custom_reward:
        with open("custom_reward.py", "w", encoding='utf-8') as file:
            file.write(custom_reward)
        print("Reward policy generated/updated for this attempt.")
    else:
        print("Could not generate reward policy for this attempt.")
        return None
    
    # 5. Execute the training process.
    process = run_training_process()
    return process

# --- Main Entry Point ---

if __name__ == "__main__":
    
    max_attempts = 3
    current_attempt = 1
    training_successful = False
    
    while not training_successful and current_attempt <= max_attempts:
        
        print(f"\n" + "="*50)
        print(f"--- TRAINING ATTEMPT N° {current_attempt}/{max_attempts} ---")
        print("="*50)

        finished_process = execute_training_attempt()

        if finished_process and finished_process.returncode == 0:
            print(f"\nSUCCESS on attempt N° {current_attempt}.")
            training_successful = True
        else:
            print(f"\nFAILURE on attempt N° {current_attempt}.")
            current_attempt += 1
            if current_attempt <= max_attempts:
                print("Waiting 5 seconds before retrying...")
                time.sleep(5)

    # --- Final Actions ---
    print("\n" + "="*50)
    if training_successful:
        print("Training process finished successfully.")
        view_trained_agent()
    else:
        print(f"The process failed after {max_attempts} attempts. Nothing will be visualized.")
    print("="*50)
