# M-Eureka 🤖✨

### A simplified implementation of the [Eureka paper by NVIDIA Research](https://eureka-research.github.io/)

[![Project Status](https://img.shields.io/badge/status-pre--prototype-yellow)](https://github.com/PonPlayJS/M-Eureka)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)

**M-Eureka** is a project that aims to distill the key concepts of the innovative "Eureka" paper into a simpler, more accessible format. It uses Large Language Models (LLMs), in this case via the **DeepSeek** API, to automatically generate reward functions for Reinforcement Learning (RL) tasks.

This pre-prototype focuses on the classic `CartPole` environment, demonstrating the main workflow: AI-driven reward code generation, evaluation, and agent training.

![Trained Agent Demonstration](https://github.com/PonPlayJS/M-Eureka/raw/main/image.png)

---

## 🎯 Key Features

* **AI-Powered Reward Generation**: Leverages the power of LLMs to autonomously write and refine reward policies.
* **Simplified Codebase**: A reduced and commented codebase, ideal for learning and experimenting.
* **Classic Environment**: Implemented on Gymnasium's `CartPole-v1` for easy understanding and execution.
* **Policy Validation**: A safety mechanism that retries the process if the AI-generated reward code is not executable.
* **Modular and Extensible**: Designed to be easily adaptable to other environments in the future.

---

## 🚀 Getting Started

Follow these steps to get the project up and running on your local machine.

### Prerequisites

Make sure you have the following installed:

* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
* Python 3.10
* Git
* A **DeepSeek API Key**: You can get one from their [official website](https://platform.deepseek.com/api_keys).

### 🔧 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/PonPlayJS/M-Eureka
    cd M-Eureka
    ```

2.  **Create and activate the Conda environment**
    ```bash
    conda create --name meureka python=3.10
    conda activate meureka
    ```
    *(To deactivate the environment later, simply run `conda deactivate`)*

3.  **Install dependencies**
    We have prepared a `requirements.txt` file to make this process easier.
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure your API Key**
    ```
    export DS_API_KEY="your_secret_key_goes_here"
    ```

---

## ▶️ Usage

Once the setup is complete, you can start the training process with a single command:

```bash
python meureka/meureka_main.py
```

The script will handle:
1.  Communicating with the DeepSeek API to generate a reward function.
2.  Validating and applying this function to the environment.
3.  Training a PPO agent from `stable-baselines3`.
4.  Saving the trained model as `ppo_custom_carpole.zip` and a video of its performance in the `videos/` folder.

---

## 🗺️ Roadmap

We have big plans to expand M-Eureka's capabilities:

* [ ] Adapt the code to work with **RoboCup** environments.
* [ ] Integrate more advanced simulators like **Mujoco** and **Isaac Sim**.
* [ ] Allow selection of different LLM models.
* [ ] Implement a more robust reward evaluation and evolution system.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for improving the project, feel free to open an *issue* to discuss it or submit a *pull request*.

---

## 🙏 Acknowledgements

* To **NVIDIA Research** for publishing the paper [Eureka: Human-Level Reward Design via Coding Large Language Models](https://eureka-research.github.io/).

---





