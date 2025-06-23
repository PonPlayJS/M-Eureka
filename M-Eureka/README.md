# M-Eureka 
- Pre-prototype
- Eureka simplified (https://eureka-research.github.io/)
- Ubuntu 24.04

## Requirements
* Python 3.10
* pip
* miniconda
* DeepSeek API key (https://api-docs.deepseek.com/)

## Configuración del entorno
```
conda create --n em python=3.10
conda activate em  # you can deactivate it with "conda deactivate"
```

## Dependency Installation
```
pip install gymnasium stable-baselines3 
pip install openai==0.28
pip install 'shimmy>=2.0'
pip install numpy==1.23.1
pip install pygame
```

## Clone Repository
```
git clone https://github.com/PonPlayJS/M-Eureka
cd M-Eureka
```

## Configuration
1. Edit meureka/meureka.py and add your DeepSeek API key on line 13.
2. The custom CartPole environment is configured in int_gym.py
3. The custom reward function is in custom_reward.py

## Training
```
python meureka.py
```
The trained model will be automatically saved as "ppo_custom_carpole.zip" in the "videos" folder of the project.

![alt text](image.png)

## Recent Changes
- If a reward policy is not executed, the entire process is redone.



