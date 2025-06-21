# M-Eureka 
- Pre-prototipo
- Eureka simplified (https://eureka-research.github.io/)
- Ubuntu 24.04

## Requisitos
* Python 3.10
* pip
* miniconda
* DeepSeek API key (https://api-docs.deepseek.com/)

## Configuración del entorno
```
conda create --n em python=3.10
conda activate em  # puedes desactivarlo con "conda deactivate"
```

## Instalación de dependencias
```
pip install gymnasium stable-baselines3 
pip install openai==0.28
pip install 'shimmy>=2.0'
pip install numpy==1.23.1
pip install pygame
```

## Clonar repositorio
```
git clone https://github.com/PonPlayJS/M-Eureka
cd M-Eureka
```

## Configuración
1. Edita meureka/meureka.py y agrega tu API key de DeepSeek en la línea 13.
2. El entorno personalizado CartPole está configurado en int_gym.py
3. La función de recompensa personalizada está en custom_reward.py

## Entrenamiento
```
python meureka.py
```
El modelo entrenado se guardará automáticamente como "ppo_custom_carpole.zip" en la carpeta "videos" del proyecto.

![CartPole Visualization](https://github.com/user-attachments/assets/c899c84a-e098-45e2-9579-eec26a2d510d)

## Cambios recientes
- ¡Ya no se necesita usar el archivo BASH, solo ejecuta "meureka.py" 



