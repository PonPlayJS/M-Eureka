# M-Eureka 
- Pre-prototipo
- Eureka simplified (https://eureka-research.github.io/)
- Ubuntu 24.04

## Requisitos
* Python 3.10+
* pip
* miniconda
* DeepSeek API key (https://api-docs.deepseek.com/)

## Configuración del entorno
```
conda create --name em python=3.10
conda activate em  # puedes desactivarlo con "conda deactivate"
```

## Instalación de dependencias
```
pip install gymnasium stable-baselines3 
pip install openai==0.28
pip install 'shimmy>=2.0'
pip install numpy==1.23.1
```

## Clonar repositorio
```
git clone https://github.com/PonPlayJS/M-Eureka
cd M-Eureka
```

## Configuración
1. Edita code_generator.py y agrega tu API key de DeepSeek
2. El entorno personalizado CartPole está configurado en int_gym.py
3. La función de recompensa personalizada está en custom_reward.py

## Entrenamiento
```
python traning.py
```
El modelo entrenado se guardará automáticamente como "trained_model" en la carpeta del proyecto.

## Visualización
```
python view.py
```

## Ejecución rápida
Puedes usar el script orden.sh:
```
chmod +x orden.sh
./orden.sh
```

![CartPole Visualization](https://github.com/user-attachments/assets/c899c84a-e098-45e2-9579-eec26a2d510d)

## Cambios recientes
- Actualizado para usar Gymnasium en lugar de Gym
- Corregido el formato de retorno de step() para cumplir con la API actual
- Mejorado el sistema de recompensas personalizadas
