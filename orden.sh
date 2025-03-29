#!/bin/bash

# Activar el entorno conda
conda activate em

# Ejecutar los scripts en orden
python code_generator.py
python int_gym.py
python traning.py
