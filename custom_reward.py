def custom_reward(observation):
    angle = observation[2]
    x_pos = observation[0]
    reward = 1.0
    
    # Penalización por ángulo excesivo (escalado con el ángulo)
    angle_penalty = min(max(abs(angle) * 5, 0), 2)
    reward -= angle_penalty
    
    # Penalización por desviación horizontal extrema
    if abs(x_pos) > 2.4:
        reward -= 2.0
    
    # Bonus por mantener el poste cerca de la vertical
    if abs(angle) < 0.1:
        reward += 0.5
    
    # Bonus de supervivencia (aumenta con el tiempo)
    reward += 0.01
    
    return float(reward)
