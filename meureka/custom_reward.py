def custom_reward(observation):
    angle = observation[2]
    x_pos = observation[0]
    reward = 0.0
    
    # Recompensa base por mantener el equilibrio
    reward += 0.1
    
    # Penalización cuadrática por ángulo (suave cerca de 0, fuerte al alejarse)
    angle_penalty = angle**2 * 10
    reward -= angle_penalty
    
    # Penalización por salirse de los límites horizontales
    if abs(x_pos) > 2.0:
        reward -= 5.0
    
    # Bonus exponencial por mantener ángulo cercano a 0
    if abs(angle) < 0.05:
        reward += 2.0
    elif abs(angle) < 0.1:
        reward += 1.0
    
    # Bonus progresivo por tiempo equilibrado (crece con ángulo controlado)
    time_bonus = 0.02 * (1.0 - min(abs(angle)/0.2, 1.0))
    reward += time_bonus
    
    return float(max(reward, -1.0))