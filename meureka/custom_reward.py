def custom_reward(observation):
    angle = observation[2]
    x_pos = observation[0]
    x_velocity = observation[1]
    angle_velocity = observation[3]
    reward = 0.0
    
    # Base reward for maintaining balance
    reward += 0.1
    
    # Quadratic penalty for angle (soft near 0, strong when moving away)
    angle_penalty = angle**2 * 5
    reward -= angle_penalty
    
    # Penalty for high angular velocity
    angular_velocity_penalty = abs(angle_velocity) * 0.1
    reward -= angular_velocity_penalty
    
    # Penalty for extreme position
    if abs(x_pos) > 1.5:
        reward -= abs(x_pos) * 0.5
    
    # Exponential bonus for angle close to zero
    if abs(angle) < 0.05:
        reward += 1.0
    elif abs(angle) < 0.1:
        reward += 0.5
    
    # Decreasing bonus for low velocity (encourages smoothness)
    if abs(angle_velocity) < 0.5:
        reward += 0.2 * (0.5 - abs(angle_velocity))
    
    # Terminal penalty for failure
    if abs(angle) > 0.5 or abs(x_pos) > 2.4:
        reward -= 5.0
    
    return float(reward)