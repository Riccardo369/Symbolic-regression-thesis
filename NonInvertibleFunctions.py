import numpy as np


def __pseudo_inverse_periodic_input(real_input, theta_base, second_candidate_func):
    
    period = 2*np.pi
    
    cand_a = theta_base
    cand_b = second_candidate_func(theta_base)
    
    #Apply the modular operation near to point
    cand_a_shifted = real_input + (cand_a - real_input + period/2) % period - period/2
    cand_b_shifted = real_input + (cand_b - real_input + period/2) % period - period/2
    
    # Calculate distance from real_point
    dist_a = np.abs(cand_a_shifted - real_input)
    dist_b = np.abs(cand_b_shifted - real_input)
    
    return np.where(dist_a < dist_b, cand_a_shifted, cand_b_shifted)


def pseudo_inverse_input_sin(real_input, output):
    return __pseudo_inverse_periodic_input(real_input, np.arcsin(output), lambda t: np.pi - t)

def pseudo_inverse_input_cos(real_input, output):
    return __pseudo_inverse_periodic_input(real_input, np.arccos(output), lambda t: -t)

