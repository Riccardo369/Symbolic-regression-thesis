import numpy as np
import Errors as er
import portion as P

def transform_domain(interval, shift = 0, scale = 0):
    
    if interval.empty:
        return P.empty()
    
    transformed = P.empty()
    for atomic in interval:
        # Calcoliamo i nuovi limiti
        new_lower = atomic.lower * scale + shift
        new_upper = atomic.upper * scale + shift
        
        # Se la scala è negativa, i limiti si invertono!
        if scale < 0:
            new_lower, new_upper = new_upper, new_lower
            new_left, new_right = atomic.right, atomic.left
        else:
            new_left, new_right = atomic.left, atomic.right
            
        transformed |= P.interval(P.AbstractInterval.from_atomic(new_left, new_lower, new_upper, new_right))
        
    return transformed


#Domain modification
def domains_shift(intervals, shift_list):
    return [transform_domain(x, shift = y) for x, y in zip(intervals, shift_list)]

def domains_scale(intervals, scale_list):
    return [transform_domain(x, scale = y) for x, y in zip(intervals, scale_list)]

def domains_complementary(intervals):
    return [~x for x in intervals]

def domains_intersection(intervals1, intervals2):
    return [x & y for x, y in zip(intervals1, intervals2)]

def domains_union(intervals1, intervals2):
    return [x | y for x, y in zip(intervals1, intervals2)]

def domains_difference(intervals1, intervals2):
    return [x - y for x, y in zip(intervals1, intervals2)]



#Sinusoidal expression
def __get_ideal_periodic_input(real_input, theta_base, second_candidate_func):
    
    period = 2*np.pi
    
    cand_a = theta_base
    cand_b = second_candidate_func(theta_base)
    
    # Riportiamo entrambi vicino al real_input usando il periodo corretto
    cand_a_shifted = real_input + (cand_a - real_input + period/2) % period - period/2
    cand_b_shifted = real_input + (cand_b - real_input + period/2) % period - period/2
    
    # Scegliamo il più vicino
    dist_a = np.abs(cand_a_shifted - real_input)
    dist_b = np.abs(cand_b_shifted - real_input)
    
    return np.where(dist_a < dist_b, cand_a_shifted, cand_b_shifted)

def __get_nearest_feature_input(features, output, f):
    return sorted([{"name": key, "err": er.MAE(f(features[key]), output)} for key in features.keys()], key = lambda x: x["err"])[0]["name"]


def get_ideal_input_sin(real_input, output):
    return __get_ideal_periodic_input(real_input, np.arcsin(output), lambda t: np.pi - t)

def get_ideal_input_cos(real_input, output):
    return __get_ideal_periodic_input(real_input, np.arccos(output), lambda t: -t)

def get_ideal_input_tan(real_input, output):
    period = np.pi
    theta = np.arctan(output)
    return real_input + (theta - real_input + period/2) % period - period/2


def get_nearest_feature_input_sin(features, output):
    return __get_nearest_feature_input(features, output, lambda x: np.sin(x))

def get_nearest_feature_input_cos(features, output):
    return __get_nearest_feature_input(features, output, lambda x: np.cos(x))

def get_nearest_feature_input_tan(features, output):
    return __get_nearest_feature_input(features, output, lambda x: np.tan(x))