from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path: sys.path.append(parent_dir)

import Indicators as ind
from Model import SymbolicRegression
import Errors as er
import Operations as op

def constant_confidence(x, y):
    return 1 / (np.var(y) + 1)

def identity_confidence(x, y):
    return ind.pearson_correlation(x, y)

def neg_confidence(x, y):
    return -ind.pearson_correlation(x, y)

def abs_confidence(x, y):
    
    In, Out = ind.prepare(x, y)
    
    mask_left = In <= 0
    mask_right = In >= 0
    
    if(np.sum(mask_left) < 2 and np.sum(mask_right) < 2): return 0
        
    result_left = ind.pearson_correlation(In[mask_left], Out[mask_left]) if np.sum(mask_left) > 1 else 1
    result_right = -ind.pearson_correlation(In[mask_right], Out[mask_right]) if np.sum(mask_right) > 1 else 1
        
    result = np.abs(result_left * result_right)
    return result

def div_confidence(x, y):
    
    In, Out = ind.prepare(x, y)
    
    mask_left = In < 0
    mask_right = In > 0
    
    if(np.sum(mask_left) < 2 and np.sum(mask_right) < 2): return 0
        
    result_left = ind.pearson_correlation(In[mask_left], Out[mask_left]) if np.sum(mask_left) > 1 else 1
    result_right = -ind.pearson_correlation(In[mask_right], Out[mask_right]) if np.sum(mask_right) > 1 else 1
        
    result = np.abs(result_left * result_right)
    return result

def ln_confidence(x, y):
    return ind.pearson_correlation(np.log(x), y) * ind.spearman_correlation(x, y)

def exp_confidence(x, y):
    return ind.pearson_correlation(np.exp(x), y) * ind.spearman_correlation(x, y)

def sin_confidence(x, y):
    return ind.pearson_correlation(np.sin(x), y) * ind.chatterjee_correlation(x, y)

def cos_confidence(x, y):
    return ind.pearson_correlation(np.cos(x), y) * ind.chatterjee_correlation(x, y)


rng = np.random.default_rng()

confidence_dict = {#"c":      lambda x, y: constant_confidence(x, y),
                   "X":      lambda x, y: identity_confidence(x, y),
                   "-X":     lambda x, y: neg_confidence(x, y),
                   "|X|":    lambda x, y: abs_confidence(x, y),
                   "1/X":    lambda x, y: div_confidence(x, y),
                   "ln(X)":  lambda x, y: ln_confidence(x, y),
                   "exp(X)": lambda x, y: exp_confidence(x, y),
                   "sin(X)": lambda x, y: sin_confidence(x, y),
                   "cos(X)": lambda x, y: cos_confidence(x, y)}

transform_dict = {#"c":       lambda x: rng.normal(loc = 0, scale = 0.5, size = len(x)),
                  "X":       lambda x: x,
                  "-X":      lambda x: -x,
                  "|X|":     lambda x: np.abs(x),
                  "1/X":     lambda x: 1/x,
                  "ln(X)":   lambda x: np.log(x),
                  "exp(X)":  lambda x: np.exp(x),
                  "sin(X)":  lambda x: np.sin(x),
                  "cos(X)":  lambda x: np.cos(x)}

original_X = rng.normal(loc = 0, scale = 200, size = 10000)


for tr in transform_dict.keys():
    
    Y = transform_dict[tr](original_X)
        
    mask = np.isfinite(Y)
        
    X = original_X[mask]
    Y = Y[mask]
    
    results = sorted([{"type": key, "confidence": confidence_dict[key](X, Y)} for key in confidence_dict.keys()], key = lambda x: -x["confidence"])
    
    winners = [x["type"] for x in results if x["confidence"] == results[0]["confidence"]]
    
    print(f"Real transformation wins: {tr in winners}. Transformations: real = {tr}, winners = ({', '.join(winners)})")


#Questo file dovrebbe dimostrare abbastanza la validità di usare le confidence per capire la funzione che si trova al di sopra, in questo modo, si può già sapere quale
# funzione mettere, perchè la X randomica rappresenta qualsiasi forma di sotto albero che proviene da sotto