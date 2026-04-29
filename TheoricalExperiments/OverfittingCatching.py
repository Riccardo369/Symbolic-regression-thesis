import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path: sys.path.append(parent_dir)

import Errors as er
import Indicators as ind

import numpy as np

import matplotlib.pyplot as plt

def linear_result(X, Y, n):
    m = (n*(np.sum(X * Y)) - (np.sum(X)*np.sum(Y))) / ((n*np.sum(X**2)) - (np.sum(X)**2))
    q = (np.sum(Y) - (m*np.sum(X))) / n
    #print("m: ", m, "q: ", q)
    return m*X + q

X = np.linspace(-20, 20, 200)
Y = np.exp(X) * X

noise_range = 1.0
Y = Y + np.random.uniform(-noise_range, noise_range, size=X.shape)

test = ["X",
        "-X",
        "np.exp(X)",
        "np.log(X)",
        "np.sin(X)",
        "np.cos(X)",
        "np.abs(X)",
        "1/X",
        
        "X*X",
        
        "-X*X",
        "np.exp(X)*X",
        "np.log(X)*X",
        "np.sin(X)*X",
        "np.cos(X)*X",
        "np.abs(X)*X",
        
        "(np.exp(X) * X) + 10"]

#raise ValueError("STOP")

MSE_error = []
Chaterjee_error = []
DCorr_error = []
BC_error = []

result = []

for t in test:
    
    Y_pred = linear_result(eval(t), Y, len(Y))
    
    mask = np.isfinite(Y_pred)
    
    result.append({"propose": t, 
                   "MSE": er.MSE(Y_pred[mask], Y[mask]), 
                   "Chaterjee": ind.chatterjee_correlation(Y_pred[mask], Y[mask]),
                   "DCorr": ind.distance_covariance_library(Y_pred[mask], Y[mask]),
                   "BC": ind.distance_covariance_library(Y_pred[mask], Y[mask])})

MAX_MSE_error = np.max([x["MSE"] for x in result])
for i in result: i["MSE"] /= MAX_MSE_error

for i in result: print(i)

