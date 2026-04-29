import time
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path: sys.path.append(parent_dir)

import Indicators as ind

def measure(X, Y, n, k, application, name):
    start = time.time()
    application(X, Y)
    finish = time.time()
    delta = finish - start
    print(f"n = {n}, Xk = {k}, delta time: ", delta, f"({name} correlation)")
    return delta


Range = (-100, 100)

N = [100, 500, 1000, 2000, 4000, 8000]
K = [1, 2, 3, 4]

for k in K:
    
    DCorr = []
    Ball = []
    
    for n in N:
          
        X = np.random.uniform(low = Range[0], high = Range[1], size = (k, n))
        Y = np.random.uniform(low = Range[0], high = Range[1], size = (1, n))
        
        DCorr.append(measure(X.T, Y.T, n, k, ind.distance_covariance_library, "DCorr"))
        #Ball.append(measure(X, Y, n, k, ind.ball_covariance_library, "Ball"))

    plt.plot(N, DCorr, color = "red", label = "DCorr time")
    #plt.plot(N, Ball, color = "blue", label = "Ball time")
    plt.title(f"{k} features", color = "black")
    plt.legend()
    plt.show()
