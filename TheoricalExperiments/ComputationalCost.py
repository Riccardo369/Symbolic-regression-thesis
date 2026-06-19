import time
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path: sys.path.append(parent_dir)

import Indicators as ind
from utils_experiment import create_random_model
import Operations as op

import Ball
import dcor


"""['CompileMode', 
 'DistanceCovarianceMethod', 
 'EstimationStatistic', 
 'HypothesisTest', 
 'RowwiseMode', 
 'Stats', 
 '__builtins__', 
 '__cached__', 
 '__doc__', 
 '__file__', 
 '__loader__', 
 '__name__', 
 '__package__', 
 '__path__', 
 '__spec__', 
 '__version__', '_dcor',
 '_dcor_internals', 
 '_dcor_internals_numba', 
 '_energy', 
 '_fast_dcov_avl', 
 '_fast_dcov_mergesort', 
 '_hypothesis', 
 '_partial_dcor',
 '_rowwise', '_utils', 
 'distance_correlation', 
 'distance_correlation_af_inv', 
 'distance_correlation_af_inv_sqr', 
 'distance_correlation_sqr', 
 'distance_covariance', 
 'distance_covariance_sqr', 
 'distance_stats', 
 'distance_stats_sqr', 
 'distances',
 'double_centered', 
 'energy_distance', 
 'homogeneity', 
 'independence', 
 'mean_product', 
 'partial_distance_correlation', 
 'partial_distance_covariance', 
 'rowwise', 
 'u_centered', 
 'u_complementary_projection', 
 'u_distance_correlation_sqr', 
 'u_distance_covariance_sqr', 
 'u_distance_stats_sqr', 
 'u_product', 
 'u_projection']"""



#Range = (-100, 100)
#n = 100
#X = np.random.uniform(low = Range[0], high = Range[1], size = (1, n))
#Y = np.random.uniform(low = Range[0], high = Range[1], size = (1, n))

#print(Ball.bcor(X, Y))

#print(dir(Ball))

#print("-"*100)

#print(help(dcor.))

#print(dcor.distance_covariance(X, Y))

#raise ValueError("STOP")

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

def test(Y_creation):
    
    DCorr_list = []
    Ball_list = []
    
    for n in N:
          
        X = np.random.uniform(low = Range[0], high = Range[1], size = (k, n))
        Y = Y_creation(X)
        
        print(X.shape, Y.shape)
        
        DCorr_list.append(measure(X, Y, n, k, dcor.distance_covariance, "DCorr"))
        Ball_list.append(measure(X, Y, n, k, Ball.bcov, "Ball"))

    plt.plot(N, DCorr_list, color = "red", label = "DCorr time")
    plt.plot(N, Ball_list, color = "blue", label = "Ball time")
    plt.title(f"{k} features", color = "black")
    plt.legend()
    plt.show()


def Y_by_model(X):
    
    features = {f"X{i}": X[i] for i in range(X.shape[0])}
    
    while(True):
    
        model = create_random_model(features, 5, [op.add, op.mul, op.neg, op.div, op.abs]) 
        output = model.get_root().forward_calculation_memory_sub_tree(features).get_memory("forward")
        
        if(np.all(np.isfinite(output)) and isinstance(output, np.ndarray)): break
        
    return output
        
    
for k in K:
    
    print("X and Y casual")
    test(lambda x: np.random.uniform(low = Range[0], high = Range[1], size = (1, x.shape[1])))
    
    print("X and Y correlated by random function")
    test(lambda x: np.array([Y_by_model(x)]))


#Già con una feature, la Ball cresce in modo quadratico, mentre la Dcorr in modo lineare. In più la ball covariance con pochi dati non
#cattura bene se c'è una funzione (ma neanche la DCorr), ed all' aumentare dei dati rallenta di tantissimo.