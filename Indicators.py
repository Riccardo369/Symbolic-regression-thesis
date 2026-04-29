from scipy.stats import pearsonr, spearmanr, rankdata
import numpy as np
import Ball
import dcor

#For linearity correlation
def pearson_correlation(x, y):
    return pearsonr(x, y)[0]

#For monoton correlation
def spearman_correlation(x, y):
    return spearmanr(x, y)[0]

#For sinusoidal and monoton correlation
def chatterjee_correlation(x, y):
    
    n = np.float64(len(x))
    if n < 2: return 0.0
    
    idx = np.argsort(x)
    y_ordered = y[idx]
    
    r = rankdata(y_ordered, method='ordinal').astype(np.float64)
    gr_r = rankdata(y_ordered, method='max').astype(np.float64)
    
    sum_abs_diff = np.sum(np.abs(np.diff(r)))
    
    denominator_term = np.sum(gr_r * (n - gr_r))
    
    if(denominator_term == 0): return 0.0
    
    return 1 - (sum_abs_diff / ((2.0 / n) * denominator_term))

def ball_covariance_library(X, Y):
    return Ball.bcor(X, Y)

def distance_covariance_library(X, Y):
    return dcor.distance_correlation(X, Y)



if(__name__ == "__main__"):

    import time

    X = np.random.random((3, 100))

    Y = np.random.random((1, 100))
    
    print(X.shape, Y.shape)

    start = time.time()
    risultato = distance_covariance_library(X, Y)
    finish = time.time()
    print(f"DCorr correlation tra X (3D) e Y (1D): {risultato}")
    print("Delta time: ", finish - start)
    
    print("")
    
    start = time.time()
    risultato = ball_covariance_library(X, Y)
    finish = time.time()
    print(f"Ball correlation tra X (3D) e Y (1D): {risultato}")
    print("Delta time: ", finish - start)

    
