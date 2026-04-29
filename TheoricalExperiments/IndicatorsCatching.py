import numpy as np

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path: sys.path.append(parent_dir)

import Indicators as ind


rng = np.random.default_rng()

def calculate_indicators(X, Y, label):
    
    try:
 
        mask = np.isfinite(Y)
        X = X[mask]    
        Y = Y[mask]
        
    except: pass
    
    try: DCorr = ind.distance_covariance_library(X.T, Y.T)
    except: DCorr = None
    
    try: Pearson = ind.pearson_correlation(X, Y)
    except: Pearson = None
    
    try: Spearman = ind.spearman_correlation(X, Y)
    except: Spearman = None
    
    try: Chaterjee = ind.chatterjee_correlation(X, Y)
    except: Chaterjee = None
    
    print("Function:  ", label)
    
    print("DCorr:     ", DCorr)
    print("Pearson:   ", Pearson)
    print("Spearman:  ", Spearman)
    print("Chaterjee: ", Chaterjee)
    print("")


n = 10000

X = np.array([np.array([rng.uniform(-100, 100) for _ in range(n)]),
              np.array([rng.uniform(-100, 100) for _ in range(n)])
              ])

test_functions = [
    
                  #Linear cases
                  #"3",
                  "5*X[0] + 3",
                  "X[0]",
                  "X[0] + X[1]",               
                  "X[0] * X[1]",
                  "3*X[0]",
                  "X[0] + 5",
                  "X[0]*X[1] + 5",
                  "3*X[0] + X[1]",
                  "3*X[0] + 6",
                  "X[0]*X[1] + X[0]",
                  
                  #Non linear cases
                  "-X[0]",
                  "1/X[0]",
                  "np.sin(X[0])",
                  "np.cos(X[0])",
                  "np.abs(X[0])",
                  "np.exp(X[0])",
                  "np.log(X[0])",

                  #No function cases
                  "np.array([rng.uniform(-100, 100) for _ in range(n)])",
                  
                  
                  #Compose Cases
                  
                  "-(X[0] + X[1])",
                  "-(X[0] * X[1])",
                  
                  "1/(X[0] + X[1])",
                  "1/(X[0] * X[1])",
                  
                  "np.sin(X[0] + X[1])",
                  "np.sin(X[0] * X[1])",
                  
                  "np.cos(X[0] + X[1])",
                  "np.cos(X[0] * X[1])",
                  
                  "np.abs(X[0] + X[1])",
                  "np.abs(X[0] * X[1])",
                  
                  "np.exp(X[0] + X[1])",
                  "np.exp(X[0] * X[1])",
                  
                  "np.log(X[0] + X[1])",
                  "np.log(X[0] * X[1])"
                  
                  
                  
                  #"np.sin(X[0] + X[1])",
                  #"np.cos(X[0] * X[1])",
                  #"np.log(np.sin(X[0]))",
                  #"np.sin(np.log(X[0]))",
                  #"np.log(np.abs(np.sin(X[1])))",
                  #"np.sin(np.abs(X[1]))",
                  #"np.log(np.abs(np.sin(X[1] + X[0])))",
                  #"np.log(np.abs(X[0])) + np.sin(X[1] + X[0])",
                
                  #"X[0]**2 + np.cos(X[0])",
                  #"X[0]**2 * np.cos(X[0])",
                  #"X[0]**2 * np.cos(X[0]**2)",
                  #"2**X[0] + np.cos(X[0])"
                  
                  
                  
                  #"X[0]**2",
                  #"X[0]**2 + 2*X[0] + 5",
                  #"2*X[0]**3 + X[0]**2 + 2*X[0] + 5",
                  #"8*X[0]**4 + 2*X[0]**3 + X[0]**2 + 2*X[0] + 5",
                  
                  #"np.sin(X[0]**2)",
                  #"2**np.cos(X[0])",
                  
                  
                  ]




for test in test_functions:
    
    Y = eval(test)
    
    mask = np.isfinite(Y)
    
    K = np.array([x[mask] for x in X])
    R = Y[mask]
    
    #ind.distance_covariance_library(K.T, np.array([R].T))
    
    #try: dist_cov = ind.distance_covariance_library(K.T, np.array([R]).T)
    #except: dist_cov = "fainted"
    
    #ball_cov = ind.ball_covariance_library(K, R)
    
    #try: ball_cov = ind.ball_covariance_library(K, R)
    #except: ball_cov = "fainted"
    
    #Max = np.max([np.max(np.abs(K)), np.max(np.abs(R))])
    #K = K / Max
    #R = R / Max
    
    #try: dist_cov_norm = ind.distance_covariance_library(K, np.array([R]))
    #except: dist_cov_norm = "fainted"
    
    #try: ball_cov_norm = ind.ball_covariance_library(K, R)
    #except: ball_cov_norm = "fainted"
    
    for i, x in enumerate(K):
        
        noise = rng.normal(loc = 0, scale = 1, size=len(x))
        
        label = test + f", corr(X[{i}], Y)"
        print(label)
        
        #print(f"Distance covariance: {dist_cov}")
        #print(f"Ball covariance: {ball_cov}")
        #print(f"Distance covariance norm: {dist_cov_norm}")
        #print(f"Ball covariance norm: {ball_cov_norm}")
        
        calculate_indicators(x, R,         label)
        #calculate_indicators(x, Y + noise, label + " (with gaussian noise N(0, 1))")
        
        print("")
        


