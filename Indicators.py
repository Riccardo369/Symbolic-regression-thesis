#import os
#import sys
#mingw_path = r'C:\mingw64\bin' 
#if os.path.exists(mingw_path): os.add_dll_directory(mingw_path)
#import Ball

#import dcor

#print(help(Ball.bcor))
#print("-"*100)
#print(help(Ball.bcov))
#print("-"*100)
#print(help(Ball.bcorsis))
#print("-"*100)

#print(Ball.bcov_test.__doc__)

from scipy.stats import pearsonr, spearmanr, rankdata
import numpy as np

#For linearity correlation
def pearson_correlation(x, y):
    return pearsonr(x, y)[0]

#For monoton correlation
def spearman_correlation(x, y):
    return spearmanr(x, y)[0]

#For monoton and sinusoidal correlation
def chatterjee_correlation(x, y):

    n = len(x)
    
    y_ordered = y[np.argsort(x)]
    r = rankdata(y_ordered, method='ordinal')
    gr_r = rankdata(y_ordered, method='max')
    
    return 1 - (n * np.sum(np.abs(np.diff(r)))) / (2 * np.sum(gr_r * (n - gr_r)))

#For calculation the dependence (therefore a possible function) between che features x and the output y
def dependence_correlation(x, y):
    
    if(len(x.shape) != 2 and len(y.shape) != 1): raise TypeError("x must be a matrix, every row is a feature, and y must be a vector")
    
    #dist_X = dcor.distances.pairwise_distances(x.T)
    
    #print(dist_X.shape)

    #return dcor.distance_correlation(x.T, y)
    #return Ball.bcov(x.T, y)[0]


