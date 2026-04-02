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
