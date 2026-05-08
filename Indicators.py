from scipy.stats import pearsonr, spearmanr, rankdata
import numpy as np

def prepare(x, y):
    
    def is_constant(x):
        return type(x) in [int, float] or np.isscalar(x)
    
    def is_numpy_array(x):
        return isinstance(x, np.ndarray)

    if(is_constant(x) and is_constant(y)): return np.array([x]*2), np.array([y]*2)
    if(is_constant(x) and is_numpy_array(y)): return np.array([x]*len(y)), y
    if(is_constant(y) and is_numpy_array(x)): return x, np.array([y]*len(x))
    if(is_numpy_array(x) and is_numpy_array(y)): return x, y

    raise ValueError(f"There is a problem, it is not possible to recognize x (type {type(x)}) or y (type {type(y)})")
    
#For linearity correlation
def pearson_correlation(x, y):
    x, y = prepare(x, y)
    result = pearsonr(x, y)[0]
    return result if not np.isnan(result) else 0

#For monoton correlation
def spearman_correlation(x, y):
    x, y = prepare(x, y)
    result = spearmanr(x, y)[0]
    return result if not np.isnan(result) else 0

#For sinusoidal and monoton correlation
def chatterjee_correlation(x, y):
    
    x, y = prepare(x, y)
    
    n = np.float64(len(x))
    if n < 2: return 0.0
    
    idx = np.argsort(x)
    y_ordered = y[idx]
    
    r = rankdata(y_ordered, method='ordinal').astype(np.float64)
    gr_r = rankdata(y_ordered, method='max').astype(np.float64)
    
    sum_abs_diff = np.sum(np.abs(np.diff(r)))
    
    denominator_term = np.sum(gr_r * (n - gr_r))
    
    if(denominator_term == 0): return 0.0
    
    result = 1 - (sum_abs_diff / ((2.0 / n) * denominator_term))
    
    return result if not np.isnan(result) else 0
