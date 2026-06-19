from scipy.stats import pearsonr, spearmanr, rankdata, kendalltau
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
    
    if(np.var(y) == 0): return 1
    if(np.var(x) == 0 and np.var(y) > 0): return 0
    
    result = pearsonr(x, y)[0]
    return result

#For monoton correlation
def spearman_correlation(x, y):
    x, y = prepare(x, y)
    
    if(np.var(y) == 0): return 1
    if(np.var(x) == 0 and np.var(y) > 0): return 0
    
    result = spearmanr(x, y)[0]
    return result

#For monoton correlation
def kendall_correlation(x, y):
    result = kendalltau(*prepare(x, y))[0]
    if(np.isnan(result)): result = 0
    return result

#For sinusoidal and monoton correlation
#With less points, chatterjee is less precise with identification
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
    
    return result




if(__name__ == "__main__"):
    
    x = 3
    y = np.array([1, 2, 3])

    f = lambda x, y: pearson_correlation(x, y)

    print("Pearson")
    print(f(x, x))
    print(f(x, y))
    print(f(y, x))
    print(f(y, y))

    f = lambda x, y: spearman_correlation(x, y)

    print("Spearman")
    print(f(x, x))
    print(f(x, y))
    print(f(y, x))
    print(f(y, y))
    
    f = lambda x, y: kendall_correlation(x, y)
    
    print("Kendall")
    print(f(x, x))
    print(f(x, y))
    print(f(y, x))
    print(f(y, y))

    f = lambda x, y: chatterjee_correlation(x, y)

    print("Chaterjee")
    print(f(x, x))
    print(f(x, y))
    print(f(y, x))
    print(f(y, y))
    