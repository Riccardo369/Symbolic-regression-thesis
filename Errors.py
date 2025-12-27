import numpy as np

def MAE(x, y):
    return np.mean(np.abs(x - y))

def MSE(x, y):
    return np.sum((x - y)**2)

def WMAE(x, y):
    diff = np.abs(x - y)
    w = diff / np.sum(diff)
    w = 1 - w
    w /= np.sum(w)
    return np.sum(diff * w)