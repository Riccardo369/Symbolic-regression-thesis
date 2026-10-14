import numpy as np
import Operations as op

def MAE(y_model, y_true):
    return np.mean(np.abs(y_model - y_true))

def MSE(y_model, y_true):
    with np.errstate(over='ignore'):
        return np.mean((y_model - y_true)**2)

def R2(y_model, y_true):
    return 1 - (MSE(y_model, y_true) / np.var(y_true))