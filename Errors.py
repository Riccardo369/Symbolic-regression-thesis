import numpy as np
import Operations as op

def MDL(Y, Y_model, model):
    
    def count_nodes(node):
        if(not issubclass(type(node), op.Leaf)): return sum([count_nodes(x)+1 for x in node.get_childrens()])
        else: return 1
        
    return ((Y.shape[0]*np.log(np.mean(np.abs(Y - Y_model)))) + (count_nodes(model.get_root())*np.log(Y.shape[0]))) / 2  


def MaxAE(x, y):
    return np.max(np.abs(x - y))

def MAE(x, y):
    return np.mean(np.abs(x - y))


def MSE(x, y):
    with np.errstate(over='ignore'):
        return np.mean((x - y)**2)
