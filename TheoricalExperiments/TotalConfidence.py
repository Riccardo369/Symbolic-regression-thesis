from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path: sys.path.append(parent_dir)

import Indicators as ind
from Model import SymbolicRegression
import Errors as er

def total_confidence(node):
    
    result = 1
    for c in node.get_childrens(): result *= total_confidence(c)
    result = result * node.calculation_hat_confidence()
    return result if np.isfinite(result) else 0


if(__name__ == "__main__"):
    
    X = np.linspace(1, 100, 100)
    Y = np.sin(4*X)

    model1 = SymbolicRegression.string_to_tree("sin(X)")
    model2 = SymbolicRegression.string_to_tree("sin(mul(X, 4))")
    model3 = SymbolicRegression.string_to_tree("sin(mul(X, X))")

    for m in [model1, model2, model3]:
        
        m.get_root().forward_calculation_memory_sub_tree({"X": X}).backward_calculation_memory_sub_tree(Y)
        print("model", m, "confidence", total_confidence(m.get_root()))
        
    plt.plot(X, Y, color = "black", label = "real function")
    #plt.plot(X, model1.get_root().get_memory("forward"), color = "purple", label = "model 1")
    #plt.plot(X, model2.get_root().get_memory("forward"), color = "green", label = "model 2")
    plt.plot(X, model3.get_root().get_memory("forward"), color = "red", label = "model 3")
    plt.legend()
    plt.show()

