import numpy as np
import matplotlib.pyplot as plt

import LearningStrategy as ls
from Model import SymbolicRegression
import Operations as op
import Errors as er

X = np.array([x for x in range(1, 100, 1)])
Y = 2*X + 1000

tree = SymbolicRegression.string_to_tree("0")

def MDL(model):
    
    def count_nodes(node):
        if(not issubclass(type(node), op.Leaf)): return sum([count_nodes(x)+1 for x in node.get_childrens()])
        else: return 1
        
    return ((Y.shape[0]*np.log(np.mean(np.abs(Y - model.get_root().get_forward_calculation_memory())))) + (count_nodes(model.get_root())*np.log(Y.shape[0]))) / 2  


def MAE(model):
    
    root = model.get_root()
    
    root.forward_calculation_memory_sub_tree({"X": X}).backward_calculation_memory_sub_tree(Y)
    result = np.mean(np.abs(root.get_forward_calculation_memory() - root.get_backward_calculation_memory()))
    return result


class Counter:
    
    def __init__(self, n):
        self.__n = n
        self.__i = 0
        
    def __call__(self):
        self.__i += 1
        return not self.__i < self.__n

counter = Counter(5)    

metric_results = tree.learn({"X": X}, Y, {"X": X},
                            Y,
                            ls.AnalysisRamification([op.add,
                                                    op.sub,
                                                    op.mul,
                                                    op.div]),
                            lambda model, metrics: #counter() or
                            metrics["MAE error"][-1] == 0,
                            metric_functions = {"tree": lambda model: str(model.get_root()),
                                                "MAE error": lambda model: MAE(model),
                                                "MDL": lambda model: MDL(model)})


#print(metric_results)


#tree.view(f = lambda x: np.mean(np.abs(x.get_forward_calculation_memory() - x.get_backward_calculation_memory())))