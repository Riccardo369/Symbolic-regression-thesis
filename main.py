import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

import LearningStrategy as ls
from Model import SymbolicRegression
import Operations as op
import Errors as er

import Indicators as ind

def chaterjee_metric(Y_output, Y_target):
    
    if(np.isscalar(Y_output) and np.isscalar(Y_target)): return -1
    
    if(np.isscalar(Y_output)): Y_output = np.array([Y_output for _ in range(len(Y_target))])
    if(np.isscalar(Y_target)): Y_target = np.array([Y_target for _ in range(len(Y_output))])
    
    return -ind.chatterjee_correlation(Y_output, Y_target)


def get_package_importance_nodes(nodes, unary_operations, TR_inputs_dict, TR_output):
        
        combinations = []
        
        for n in nodes:
            combinations.append({"typology": "constant", "node": n, "unary operation": None, "feature": np.mean(n.get_memory("backward"))})
            for f in TR_inputs_dict.keys():
                combinations.append({"typology": "feature", "node": n, "unary operation": None, "feature": f if isinstance(n, op.Leaf) else None})
                for o in unary_operations:
                    combinations.append({"typology": "feature - operation", "node": n, "unary operation": o, "feature": f if isinstance(n, op.Leaf) else None})
                
        return [combinations]





X = np.array([x for x in range(1, 10000, 1)], dtype = np.float64) / 10
Y = 2*X

tree = SymbolicRegression.string_to_tree("0")

metric_results = tree.learn({"X": X}, Y,
                            ls.AnalysisRamification([op.neg, op.div, op.sin, op.cos, op.abs, op.exp, op.ln], 
                                                    [(op.mul, 1), (op.add, 0)],
                                                    get_package_importance_nodes,
                                                    {"MAE": lambda model: er.MAE(model.get_root().get_memory("forward"), Y),
                                                     "Chaterjee": lambda model: chaterjee_metric(model.get_root().get_memory("forward"), Y)}),
                            
                            
                            metric_functions = {"tree": lambda model: str(model.get_root()),
                                                #"MAE error": lambda model: er.MAE(model.get_root().get_memory("forward"), Y),
                                                #"Chaterjee": lambda model: ind.chatterjee_correlation(model.get_root().get_memory("forward"), Y)
                                                }
                            )

print(metric_results)

