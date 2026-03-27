import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import LearningStrategy as ls
from Model import SymbolicRegression
import Operations as op
import Errors as er

from Indicators import pearson_correlation, spearman_correlation, chatterjee_correlation, dependence_correlation

#from ball import bcor


X = np.array([x for x in range(1, 100, 1)], dtype = np.float64)
#Y = np.array(2*X + 1000, dtype = np.float64)
Y = np.cos(X)

tree = SymbolicRegression.string_to_tree("add(mul(X, 3), 5)")

#print([x for x in dir(xicorpy.XiCorrelation(X, Y)) if x[0] != "_"])


print("Pearson", pearson_correlation(X, Y))
print("Spearman", spearman_correlation(X, Y))
print("Chaterjee", chatterjee_correlation(X, Y))
#print("Dependence", dependence_correlation(np.array([X, 2*X, 3*X]), Y))

#print(BallCorr().test(np.array([X, Y]), Y)[0])


#tree.get_root().view_sub_tree()


#class Counter:
#    
#    def __init__(self, n):
#        self.__n = n
#        self.__i = 0
#        
#    def __call__(self):
#        self.__i += 1
#        return not self.__i < self.__n
#
#counter = Counter(5)    
#
#metric_results = tree.learn({"X": X}, Y, {"X": X},
#                            Y,
#                            ls.AnalysisRamification([op.add,
#                                                    op.sub,
#                                                    op.mul,
#                                                    op.div]),
#                            lambda model, metrics: #counter() or
#                            metrics["MAE error"][-1] == 0,
#                            metric_functions = {"tree": lambda model: str(model.get_root()),
#                                                "MAE error": lambda model: er.MAE(Y, model.get_root().get_memory("forward")),
#                                                "MDL": lambda model: er.MDL(Y, model.get_root().get_memory("forward"), model)
#                                                }
#                            )
#
#
#print(metric_results)