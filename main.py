import numpy as np
import matplotlib.pyplot as plt

import LearningStrategy as ls
from Model import SymbolicRegression
import Operations as op
import Errors as er

X = np.array([x for x in range(1, 100, 1)])
Y = 2*X + 1000

tree = SymbolicRegression.string_to_tree("0")


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
                                                "MAE error": lambda model: er.MAE(Y, model.get_root().get_memory("forward")),
                                                "MDL": lambda model: er.MDL(Y, model.get_root().get_memory("forward"), model)
                                                }
                            )


print(metric_results)