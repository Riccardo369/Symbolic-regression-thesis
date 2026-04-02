import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import LearningStrategy as ls
from Model import SymbolicRegression
import Operations as op
import Errors as er

X = np.array([x for x in range(1, 100, 1)], dtype = np.float64)
Y = 2

tree = SymbolicRegression.string_to_tree("0")



metric_results = tree.learn({"X": X}, Y, {"X": X},
                            Y,
                            ls.AnalysisRamification([op.add,
                                                    op.mul,
                                                    op.div,
                                                    op.cos]),
                            metric_functions = {"tree": lambda model: str(model.get_root()),
                                                "MAE error": lambda model: er.MAE(Y, model.get_root().get_memory("forward"))}
                            )


print(metric_results)