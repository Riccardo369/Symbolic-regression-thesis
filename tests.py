import numpy as np
import matplotlib.pyplot as plt

import LearningStrategy as ls
from Model import SymbolicRegression
import Operations as op
import Errors as er

show_plot = True

test_functions = ["3",
                  "X",
                  "np.cos(X)",
                  "X**2 + 2*X",
                  "X**2 + np.cos(X)",
                  "X**2 * np.cos(X)",
                  "X**2 * np.cos(X**2)",
                  "2**X + np.cos(X)"
                  ]

X = np.array([x for x in range(1, 1000, 1)], dtype = np.float64) / 100


for test in test_functions:
    
    model = SymbolicRegression.string_to_tree("0")
    
    Y = eval(f"lambda X: {test}")(X)
    
    metric_results = model.learn({"X": X}, Y, {"X": X},
                            Y,
                            ls.AnalysisRamification([op.add,
                                                    op.mul,
                                                    op.div,
                                                    op.cos]),
                            metric_functions = {"tree": lambda model: str(model.get_root()),
                                                "MAE error": lambda model: er.MAE(Y, model.get_root().get_memory("forward"))}
                            )
    
    print("*"*100)
    print("Ideal:   ", test)
    print("inferred:", metric_results["result"]["tree"][-1])
    print("*"*100)
    
    if(show_plot):
        
        inferred_Y = model.get_calculation_function()({"X": X})
        
        if(type(Y) in [int, float]): Y = np.array([Y] * len(X))
        if(type(inferred_Y) in [int, float]): inferred_Y = np.array([inferred_Y] * len(X))
        
        fig = plt.figure(facecolor='black')
        ax = plt.axes()

        ax.set_facecolor('black')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        plt.plot(X, Y, color = "red", label = f"ideal: {test}")
        plt.plot(X, inferred_Y, color = "blue", label = f"inferred: {model.get_root().get_string_calculation_function()['body']}")

        plt.legend()
        plt.show(block = False)
    
if(show_plot): plt.show(block = True)
