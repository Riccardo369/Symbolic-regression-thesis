import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import LearningStrategy as ls
from Model import SymbolicRegression
import Operations as op
import Errors as er

X = np.array([x for x in range(1, 1000, 1)], dtype = np.float64) / 10
Y = np.cos(X)

tree = SymbolicRegression.string_to_tree("0")

#metric_results = tree.learn({"X": X}, Y, {"X": X},
#                            Y,
#                            ls.AnalysisRamification([op.add,
#                                                    op.mul,
#                                                    op.div,
#                                                    op.cos]),
#                            metric_functions = {"tree": lambda model: str(model.get_root()),
#                                                "MAE error": lambda model: er.MAE(Y, model.get_root().get_memory("forward"))}
#                            )


#print(metric_results)



gradient = ls.AnalysisRamification.estimate_gradient({"X": X}, Y)
hessian = ls.AnalysisRamification.estimate_hessian({"X": X}, Y)

fig = plt.figure(facecolor='black')
ax = plt.axes()

ax.set_facecolor('black')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

print("function: ", Y.shape)
print("gradient: ", gradient.shape)
print("hessian: ", hessian.shape)
      
plt.plot(X, Y, color = "red", label = f"function")
plt.plot(X, gradient, color = "blue", label = f"inferred gradient")
plt.plot(X, hessian.squeeze(0).squeeze(1), color = "green", label = f"inferred hessian")

plt.legend()
plt.show()


#tree = SymbolicRegression.string_to_tree("mul(cos(X), sin(X))")
#gradient = tree.get_gradient_function(["X", "Y"])
#hessian = tree.get_hessian_function(["X", "Y"])
#print(np.mean(np.array(gradient({"X": X, "Y": Y})), axis = 1))
#print(np.mean(np.array(hessian({"X": X, "Y": Y})), axis = 2))