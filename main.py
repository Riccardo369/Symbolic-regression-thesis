import numpy as np
import matplotlib.pyplot as plt

import LearningStrategy as ls
from Model import SymbolicRegression
import Operations as op
import Errors as er

X = np.array([x for x in range(1, 10, 2)])
Y = 3*X + 4

tree = SymbolicRegression.string_to_tree("0")

#plt.show()

def MAE(model):
    
    root = model.get_root()
    
    root.forward_calculation_memory_sub_tree({"X": X}).backward_calculation_memory_sub_tree(Y)
    result = np.mean(np.abs(root.get_forward_calculation_memory() - root.get_backward_calculation_memory()))
    root.clear_all_calculation_memory_sub_tree()
    return result
    
metric_results = tree.learn({"X": X}, Y, {"X": X}, Y, ls.IndipendentRamification([op.add, op.sub, op.mul, op.div]), metric_functions = {"tree": lambda model: str(model.get_root()), "MAE error": lambda model: MAE(model)})

print(metric_results)