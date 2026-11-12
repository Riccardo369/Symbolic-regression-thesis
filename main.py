import numpy as np
import time

from Model import SymbolicRegression

class Counter:
        
        def __init__(self, i):
            self.__i, self.__r = i, 0
            
        def __call__(self):
            self.__r += 1
            return self.__r > self.__i


def task(TR_inputs, TR_output, name_json):

    counter = Counter(10)

    tree = SymbolicRegression.string_to_tree("0")
    
    metric_results = tree.learn(TR_inputs,
                                TR_output,
                                stop_criteria = lambda ls: counter(),
                                metric_functions = {"tree": lambda model: str(model), "timestamp": lambda model: time.time()})
    
    print(tree)
    
    
#print(SymbolicRegression.string_to_tree("add(0, cos(X))").simplify_model())
#print(SymbolicRegression.string_to_tree("mul(0, cos(X))").simplify_model())
#print(SymbolicRegression.string_to_tree("add(1, cos(X))").simplify_model())
#print(SymbolicRegression.string_to_tree("mul(1, cos(X))").simplify_model())
#print(SymbolicRegression.string_to_tree("mul(sqrt(X0), sqrt(add(sqrt(X0), sqrt(X0))))").simplify_model())

#raise ValueError("STOP")

TR_inputs = {"X": np.linspace(-10, 10, 10000, dtype = np.float64)}
TR_output = np.sin(TR_inputs["X"]) + np.sin(np.exp(TR_inputs["X"]))

task(TR_inputs, TR_output, "prova1.json")

