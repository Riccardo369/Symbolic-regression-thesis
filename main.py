import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os

import LearningStrategy as ls
from Model import SymbolicRegression
import Operations as op
import Errors as er
import Indicators as ind
import InformationSOTA.artificial_formulas as af
import utils







class Counter:
    
    def __init__(self, i):
        self.__i = i
        self.__r = 0
        
    def __call__(self):
        self.__r += 1
        return self.__r > self.__i
    



def stop_criteria_function(counter):
    return counter()



#-----------------------------------------------------------------------------------------------------------------------

counter = Counter(250)

TR_inputs = {"X": np.linspace(1, 10, 5, dtype = np.float64)}
TR_output = 12 * TR_inputs["X"]

tree = SymbolicRegression.string_to_tree("mul(X, 12)")

metric_results = tree.learn(TR_inputs,
                            TR_output,
                            ls.HeuristicLearning(),
                            stop_criteria = lambda ls: stop_criteria_function(counter),
                            metric_functions = {"tree": lambda model: str(model), "timestamp": lambda model: time.time()})



raise ValueError("TOTAL STOP")



#-----------------------------------------------------------------------------------------------------------------------


art_formulas = af.artificial_formulas

for a in range(len(choice_nodes)):
    
    for b in range(len(regularization_MDL)):
        
        for i, c in enumerate(list(art_formulas.keys())[0:13]):
            
            print("*"*100)
            
            #if(os.path.exists(f"MOCK_save_experiments/{a}_{b}_{c}.json")): continue
            
            ideal_function = c

            X = art_formulas[c]
            Y = eval(ideal_function)
            
            if(type(Y) in [int, float]): Y = np.array([Y]*len(X), dtype = np.float64)
            
            mask = np.isfinite(Y)
            X = X[mask]
            Y = Y[mask]

            TR_inputs = {"X": X}
            TR_output = Y

            counter = Counter(50)

            tree = SymbolicRegression.string_to_tree("0")
            
            print("DATASET: ", ideal_function)

            metric_results = tree.learn(TR_inputs,
                                        TR_output,
                                        ls.HeuristicLearning(),
                                        
                                        stop_criteria = lambda ls: stop_criteria_function(ls, counter, TR_output),
                                        
                                        metric_functions = {"tree": lambda model: str(model),
                                                            #"MAE": lambda model: er.MAE(model.get_root().get_memory("forward"), TR_output),
                                                            #"Diff": lambda model: model.get_root().get_memory("forward") - TR_output,
                                                            "timestamp": lambda model: time.time(),
                                                            "MDL": regularization_MDL[b]
                                                            }
                                        )
            
            
            print(f"REAL FORMULA:     {ideal_function}")
            print(f"INFERRED FORMULA: {metric_results['results']['tree'][-1]}")
            
            #json.dump(metric_results | {"real function": ideal_function}, open(f"MOCK_save_experiments/{a}_{b}_{i}.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
            
            #break
            
        raise ValueError("BASE VERSION ARE COMPLETED")

