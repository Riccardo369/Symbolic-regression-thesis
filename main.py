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



def task(TR_inputs, TR_output, name_json):

    class Counter:
        
        def __init__(self, i):
            self.__i, self.__r = i, 0
            
        def __call__(self):
            self.__r += 1
            return self.__r > self.__i

    counter = Counter(10)

    tree = SymbolicRegression.string_to_tree("0")

    metric_results = tree.learn(TR_inputs,
                                TR_output,
                                ls.HeuristicLearning(),
                                stop_criteria = lambda ls: counter(),
                                metric_functions = {"tree": lambda model: str(model), "timestamp": lambda model: time.time()})
    
    
    json.dump(metric_results, open(name_json, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    
    
    
TR_inputs = {"X": np.linspace(-10, 10, 10000, dtype = np.float64)}
TR_output = np.sin(TR_inputs["X"]) + np.sin(np.exp(TR_inputs["X"]))

task(TR_inputs, TR_output, "prova1.json")

