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

#Foreach node:

#Nodo interno singolo: ou
#Nodo a catena: f * ou
#foglia: 2 + ou

# Nodo interno: aggiungo tutte le varie possibili operazioni sopra
# Foglia: aggiungo tutte le feature + costante, non ci metto la combinazione di operazione unaria + feature per osservare il comportamento del modello senza provare a 
# scavalcare con dei pattern già complessi (perchè sennò bisognava provare anche con pattern complessi di 2 livello, 3 livello etc.), e da lì poi l' esplosione,
# quindi per vedere se il modello riesce a catturare le relazioni strutturali giuste senza fare i salti


def default_package_extensions(nodes, unary_operations, TR_inputs_dict):

    def get_constant(memory):
        
        if(isinstance(memory, np.ndarray)):
            mask = memory[np.isfinite(memory)]
            if(len(mask) == 0): raise ValueError("It is not possible to do the mean with empty list")

        return np.mean(memory[np.isfinite(memory)]) if isinstance(memory, np.ndarray) else memory
    
    combinations = []
    
    #n nodes
    for n in nodes:
        
        forward = n.get_memory("forward")
        backward = n.get_memory("backward")
        
        if(isinstance(n, op.Leaf)):
            
            choice_feature = sorted([{"f": x, "score": np.abs(ind.chatterjee_correlation(TR_inputs_dict[x], backward))} for x in TR_inputs_dict.keys()], key = lambda x: -x["score"])
            
            combinations.append({"type": "change_base", "node": n, "new_feature": choice_feature[0]["f"], "score": choice_feature[0]["score"]})
           
            combinations.append({"type": "change_base", 
                                 "node": n, 
                                 "new_feature": get_constant(n.get_memory("backward")),
                                 "score": 1 / (1 + np.var(backward))})
            
        if(n.is_constant()): continue
            
        total_analysis = utils.total_analysis(forward, backward, ["monoton", "symmetry", "sinusoidal", "vertical_asymptote"])
        
        for ou in unary_operations:
            
            oper = ou()
            
            if(not oper.check_input_domain(0, forward)): continue
            
            combinations.append({"type": "add_operation", "node": n, "new operation": ou, "score": ou.calculate_statistical_score(total_analysis)})
        
    combinations = sorted(combinations, key = lambda x: -x["score"])
    
    combinations = [x for x in combinations if not (x["type"] == "change_base" and str(x["node"]) == str(x["new_feature"]))]

    return combinations




class Counter:
    
    def __init__(self, i):
        self.__i = i
        self.__r = 0
        
    def __call__(self):
        self.__r += 1
        return self.__r > self.__i
    



def only_leafs(n):
    return [x for x in n if isinstance(x, op.Leaf)]

def sorted_by_chatterjee(n):
    return sorted(n, key = lambda x: (np.abs(ind.chatterjee_correlation(x["node"].get_memory("forward"), x["node"].get_memory("backward"))), -x["score"]))



choice_nodes = [lambda n, uo, TR_inputs: default_package_extensions(n, uo, TR_inputs),
                lambda n, uo, TR_inputs: default_package_extensions(only_leafs(n), uo, TR_inputs),
                lambda n, uo, TR_inputs: sorted_by_chatterjee(default_package_extensions(n, uo, TR_inputs)),
                lambda n, uo, TR_inputs: sorted_by_chatterjee(default_package_extensions(only_leafs(n), uo, TR_inputs))]

regularization_MDL = [lambda model: 0,
                      lambda model: er.MDL(TR_output, model.get_root().get_memory("forward"), model)
                      ]



def stop_criteria_function(strategy_learning, counter, TR_output):
    
    metrics_MDL = strategy_learning.get_metrics()["MDL"]
    
    model = strategy_learning.get_model()
    
    Y_model = model.get_root().get_memory("forward")
    
    go = counter() or er.MAE(Y_model, TR_output) <= 1e-6 or (False if len(metrics_MDL) < 2 else metrics_MDL[-2] < metrics_MDL[-1])
    
    return go



art_formulas = af.artificial_formulas

for a in range(len(choice_nodes)):
    
    for b in range(len(regularization_MDL)):
        
        for c in range(len(art_formulas))[3:8]:
            
            #if(os.path.exists(f"MOCK_save_experiments/{a}_{b}_{c}.json")): continue
            
            ideal_function = art_formulas[c]

            X = np.array([x for x in range(-1000, 1000, 1)], dtype = np.float64) / 10
            Y = eval(ideal_function)
            
            if(type(Y) in [int, float]): Y = np.array([Y]*len(X))

            mask = np.isfinite(Y)
            X = X[mask]
            Y = Y[mask]

            TR_inputs = {"X": X}
            TR_output = Y

            counter = Counter(10)

            tree = SymbolicRegression.string_to_tree("0")

            metric_results = tree.learn(TR_inputs,
                                        TR_output,
                                        ls.AnalysisRamification([op.div, op.abs, op.exp, op.ln, op.sin, op.arcsin],
                                                                
                                                                choice_nodes[a]
                                                                
                                                                ),
                                        
                                        stop_criteria = lambda ls: stop_criteria_function(ls, counter, TR_output),
                                        
                                        metric_functions = {"tree": lambda model: str(model),
                                                            "MAE": lambda model: er.MAE(model.get_root().get_memory("forward"), TR_output),
                                                            #"timestamp": lambda model: time.time(),
                                                            "MDL": regularization_MDL[b]
                                                            }
                                        )
            
            
            print(f"REAL FORMULA:     {ideal_function}")
            print(f"INFERRED FORMULA: {metric_results['results']['tree'][-1]}")
            
            #json.dump(metric_results, open(f"MOCK_save_experiments/{a}_{b}_{c}.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
            
            break
            
        raise ValueError("BASE VERSION ARE COMPLETED")


