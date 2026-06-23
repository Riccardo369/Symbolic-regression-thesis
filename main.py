import numpy as np
import matplotlib.pyplot as plt

import LearningStrategy as ls
from Model import SymbolicRegression
import Operations as op
import Errors as er
import Indicators as ind

import time



#Foreach node:

#Nodo interno: ou
#foglia feature : f + ou
#foglia const   : f+1 + ou

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
     
            for x in list(TR_inputs_dict.keys()): combinations.append({"type": "change_base", 
                                                                       "node": n, 
                                                                       "new_feature": x, 
                                                                       "score": np.abs(ind.chatterjee_correlation(TR_inputs_dict[x], backward))})

            combinations.append({"type": "change_base", 
                                 "node": n, 
                                 "new_feature": get_constant(n.get_memory("backward")),
                                 "score": 1 / (1 + np.var(backward))})
            
        if(n.is_constant()): continue
            
        total_analysis = ind.total_analysis(forward, backward, ["monoton", "symmetry", "sinusoidal", "vertical_asymptote"])
        
        for ou in unary_operations:
            
            oper = ou()
            
            if(not oper.check_input_domain(0, forward)): continue
            
            combinations.append({"type": "add_operation", "node": n, "new operation": ou, "score": ou.calculate_statistical_score(total_analysis)})
        
    combinations = sorted(combinations, key = lambda x: -x["score"])

    return [x for x in combinations]





ideal_function = "np.sin(X) * X**2"

X = np.array([x for x in range(-100, 100, 1)], dtype = np.float64) / 10
Y = eval(ideal_function)

mask = np.isfinite(Y)
X = X[mask]
Y = Y[mask]

TR_inputs = {"X": X}
TR_output = Y

tree = SymbolicRegression.string_to_tree("0")

class Counter:
    
    def __init__(self, i):
        self.__i = i
        self.__r = 0
        
    def __call__(self):
        self.__r += 1
        return self.__r >= self.__i
    
counter = Counter(400)



def stop_criteria_function(strategy_learning, counter, TR_output):
    
    model = strategy_learning.get_model()
    
    go = counter() or er.MAE(model.get_root().get_memory("forward"), TR_output) <= 1e-6
    
    return go


metric_results = tree.learn(TR_inputs,
                            TR_output,
                            ls.AnalysisRamification([op.div, op.abs, op.exp, op.ln, op.sin, op.arcsin],
                                                    
                                                    lambda candidates: candidates,  #Filter phase
                                                    
                                                    lambda n, uo, TR_inputs: default_package_extensions(n, uo, TR_inputs)
                                                    
                                                    ),
                            
                            stop_criteria = lambda ls: stop_criteria_function(ls, counter, TR_output),
                            
                            metric_functions = {"tree": lambda model: str(model), "timestamp": lambda model: time.time()}
                            )

#MAE_list = [x / max(MAE_list) for x in MAE_list]

raise ValueError("STOP")



def plot(x, color, label):

    plt.plot(range(len(x)), x, color = color, label = label)
    plt.legend()
    plt.show()
    
plot(metric_results["results"]["Param"], "black", "Count nodes")
plot(metric_results["results"]["Total confidence"], "green", "Total confidence")
plot(metric_results["results"]["MAE error"], "red", "MAE error")
plot(metric_results["results"]["Chaterjee"], "purple", "Chaterjee")


model = SymbolicRegression.string_to_tree(metric_results["results"]["tree"][-1])

Y_calculated = model.get_calculation_function()({"X": X})

if(np.isscalar(Y_calculated)): Y_calculated = [Y_calculated]*len(X)

plt.plot(X, Y, color = "green", label = f"Ideal function {ideal_function}")
plt.plot(X, Y_calculated, color = "blue", label = f"Inferred function {model.get_root().get_string_calculation_function()['body']}")
plt.legend()
plt.show()




