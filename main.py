import numpy as np
import matplotlib.pyplot as plt

import LearningStrategy as ls
from Model import SymbolicRegression
import Operations as op
import Errors as er
import Indicators as ind

import time


#Identification a secco di sin:  chatterjee * (1 - spearman)


def indicators_package_extensions(nodes, unary_operations, TR_inputs_dict):
    
    combinations = default_package_extensions(nodes, unary_operations, TR_inputs_dict)[0]
    
    result = []
    
    analyzed_nodes = [{"node": x, 
                       "constant": 1 / (np.var(x.get_memory("backward")) + 1),
                       "pearson": ind.pearson_correlation(x.get_memory("forward"), x.get_memory("backward")),
                       "chaterjee": ind.chatterjee_correlation(x.get_memory("forward"), x.get_memory("backward"))} for x in nodes]
    
    #Consider the node like: the linear distance correlation between output and excepted output (explain this), and if 2 nodes are with same distance linear correlation.
    #considering the node with more potential construction with chaterjee (explain this).
    for s in sorted(analyzed_nodes, key = lambda x: ((np.abs(x["pearson"]) if not x["node"].is_constant() else x["constant"]), -np.abs(x["chaterjee"]))):
        result.append([x for x in combinations if x["old node"] is s["node"]])

    return result



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
        
        #f or f + 1 (about leaf cost)
        if(isinstance(n, op.Leaf)):
            
            leafs = [op.Leaf(x) for x in TR_inputs_dict.keys() if x != n.get_name()]
            leafs.append(op.Leaf(get_constant(n.get_memory("backward"))))
                
            for l in leafs: combinations.append({"old node": n, "new node": l, "type": "base"})
        
        
        #ou (add unary operation)
        for ou in unary_operations:
            
            oper = ou()
            
            if(not oper.check_input_domain(0, n.get_memory("forward"))): continue
            
            if(not n.is_constant()):
            
                leaf = n.copy_sub_tree()

                oper.new_connection(leaf, 0)
                combinations.append({"old node": n, "new node": oper.copy_sub_tree(), "type": "hat"})
                oper.free_connection(0)


    for i in range(len(combinations)):
        
        combinations[i]["new node"].forward_calculation_memory_sub_tree(TR_inputs_dict).set_memory("backward", combinations[i]["old node"].get_memory("backward"))

    
    return [combinations]







ideal_function = "X * np.sin(X)"

X = np.array([x for x in range(-100, 100, 1)], dtype = np.float64) / 10
Y = eval(ideal_function)

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
    
counter = Counter(100)


strategies_package_nodes = [default_package_extensions,
                            indicators_package_extensions]

strategies_filter_nodes = [lambda x: True,
                           lambda x: isinstance(x, op.Leaf)]


def stop_criteria_function(strategy_learning, counter, TR_output):
    
    model = strategy_learning.get_model()
    
    go = counter() or er.MAE(model.get_root().get_memory("forward"), TR_output) <= 1e-6
    
    return go


metric_results = tree.learn(TR_inputs,
                            TR_output,
                            ls.AnalysisRamification([op.div, op.abs, op.exp, op.ln, op.sin, op.arcsin],
                                                    
                                                    lambda candidates: candidates,  #Filter phase
                                                    
                                                    lambda n, uo, TR_inputs: strategies_package_nodes[0]([x for x in n if strategies_filter_nodes[1](x)], uo, TR_inputs)
                                                    
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




