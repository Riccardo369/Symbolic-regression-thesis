import numpy as np
import matplotlib.pyplot as plt

import LearningStrategy as ls
from Model import SymbolicRegression
import Operations as op
import Errors as er
import utils

import Indicators as ind
import TheoricalExperiments.TotalConfidence as tc





def chaterjee_package_extensions(nodes, unary_operations, TR_inputs_dict, TR_output):
    
    combinations = default_package_extensions(nodes, unary_operations, TR_inputs_dict, TR_output)[0]
    
    result = []
    
    analyzed_nodes = [{"node": x, "chaterjee": ind.chatterjee_correlation(x.get_memory("forward"), x.get_memory("backward"))} for x in nodes]
    
    for s in sorted([x for x in analyzed_nodes if np.isfinite(x["Chaterjee"])], key = lambda x: x["chaterjee"])[::-1]:
        result.append([x for x in combinations if x["node"] == s])

    return result



#Foreach node:

#unary operation: 2*ou
#Nodo interno: ou
#foglia feature : f + ou*f
#foglia const   : f+1 + ou*f

#f su foglia feature perchè: (se costante ci metti tutte le f + costante, altrimenti tutte le f - quella attuale + la costante)

def default_package_extensions(nodes, unary_operations, TR_inputs_dict, TR_output):

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
                
            for l in leafs: combinations.append({"old node": n, "new node": l})
        
        #ou - 1 (substitution unary operation) + directly child (without node so)
        else:
            
            if(n.get_channels() == 1):
                
                child_copy = n.get_childrens()[0]
                
                combinations.append({"old node": n, "new node": child_copy.copy_sub_tree()})
                
                for ou in [x for x in unary_operations if not isinstance(n, x)]:
                    
                    oper = ou()
                    
                    if(not oper.check_input_domain(0, n.get_memory("forward"))): continue
                    
                    oper.new_connection(child_copy.copy_sub_tree(), 0)
                    
                    combinations.append({"old node": n, "new node": oper.copy_sub_tree()})
                    
                    oper.free_connection(0)
            
        

        #ou (add unary operation)
        for ou in unary_operations:
            
            oper = ou()
            
            if(not oper.check_input_domain(0, n.get_memory("forward"))): continue
            
            if(not n.is_constant()):
            
                leaf = n.copy_sub_tree()

                oper.new_connection(leaf, 0)
                combinations.append({"old node": n, "new node": oper.copy_sub_tree()})
                oper.free_connection(0)
            
            #f (only for leaf)
            if(isinstance(n, op.Leaf)):
            
                leafs = [op.Leaf(x) for x in TR_inputs_dict.keys() if x != n.get_name()]
                        
                for leaf in leafs:
                    
                    oper.new_connection(leaf, 0)
                    
                    oper.forward_calculation_memory_sub_tree(TR_inputs_dict).set_memory("backward", n.get_memory("backward"))
                    combinations.append({"old node": n, "new node": oper.copy_sub_tree()})   
                    oper.free_connection(0)

    for i in range(len(combinations)):
        
        combinations[i]["new node"].forward_calculation_memory_sub_tree(TR_inputs_dict).set_memory("backward", combinations[i]["old node"].get_memory("backward"))
        

    #print("COMBINATIONS" + '-'*100)
    #for i in combinations: print(i)
    
    #raise ValueError("STOP")
    
    return [combinations]



ideal_function = "4 * np.sin(X) + 5"

X = np.array([x for x in range(-1000, 1000, 1)], dtype = np.float64) / 10
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
    

counter = Counter(20)

def filter_phase(candidates):
    return candidates

def sort_phase(candidates):
    return sorted(candidates, key = lambda x: abs(x["delta heuristic metrics"]["MAE"]) * x["delta heuristic metrics"]["Chaterjee"])


metric_results = tree.learn(TR_inputs,
                            TR_output,
                            ls.AnalysisRamification([op.neg, op.div, op.abs, op.exp, op.ln, op.sin, op.cos],
                                                    filter_phase,
                                                    
                                                    sort_phase,
                                                    
                                                    default_package_extensions,
                                                    
                                                    {"MAE": lambda model: er.MAE(model.get_calculation_function()(TR_inputs), TR_output),
                                                     "Chaterjee": lambda model: abs(ind.chatterjee_correlation(model.get_calculation_function()(TR_inputs), TR_output)),
                                                     "Total confidence": lambda model: tc.total_confidence(model.get_root())}),
                            
                            stop_criteria = lambda x: counter(),
                                
                            metric_functions = {"tree": lambda model: str(model),
                                                "Count nodes": lambda model: len(model.get_root().get_all_subtree_nodes()),
                                                "MAE error": lambda model: er.MAE(model.get_root().get_memory("forward"), Y),
                                                "formula": lambda model: model.get_root().get_string_calculation_function()["body"],
                                                "Chaterjee": lambda model: ind.chatterjee_correlation(model.get_calculation_function()(TR_inputs), TR_output),
                                                "Total confidence": lambda model: tc.total_confidence(model.get_root()),
                                                }
                            )

#MAE_list = [x / max(MAE_list) for x in MAE_list]

def plot(x, color, label):

    plt.plot(range(len(x)), x, color = color, label = label)
    plt.legend()
    plt.show()
    
plot(metric_results["results"]["Count nodes"], "black", "Count nodes")
plot(metric_results["results"]["Total confidence"], "green", "Count nodes")
plot(metric_results["results"]["MAE error"], "red", "MAE error")
plot(metric_results["results"]["Chaterjee"], "purple", "Chaterjee")


model = SymbolicRegression.string_to_tree(metric_results["results"]["tree"][-1])

Y_calculated = model.get_calculation_function()({"X": X})

if(np.isscalar(Y_calculated)): Y_calculated = [Y_calculated]*len(X)

plt.plot(X, Y, color = "green", label = f"Ideal function {ideal_function}")
plt.plot(X, Y_calculated, color = "blue", label = f"Inferred function {model.get_root().get_string_calculation_function()['body']}")
plt.legend()
plt.show()






#SISTEMARE I CALCOLI DELLE CONFIDENZE DEI VARI NODI, GESTENDO LE VARIE SITUAZIONI DI OGNI ELEMENTO. QUANDO CALCOLI LA CONFIDENZA, DEVI PRENDERE IL FORWARD DEL NODO SOTTO SE
#SEI UN OPERAZIONE.

#SISTEMARE IL FATTO CHE CI POSSONO ESSERE DELLE ESTENSIONI LINEARI CHE NASCOSTE CHE NON SI CONOSCONO, QUINDI ADATTARE IL PROPRIO BACKWARD CON FUNZIONI FLESSIBILI. SE CE 
# BISOGNO DI RISCALARE CON UNA TRASFORMAZIONE LINEARE UN ELEMENTO, ALLORA LO AGGIUNGI ANCHE SUL MODELLO, PERCHè SE FAI UNA BACKWARD DEVIATA, POI HAI UNA FORWARD SCOLLEGATA 
# DALLA BACKWARD, E QUESTO PORTA AD INCOERENZE, PERCHè TI PERDI I COEFFICIENTI DELLE ESTENSIONI LINEARI

# E' ORA DI CERCARE I COMPETITOR E FARE LE PROVE, PER FAVORE