import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

import LearningStrategy as ls
from Model import SymbolicRegression
import Operations as op
import Errors as er

import Indicators as ind

def chaterjee_metric(Y_output, Y_target):
    
    if(np.isscalar(Y_output) and np.isscalar(Y_target)): return -1
    
    if(np.isscalar(Y_output)): Y_output = np.array([Y_output for _ in range(len(Y_target))])
    if(np.isscalar(Y_target)): Y_target = np.array([Y_target for _ in range(len(Y_output))])
    
    return -ind.chatterjee_correlation(Y_output, Y_target)

tree = SymbolicRegression.string_to_tree("0")

def get_package_importance_nodes(nodes, unary_operations, TR_inputs_dict, TR_output):
        
    combinations = []
    
    #Se node è una foglia:      costante + ogni feature + ogni feature x ogni operazione
    #Se node è un nodo interno:                           ogni feature x ogni operazione
        
    for n in nodes:
        
        if(isinstance(n, op.Leaf)): combinations.append({"old node": n, "new node": op.Leaf(np.mean(n.get_memory("backward")))})
            
        for f in TR_inputs_dict.keys():
                
            if(isinstance(n, op.Leaf)): combinations.append({"old node": n, "new node": op.Leaf(f)})
                
            for o in unary_operations:
                    
                oper = o()
                
                if(not oper.check_input_domain(0, TR_inputs_dict[f] if isinstance(n, op.Leaf) else n.get_memory("forward"))): continue
                if(n.get_father() != None and not oper.check_output_domain(n.get_memory("backward") if n.get_father() != None else TR_output)): continue
                                
                oper.new_connection(op.Leaf(f) if(isinstance(n, op.Leaf)) else n.copy_sub_tree(), 0)
                            
                combinations.append({"old node": n, "new node": oper})

    return [combinations]




X = np.array([x for x in range(-10000, 10000, 1)], dtype = np.float64) / 1


ideal_function = "X**6"
Y = eval(ideal_function)

TR_inputs = {"X": X}
TR_output = Y

#add(mul(abs(X), 10000.0), -16666666.5

#y1 = SymbolicRegression.string_to_tree("add(mul(abs(X), 10000.0), -16666666.5)").get_calculation_function()(TR_inputs)
#y2 = SymbolicRegression.string_to_tree("add(mul(abs(X), abs(X)), -16666666.5)").get_calculation_function()(TR_inputs)

#y3 = SymbolicRegression.string_to_tree("add(add(3, mul(W, add(X, add(Y, 3)))), mul(-1, mul(W, Y)))")
#y3 = SymbolicRegression.string_to_tree("add(mul(add(mul(X, 5), 4), 3), 3)")

#y3 = SymbolicRegression.string_to_tree("mul(mul(mul(div(mul(3, X)), add(X, X)), mul(3, E)), X)")
#y3.simplify_model()
#print(y3.get_root().get_string_calculation_function()["body"])


#y3 = SymbolicRegression.string_to_tree("0")
#y3.simplify_model()

#print(y3.get_root().get_string_calculation_function()["body"])

#print("Original:", y2)

#nodes = y2.get_root().get_all_subtree_nodes()
#x = op.Leaf("r")
#
#for n in y2.get_root().get_all_subtree_nodes():
#    
#    #root = n
#    #while(root.get_father() != None): root = root.get_father()
#    
#    #print(root == y2.get_root())
#    
#    #if(n == y2.get_root()): continue
#    
#    y2.replace_subtree(n, x)
#    print(y2, f"({n}) --> ({x})")
#    y2.replace_subtree(x, n)
#    print(y2, f"({x}) --> ({n})")
#    
#    #raise ValueError("STOP")
#
#raise ValueError("STOP")


#print(y1)
#print(y2)

#print(chaterjee_metric(y1, Y))
#print(chaterjee_metric(y2, Y))
#
#print(er.MAE(y1, Y))
#print(er.MAE(y2, Y))
#
#raise ValueError("STOP")


#tree = SymbolicRegression.string_to_tree("0")

metric_results = tree.learn(TR_inputs, TR_output,
                            ls.AnalysisRamification([op.neg, op.div, op.sin, op.cos, op.abs, op.exp, op.ln],
                                                    lambda x: (np.isfinite(x["Chaterjee"]) and x["Chaterjee"] >= 0) # and (np.isfinite(x["MAE"]) and x["MAE"] > 0)
                                                    ,
                                                    lambda x: (-x["Chaterjee"], -x["MAE"]),
                                                    get_package_importance_nodes,
                                                    {"MAE": lambda model: er.MAE(model.get_calculation_function()(TR_inputs), TR_output),
                                                     "Chaterjee": lambda model: chaterjee_metric(model.get_calculation_function()(TR_inputs), TR_output)}),
                            
                            
                            metric_functions = {#"tree": lambda model: str(model.get_root()),
                                                "Count_nodes": lambda model: len(model.get_root().get_all_subtree_nodes()),
                                                "MAE error": lambda model: er.MAE(model.get_root().get_memory("forward"), Y),
                                                "formula": lambda model: model.get_root().get_string_calculation_function()["body"],
                                                #"Chaterjee": lambda model: ind.chatterjee_correlation(model.get_root().get_memory("forward"), Y)
                                                }
                            )

print(metric_results)

MAE_list = []
Chaterjee_list = []

for tree in metric_results["results"]["tree"]:
    
    Yo = SymbolicRegression.string_to_tree(tree).get_calculation_function()({"X": X})
    
    MAE_list.append(er.MAE(Yo, Y))
    Chaterjee_list.append(-chaterjee_metric(Yo, Y))

#MAE_list = [x / max(MAE_list) for x in MAE_list]

plt.plot(range(len(MAE_list)), MAE_list, color = "red", label = "MAE")
plt.legend()
plt.show()

plt.plot(range(len(Chaterjee_list)), Chaterjee_list, color = "orange", label = "Chaterjee")
plt.legend()
plt.show()

model = SymbolicRegression.string_to_tree(metric_results["results"]["tree"][-1])

plt.plot(X, Y, color = "green", label = f"Ideal function {ideal_function}")
plt.plot(X, model.get_calculation_function()({"X": X}), color = "blue", label = f"Inferred function {model.get_root().get_string_calculation_function()['body']}")
plt.legend()
plt.show()

#CAPIRE SE, MENTRE FACCIO I CALCOLI, è SEMPRE MEGLIO SILENZIARE I VARI COEFFICIENTI PER FARE LE SCELTE MIGLIORI, E CAPIRE QUALI COEFFICIENTI VANNO SILENZIATI NEL CASO
#TANTO I COEFFICIENTI SONO SEMPRE FIGLI DI ADD O MUL.
#TENERE CONTO DELLA TRASPARENZA, E DEI DOMINI

#PROVARE CON METTERE A TACERE SOLO I COEFFICIENTI DI ADD E MUL CHE SONO SOPRA DI TUTTI, PERCHè QUELLI INTERNI POTREBBERO SERIAMENTE COMPROMETTERE TUTTE LE TRASFORMAZIONI 
#NON LINEARI INTERNE ED AVERE PROBLEMI DI DOMINIO, QUINDI TOGLIERE I COEFFICIENTI SOLO QUANDO SI CALCOLANO GLI ERRORI (ED ANCHE CHATERJEE PER SICUREZZA)