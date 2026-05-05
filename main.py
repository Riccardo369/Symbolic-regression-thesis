import numpy as np
import matplotlib.pyplot as plt

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





def get_package_importance_nodes(nodes, unary_operations, TR_inputs_dict, TR_output):
    
    combinations = []
    
    #Per ogni foglia, che sia nodo interno o foglia: prendi il nodo, lo metti
    
    #Se node è una foglia:      costante + ogni feature + ogni feature x ogni operazione
    #Se node è un nodo interno:                           ogni feature x ogni operazione
        
    for n in nodes:

        for o in unary_operations:
                
            for f in TR_inputs_dict.keys():
         
                if(not o().check_input_domain(0, TR_inputs_dict[f] if isinstance(n, op.Leaf) else n.get_memory("forward"))): continue
                    
                for b in [op.add, op.mul]:
                    
                    #Essendo che b è un operazione a catena (cioè associativa e commutativa), se applico l' estensione con l' operazione in questione sulla radice
                    #od in uno dei figli è uguale (l' effetto sarà lo stesso), quindi per evitare estensioni inutili, lo si fa solo sulle radici di queste catene,
                    #quindi se n è un nodo che fa parte di questa catena (cioè catena di operazione b), la si salta proprio
                    if(isinstance(n.get_father(), b)): continue
                    
                    oper = o()
                           
                    oper.new_connection(n.copy_sub_tree(), 0)
                    
                    feature_leaf = op.Leaf(f)
                        
                    binary_oper = b()
                    binary_oper.new_connection(oper.copy_sub_tree(), 0)
                    binary_oper.new_connection(feature_leaf, 1)
                                        
                    combinations.append({"old node": n, "new node": binary_oper.copy_sub_tree()})
                    
                    binary_oper.forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
                    
                    if(np.all(np.isfinite(feature_leaf.get_memory("backward")))):
                        
                        feature_leaf.set_name(np.mean(feature_leaf.get_memory("backward")))
                        combinations.append({"old node": n, "new node": binary_oper.copy_sub_tree()})

    return [combinations]



ideal_function = "2*(X**3) + 5"

X = np.array([x for x in range(1, 100, 1)], dtype = np.float64) / 1
Y = eval(ideal_function)

TR_inputs = {"X": X}
TR_output = Y


tree = SymbolicRegression.string_to_tree("0")

metric_results = tree.learn(TR_inputs, TR_output,
                            ls.AnalysisRamification([op.neg, op.div, op.sin, op.cos, op.abs, op.exp, op.ln],
                                                    lambda x: (np.isfinite(x["Chaterjee"]) and x["Chaterjee"] > 0) or 
                                                              (np.isfinite(x["Chaterjee"]) and x["Chaterjee"] == 0 and np.isfinite(x["MAE"]) and x["MAE"] > 0)
                                                    ,
                                                    lambda x: (-x["Chaterjee"], -x["MAE"]),
                                                    get_package_importance_nodes,
                                                    {"MAE": lambda model: er.MAE(model.get_calculation_function()(TR_inputs), TR_output),
                                                     "Chaterjee": lambda model: chaterjee_metric(model.get_calculation_function()(TR_inputs), TR_output)}),
                            
                            
                            metric_functions = {"tree": lambda model: str(model.get_root()),
                                                "Count_nodes": lambda model: len(model.get_root().get_all_subtree_nodes()),
                                                "MAE error": lambda model: er.MAE(model.get_root().get_memory("forward"), Y),
                                                #"formula": lambda model: model.get_root().get_string_calculation_function()["body"],
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

plt.plot(range(len(metric_results["results"]["Count_nodes"])), metric_results["results"]["Count_nodes"], color = "black", label = "Total nodes")
plt.legend()
plt.show()

plt.plot(range(len(MAE_list)), MAE_list, color = "red", label = "MAE")
plt.legend()
plt.show()

plt.plot(range(len(Chaterjee_list)), Chaterjee_list, color = "orange", label = "Chaterjee")
plt.legend()
plt.show()

model = SymbolicRegression.string_to_tree(metric_results["results"]["tree"][-1])

Y_calculated = model.get_calculation_function()({"X": X})

if(np.isscalar(Y_calculated)): Y_calculated = [Y_calculated]*len(X)

plt.plot(X, Y, color = "green", label = f"Ideal function {ideal_function}")
plt.plot(X, Y_calculated, color = "blue", label = f"Inferred function {model.get_root().get_string_calculation_function()['body']}")
plt.legend()
plt.show()



#Il fatto di mettere sul filtro le estensioni sul delta rispetto al MAE, non implica che l' errore si abbassi per forza, perchè durante l' analisi di ogni 
#estensione, può capitare che l' estensione è quella standard, ma le metriche sono fatte sulla versione linear extension con i coefficienti calcolati, per avere
#una proiezione più accurata, calcolando il suo "potenziale", ma senza tuffandoci sopra, così si guarda il potenziale di quella estensione e si va a passo piano.


# E' ORA DI CERCARE I COMPETITOR E FARE LE PROVE, PER FAVORE