import numpy as np
import matplotlib.pyplot as plt

import LearningStrategy as ls
from Model import SymbolicRegression
import Operations as op
import Errors as er
import utils

import Indicators as ind

def chaterjee_metric(X, Y):
    
    if(np.isscalar(X) and np.isscalar(Y)): return -1
    
    if(np.isscalar(X)): X = np.array([X for _ in range(len(Y))])
    if(np.isscalar(Y)): Y = np.array([Y for _ in range(len(X))])
    
    return -ind.chatterjee_correlation(X, Y)






def chaterjee_package_extensions(nodes, unary_operations, TR_inputs_dict, TR_output):
    
    combinations = default_package_extensions(nodes, unary_operations, TR_inputs_dict, TR_output)[0]
    
    result = []
    
    new_nodes = [{"node": x, 
                  "Chaterjee": chaterjee_metric(x.get_memory("forward"), x.get_memory("backward")),
                  "confidence": x["confidence"]
              } for x in nodes]
    
    
    for s in sorted(new_nodes, key = lambda x: (-x["Chaterjee"], -x["confidence"])):
        result.append([x for x in combinations if x["old node"] == s])

    return result
    

#Foreach node:

#Combination cost if node is a constant: (f)*(1 + ou + ob) - 1
#Combination cost if node is a not constant: (f+1)*(1 + ou + ob) - 1 + 1 + ob

#LEGEND:
#f+1 = features + node
#(1 + ou + ob) = 1 feature of reference + unary operations (oper with feature of reference) + binary operations (node, feature of reference)
# - 1 = minus the node itself
#1 + ob (non constant): 1 constant + binary operations (node, constant)

#DA RICOSTRUIRE (PERCHé è ANCOR DI MENO IL COSTO)


#Nodo interno: ou + ob*f + 3
#foglia: (f or f+1) + ou*f + ob*f + 3

#f or (f+1) perchè: (se costante non ci metti tutte le f, altrimenti tutte le f - quella attuale + la costante)
# + 3: sono le combinazioni dell' estensione lineari: solo add, solo mul e add e mul


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
                
            for l in leafs: combinations.append(l)

        #ou operations
        for ou in unary_operations:
            
            oper = ou()
            
            #Essendo che il nodo andrà sotto l' operazione unaria, se l' output forward del nodo non è compatibile con il dominio di input dell' operazione unaria,
            #è inutile allora fare tutte le prove che ne derivano.
            if(not oper.check_input_domain(0, n.get_memory("forward")) or n.is_constant()): continue
            
            leaf = n.copy_sub_tree()

            oper.new_connection(leaf, 0)
            combinations.append(oper.copy_sub_tree())
            oper.free_connection(0)
            
            #f (only for leaf)
            if(isinstance(n, op.Leaf)):
            
                leafs = [op.Leaf(x) for x in TR_inputs_dict.keys() if x != n.get_name()]
                        
                for leaf in leafs:
                    
                    oper.new_connection(leaf, 0)
                    
                    oper.forward_calculation_memory_sub_tree(TR_inputs_dict).set_memory("backward", n.get_memory("backward"))
                    combinations.append({"old node": n, "new node": oper.copy_sub_tree(), "confidence": oper.calculation_confidence()})   
                    oper.free_connection(0)
          
        #ob
        for ob in [op.add, op.mul]:
                
            #Essendo che b è un operazione a catena (cioè associativa e commutativa), se applico l' estensione con l' operazione in questione sulla radice
            #od in uno dei figli è uguale (l' effetto sarà lo stesso), quindi per evitare estensioni inutili, lo si fa solo sulle radici di queste catene,
            #quindi se n è un nodo che fa parte di questa catena (cioè catena di operazione b), la si salta proprio.
            if(isinstance(n.get_father(), ob)): continue

            #f
            for f in TR_inputs_dict.keys():
                        
                binary_oper = ob()

                binary_oper.new_connection(n.copy_sub_tree(), 0)
                binary_oper.new_connection(op.Leaf(f), 1)
                
                combinations.append(binary_oper)

    
        if(n.is_constant()): continue
                
        linear_ext = utils.linear_extension(n.get_memory("forward"), n.get_memory("backward"))
        
        m = linear_ext["m"]
        q = linear_ext["q"]
        
        m = m if np.isfinite(m) else 1
        q = q if np.isfinite(q) else 0
                
        if(m != 1 and q != 0):
            
            add = op.add()
            mul = op.mul()
            
            mul.new_connection(n.copy_sub_tree(), 0)
            mul.new_connection(op.Leaf(m), 1)
            
            add.new_connection(mul, 0)
            add.new_connection(op.Leaf(q), 1)
            
            combinations.append(add)
            
        if(m != 1):

            mul = op.mul()
            
            mul.new_connection(n.copy_sub_tree(), 0)
            mul.new_connection(op.Leaf(m), 1)
            
            combinations.append(mul)
            
        if(q != 0):
            
            add = op.add()
            
            add.new_connection(n.copy_sub_tree(), 0)
            add.new_connection(op.Leaf(q), 1)

            combinations.append(add)
            


    for i in range(len(combinations)):
        
        combinations[i].forward_calculation_memory_sub_tree(TR_inputs_dict).set_memory("backward", n.get_memory("backward"))
        combinations[i] = {"old node": n, "new node": combinations[i], "confidence": combinations[i].calculation_confidence()}
   

    #print("COMBINATIONS" + '-'*100)
    #for i in combinations: print(i)
    
    return [combinations]



ideal_function = "3*(X**2) + 5"

X = np.array([x for x in range(1, 1000, 1)], dtype = np.float64) / 1000
Y = eval(ideal_function)

TR_inputs = {"X": X}
TR_output = Y

tree = SymbolicRegression.string_to_tree("0")

def f(node):
    for c in node.get_childrens(): f(c)
    print(node)
    print("forward", node.get_memory("forward"))
    print("backward", node.get_memory("backward"))
    print("")

#tree.get_root().forward_calculation_memory_sub_tree(TR_inputs).backward_calculation_memory_sub_tree(TR_output)

#f(tree.get_root())

#tree.view(lambda x: f"{x.calculation_confidence()} | {ind.pearson_correlation(x.get_memory('forward'), x.get_memory('backward'))} | {er.MAE(x.get_memory('forward'), x.get_memory('backward'))}")

#plt.plot(X, Y, color = "red", label = "Ideal function")
#plt.plot(X, tree.get_calculation_function()(TR_inputs), color = "blue", label = "Ideal function")
#plt.legend()
#plt.show()

#raise ValueError("STOP")


metric_results = tree.learn(TR_inputs, TR_output,
                            ls.AnalysisRamification([op.neg, op.div, op.sin, op.cos, op.abs, op.exp, op.ln],
                                                    lambda x: (np.isfinite(x["delta heuristic metrics"]["Chaterjee"]) and x["delta heuristic metrics"]["Chaterjee"] >= 0)
                                                              ,
                                                    
                                                    lambda x: (-np.abs(x["confidence"]),
                                                                -x["delta heuristic metrics"]["Chaterjee"],
                                                               -x["delta heuristic metrics"]["MAE"]),
                                                    
                                                    default_package_extensions,
                                                    {"MAE": lambda model: er.MAE(model.get_calculation_function()(TR_inputs), TR_output),
                                                     "Chaterjee": lambda model: chaterjee_metric(model.get_calculation_function()(TR_inputs), TR_output)}),
                            
                            
                            metric_functions = {"tree": lambda model: str(model),
                                                "Count_nodes": lambda model: len(model.get_root().get_all_subtree_nodes()),
                                                "MAE error": lambda model: er.MAE(model.get_root().get_memory("forward"), Y),
                                                "formula": lambda model: model.get_root().get_string_calculation_function()["body"],
                                                "Chaterjee": lambda model: chaterjee_metric(model.get_calculation_function()(TR_inputs), TR_output)
                                                }
                            )

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






#SISTEMARE I CALCOLI DELLE CONFIDENZE DEI VARI NODI, GESTENDO LE VARIE SITUAZIONI DI OGNI ELEMENTO. QUANDO CALCOLI LA CONFIDENZA, DEVI PRENDERE IL FORWARD DEL NODO SOTTO SE
#SEI UN OPERAZIONE.

#SISTEMARE IL FATTO CHE CI POSSONO ESSERE DELLE ESTENSIONI LINEARI CHE NASCOSTE CHE NON SI CONOSCONO, QUINDI ADATTARE IL PROPRIO BACKWARD CON FUNZIONI FLESSIBILI. SE CE 
# BISOGNO DI RISCALARE CON UNA TRASFORMAZIONE LINEARE UN ELEMENTO, ALLORA LO AGGIUNGI ANCHE SUL MODELLO, PERCHè SE FAI UNA BACKWARD DEVIATA, POI HAI UNA FORWARD SCOLLEGATA 
# DALLA BACKWARD, E QUESTO PORTA AD INCOERENZE, PERCHè TI PERDI I COEFFICIENTI DELLE ESTENSIONI LINEARI

# E' ORA DI CERCARE I COMPETITOR E FARE LE PROVE, PER FAVORE