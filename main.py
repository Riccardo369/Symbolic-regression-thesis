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





def standard_package_extensions(nodes, unary_operations, TR_inputs_dict):

    def get_constant(memory):
        
        if(isinstance(memory, np.ndarray)):
            mask = np.isfinite(memory)
            memory = memory[mask]

        return np.mean(memory[np.isfinite(memory)]) if isinstance(memory, np.ndarray) else memory
    
    combinations = []
    
    #n nodes
    for n in nodes:
        
        #Take backward of the context
        backward = n.get_memory("backward")
        
        #Decide which leaf must be analyze (if we are in a leaf, analyze all features, otherwise, analyze the node itself)
        leafs = []
        
        if(isinstance(n, op.Leaf)):
            
            leafs = [op.Leaf(x) for x in TR_inputs_dict.keys()]
            for l in leafs: l.set_memory("forward", TR_inputs_dict[l.get_name()])
            
            leafs = sorted(leafs, key = lambda x: -ind.chatterjee_correlation(TR_inputs_dict[x.get_name()], backward))
                
        else:
            
            copy_node = n.copy_sub_tree()
            copy_node.forward_calculation_memory_sub_tree(TR_inputs).backward_calculation_memory_sub_tree(n.get_memory("backward"))
            leafs = [copy_node]
        
        
        #Start analysis
        
        #Calculate score for constant  
        combinations.append({"node": n, "new node": op.Leaf(get_constant(n.get_memory("backward"))), "score": 1 / (1 + np.var(backward))})
        
        #If node is a constant, has no any sense to continue
        if(n.is_constant()): continue

        for l in leafs:

            #If leaf is a chain
            if(isinstance(l, op.ChainOperationNode)):
                
                #FOCI algorithm (for get the index of most important childrens)
                len_vector = len(list(TR_inputs_dict.values())[0])
                matrix_features = np.array([x.get_memory("forward") if not x.is_constant() else np.array([x.get_memory("forward")]*len_vector) for x in l.get_childrens()]).T
                important_childrens = ind.most_important_features(matrix_features, backward)

                #Create the new node (with the 2 sub_set_divided)
                new_chain = type(l)()
                sub_set_chain = type(l)()
                residual_chain = type(l)()
                new_chain.new_connection(sub_set_chain)
                new_chain.new_connection(residual_chain)
                
                #Divide the childrens
                for i, c in enumerate(l.get_childrens()):
                    
                    if(i in important_childrens): sub_set_chain.new_connection(c.copy_sub_tree())
                    else:                        residual_chain.new_connection(c.copy_sub_tree())
                    
                #Calculate forward and backward of sub_set_chain
                new_chain.forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(l.get_memory("backward"))
                
                #Free sub_set from new_chain
                new_chain.free_connection(sub_set_chain)
                
                #Analyze the sub set
                total_analysis = utils.total_analysis(sub_set_chain.get_memory("forward"), sub_set_chain.get_memory("backward"), ["linear", "monoton", "symmetry", "sinusoidal", "vertical_asymptote"])
                    
            
                #Calculate score for each operation that will be up the chain
                for ou in unary_operations:
                        
                    oper = ou()
                    r = op.Leaf(1) 
                    oper.new_connection(r)   
                
                    if(not oper.check_input_domain(0, sub_set_chain.get_memory("forward"))): continue
                    
                    oper.free_connection(r)
                    
                    #Hook up all, new_chain (last child) - oper - sub_set_chain
                    oper.new_connection(sub_set_chain)
                    new_chain.new_connection(oper)
                        
                    combinations.append({"node": n, "new node": new_chain.copy_sub_tree(), "score": ou.calculate_statistical_score(total_analysis)})
                    
                    #Unhook all, new_chain (last child) - oper - sub_set_chain
                    new_chain.free_connection(oper)
                    oper.free_connection(sub_set_chain)
         
                
                #In this way, all childrens that are not important for the chain, will be deleted, also because the linear are refee only to sub_set_chain
                combinations.append({"node": n, "new node": sub_set_chain.copy_sub_tree(), "score": np.abs(total_analysis["linear"])})
                            

            #If node si a single node 
            else:
                        
                total_analysis = utils.total_analysis(l.get_memory("forward"), backward, ["linear", "monoton", "symmetry", "sinusoidal", "vertical_asymptote"])
                    
                #Calculate score for each operation
                for ou in unary_operations:
                        
                    oper = ou()
                    oper.new_connection(l.copy_sub_tree())
                    
                    if(not oper.check_input_domain(0, l.get_memory("forward"))): continue
                    
                    combinations.append({"node": n, "new node": oper, "score": ou.calculate_statistical_score(total_analysis)})
                    
                combinations.append({"node": n, "new node": l, "score": np.abs(total_analysis["linear"])})

        
        
    combinations = [x for x in combinations if(str(x["node"]) != str(x["new node"]) and np.isfinite(x["score"]))]

    combinations =  sorted(combinations, key = lambda x: -x["score"])
              
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



choice_nodes = [lambda n, uo, TR_inputs: standard_package_extensions(n, uo, TR_inputs),
                #lambda n, uo, TR_inputs: default_package_extensions(only_leafs(n), uo, TR_inputs),
                #lambda n, uo, TR_inputs: sorted_by_chatterjee(default_package_extensions(n, uo, TR_inputs)),
                #lambda n, uo, TR_inputs: sorted_by_chatterjee(default_package_extensions(only_leafs(n), uo, TR_inputs))
                ]



def stop_criteria_function(strategy_learning, counter, TR_output):
    
    metrics_MDL = strategy_learning.get_metrics()["MDL"]
    
    model = strategy_learning.get_model()
    
    Y_model = model.get_root().get_memory("forward")
    
    go = counter() or er.MAE(Y_model, TR_output) <= 1e-6 or (False if len(metrics_MDL) < 2 else metrics_MDL[-2] < metrics_MDL[-1])
    
    return go



#-----------------------------------------------------------------------------------------------------------------------


X = {"X": np.linspace(-1, 1, 5, dtype = np.float64)}
Y = np.arcsin(np.sin(3 * X["X"] * 3))

#TR_inputs = {"X": X, "S": np.linspace(-100, 100, 1000, dtype = np.float64)*10}
#TR_output = Y

m = SymbolicRegression.string_to_tree("arcsin(sin(mul(3, X, 3)))")

m.get_root().forward_calculation_memory_sub_tree(X).backward_calculation_memory_sub_tree(Y)

m.view(lambda x: f"{x.get_memory('forward')} / {x.get_memory('backward')}")

#standard_package_extensions(m.get_root().get_all_subtree_nodes(), [op.div, op.exp, op.sqrt, op.sin, op.cos], TR_inputs)

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

