import numpy as np
import matplotlib.pyplot as plt
import random

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path: sys.path.append(parent_dir)

import Indicators as ind
from Model import SymbolicRegression
import Errors as er
import Operations as op
from utils_experiment import create_random_model

def get_nodes(model):
    
    def f(node):
        result = []
        if(node != None):
            for c in node.get_childrens(): result += f(c)
        result += [node]
        return result

    return f(model.get_root())

def get_indicator_internal_nodes(model, f):
    
    nodes = [x for x in get_nodes(model) if not isinstance(x, op.Leaf) and x != None]
    
    result = []
    
    for n in nodes:
        
        result.append(f(n.get_memory("forward"), n.get_memory("backward")))
        
    return result

def get_constants(model):
    return [x for x in get_nodes(model) if x != None and isinstance(x, op.Leaf) and x.is_constant()]



if(__name__ == "__main__"):

    k = 0

    while(k < 20):
        
        rng = np.random.default_rng()
        features = {f"X{i}": rng.normal(loc = 0, scale = 200, size = 10000) for i in range(10)}

        model = create_random_model(features, 10, [op.add, op.mul, op.exp, op.ln, op.div, op.abs, op.cos, op.sin, op.neg])

        #Calculate first output (to compare with others)
        output = model.get_root().forward_calculation_memory_sub_tree(features).get_memory("forward")

        #Catch the constant
        constants = get_constants(model)
        
        #If there is no any constants, skip, because the experiment will not valid (there is no any constant to modify)
        if(len(constants) == 0): continue

        results = []

        for _ in range(10):
            for const in constants: const.set_name(random.uniform(-1000, 1000))
            model.get_root().forward_calculation_memory_sub_tree(features).backward_calculation_memory_sub_tree(output)
            
            result = {"model": str(model),
                    "P": [abs(x) for x in get_indicator_internal_nodes(model, ind.pearson_correlation)],
                    "S": [abs(x) for x in get_indicator_internal_nodes(model, ind.spearman_correlation)],
                    "C": [abs(x) for x in get_indicator_internal_nodes(model, ind.chatterjee_correlation)]}
            
            results.append(result)
            
            #print("MODEL:", result["model"])
            #for i in range(len(result["Captured nodes"])):    
            #    print(f"{i}) {result['Captured nodes'][i]}, P: {result['P'][i]}, S: {result['S'][i]}, C: {result['C'][i]}")
            #    
            #print("")
            
        
        for n in [x for x in get_nodes(model) if not isinstance(x, op.Leaf) and x != None]: print(n)
        
        for indicator in ["P", "S", "C"]:
            
            var_nodes = np.var(np.array([i[f"{indicator}"] for i in results]), axis = 0).tolist()
            print(f"TOTAL VAR {indicator}:", var_nodes)
        
        print(f"\n{'*'*100}")

        k += 1

 
        

#Empiricamente parlando, la correlation tra forward e backward di ogni nodo (con pearons, spearman e chaterjee), non varia di tanto al variare delle costanti di tutto
# il modello. Ma è un falso segnale, perchè ci sono alcuni casi in cui si possono avere anche varianze da 0.2 (che è molto), quindi, al variare delle costanti, variano 
# anche le varie correlazioni (questo vale anche per la radice). C'è ora da capire in quali casi quelle correlazioni sono così violente.

