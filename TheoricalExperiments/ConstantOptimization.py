import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path: sys.path.append(parent_dir)

from Model import SymbolicRegression
import Operations as op
import utils
import Errors as er

def linear_extension_optimization(tree, TR_inputs_dict, TR_output):
    
    def get_nodes_to_consider(node):
        result = []
        for c in node.get_childrens(): result += get_nodes_to_consider(c)
        add = []
        if(not node.is_constant()): add += [node]
        return result + add

    def check_domain(model):
        model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        return np.all(np.isfinite(model.get_root().get_memory("forward"))) & np.all([np.all(np.isfinite(x.get_memory("backward"))) for x in model.get_root().get_all_subtree_nodes() if isinstance(x, op.Leaf)])

    history = []

    while(True):
        
        root = tree.get_root()
        
        root.forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        
        nodes = sorted(get_nodes_to_consider(root), key = lambda x: x.calculation_hat_confidence())
        
        while(len(nodes) > 0 and nodes[0] in history): del nodes[0]
        
        if(len(nodes) == 0): break
        
        node = nodes[0]
        
        history.append(node)
        
        lin_ext = utils.linear_extension(node.get_memory("forward"), node.get_memory("backward"))
        
        m = lin_ext["m"] if np.isfinite(lin_ext["m"]) else 1
        q = lin_ext["q"] if np.isfinite(lin_ext["q"]) else 0

        copy_tree = tree.copy()
        
        tree.replace_subtree(node, SymbolicRegression.string_to_tree(f"add(mul({node}, {m}), {q})").get_root())
        
        tree.simplify_model()
        
        #print(tree, "domain valid", check_domain(tree))
        
        if(not check_domain(tree)): tree.replace_subtree(tree.get_root(), copy_tree.get_root())
        
        if(str(copy_tree) != str(tree)): history = []

        error_copy_tree = er.MAE(copy_tree.get_calculation_function()(TR_inputs_dict), TR_output)
        error_tree = er.MAE(tree.get_calculation_function()(TR_inputs_dict), TR_output)
        
        #print("Error previous model:", error_copy_tree)
        #print("Error model:         ", error_tree)
        
        if(error_tree > error_copy_tree):
            tree.replace_subtree(tree.get_root(), copy_tree.get_root())
            break
        
    if(not check_domain(tree)): raise ValueError("It is not possible to get a model that satisfied the internal domain constraints")
        
    return tree
     

if(__name__ == "__main__"):
    
    ideal_function = "(5 + np.log(10*X + 3) * 3)"

    X = np.array([x for x in range(10, 100, 1)], dtype = np.float64) / 10
    Y = eval(ideal_function)

    TR_inputs = {"X": X}
    TR_output = Y

    tree = SymbolicRegression.string_to_tree("ln(add(X, 30))")

    linear_extension_optimization(tree, TR_inputs, TR_output)

    plt.plot(X, Y, color = "red", label = "ideal function")
    plt.plot(X, tree.get_calculation_function()(TR_inputs), color = "blue", label = "calculated function")
    plt.legend()
    plt.show()




#ALGORITMO PROMETTENTE. NON FUNZIONA CON FUNZIONI SURRIETTIVE, PER IL SEMPLICE FATTO CHE DURANTE LA BACKWARD, NON SI TIENE CONTO DI UN INSIEME DI POSSIBILI VALORI, MA DEI 
# VALORI PIù VICINI A QUELLI DI FORWARD. QUINDI SE L' INPUT è SCALATO E TRASLATO, IL CALCOLO DI BACKWARD PROPRIO LO ROMPE, ALL' INPUT INIZIALE DELLA FUNZIONE UNARIA
# AMMETTO CHE CI POSSONO ESSERE ALGORITMI MIGLIORI, MA QUESTO è STATO FATTO SOPRATUTTO PER ESSERE FEDELI AL NON SCEGLIERE IPER-PARAMETRI

#PERCHè USI MAE PIUTTOSTO CHE CHATERJEE? PERCHè ... ??? (DAMMI UNA RISPOSTA)

#FORSE è MEGLIO CERCARE UN ALTRO ALGORITMO