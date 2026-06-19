from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path: sys.path.append(parent_dir)

import Indicators as ind
from Model import SymbolicRegression
import Errors as er
import random
import Operations as op

from utils_experiment import create_random_model 

def total_confidence(node):
    
    def f(node):
    
        result = []
        for c in node.get_childrens(): result += f(c)
        return result + [ind.pearson_correlation(node.get_memory("forward"), node.get_memory("backward"))]
    
    result = f(node)
    
    return np.mean(result)


def comparition(original_string, other_strings):
    
    X = np.linspace(-100, 100, 1000)
    
    models = {"Original model": SymbolicRegression.string_to_tree(original_string)} | {f"Model {i}": SymbolicRegression.string_to_tree(s) for i, s in enumerate(other_strings)}
    
    original_root = models["Original model"].get_root()

    for m in models.keys():
        
        models[m].get_root().forward_calculation_memory_sub_tree({"X": X}).backward_calculation_memory_sub_tree(original_root.get_memory("forward"))
        print("model", models[m], "error", er.MAE(models[m].get_root().get_memory("forward"), original_root.get_memory("forward")))
        
        models[m].view(lambda x: f"P: {ind.pearson_correlation(x.get_memory('forward'), x.get_memory('backward'))} | C: {ind.chatterjee_correlation(x.get_memory('forward'), x.get_memory('backward'))}")
        
        plt.plot(X, original_root.get_memory("forward"), color = "blue", label = "real function")
        plt.plot(X, models[m].get_root().get_memory("forward"), color = "red", label = m)
        plt.legend()
        plt.show()
        
        
def check_domain(model, features, Y):
    model.get_root().forward_calculation_memory_sub_tree(features).backward_calculation_memory_sub_tree(Y)
    return np.all(np.isfinite(model.get_root().get_memory("forward"))) & np.all([np.all(np.isfinite(x.get_memory("backward"))) for x in model.get_leafs()])
   

def catch_the_hole():
    
    def create_the_model_and_the_hole():
        
        rng = np.random.default_rng()
        features = {f"X{i}": rng.normal(loc = 0, scale = 200, size = 10000) for i in range(1)}
        
        while(True):
            
            model = create_random_model(features, 3, [op.add, op.mul, op.neg, op.sin])
            
            Y = model.get_root().forward_calculation_memory_sub_tree(features).get_memory("forward")

            if(check_domain(model, features, Y) and np.sum([x.get_channels() == 1 for x in model.get_root().get_all_subtree_nodes()]) > 0):                
                break

        new_model = model.copy()
        nodes = new_model.get_root().get_all_subtree_nodes()
        
        valid = False
        
        while(True):
            
            old_node = random.choice(nodes)
            del nodes[nodes.index(old_node)]
            if(old_node.get_channels() != 1): continue
            
            new_node = old_node.get_childrens()[0].copy_sub_tree()
            new_model.replace_subtree(old_node, new_node)
            valid = check_domain(new_model, features, Y)

            mask = []
            
            for x in new_model.get_leafs():
                
                new_adding = np.isfinite(x.get_memory("forward")) & np.isfinite(x.get_memory("backward"))
                
                if(isinstance(new_adding, np.ndarray)): mask.append(new_adding)
                
            mask = np.array(mask)
            
            print(mask.shape)
            
            mask = np.all(mask, axis = 0)
            
            if(valid): break
            
            new_model.replace_subtree(new_node, old_node)
            
            
        if(not valid): raise TypeError("The model created (or his succession with a hole) is not aviable")

        print(f"With {np.sum(mask)} points")
        
        def prepare(memory):
            try: return memory[mask]
            except: return memory
         
        def string_ind(x):
            
            if(not isinstance(x.get_memory("backward"), np.ndarray) and x.is_constant()):
                
                P = 1
                C = 1
                
            else:
   
                P = ind.pearson_correlation(prepare(x.get_memory('forward')), prepare(x.get_memory('backward')))    if not x.is_constant() else 1 / (1 + np.var(prepare(x.get_memory('backward'))))
                C = ind.chatterjee_correlation(prepare(x.get_memory('forward')), prepare(x.get_memory('backward'))) if not x.is_constant() else 1 / (1 + np.var(prepare(x.get_memory('backward'))))
                
            return f"P: {P} | C: {C}, MAE: {er.MAE(x.get_memory('forward'), x.get_memory('backward'))}"
        
        new_model_nodes = [{"node": x,
                            "P": np.abs(ind.pearson_correlation(prepare(x.get_memory('forward')),     prepare(x.get_memory('backward')))) if not x.is_constant() else 1 / (1 + np.var(prepare(x.get_memory('backward')))),
                            "C": -np.abs(ind.chatterjee_correlation(prepare(x.get_memory('forward')), prepare(x.get_memory('backward')))) if not x.is_constant() else 1 / (1 + np.var(prepare(x.get_memory('backward'))))}
                            for x in new_model.get_root().get_all_subtree_nodes()] # if not isinstance(x, op.Leaf)]
        
        new_model_nodes = sorted(new_model_nodes, key = lambda x: (x["P"], x["C"]))
        
        print("original model:", model)
        print("hole:", old_node)
        
        for i in new_model_nodes: print(i)
        
        model.view(string_ind)
        new_model.view(string_ind)
        
        

    while(True):
        
        try:
            
            create_the_model_and_the_hole()
            break
        
        except TypeError:
            print("Failed another trying to create model and succession with a hole")
            continue



if(__name__ == "__main__"):
    
    
    pass
    
    #comparition("mul(2, mul(X, X))", ["add(mul(5, mul(X, X)), mul(-3, mul(X, X)))"])
    
    catch_the_hole()
    


#SE LA FUNZIONE è UGUALE, LA CONFIDENCE è 1 DA TUTTE LE PARTI. QUESTO PERCHè SI VEDONO SOLO I DATI, NON LA FORMA DELLA FUNZIONE IDEALE
#QUESTO PUò PORTARE TRANQUILLAMENTE AD OVERFITTING AD ESEMPIO

#LA TECNICA DI TROVARE DIRETTAMENTE IL DATO BASANDOSI SUL PEARSON E CHATTERJEE NON GARANTISCE DI TROVARE IL MIGLIOR NODO DA ESTENDERE,
# IN MOLTI ESPERIMENTI IL VERO NODO DA PRENDERE NON SI TROVA IN CIMA ALLA CLASSIFICA, A VOLTE NEANCHE A PARIMETRO CON ALTRI NODI CANDIDATI