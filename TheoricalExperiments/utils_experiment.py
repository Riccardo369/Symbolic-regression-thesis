import random

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path: sys.path.append(parent_dir)

from Model import SymbolicRegression
import Operations as op



def create_random_model(features, n, operations):
    
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

    def count_nodes(model):
        return len(get_nodes(model))

    def count_leafs(model):
        return len([x for x in get_nodes(model) if x == None or isinstance(x, op.Leaf)])
        
    def replace(model, i, node_of_replacing):
        
        counter = [0]
        
        def f(node, counter, i, node_of_replacing):
                
            for e, c in enumerate(node.get_childrens()):
                
                if(isinstance(c, op.Leaf) or c == None):
                    
                    if(counter[0] == i):
                        
                        if(isinstance(c, op.Leaf)): node.free_connection(e)
                        node.new_connection(node_of_replacing, e)
                    
                    counter[0] += 1
        
                else: f(c, counter, i, node_of_replacing)
        
        f(model.get_root(), counter, i, node_of_replacing)


    #Create model
    model = SymbolicRegression.string_to_tree("0")

    for n in [random.choice(operations)() for _ in range(n)]:
        if(count_nodes(model) == 1): model = SymbolicRegression.string_to_tree(f"{n}")
        else: replace(model, random.choice(list(range(count_leafs(model)))), n)

    #Fix the leafs
    for i in range(count_leafs(model)):
        choose = random.choice([op.Leaf(f) for f in features.keys()] + [op.Leaf(random.uniform(-1000, 1000)) for _ in features.keys()])
        replace(model, i, choose)
        
    model.simplify_model()
    
    return model