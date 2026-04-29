import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path: sys.path.append(parent_dir)

import LearningStrategy as ls
from Model import SymbolicRegression
import Operations as op
import Errors as er
import Indicators as ind

def replace_subtree(model, node1, node2):
        
    #Get the father
    father_leaf = node1.get_father()
                
    #The leaf is the root
    if(father_leaf == None): model.set_root(node2)
                   
    else:
            
        i = father_leaf.get_childrens().index(node1)

        op.OperationNode.free_connection(father_leaf, node1)
        op.OperationNode.new_connection(father_leaf, node2, i)
        

X = np.array([0.5, 1, np.pi/2, 2, 3])
Y = (np.sin(X) * 3)



def f(node, target, derative = False):
    
    if(node.get_father() == None): return target if not derative else 1

    var = "r"
    
    root_to_consider = node
    while(root_to_consider.get_father() != None): root_to_consider = root_to_consider.get_father()
     
    def recursion(n):
        if(n == node): return var
        if(issubclass(type(n), op.Leaf)):
            return n.get_name()
            
        return f"{type(n).__name__}({', '.join([str(recursion(child)) for child in n.get_childrens()])})"
    
    #Create the function with var enter 
    function = SymbolicRegression.string_to_tree(recursion(root_to_consider))
    
    #Take the leaf of enter
    leaf = None
    
    for l in function.get_leafs(): 
        if(l.get_name() == var): leaf = l
        
    copy_node = node.copy_sub_tree()
      
    #Set name of enter to primal name (only for calculate the backward for other leaf)
    replace_subtree(function, leaf, copy_node)
    
    #Calculate backward
    function.get_root().forward_calculation_memory_sub_tree({"X": X, "Y": Y}).backward_calculation_memory_sub_tree(target)

    replace_subtree(function, copy_node, leaf)

        
    if(derative): function = function.derivate_model(var)
    
    #print(function)

    return function.get_root().forward_calculation_memory_sub_tree({"X": X, "Y": Y, var: copy_node.get_memory("backward")}).get_memory("forward")



Model = SymbolicRegression.string_to_tree("mul(sin(X), 3)")

Model.get_root().view_sub_tree(lambda x: f(x, Y, False) - Y, name = "Model - target")
Model.get_root().view_sub_tree(lambda x: f(x, Y, True), name = "Derative Model")

Loss = SymbolicRegression.string_to_tree(f"pow(sub(m, Y), 2)")
original_loss_leafs = Loss.get_leafs()
Loss.get_root().set_memory("backward", 0)
replace_subtree(Loss, original_loss_leafs[0], Model.get_root())

Loss.get_root().view_sub_tree(lambda x: f(x, 0, False), name = "Quadratic loss")
Loss.get_root().view_sub_tree(lambda x: f(x, 0, True), name = "Derative Quadratic Loss")

