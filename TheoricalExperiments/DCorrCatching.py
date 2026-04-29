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
        
        
def linear_result(X, Y, n):
    m = (n*(np.sum(X * Y)) - (np.sum(X)*np.sum(Y))) / ((n*np.sum(X**2)) - (np.sum(X)**2))
    q = (np.sum(Y) - (m*np.sum(X))) / n
    #print("m: ", m, "q: ", q)
    return m*X + q
 
 
X = np.array([x for x in range(-100, 100, 1)]) / 10

def f(text_function):
    
    Y = eval(text_function)

    print(f"DCorr(X, {text_function}):", ind.distance_covariance_library(X, Y))
    print(f"DCorr(m*np.sin(X) + q, {text_function}):", ind.distance_covariance_library(linear_result(np.sin(X), Y, len(X)), Y))
    print(f"DCorr(m*X + q, {text_function}):", ind.distance_covariance_library(linear_result(X, Y, len(X)), Y))
    print("")
    
f("np.sin(X) + 3")
f("np.sin(X) + X")
f("np.sin(X) * 3")
f("np.sin(X) * X")
f("np.sin(X) * 3 + X")
f("np.sin(X) * X + X")
f("np.sin(X) * 3 * X")
f("np.sin(X) * X * X")







