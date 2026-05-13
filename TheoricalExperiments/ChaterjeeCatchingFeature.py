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

unary_opers = [op.exp, op.ln, op.div, op.abs, op.cos, op.sin, op.neg]

k = 100
c = 0

oper_range_n = (10, 15)
feature_n = 10

while(k < 100):

    rng = np.random.default_rng()
    features = {f"X{i}": rng.normal(loc = 0, scale = 200, size = 10000) for i in range(feature_n)}

    feature_choiced = random.choice(list(features.keys()))
    root = op.Leaf(feature_choiced)

    for _ in range(random.randint(*oper_range_n)):
        
        new_oper = random.choice(unary_opers)()
        
        m = random.uniform(-1000, 1000)
        q = random.uniform(-1000, 1000)
        
        new_oper.new_connection(SymbolicRegression.string_to_tree(f"add(mul({root}, {m}), {q})").get_root(), 0)
        
        root = new_oper

    output = root.forward_calculation_memory_sub_tree(features).get_memory("forward")
    
    #if(~np.all(np.isfinite(output))): continue    #Comment this line for do experiment only with model without forward domain problems

    mask = np.isfinite(output)

    results_catching = sorted([{"feature": f, "Chaterjee": ind.chatterjee_correlation(features[f][mask], output[mask])} for f in features.keys()], key = lambda x: -x["Chaterjee"])

    winners = [x["feature"] for x in results_catching if x["Chaterjee"] == results_catching[0]["Chaterjee"]]

    #print("Function:", root)
    #print("Real feature:", feature_choiced)

    #print("Result chaterjee catching:")
    #for i in results_catching: print(i)
    
    if(feature_choiced in winners): c += 1
    
    k += 1

print(f"Success: {c} / {k}")

