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

def test(total_features):

    k = 0
    c = 0
    
    n = 10000

    feature_n = total_features

    while(k < 100):

        rng = np.random.default_rng()
        features = {f"X{i}": rng.normal(loc = 0, scale = 200, size = n) for i in range(feature_n)}

        feature_choiced = random.choice(list(features.keys()))
        root = create_random_model({feature_choiced: features[feature_choiced]}, 10, [op.add, op.mul, op.div, op.abs, op.cos, op.sin, op.neg]).get_root()

        output = root.forward_calculation_memory_sub_tree(features).get_memory("forward")
        
        mask = np.isfinite(output)
        
        def prepare(memory):
            try: return memory[mask]
            except: return memory

        results_catching = sorted([{"feature": f, "Chaterjee": ind.chatterjee_correlation(prepare(features[f]), prepare(output))} for f in features.keys()], key = lambda x: -x["Chaterjee"])

        winners = [x["feature"] for x in results_catching if x["Chaterjee"] == results_catching[0]["Chaterjee"]]

        #print("Function:", root.get_string_calculation_function()["body"])

        #print("Result chaterjee catching:")
        #for i in results_catching: print(i)
        
        if(feature_choiced in winners): c += 1
        
        print(f"Winners {len(winners)}, with {np.sum(mask)} points")
        
        k += 1

    print(f"With {total_features} features, Success: {c} / {k}")


test(10)
print("-"*100)
test(50)

#Anche con funzioni complesse, chaterjee riesce ad identificare tutto una feature su tutto un sotto albero complesso che intercetta quella 
# sotto albero che ha una sola feature, anche se si aumentano il numero di features. Il problema è che quando ci sono pochi punti,
# chatterjee ridà lo stesso valore per tutti, e da lì tutte le feature hanno lo stesso valore, poche volte succede quando ci sono tanti punti.
# Ma questo è legato alla natura di chatterjee.