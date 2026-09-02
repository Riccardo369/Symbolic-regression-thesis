import numpy as np
import json
import matplotlib.pyplot as plt

import Indicators as ind
import Errors as er
import NonInvertibleFunctions as nif


#Some functions, like sin or cos, are invertible function. That create some problems to backward phase. So in this script are written some tests for apply and compare
#more attempting to invert sin and cos, with intention to see which technique are the best and why.

#Lets consider the problem: f(g(x)) = y, where you have x, y and f (that is sin or cos). Try to return to x starting from y (for backward phase).
#This experiment do not try to estimate g(x) for aliasing and acceleration that can change (so it is a problem almost impossible to solve).
#Reconstructing the input (with some details about error) allow to have some information for backward for subtree of input
#Instead, if you don't "add" some information, the backward of input is the forward of input, and that dosen't add any information to input subtree.



#Various g(x)
g_functions = ["(2*X) + (np.pi/4)",
               "np.exp(0.5*X)",
               "(X**3) - (3*X)",
               "1/X",
               "np.abs(X)",
               "X + (0.5*np.sin(3*X))"]


approaches = [{"typology": "pseudo_inverse", "sin": nif.pseudo_inverse_input_sin, "cos": nif.pseudo_inverse_input_cos}]


#Apply f(g(x)), reverse f with various technique (giving X like input and y like output), and calculate pearson / MAE between g(x) and reversed f
result = []

#Numero di esperimenti (numero di seed diversi) = 20. In questo caso si possono usare gli stessi seed, perchè il dominio è in R in tutti i casi (facciamo da -10 a 10)
for seed in [84, 192, 317, 429, 501, 683, 715, 842, 906, 958]:
    
    generator = np.random.default_rng(seed)
      
    #Variazione del numero di dati:
    for n in [100, 1000, 10000]:
        
        X = generator.uniform(low = -10, high = 10, size = n)
            
        for g in g_functions:
            
            for tecnique in approaches:
            
                print(f"Non invertible test: random seed {seed}, g = {g}, with {n} points, approach = {tecnique['typology']}")

                g_output = eval(g)
                
                Y_sin = np.sin(g_output)
                Y_cos = np.cos(g_output)
                
                inverse_Y_sin = tecnique["sin"](X, Y_sin)
                inverse_Y_cos = tecnique["cos"](X, Y_cos)
            
                result.append({
                            'seed': seed,
                            'n_samples': n,
                            'function': g,
                            'mae_sin': np.abs(er.MAE(X, inverse_Y_sin)),
                            'mae_cos': np.abs(er.MAE(X, inverse_Y_cos)),
                            "typology": tecnique["typology"],
                            "pearson_sin": np.abs(ind.pearson_correlation(X, inverse_Y_sin)),
                            "pearson_cos": np.abs(ind.pearson_correlation(X, inverse_Y_cos))
                        })
    
    
print("Save results...")
json.dump(result, open("TestNonInvertibleResults.json", "w", encoding="utf-8"), indent=4)
print("Statistical tests done!")

