import matplotlib.pyplot as plt
import numpy as np
import NonInvertibleFunctions as nif

import json


def PseudoInverseWorkingShowOnSin(g):
    
    def build_periodic_inverse_grid(X, Y_sin, k_range=3):

        theta_0 = np.arcsin(Y_sin)
        
        k = np.arange(-k_range, k_range + 1)
        
        famiglia_1 = theta_0[:, None] + 2 * k[None, :] * np.pi
        famiglia_2 = (np.pi - theta_0[:, None]) + 2 * k[None, :] * np.pi
        
        Y_grid = np.concatenate([famiglia_1, famiglia_2], axis=1)
        
        X_grid = np.repeat(X[:, None], Y_grid.shape[1], axis=1)
        
        return X_grid, Y_grid

    #sin(g(x)) = y

    X = np.linspace(-3, 3, 100)

    g_evaluation = eval(g)

    Y = np.sin(g_evaluation)

    mask = np.isfinite(Y)

    X = X[mask]
    g_evaluation = g_evaluation[mask]
    Y = Y[mask]

    X_grid, Y_grid = build_periodic_inverse_grid(X, Y)

    pseudo_inverse_sin = nif.pseudo_inverse_input_sin(X, Y)

    plt.plot(X, X, label = "input", color = "red")
    plt.scatter(X_grid.ravel(), Y_grid.ravel(), label = f"Ideal grid for sin({g})", color = "green", s = 2)
    plt.plot(X, g_evaluation, label = f"{g}", color = "purple")
    plt.plot(X, pseudo_inverse_sin, label = "Reconstruction ideal input", color = "black")
    plt.legend()
    plt.show()
    


def ShowTestNonInvertibleFunctions(text_file = "TestNonInvertibleResults.json"):
    
    print("SHOW INVERSION OF NON INVERTIBLE FUNCTIONS TESTS")
    
    data = json.load(open(text_file, 'r', encoding='utf-8'))

    n_samples = set([x["n_samples"] for x in data])
    functions = set([x["function"] for x in data])
    typology = set([x["typology"] for x in data])
    
    for n in n_samples:
        
        for f in functions:
            
            for t in typology:
                
                results = [x for x in data if x["n_samples"] == n and x["function"] == f and x["typology"] == t]
                
                print("")
                
                print(f'Results with = {n} samples, {f} function, {t} typology of inversion')
                
                mae_sin = [x["mae_sin"] for x in results]
                mae_cos = [x["mae_cos"] for x in results]
                pearson_sin = [x["pearson_sin"] for x in results]
                pearson_cos = [x["pearson_cos"] for x in results]
                
                print(f"MAE(X, inverse_Y sin): mean = {np.mean(mae_sin)}, var = {np.var(mae_sin)}")
                print(f"MAE(X, inverse_Y cos): mean = {np.mean(mae_cos)}, var = {np.var(mae_cos)}")
                print(f"Pearson(X, inverse_Y sin): mean = {np.mean(pearson_sin)}, var = {np.var(pearson_sin)}")
                print(f"Pearson(X, inverse_Y cos): mean = {np.mean(pearson_cos)}, var = {np.var(pearson_cos)}")
                
                
                
                

def ShowTestStatisticalHeuristic(text_file = "TestStatisticalHeuristicResults.json"):
    
    print("SHOW STATISTICAL HEURISTIC FOR IDENTIFYING FUNCTIONS")
        
    data = json.load(open(text_file, 'r', encoding='utf-8'))
    
    for key in data.keys():
        
        values = [x for x in data[key] if x != None]
        
        print("")
        #print(f"{key}: mean = {np.mean(values)}, var = {np.var(values)}, valid values = {len(values)} / {len(data[key])}")
        print(f"{key}: \n{data[key]}")
        




#PseudoInverseWorkingShowOnSin("X")
#PseudoInverseWorkingShowOnSin("1/X")
#PseudoInverseWorkingShowOnSin("np.exp(X)")
#PseudoInverseWorkingShowOnSin("np.log(X)")


#ShowTestNonInvertibleFunctions()


#ShowTestStatisticalHeuristic()