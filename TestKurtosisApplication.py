import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from dotenv import load_dotenv
import os
import json
import Indicators as ind


SHOW_PLOT = False

load_dotenv()
seeds_str = os.getenv("SEEDS_LIST")
seeds_list = [int(seed.strip()) for seed in seeds_str.split(",") if seed.strip()]

functions = ["np.log(X)", "np.arcsin(X)", "1000000*np.sin(X)", "1/X", "1/np.abs(X)", "np.sin(X)/np.cos(X)", "X"]

results = []

i = 0

while(i in seeds_list): i += 1

base_seed = i

for seed in seeds_list + [None]:

    for Y_string in functions:
        
        for n in [100, 1000, 10000]:
            
            X = np.random.default_rng(seed).uniform(low = -10, high = 10, size = n) if seed != None else np.linspace(-10, 10, n)
            
            for noise in [0, 0.5, 1]:
            
                print(f"Kurtosis application test: random seed {seed}, f = {Y_string}, n = {n} and noise level = {noise}")

                Y = eval(Y_string)
                
                mask = np.isfinite(Y)
                
                X = X[mask]
                Y = Y[mask]
                
                Y += np.random.default_rng(seed if seed != None else base_seed).normal(0, noise * np.std(Y), size = Y.shape)

                kurtosis_value = kurtosis(Y, fisher = True) + 2
                
                results.append({"seed": seed, "f": Y_string, "n": n, "noise": noise, "kurtosis": kurtosis_value})


                if(SHOW_PLOT):
                    
                    plt.plot(X, Y, color = "black", label = Y_string)
                    plt.legend()
                    plt.show()
                    
                    
    
print("Save results...")
json.dump(results, open("TestKurtosisApplicationResults.json", "w", encoding="utf-8"), indent=4)
print("Kurtosis application tests done!")



