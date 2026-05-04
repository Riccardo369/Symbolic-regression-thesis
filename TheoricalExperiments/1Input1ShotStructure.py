import numpy as np

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path: sys.path.append(parent_dir)

import Indicators as ind

rng = np.random.default_rng()

n = 10000

X = [rng.permutation(rng.standard_normal(n)),
     #rng.permutation(rng.standard_normal(n))
     ]

test_functions = [#"np.sin(np.exp(X[0]))",
                  #"np.exp(X[0])",
                  
                  #"1 / (1 + np.exp(-X[0]**3))",
                  #"1 + np.exp(-X[0]**3)",
                  #"np.exp(-X[0]**3)",
                  #"-X[0]**3",
                  
                  #"np.sqrt(X[0] + 100)",
                  #"X[0] + 100",
                  
                  "np.log(X[0]**2 + np.sin(X[0]) + np.exp(X[0]))",
                  "X[0]**2 + np.sin(X[0]) + np.exp(X[0])",
                  
                  #"2 ** (np.log(X[0] * 3))",
                  #"np.log(X[0] * 3)",
                  #"X[0] * 3",
                  
                  #"np.log(np.log(X[0] * 3))",
                  #"np.log(X[0] * 3)",
                  #"X[0] * 3",
                  
                  #"np.log(2 ** (X[0] * 3))",
                  #"2 ** (X[0] * 3)",
                  #"X[0] * 3",
                  
                  #"np.log(np.sin(X[0]**2))",
                  #"np.sin(X[0]**2)",
                  #"X[0]**2",
                  ]

for test in test_functions:
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Y = eval(test)
    mask = np.isfinite(Y)
    
    for i, x in enumerate(X):
        
        noise = rng.normal(loc = 0, scale = 1, size=len(x))
        
        label = test + f", corr(X{i+1}, Y)"
        
        print(label)
        print(ind.pearson_correlation(x[mask], Y[mask]))
        print(ind.spearman_correlation(x[mask], Y[mask]))
        print(ind.chatterjee_correlation(x[mask], Y[mask]))
        print("")
        
        
        
"""
In situazioni in cui c' è da capire se una pila di funzioni, ipotesi:

1) se funzioni monotone stanno sopra una funzione lineare, la funzione lineare sparisce sotto la funzione monotona
2) i 3 coefficienti non dicono sempre, quale sia la funzione che sta sopra a tutte, a prescindere da quello che c'è sotto.
"""

"""
Se si usasse una tecnica che dà per scontato il contrario del punto 2, l' euristica è: scelgo, tra tutte le possibili operazioni, quella che mi dà, sul passo successivo, 
il valore massimo tra tutti i coefficienti, se il valore massimo tra tutti i coefficienti è minore rispetto al valore massimo attuale, ti fermi. 
Varrebbe lo stesso come ottima euristica per arrivare al risultato finale? Se funziona, più strade potrebbero sempre portare allo stesso valore del passo dopo.
"""