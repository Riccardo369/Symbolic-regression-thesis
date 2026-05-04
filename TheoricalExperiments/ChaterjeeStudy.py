import numpy as np

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path: sys.path.append(parent_dir)

import Indicators as ind

#CHATERJEE è INVARIANTE ALLE SCALE ED ALLE TRASLAZIONI. DIMOSTRAZIONE (FAI PARTIRE PIù VOLTE IL CODICE E DOVRESTI VEDERE CHE CHATERJEE RIDà LO STESSO VALORE)

rng = np.random.default_rng()

a = -100
b = 100

non_linear_transformation = ["X",
                             "np.sin(X)",
                             "np.exp(X)",
                             "X**2"]

for nlt in non_linear_transformation:

    X = rng.uniform(a, b, 100)
    
    X = eval(nlt) #Apply non linear transformation

    Y1 = X

    # Genera coefficienti lineari casuali
    m = rng.uniform(a, b)
    q = rng.uniform(a, b)

    Y2 = (m * X) + q

    print("Transformation:", nlt, "m:", m, "q:", q, "delta situation:", np.abs(ind.chatterjee_correlation(X, Y1) - ind.chatterjee_correlation(X, Y2)))
    
    
#A quanto pare, solo per la EXP cambia di un pò