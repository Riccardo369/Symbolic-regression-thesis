import numpy as np

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path: sys.path.append(parent_dir)

import Indicators as ind

def approx(n, k = 5):
    if(np.isnan(n)): return n
    return int(n * (10**k)) / (10**k)


#Questo script mostra l' andamento dei 3 coefficienti al variare della funzione ed al variare della scalatura lineare di queste funzioni ad una variabile X randomica

rng = np.random.default_rng()

a = 10
b = 50


def linear_extension_invariant():
    
    non_linear_transformation = ["X",
                             "-X",
                             "1/X",
                             "X**2",
                             "X**3",
                             "np.exp(X**3)",
                             "np.sin(X)",
                             "np.cos(X)",
                             "np.abs(X)",
                             "np.exp(X)",
                             "np.log(X)"
                             ]

    for nlt in non_linear_transformation:
        
        for _ in range(10):

            X = np.array(rng.uniform(a, b, int(rng.uniform(10, 10000))))
            #X = X[np.isfinite(X) & ~np.isnan(X)]
            
            #print(np.any(np.isnan(X)))
            
            m = rng.uniform(a, b)
            q = rng.uniform(a, b)
            
            Y1 = eval(nlt) #Apply non linear transformation

            m = rng.uniform(a, b)
            q = rng.uniform(a, b)

            Y2 = (m * Y1) + q
            
            result = {"f": nlt,
                    "P X - (Y1 delta Y2)": abs(approx(ind.pearson_correlation(X, Y1)) - approx(ind.pearson_correlation(X, Y2))),
                    "S X - (Y1 delta Y2)": abs(approx(ind.spearman_correlation(X, Y1)) - approx(ind.spearman_correlation(X, Y2))),
                    "C X - (Y1 delta Y2)": abs(approx(ind.chatterjee_correlation(X, Y1)) - approx(ind.chatterjee_correlation(X, Y2))),
                    "P Y1 - Y2": approx(ind.pearson_correlation(Y1, Y2)),
                    "S Y1 - Y2": approx(ind.spearman_correlation(Y1, Y2)),
                    "C Y1 - Y2": approx(ind.chatterjee_correlation(Y1, Y2))
                    }
        
            print(result)
        #print({i: result[i] for i in dict_aviable[nlt]})
  

def indicator_behaivour():
    
    non_linear_transformation = ["X",
                             "-X",
                             "1/X",
                             "X**2",
                             "X**3",
                             "np.exp(X**3)",
                             "np.sin(X)",
                             "np.cos(X)",
                             "np.abs(X)",
                             "np.exp(X)",
                             "np.log(X)"
                             ]
    
    X = np.linspace(-1000, 1000, 10000)
    
    for nlt in non_linear_transformation:
        
        Y = eval(nlt)
        
        mask = np.isfinite(Y)
 
        result = {"f": nlt,
                "P": approx(ind.pearson_correlation(X[mask], Y[mask])),
                "S": approx(ind.spearman_correlation(X[mask], Y[mask])),
                "C": approx(ind.chatterjee_correlation(X[mask], Y[mask]))
                }
        
        print(result)
        

indicator_behaivour()


#Pearson, Spearman e Chaterjee, sono tutti invarianti rispetto alle scale (basta vedere i delta cche sono tutti = 0)

#SE METTIAMO IN RELAZIONE Y1 E Y2 CON PEARSON E SPEARMAN, SI OTTERRRà SEMPRE UN VALORE = 1

#SE SI FA UN ESTENSIONE LINEARE A SU UN ALTRA ESTENSIONE LINEARE B, CI PUò OTTENERE UNA ESTENSIONE LINEARE C, DOVE B = C, PERCHè:

#B = mb*f(x) + qb

#C = mc * f(x) + qc

#Si fa un estensione A su B:

#A = ma*(mb*f(x) + qb) + qa
#A = ma*mb*f(x) + ma*qb + qa

#Quindi A = C, dove:

#ma*mb = mc
#ma*qb + qa = qc

#Quindi anche se non si conoscono i coefficienti di B e di C, si può tranquillamente arrivare da B a C con un estensione lineare, ed essendo che pearson e spearman sono 
#invarianti rispetto alla scala e traslazione, ridaranno sempre lo stesso valore sia in B che in C.

#Chaterjee ridà sempre un valore più alto, ma è meno affidabile rispetto a pearson e spearman