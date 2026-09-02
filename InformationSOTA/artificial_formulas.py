import numpy as np

opers = ["1/", "np.abs", "np.exp", "np.sqrt", "np.sin", "np.log"]

compose_first_level = dict()

for a in opers:
    
    for b in opers:
        
        compose_first_level[f"{a}({b}(X))"] = np.linspace(-100, 100, 1000, dtype = np.float64)



artificial_formulas = {"3": np.linspace(-100, 100, 1000, dtype = np.float64),
                       "X": np.linspace(-100, 100, 1000, dtype = np.float64),
                       "1/X": np.linspace(-100, 100, 1000, dtype = np.float64),
                       "np.abs(X)": np.linspace(-100, 100, 1000, dtype = np.float64),
                       "np.exp(X)": np.linspace(-100, 100, 1000, dtype = np.float64),
                       "np.sqrt(X)": np.linspace(0, 100, 1000, dtype = np.float64),
                       "np.sin(X)": np.linspace(-100, 100, 1000, dtype = np.float64),
                       
                       "(4 * X) + 5": np.linspace(-100, 100, 1000, dtype = np.float64),
                       "(3 * (1/X)) + 8": np.linspace(-100, 100, 1000, dtype = np.float64),
                       "(5 * np.abs(X)) + 4": np.linspace(-100, 100, 1000, dtype = np.float64),
                       "(3 * np.exp(X)) + 6": np.linspace(-100, 100, 1000, dtype = np.float64),
                       "(8 * np.sqrt(X)) + 12": np.linspace(0, 100, 1000, dtype = np.float64),
                       "(2 * np.sin(X)) + 34": np.linspace(-100, 100, 1000, dtype = np.float64)} | compose_first_level | {
                        
                       
                       #"X**2 + 2*X": np.linspace(-100, 100, 1000, dtype = np.float64),
                       #"2*(X**2) + 5*X + 6": np.linspace(-100, 100, 1000, dtype = np.float64),
                       #"X**2 + np.sin(X)": np.linspace(-100, 100, 1000, dtype = np.float64),
                       #"X**2 * np.sin(X)": np.linspace(-100, 100, 1000, dtype = np.float64),
                       #"X**2 * np.sin(X**2)": np.linspace(-100, 100, 1000, dtype = np.float64),
                       #"2**X + np.sin(X)": np.linspace(-100, 100, 1000, dtype = np.float64)
                       }


