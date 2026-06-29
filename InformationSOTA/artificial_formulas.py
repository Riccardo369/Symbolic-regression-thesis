opers = ["1/", "np.abs", "np.exp", "np.log", "np.sin", "np.arcsin"]

compose_first_level = []

for a in opers:
    
    for b in opers:
        
        compose_first_level.append(f"{a}({b}(X))")
        
#op.div, op.abs, op.exp, op.ln, op.sin, op.arcsin

artificial_formulas = ["3",
                       "X",
                       "1/X",
                       "np.abs(X)",
                       "np.exp(X)",
                       "np.log(X)",
                       "np.sin(X)",
                       "np.arcsin(X)",
                       
                       "4 * X + 5",
                       "3* (1/X) + 8",
                       "5 * np.abs(X) + 4",
                       "3 * np.exp(X) + 6",
                       "8 * np.log(X) + 12",
                       "2 * np.sin(X) + 34",
                       "0.4 * np.arcsin(X) + 23",
                       
                       #Tutte quelle composte ad un livello senza costanti di scala e traslazione
                        
                       ] + compose_first_level + [
                            
                       #Lascia questi per dopo
                       
                       #"X**2 + 2*X",
                       #"2*(X**2) + 5*X + 6",
                       #"X**2 + np.cos(X)",
                       #"X**2 * np.cos(X)",
                       #"X**2 * np.cos(X**2)",
                       #"2**X + np.cos(X)"
                       ]

