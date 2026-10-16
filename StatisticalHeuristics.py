import numpy as np
from scipy.stats import kurtosis
import Indicators as ind


epsilon_classification = 1e-20


def invert(X):
    return np.where(X != 0, 1/X, float("inf"))



#Può distruggere le scale, distruggere le traslazioni, e permette di fare calcoli più veloci, senza perdere le correlazioni spaziali tra i valori
def rescaling(X, a, b, c, d):
    return c + (((X - a) * (d - c)) / (b - a))



#NON è ROBUSTO AL RUMORE / OUTLIERS, DA RISOLVERE
def linear_correlation(X, Y):
    return ind.pearson_correlation(X, Y)




#[0, 1]
def sinusoidal_coefficient(Y):

    diff_Y = np.diff(Y)
    diff_Y = diff_Y[diff_Y != 0]
    total_change = np.sum(diff_Y[1:] * diff_Y[:-1] < 0)

    return (total_change / (total_change + 1))




def is_sin(X, Y):
    
    # Normalizziamo Y tra -1 e 1 per valutare la posizione
    y_min, y_max = np.min(Y), np.max(Y)
    
    if y_max == y_min: return False
    
    Y_norm = 2 * (Y - y_min) / (y_max - y_min) - 1

    # Indice del punto più vicino a X = 0 (o il primo punto della serie)
    idx_0 = np.argmin(np.abs(X)) if np.min(X) <= 0 <= np.max(X) else 0

    y_start = abs(Y_norm[idx_0])  # Quanto è distante dallo zero? (da 0 a 1)

    # Per il SENO: y_start deve essere vicino a 0  -> sin_weight = 1 - y_start
    # Per il COSENO: y_start deve essere vicino a 1 -> cos_weight = y_start
    sin_weight = 1.0 - y_start
    cos_weight = y_start
    
    result = bool(sin_weight >= cos_weight)

    return result

def is_arcsin(X, Y):
    return ind.spearman_correlation(X, Y) >= 0


def __monoton_acceleration(X, dy_abs, f):
    
    lin_dy_abs = f(dy_abs)
    
    mask = np.isfinite(lin_dy_abs)
        
    return np.abs(ind.spearman_correlation(X[mask], lin_dy_abs[mask]))
    
    


#Some heuristic has to repeat the same operation, so for optimizing the computation, it is created this function that calculates all statistical heuristics
#Method used for learning algorithm in LearningStrategy ("StatisticalHeuristicLearning")
def all_heuristic_statisticals(X, Y, X_name = None):
    
    if(len(X) == 0 or len(Y) == 0): return []
    
    #Sort the elements considering the X values
    index_sort = np.argsort(X)
    X = X[index_sort]
    Y = Y[index_sort]

    result = dict()
    
    #SIN / COS
    sin_coeff = sinusoidal_coefficient(Y)
    
    result["sin"] = sin_coeff + (int(is_sin(X, Y)) * epsilon_classification)
    result["cos"] = sin_coeff + ((1 - int(is_sin(X, Y))) * epsilon_classification)

    dY = np.gradient(Y, X)
    abs_dY = np.clip(np.abs(dY), 1e-20, None)


    #EXP
    result["exp"] = __monoton_acceleration(X, abs_dY, lambda x: np.log(x))
    
    #LN
    result["ln"] = __monoton_acceleration(X, abs_dY, lambda x: 1/x)
    
    #SQRT
    result["sqrt"] = __monoton_acceleration(X, abs_dY, lambda x: 2*(x**(3/2)))

    #DIV

    # Calcolo dell'asimmetria della pendenza
    k = kurtosis(Y, fisher = True) + 2      
    result["div"] = k / (1.0 + k)


    #ARCSIN / ARCCOS: PER QUESTO, C'è SUI LIMITI DELLA FUNZIONE, UN ACCRESCIMENTO CHE FINISCE VERSO L' ALTO / BASSO, COME SE CI FOSSE UN MURO SU CUI SBATTI IL MUSO PER VERTICALE    
    #Y_norm = rescaling(Y, np.min(Y), np.max(Y), 0, 1)
    dist_from_center = Y - np.mean(Y)
    
    #Y - np.mean(Y) serve per calcolare la distanza tra ogni punto della forma ed il punto centrale. Il punto centrale è la media così si ha una robustezza agl' outlier
    #Il trucco sta nel fare un folding di dist_from center (con un abs), in modo tale da avere un unica direzione che parte dal centro, e analizzarla con spearman con l' andamento della derivata (sempre in abs).
    #In questo modo spearman vedrà se il comportamento di entrambe le direzioni sia unico, quindi simmetrico.
    
    
    # Nelle funzioni arcsin/arccos, la pendenza cresce man mano che ci si allontana dal centro
    arcsin_score = np.abs(ind.spearman_correlation(np.abs(dist_from_center), np.abs(dY)))
    
    is_arcsin_val = is_arcsin(X, Y)

    result["arcsin"] = arcsin_score + ((is_arcsin_val) * epsilon_classification)
    result["arccos"] = arcsin_score + ((1 - is_arcsin_val) * epsilon_classification)
    
    #Prepare result
    result = [{"variable": X_name, "operation": k, "value": result[k]} for k in result.keys()]
    
    return result


#Method used for TestStatisticalHeuristic
def all_heuristic_statisticals_complete(X, Y, X_name = None):
    return all_heuristic_statisticals(X, Y, X_name) + [{"variable": X_name, "operation": "constant", "value": 1 / (1 + np.std(Y))},
                                                       {"variable": X_name, "operation": "linear", "value": np.abs(linear_correlation(X, Y))}]

