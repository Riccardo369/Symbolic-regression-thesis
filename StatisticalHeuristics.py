import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import json
from scipy.stats import skew

import InformationSOTA.artificial_formulas as af
#import Domains as dm
import utils
import Operations as op
import Indicators as ind
from Model import SymbolicRegression



epsilon_classification = 1e-20
rng = np.random.default_rng()


def invert(X):
    return np.where(X != 0, 1/X, float("inf"))



#Può distruggere le scale, distruggere le traslazioni, e permette di fare calcoli più veloci, senza perdere le correlazioni spaziali tra i valori
def rescaling(X, a, b, c, d):
    return c + (((X - a) * (d - c)) / (b - a))


#Transforma il valore di una pendenza in un valore da 0 ad 1, più la pendenza è violenta (lontana dallo 0), più è vicina ad 1, più è vicina a 0, più è 0
def power_pendency(x):
    return (2 / np.pi) * np.arctan(np.abs(x))

#Calcola le pendenze stimate nei punti intermezzi a quelli trovati
def calculation_pendency(X, Y):
    
    Min_X, Max_X = np.min(X), np.max(X)
    Min_Y, Max_Y = np.min(Y), np.max(Y)
    
    X = rescaling(X, Min_X, Max_X, 0, 1)
    Y = rescaling(Y, Min_Y, Max_Y, 0, 1)

    new_X = X[:-1] + ((X[1:] - X[:-1]) / 2)
    pendency = (Y[1:] - Y[:-1]) / (X[1:] - X[:-1])

    new_X = rescaling(new_X, 0, 1, Min_X, Max_X)

    return new_X, pendency



#NON è ROBUSTO AL RUMORE / OUTLIERS, DA RISOLVERE
def linear_correlation(X, Y):
    return ind.pearson_correlation(X, Y)




#[0, 1]
def sinusoidal_coefficient(Y):

    diff_Y = np.diff(Y)
    diff_Y = diff_Y[diff_Y != 0]
    total_change = np.sum(diff_Y[1:] * diff_Y[:-1] < 0)

    return (total_change / (total_change + 1))




def arcsinusoidal_coefficient1(X, Y):

    y_min, y_max = np.min(Y), np.max(Y)
    if y_max == y_min: return 0.0
    
    Y_norm = 2 * (Y - y_min) / (y_max - y_min) - 1
    
    Z = np.sin((np.pi / 2) * Y_norm)
    
    mask = np.isfinite(Z) & np.isfinite(X)

    score = np.abs(ind.linear_correlation(X[mask], Z[mask]))
    
    return score


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







#Some heuristic has to repeat the same operation, so for optimizing the computation, it is created this function that calculates all statistical heuristics
def all_heuristic_statisticals(X, Y, X_name = None):
    
    #Sort the elements considering the X values
    index_sort = np.argsort(X)
    X = X[index_sort]
    Y = Y[index_sort]
    
    result = dict()
    
    #CONSTANT / LINEAR    X
    result["constant"] = 1 / (1 + np.std(Y))
    result["linear"] =  np.abs(linear_correlation(X, Y))
    
    #SIN / COS     X
    sin_coeff = sinusoidal_coefficient(Y)
    
    result["sin"] = sin_coeff + (int(is_sin(X, Y)) * epsilon_classification)
    result["cos"] = sin_coeff + ((1 - int(is_sin(X, Y))) * epsilon_classification)

    dY = np.gradient(Y, X)
    
    

    #EXP
    # L'esponenziale accelera: consideriamo il valore assoluto della pendenza
    abs_dY = np.abs(dY)
    abs_dY = np.clip(abs_dY, 1e-20, None) # Evita log(0)
    ln_dY = np.log(abs_dY)

    mask_exp = np.isfinite(ln_dY)
    
    # Spearman cattura la crescita logaritmica della pendenza anche se distorta da composte
    result["exp"] = np.abs(ind.spearman_correlation(X[mask_exp], ln_dY[mask_exp]))

    #LN
    # Il logaritmo decelera: l'inverso della pendenza ha una relazione monotona con X
    inv_dY = invert(dY)
    mask_ln = np.isfinite(inv_dY)

    result["ln"] = np.abs(ind.spearman_correlation(X[mask_ln], inv_dY[mask_ln]))
    
    #SQRT
    mask_pos = dY > 0 # La radice reale ha pendenza positiva
 
    # Spearman tra X e dY sarà FORTEMENTE NEGATIVO (pendenza che scende)
    corr_sqrt = ind.spearman_correlation(X[mask_pos], dY[mask_pos])
        
    # Se la pendenza scende fortemente con X, è un ottimo candidato SQRT
    result["sqrt"] = np.abs(corr_sqrt) if corr_sqrt < 0 else 0.0
    
    #DIV
    abs_dY = np.abs(dY)
    mask = np.isfinite(abs_dY)

    # Calcolo dell'asimmetria della pendenza
    s = np.abs(skew(abs_dY[mask]))
        
    # Mappatura naturale in [0, 1) priva di costanti arbitrarie
    result["div"] = s / (1.0 + s)

    
    
    #ARCSIN / ARCCOS: PER QUESTO, C'è SUI LIMITI DELLA FUNZIONE, UN ACCRESCIMENTO CHE FINISCE VERSO L' ALTO / BASSO, COME SE CI FOSSE UN MURO SU CUI SBATTI IL MUSO PER VERTICALE    
    Y_norm = rescaling(Y, np.min(Y), np.max(Y), 0, 1)              # Normalizziamo in [0, 1]
    dist_from_center = np.abs(Y_norm - 0.5)                        # Massima agli estremi, minima al centro
    
    # Nelle funzioni arcsin/arccos, la pendenza cresce man mano che ci si allontana dal centro
    arcsin_score = ind.spearman_correlation(dist_from_center, np.abs(dY))
    
    is_arcsin_val = int(arcsin_score > 0)
    
    arcsin_score = np.abs(arcsin_score) if not np.isnan(arcsin_score) else 0.0


    
    result["arcsin"] = arcsin_score + ((is_arcsin_val) * epsilon_classification)
    result["arccos"] = arcsin_score + ((1 - is_arcsin_val) * epsilon_classification)
    
    #Prepare result
    result = [{"variable": X_name, "operation": k, "value": result[k]} for k in result.keys()]
    
    return result


    
    

def function_base_score(X, Y, f):
    Y1 = f(X)
    mask = np.isfinite(Y1)
    return np.abs(ind.linear_correlation(Y[mask], Y1[mask]))

    
    
    
  
#X_original = np.linspace(-10, 1, 100)
#
#for Y_string in ["np.sin(np.exp(X))", "np.sin(np.log(X))"]: #["np.arcsin(np.exp(X))", "np.exp(np.arcsin(X))", "np.log(X)+np.arcsin(X)", "np.exp(np.arcsin(X))"]:
#    
#    X = X_original.copy()
#
#    Y = eval(Y_string)
#
#    mask = np.isfinite(Y)
#    
#    X = X[mask]
#    Y = Y[mask]
#    
#    
#
#    plt.plot(X, Y, color = "black", label = Y_string)
#    plt.plot(X, monoton_reconstruction(Y), color = "red", label = "Y reconstruction")
#    plt.legend()
#    plt.show()





#Funzionamento per ogni operazione:

#Tipologie di operazioni da cui provengono i nodi: costanti, operazioni, variabili

#Se la lineare vince sugl' altri, ed il nodo è operazione o variabile, si eliminano tutti i candidati dalla gara, perchè essendo lineare, va tutto bene,
#altrimenti se è una costante, si va con l' applicazione della variabile


    
    
    







#LN e EXP: considerare la loro velocità

#SIN e COS: considerare per entrambi il cambio di segno, ed il fatto che la distanza sia regolare. Per distinguere tra i due, è molto difficile, ma nel pratico poco
#importa, perchè cos(x) = sin(x + pi/2)

#ARCSIN, ARCCOS e SQRT: sono deboli, nel senso che mescolate ed annidate altre funzioni, non cambiano di molto il comportamento, apparte qualche allungamento





################################################################################################################
################################################################################################################
################################################################################################################

#DESCRIZIONE INVARIANTE ALLE SCALE E TRASLAZIONI INTERNE ED ESTERNE:

#sin e cos, i fratelli sinusoidali, il loro andamento oscilla, lento giù, veloce giù, lento giù, punto di fermo, lento su, veloce su, lento su, punto di fermo, e ricomincia
#cos parte pi/2 più avanti rispetto a sin, andamento uguale.
#CON LE 2 SCALE E LE TRASLAZIONI, POSSONO DIVENTARE UNO UGUALE ALL' ALTRO, RISCHIO DI CONFONDERLI

#arcsin e arccos, frateli antisinusoidali, il loro andamento è monotono, sul centro pendenza veloce, sugl' estremi pendenza velocissima, come se avessero un muro simmetrico da qualche parte
#arcsin: andamento verso su
#arccos: andamento verso giù
#CON LE 2 SCALE E LE TRASLAZIONI, POSSONO DIVENTARE UNO UGUALE ALL' ALTRO, RISCHIO DI CONFONDERLI

#ln e exp, fratelli rivali con crescita monotona:
#ln crescita rapidissima all' inizio, poi si rallenta bruscamente
#exp crescita lentissima, poi accelera bruscamente
#CON LE 2 SCALE E LE TRASLAZIONI, POSSONO DIVENTARE UNO UGUALE ALL' ALTRO, RISCHIO DI CONFONDERLI. Difficilmente si confondono: la crescita lenta di exp è più lenta della descrescita di ln.

#cot: asintoti verticali continui su: k*pi per ogni k nei numeri naturali. L' andamento è velocissimo quando si scompare e compare sull' asintoto, quando è vicino ad una soglia  in y, l' andamento rallenta.
#In ogni ciclo (tra un asintoto ed un altro), la funzione è simmetrica (con segno opposto) su quella soglia in y.

#div: asintoto su uno specchio, vicino ad una soglia in x, compare e scompare velocissimamente, seguendo il segno del dominio (se dietro alla soglia è negativo, d' avanti è positivo e viceversa).
#Andando appena più distanti da quello specchio, la funzione rallenta di tantissimo, con una curva molto violenta.

#abs: tutto ciò che è sotto una soglia in x, viene proiettato verso su, come fosse ribaltato usando quella soglia come specchio

#sqrt: la y ha un andamento quadratico rispetto alla x (e vale solo dove la x e y sono positive).
#I coefficienti interni ed esterni di scala e transizione sono gli stessi come se la y diventa x e la x diventa y



#SIN e COS: comportamento sinusoidale

#ARCSIN e ARCCOS: comportamento di spinta contro il muro

#LN e EXP: cambio continuo ed improvviso di accelerazione (|spearman| * un qualcosa che faccio capire l' accelerazione, se aumenta diminuisce in modo brusco)

#COT: 

#DIV: tutto piatto, tranne vicino all' asintoto che accelera e decelera in modo brusco

#ABS: sotto una certa soglia, viene ribaltato tutto quanto verso su

#SQRT: da vedere al contrario (invertendo Y ed X), deve essere un comportamento quadratico tra Y ed X (ma solo da una parte di un punto cardine).


