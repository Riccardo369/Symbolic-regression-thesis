import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import json
from scipy.signal import hilbert

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






#[0, 1]
def sinusoidal_coefficient(Y):

    diff_Y = np.diff(Y)
    diff_Y = diff_Y[diff_Y != 0]
    total_change = np.sum(diff_Y[1:] * diff_Y[:-1] < 0)

    return (total_change / (total_change + 1))

def arcsinusoidal_coefficient(X, Y):

    y_min, y_max = np.min(Y), np.max(Y)
    if y_max == y_min: return 0.0
    
    Y_norm = 2 * (Y - y_min) / (y_max - y_min) - 1
    
    Z = np.sin((np.pi / 2) * Y_norm)
    
    mask = np.isfinite(Z) & np.isfinite(X)

    score = np.abs(ind.pearson_correlation(X[mask], Z[mask]))
    
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


def constant_score(Y):
    return 1 / (1 + np.std(Y))

def linear_score(X, Y):
    return np.abs(ind.pearson_correlation(X, Y))

def sin_score(X, Y):
    return sinusoidal_coefficient(X, Y) + (int(is_sin(X, Y)) * epsilon_classification)

def cos_score(X, Y):
    return sinusoidal_coefficient(X, Y) + ((1 - int(is_sin(X, Y))) * epsilon_classification)

def arcsin_score(X, Y):
    return arcsinusoidal_coefficient(X, Y) + (int(is_arcsin(X, Y)) * epsilon_classification)

def arccos_score(X, Y):
    return arcsinusoidal_coefficient(X, Y) + ((1 - int(is_arcsin(X, Y))) * epsilon_classification)
      
def exp_score(X, Y):
    
    pendency_X, pendency_Y = calculation_pendency(X, Y)       
    pendency_Y = power_pendency(pendency_Y)

    #PRINT MATCHING
    ln_pendency_Y = np.log(pendency_Y)
    
    mask = np.isfinite(pendency_X)
    
    return np.abs(ind.pearson_correlation(pendency_X[mask], ln_pendency_Y[mask]))

def ln_score(X, Y):
    
    pendency_X, pendency_Y = calculation_pendency(X, Y)
    
    inv_P = invert(pendency_Y)
    
    # 3. Maschera i valori finiti
    mask = np.isfinite(inv_P) & np.isfinite(pendency_X)
        
    # 4. Pearson tra X e 1/Y' deve essere ~1.0 per il logaritmo
    return np.abs(ind.pearson_correlation(pendency_X[mask], inv_P[mask]))

def div_score(X, Y):
    
    pendency_X, pendency_Y = calculation_pendency(X, Y)
    
    pendency_Y = invert(np.sqrt(np.abs(pendency_Y)))
    
    lin_ext = utils.linear_extension(pendency_X, pendency_Y)
    
    print_matching = lin_ext["m"]*pendency_X + lin_ext["q"]
    
    mask = np.isfinite(print_matching)
    
    print_matching = print_matching[mask]
    pendency_X = pendency_X[mask]
    pendency_Y = pendency_Y[mask]
    
    K = X - invert(Y)
    k = np.mean(K)
    
    left = pendency_X < k
    right = pendency_X > k
    
    if(np.sum(left) + np.sum(right) < 2): return 0
    
    left_coeff = np.abs(ind.pearson_correlation(print_matching[left], pendency_Y[left])) if np.sum(left) >= 2 else 1
    right_coeff = np.abs(ind.pearson_correlation(print_matching[right], pendency_Y[right])) if np.sum(right) >= 2 else 1
    
    return left_coeff * right_coeff
  
def sqrt_score(X, Y):
    return np.abs(ind.pearson_correlation(X, Y**2))





#f(a, b, c, d, x) = c * ln(a * x + b) + d

#Df / da = cx / (ax + b)
#Df / db = c  / (ax + b)
#Df / dc = ln(ax + b) 
#Df / dd = 1


#Da / dx = 0 * 1/x + a/b = a/b = 0

#a/b = 0
#a = b*0
#a = 0

#





#Some heuristic has to repeat the same operation, so for optimizing the computation, it is created this function that calculates all statistical heuristics
def all_heuristic_statisticals(X, Y, X_name = None):
    
    #Sort the elements considering the X values
    index_sort = np.argsort(X)
    X = X[index_sort]
    Y = Y[index_sort]
    
    result = dict()
    
    #CONSTANT / LINEAR
    result["constant"] = 1 / (1 + np.std(Y))
    result["linear"] =  np.abs(ind.pearson_correlation(X, Y))
    
    #SIN / COS
    sin_coeff = sinusoidal_coefficient(Y)
    
    result["sin"] = sin_coeff + (int(is_sin(X, Y)) * epsilon_classification)
    result["cos"] = sin_coeff + ((1 - int(is_sin(X, Y))) * epsilon_classification)
    
    #ARCSIN / ARCCOS
    arcsin_coeff = arcsinusoidal_coefficient(X, Y)
    
    result["arcsin"] = arcsin_coeff + (int(is_arcsin(X, Y)) * epsilon_classification)
    result["arccos"] = arcsin_coeff + ((1 - int(is_arcsin(X, Y))) * epsilon_classification)
    
    #EXP
    pendency_X, pendency_Y = calculation_pendency(X, Y)
    mask = np.isfinite(pendency_X)

    ln_pendency_Y = np.log(power_pendency(pendency_Y))
    result["exp"] = np.abs(ind.pearson_correlation(pendency_X[mask], ln_pendency_Y[mask]))
    
    #LN
    inv_P = invert(pendency_Y)
    mask = np.isfinite(inv_P) & mask     
    result["ln"] = np.abs(ind.pearson_correlation(pendency_X[mask], inv_P[mask]))
    
    #SQRT
    result["sqrt"] = np.abs(ind.pearson_correlation(X, Y**2))
    
    #DIV
    pendency_Y = invert(np.sqrt(np.abs(pendency_Y)))
    mask = np.isfinite(pendency_Y)
        
    K = X - invert(Y)
    k = np.mean(K)
        
    left = pendency_X < k
    right = pendency_X > k
        
    if(np.sum(left) + np.sum(right) < 2): result["div"] = 0
    
    else: 
    
        left_coeff = np.abs(ind.pearson_correlation(pendency_X[left], pendency_Y[left])) if np.sum(left) >= 2 else 1
        right_coeff = np.abs(ind.pearson_correlation(pendency_X[right], pendency_Y[right])) if np.sum(right) >= 2 else 1
            
        result["div"] = left_coeff * right_coeff
        
        
    #Prepare result
    result = [{"variable": X_name, "operation": k, "value": result[k]} for k in result.keys()]
    
    return result


    
    

def function_base_score(X, Y, f):
    Y1 = f(X)
    mask = np.isfinite(Y1)
    return np.abs(ind.pearson_correlation(Y[mask], Y1[mask]))

    
    
    




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


