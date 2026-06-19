#g(t(x)) = y
#f(x) + e = y 

#t = modello
#e = errore
#x = input
#y = output target
#g = funzione ipotetica (dove la sua natura viene identificata da Chaterjee) 

#g(t(x)) = f(x) + e 

#Chaterjee(t(x), y) è inversamente proporzionale alla complessità di g e direttamente proporzionale alla complessità di t


#NON PUOI FARE UNA DIMOSTRAZIONE USANDO UNA SIMULAZIONE DI ALGORITMO, PERCHè POTREBBE ESSERE UNA SIMULAZIONE FATTA APPOSTA

#PROVA AD USARE DEI POLINOMI

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path: sys.path.append(parent_dir)

import Indicators as ind
from Model import SymbolicRegression
import Errors as er
import Operations as op


#Si tiene solo le confidence dei nodi perchè ...
def total_confidence(node):
    
    def f(node):
    
        result = []
        for c in node.get_childrens(): result += f(c)
        return result + [node.calculation_hat_confidence()] if isinstance(node, op.Leaf) else result
    
    result = f(node)
    
    return np.mean(result)


def polynomial_experiment():

    rng = np.random.default_rng()

    coeffs = rng.integers(low=-100, high=100, size= rng.integers(low = 6, high = 10))

    X = np.linspace(-100, 100, 10000)
    Y = np.sum([coeffs[i]*np.power(X, i) for i in range(len(coeffs))], axis=0)

    #Add noise
    Y += rng.normal(loc = 0, scale = 0.5, size = len(Y))

    print("Formula", ' + '.join([f'{coeffs[len(coeffs) - i - 1]}X^{len(coeffs) - i - 1}' for i in range(len(coeffs))]))

    results = []

    for i in range(0, len(coeffs) * 2):
        
        calculated_coeffs = np.polyfit(X, Y, i)
        
        a = ['mul(' * r + str(calculated_coeffs[len(calculated_coeffs) - r - 1]) + ', ' + ', '.join(['X)']*r) if r > 0 else str(calculated_coeffs[len(calculated_coeffs) - r - 1]) for r in range(len(calculated_coeffs))]
        
        r = len(a) - 1
        string_model = 'add(' * r + str(a[0]) + ', ' + ', '.join([f'{a[k]})' for k in range(1, len(a))]) if r > 0 else str(a[0])
        
        model = SymbolicRegression.string_to_tree(string_model)
        
        model.get_root().forward_calculation_memory_sub_tree({"X": X}).backward_calculation_memory_sub_tree(Y)

        Y_calculated = np.polyval(calculated_coeffs, X)
        
        #print(f"ERROR MODEL E Y_CALCULATED: {er.MAE(model.get_root().get_memory('forward'), Y_calculated)}")
        
        results.append({
            "degree": i,
            "pearson": ind.pearson_correlation(Y_calculated, Y),
            "spearman": ind.spearman_correlation(Y_calculated, Y),
            "kendall": ind.kendall_correlation(Y_calculated, Y),
            "chatterjee": ind.chatterjee_correlation(Y_calculated, Y),
            "total_confidence": abs(total_confidence(model.get_root())),
            "MAE": er.MAE(Y_calculated, Y)})
        
    #USARE LA TECNICA DI PRENDERE IL MAGGIORE NON FUNZIONA
    #USARE LA TECNICA DI PRENDERE L' ULTIMO DEGREE CHE SI ALZA NON FUNZIONA
    def strategy(confidences, errors):
        
        timeline = [{"confidence": c, "error": e} for c, e in zip(confidences, errors)]
        
        r = 0
        
        for i in range(len(timeline)):
            
            if(i == 0): continue
            
            if(timeline[i]["confidence"] > timeline[i-1]["confidence"]): r = i
            
        return r+1

    print(f"Real degree: {len(coeffs)}, Inferred degree: {strategy([x['total_confidence'] for x in results], [x['MAE'] for x in results])}")
    
    #print([x["MAE"] for x in results])

    max_MAE = np.max([x["MAE"] for x in results])
      
    for i in results:
        i["MAE"] = i["MAE"] / max_MAE
        #i["total_confidence"] = i["total_confidence"] / max_confidence


    plt.plot([x["degree"] for x in results], [x["total_confidence"] for x in results], color = "orange", label = "Total confidence")
    plt.plot([x["degree"] for x in results], [x["MAE"] for x in results], color = "black", label = "Scaled MAE")
    plt.plot([x["degree"] for x in results], [x["pearson"] for x in results], color = "red", label = "Pearson")
    plt.plot([x["degree"] for x in results], [x["spearman"] for x in results], color = "blue", label = "Spearman")
    plt.plot([x["degree"] for x in results], [x["kendall"] for x in results], color = "purple", label = "Kendall")
    plt.plot([x["degree"] for x in results], [x["chatterjee"] for x in results], color = "green", label = "Chatterjee")
    plt.axvline(x=len(coeffs), color='purple', linestyle='--', label = 'Real degree')
    plt.legend()
    plt.title("Polynomial experiments")
    plt.show()

polynomial_experiment()



#COSA MI ASPETTAVO:
# 
# CHATERJEE: PARTISSE BASSO, DALL' INIZIO FINO AL BEST FITTING, SI ALZASSE, SUPERATO IL BEST FITTING SI RIABBASSASSE
# MAE: SI ABBASSE SEMPRE
# Total confidence: niente

#COSA HO OTTENUTO (polynomial):

# CHATERJEE: PARTE ALTO, SI ABBASSA PRIMA DI ARRIVARE ALL BEST FITTING, FINO AL BEST FITTING SI ALZA, E NELL' OVERFITTING RIMANE ALTO (MA NELL OVERFITTING AD UN
# CERTO PUNTO INIZIA AD OSCILLARE). COME MAI? OD è IL POLINOMIALE CHE è UNA FORMA CHE NON PERMETTE QUESTA LOGICA PERCHè è TROPPO SEMPLICE?
# MAE: CIò CHE MI ASPETTAVO
# PEARSON E SPEARMAN NON MI DICONO NIENTE DI INTERESSANTE
# Total confidence: FA SU E GIù, MA QUANDO SI ENTRA IN OVERFITTING, VA SEMPRE GIù. DA CAPIRE MEGLIO QUESTO COMPORTAMENTO, MA SEMBRA INTERESSANTE

# La total confidence, come chaterjee, hanno lo stesso valore su 2 funzioni se quest' ultime passano nello stesso insieme di punti con gli stessi valori. Allora perchè
# cambia la total confidence e si abbassa quando siamo in overfitting?

#In overfitting, tutti gli indicatori, MAE e total confidence non si muovono, perchè possono vedere solo i valori output, ed essendo in overfitting i valori quasi uguali, 
# i valori degl indicatori non cambiano. Ciò che cambia è il loro atteggiamento in underfitting, quello si.


# CAPIRE PERCHé LA TOTAL CONFIDENCE DIMININUISCE QUANDO SIAMO IN OVERFITTING, GUARDANDO SOLO I DATI, SE NON C'è NESSUN ERRORE E PIù FUNZIONI PASSANO PER LA STESSA 
# FUNZIONE, NON CAPISCO PERCHè DIMINUISCE ALLORA

