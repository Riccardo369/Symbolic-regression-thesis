from scipy.stats import pearsonr, spearmanr, rankdata, kendalltau
import numpy as np

def prepare(x, y):
    
    def is_constant(x):
        return type(x) in [int, float] or np.isscalar(x)
    
    def is_numpy_array(x):
        return isinstance(x, np.ndarray)

    if(is_constant(x) and is_constant(y)): return np.array([x]*2), np.array([y]*2)
    if(is_constant(x) and is_numpy_array(y)): return np.array([x]*len(y)), y
    if(is_constant(y) and is_numpy_array(x)): return x, np.array([y]*len(x))
    if(is_numpy_array(x) and is_numpy_array(y)): return x, y

    raise ValueError(f"There is a problem, it is not possible to recognize x (type {type(x)}) or y (type {type(y)})")
    
#For linearity correlation
def pearson_correlation(x, y):
    x, y = prepare(x, y)
    
    if(np.var(y) == 0): return 1
    if(np.var(x) == 0 and np.var(y) > 0): return 0
    
    result = pearsonr(x, y)[0]
    return result

#For monoton correlation
def spearman_correlation(x, y):
    x, y = prepare(x, y)
    
    if(np.var(y) == 0): return 1
    if(np.var(x) == 0 and np.var(y) > 0): return 0
    
    result = spearmanr(x, y)[0]
    return result

#For monoton correlation
def kendall_correlation(x, y):
    result = kendalltau(*prepare(x, y))[0]
    if(np.isnan(result)): result = 0
    return result

#For sinusoidal and monoton correlation
#With less points, chatterjee is less precise with identification
def chatterjee_correlation(x, y):
    
    x, y = prepare(x, y)
    
    n = np.float64(len(x))
    if n < 2: return 0.0
    
    idx = np.argsort(x)
    y_ordered = y[idx]
    
    r = rankdata(y_ordered, method='ordinal').astype(np.float64)
    gr_r = rankdata(y_ordered, method='max').astype(np.float64)
    
    sum_abs_diff = np.sum(np.abs(np.diff(r)))
    
    denominator_term = np.sum(gr_r * (n - gr_r))
    
    if(denominator_term == 0): return 0.0
    
    result = 1 - (sum_abs_diff / ((2.0 / n) * denominator_term))
    
    return result


#--------------------------------------------------------------------------------------------------------------


def prepare_directional_dataset(x, y, k_x, direction = None):
                
    if(direction == None):
                    
        k_x = 0
        mask = x <= float("inf")
                    
    elif(direction == "left"): mask = x <= k_x
    elif(direction == "right"): mask = x >= k_x
    else: raise ValueError(f"direction must be 'left', 'right' or None, not {direction}")
                
    x_filtered = x[mask]
    y_filtered = y[mask]
                
    if len(x_filtered) == 0: return np.array([]), np.array([])
                
    distance = np.abs(x_filtered - k_x)
    idx = np.argsort(distance)
                
    return x_filtered[idx], y_filtered[idx]

      
def total_analysis(x, y, requirements, k = 0):
            
    result = dict()
                            
    if("linear" in requirements):
                
        #LINEAR
        result["linear"] = pearson_correlation(x, y)
                
    if("monoton" in requirements):
                        
        #MONOTON
        result["monoton"] = spearman_correlation(x, y)
                
    if("general" in requirements):
            
        #GENERAL
        result["general"] = chatterjee_correlation(x, y)
            
    if("symmetry" in requirements):
                    
        #SIMMETRY
        x_c = x - k
        y_c = y - np.median(y)
        result["symmetry"] = chatterjee_correlation(np.abs(x_c), np.abs(y_c)) * spearman_correlation(y_c, np.flip(y_c))
        del x_c, y_c
                
        #Order the values
        idx = np.argsort(x)
        x_sorted = x[idx]
        y_sorted = y[idx]
                
        dx, dy = np.diff(x_sorted), np.abs(np.diff(y_sorted))
        pendency = np.divide(dy, dx, out = np.zeros_like(dy), where = dx > 0)
        
        del idx, dx, dy
                
    if("sinusoidal" in requirements):
                
        #SINUSOIDAL   
        oscillation_base = chatterjee_correlation(x_sorted, y_sorted) * (1.0 - spearman_correlation(x_sorted, y_sorted))
        
        segni = np.sign(pendency)
        segni_filtrati = segni[segni != 0]

        cambi_direzione = np.sum(segni_filtrati[:-1] * segni_filtrati[1:] == -1)
        moltiplicatore_saturato = np.minimum(1.0, np.maximum(0.0, (cambi_direzione - 1) / (cambi_direzione + 1e-9)) * (1.0 + (cambi_direzione / len(segni_filtrati))))
                    
        result["sinusoidal"] = oscillation_base * moltiplicatore_saturato
                    
        del oscillation_base, segni, segni_filtrati, cambi_direzione, moltiplicatore_saturato
                
                
    if("horizontal_asymptote" in requirements):
                
        #HORIZONTAL ASYMPTOTE
        somme_pendenze_future = np.cumsum(pendency[::-1])[::-1]
        punti_rimasti = np.arange(len(pendency), 0, -1)
        pendenza_media_futura = somme_pendenze_future / punti_rimasti
                    
        # 5. Verifica della monotonicità (La pendenza cala verso destra?)
        linea_tempo = np.arange(len(pendenza_media_futura))
        s_score = spearman_correlation(linea_tempo, pendenza_media_futura)
                    
        # 6. Verifica dello spegnimento finale (Arriva a zero?)
        pendenza_iniziale = pendenza_media_futura[0]
        pendenza_finale = pendenza_media_futura[-1]
                    
        rapporto_velocita = np.divide(pendenza_finale, pendenza_iniziale, out = np.ones_like(pendenza_finale), where = pendenza_iniziale > 0)
                    
        result["horizontal_asymptote"] = np.maximum(0, -s_score) * np.maximum(0, 1.0 - rapporto_velocita)
                 
    if("vertical_asymptote" in requirements):   
                    
        #VERTICAL ASYMPTOTE
        x_left, y_left = prepare_directional_dataset(x, y, k_x=k, direction="left")
        x_right, y_right = prepare_directional_dataset(x, y, k_x=k, direction="right")
                    
        # Sotto-funzione interna per evitare di duplicare il codice di calcolo
        def compute_explosion(x_prep, y_prep):
                        
            # Invertiamo l'ordine: ora l'indice 0 è il punto più LONTANO da k
            # e l'ultimo indice è il punto VICINISSIMO a k (l'asintoto)
            x_towards = x_prep[::-1]
            y_towards = y_prep[::-1]
                            
            # Calcoliamo i passi punto-punto 
            dx = np.abs(np.diff(x_towards))
            dy = np.abs(np.diff(y_towards))
                            
            # Logica di divisione protetta dagli zeri
            pendenza = np.divide(dy, dx, out=np.zeros_like(dy), where=dx > 0)
                            
            # Inviluppo cumulativo
            pendenza_media_accumulata = np.cumsum(pendenza) / np.arange(1, len(pendenza) + 1)
                            
            # Test di monotonicità dell'esplosione (Spearman)
            s_score = spearman_correlation(np.arange(len(pendenza_media_accumulata)), pendenza_media_accumulata)
            punteggio_esplosione = np.maximum(0, s_score)
                            
            # FILTRO AD ALTA SEVERITÀ (Verifica l'impennata reale rispetto a k)
            p_start = pendenza_media_accumulata[0]   # Lontano da k
            p_max = np.max(pendenza_media_accumulata) # Vicino a k
                            
            # Rapporto di moltiplicazione
            rapporto_moltiplicazione = p_max / p_start if p_start > 0 else 0 
                    
            # Funzione di attivazione morbida
            fattore_impennata = np.minimum(1.0, rapporto_moltiplicazione / 5.0)
                            
            return punteggio_esplosione * fattore_impennata
                        
        try: punteggio_sinistro = compute_explosion(x_left, y_left)
        except: punteggio_sinistro = 1
                        
        try: punteggio_destro = compute_explosion(x_right, y_right)
        except: punteggio_destro = 1
                        
        # 3. Restituiamo il valore massimo (l'asintoto può essere anche solo monolaterale)
        result["vertical_asymptote"] = punteggio_sinistro * punteggio_destro
                    
        return result
                    
            




if(__name__ == "__main__"):
    
    x = 3
    y = np.array([1, 2, 3])

    f = lambda x, y: pearson_correlation(x, y)

    print("Pearson")
    print(f(x, x))
    print(f(x, y))
    print(f(y, x))
    print(f(y, y))

    f = lambda x, y: spearman_correlation(x, y)

    print("Spearman")
    print(f(x, x))
    print(f(x, y))
    print(f(y, x))
    print(f(y, y))
    
    f = lambda x, y: kendall_correlation(x, y)
    
    print("Kendall")
    print(f(x, x))
    print(f(x, y))
    print(f(y, x))
    print(f(y, y))

    f = lambda x, y: chatterjee_correlation(x, y)

    print("Chaterjee")
    print(f(x, x))
    print(f(x, y))
    print(f(y, x))
    print(f(y, y))
    