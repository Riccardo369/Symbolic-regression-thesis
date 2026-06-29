import numpy as np
import Indicators as ind

#Method that flat all elements inside a type chain (like a chain of add or mul)
def flat_elements(node, binary_type):
        
    if(isinstance(node, binary_type)):
        result = []
        for c in node.get_childrens():  result += flat_elements(c, binary_type)
        return result
        
    else: return [node]
    
def flat_elements_function(node, binary_type):
    
    if(isinstance(node, binary_type)):
        result = []
        for c in node.get_childrens():  result += flat_elements(c, binary_type)
        return result
        
    else: return [node.get_string_calculation_function()["body"]]
    

def linear_extension(X, Y):
    
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)
    
    m = np.sum((X - mean_X)*(Y - mean_Y)) / np.sum((X - mean_X)**2)
    
    q = mean_Y - (m*mean_X)
    
    return {"m": m, "q": q}







def __prepare_directional_dataset(x, y, k_x, direction = None):
                
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
        result["linear"] = ind.pearson_correlation(x, y)
                
    if("monoton" in requirements):
                        
        #MONOTON
        result["monoton"] = ind.spearman_correlation(x, y)
                
    if("general" in requirements):
            
        #GENERAL
        result["general"] = ind.chatterjee_correlation(x, y)
            
    if("symmetry" in requirements):
                    
        #SIMMETRY
        x_c = x - k
        y_c = y - np.median(y)
        result["symmetry"] = ind.chatterjee_correlation(np.abs(x_c), np.abs(y_c)) * ind.spearman_correlation(y_c, np.flip(y_c))
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
        oscillation_base = ind.chatterjee_correlation(x_sorted, y_sorted) * (1.0 - ind.spearman_correlation(x_sorted, y_sorted))
        
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
        s_score = ind.spearman_correlation(linea_tempo, pendenza_media_futura)
                    
        # 6. Verifica dello spegnimento finale (Arriva a zero?)
        pendenza_iniziale = pendenza_media_futura[0]
        pendenza_finale = pendenza_media_futura[-1]
                    
        rapporto_velocita = np.divide(pendenza_finale, pendenza_iniziale, out = np.ones_like(pendenza_finale), where = pendenza_iniziale > 0)
                    
        result["horizontal_asymptote"] = np.maximum(0, -s_score) * np.maximum(0, 1.0 - rapporto_velocita)
                 
    if("vertical_asymptote" in requirements):   
                    
        #VERTICAL ASYMPTOTE
        x_left, y_left = __prepare_directional_dataset(x, y, k_x=k, direction="left")
        x_right, y_right = __prepare_directional_dataset(x, y, k_x=k, direction="right")
                    
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
            s_score = ind.spearman_correlation(np.arange(len(pendenza_media_accumulata)), pendenza_media_accumulata)
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
                    
            
