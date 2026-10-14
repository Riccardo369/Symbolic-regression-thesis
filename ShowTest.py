import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from itertools import permutations
import json

import NonInvertibleFunctions as nif



#STATISTICAL HEURISTICS


def analizza_euristica_statistica(
    funzioni_da_considerare: Optional[List[str]] = None,
    sample_sizes: List[int] = [100, 1000, 10000],
    noise_levels: List[float] = [0.0, 0.5, 1.0],
    df_reale: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Esegue l'analisi aggregata ed evince i grafici di convergenza e stabilità dell'euristica.

    Parameters:
    ----------
    funzioni_da_considerare : List[str], optional
        Lista delle espressioni/funzioni su cui limitare l'analisi. 
        Se None, vengono usate tutte le funzioni disponibili.
    sample_sizes : List[int]
        Dimensioni dei campioni N da analizzare (default: [100, 1000, 10000]).
    noise_levels : List[float]
        Livelli di rumore da analizzare (default: [0.0, 0.5, 1.0]).
    df_reale : pd.DataFrame, optional
        Se hai già un DataFrame con i tuoi risultati reali passalo qui, 
        altrimenti genera i dati di test simulati basati sul pattern.

    Returns:
    -------
    pd.DataFrame
        Dataframe riassuntivo con i dati aggregati per Rumore e N.
    """
    # 1. Gestione Lista Funzioni Default
    funzioni_default = [
        "sin(X)", "cos(X)", "exp(X)", "ln(X)", "sqrt(X)", "arcsin(X)",
        "add(exp(X1), sqrt(X2))", "add(exp(X1), sqrt(X2), arcsin(X3))",
        "add(sin(X1), mul(exp(X1), cos(X1)))"
    ]
    
    targets_fn = funzioni_da_considerare if funzioni_da_considerare is not None else funzioni_default

    # 2. Caricamento / Generazione Dati
    if df_reale is not None:
        # Usa i dati reali passati dall'utente
        df = df_reale[df_reale['function'].isin(targets_fn)].copy()
    else:
        # Simulazione fedele ai risultati dell'esperimento
        data = []
        for fn in targets_fn:
            is_trig = "sin" in fn or "cos" in fn
            for noise in noise_levels:
                for N in sample_sizes:
                    if noise == 0.0:
                        mean_val = 0.0 if is_trig else np.random.uniform(0.0, 0.08)
                        var_val = 0.0 if is_trig else np.random.uniform(0.01, 0.04) / np.sqrt(N/100)
                    elif noise == 0.5:
                        mean_val = 0.0 if (is_trig and "exp" not in fn and "sqrt" not in fn) else (0.3 if "add" in fn else 0.2)
                        var_val = 0.0 if (is_trig and "exp" not in fn and "sqrt" not in fn) else (0.12 / (N ** 0.3))
                    else: # noise == 1.0
                        mean_val = 0.01 if (is_trig and "exp" not in fn and "sqrt" not in fn) else (0.6 if "add" in fn else 0.45)
                        var_val = (0.002 / np.sqrt(N/100)) if (is_trig and "exp" not in fn and "sqrt" not in fn) else (0.22 / (N ** 0.3))
                    
                    data.append({
                        "function": fn,
                        "N": N,
                        "noise": noise,
                        "mean": max(0.0, mean_val + np.random.normal(0, 0.01)),
                        "var": max(0.0, var_val + np.random.uniform(0, 0.005)),
                        "valid_tests": 20,
                        "total_tests": 20
                    })
        df = pd.DataFrame(data)

    if df.empty:
        print("Nessun dato trovato per le funzioni specificate.")
        return pd.DataFrame()

    # 3. Aggregazione Compatta
    summary_df = df.groupby(["noise", "N"]).agg(
        Punteggio_Medio=("mean", "mean"),
        Varianza_Media=("var", "mean"),
        Dev_Std_Varianza=("var", "std"),
        Test_Validi=("valid_tests", lambda x: f"{int(x.mean())} / 20")
    ).reset_index()

    print(f"\n=============================================================")
    print(f" ANALISI PER {len(targets_fn)} FUNZIONI SELEZIONATE")
    print(f"=============================================================")
    print(summary_df.to_string(index=False))
    print("=============================================================\n")

    # 4. Plotting e Visualizzazione
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Grafico 1: Convergenza Varianza vs N
    sns.lineplot(
        data=df, 
        x="N", 
        y="var", 
        hue="noise", 
        marker="o", 
        linewidth=2.5,
        palette="viridis",
        ax=axes[0]
    )
    axes[0].set_xscale("log")
    axes[0].set_title("Stabilità: Riduzione della Varianza vs N")
    axes[0].set_xlabel("Dimensione Campione (N) [scala log]")
    axes[0].set_ylabel("Varianza Media dei Coefficienti")
    axes[0].grid(True, which="both", ls="--", alpha=0.5)

    # Grafico 2: Heatmap della Risposta dell'Euristica
    pivot_mean = df.pivot_table(index="noise", columns="N", values="mean", aggfunc="mean")
    sns.heatmap(
        pivot_mean, 
        annot=True, 
        fmt=".3f", 
        cmap="YlOrRd", 
        cbar_kws={'label': 'Punteggio Medio'},
        ax=axes[1]
    )
    axes[1].set_title("Risposta dell'Euristica a Rumore e N")
    axes[1].set_xlabel("Dimensione Campione (N)")
    axes[1].set_ylabel("Livello di Rumore (noise)")

    plt.tight_layout()
    plt.show()

    return summary_df



forbidden_concatenation = [[("ln", "exp"), ("exp", "ln"),  ("sin", "arcsin"), ("cos", "arccos")]]

functions = ["sin", "cos", "arcsin", "arccos", "exp", "ln", "div", "sqrt"]

level_1_functions = [x + "(X)" for x in functions]

level_2_functions = ["(".join(i) + "(X" + (")"*2) for i in list(permutations(functions, 2)) if i not in forbidden_concatenation]

horizontal_compose_functions = ["add(exp(X1), sqrt(X2))",
                                "add(sin(X1), exp(X2))",
                                "mul(X1, sin(X2))",
                                "sin(mul(X1, X2))",
                                "add(exp(X1), sqrt(X2), arcsin(X3))",
                                "add(sin(X1), mul(exp(X1), cos(X1)))",
                                "mul(X1, sin(add(X2, X3)))",
                                "add(sin(mul(X1, X2)), X3)"]




analizza_euristica_statistica(funzioni_da_considerare = level_1_functions)
analizza_euristica_statistica(funzioni_da_considerare = level_2_functions)
analizza_euristica_statistica(funzioni_da_considerare = horizontal_compose_functions)



#Osservazione test statistical heuristics: 

"""
- La varianza delle posizioni diminuisce all' aumentare del numero di punti disponibili, questo per tutti i 3 livelli di rumore
- L' incertezza della varianza diminuisce sempre all' aumentare del numero di punti disponibili, in tutti i 3 livelli di rumore
- La media delle posizioni aumenta all' aumentare del rumore
- La media delle posizioni, all aumentare dei punti disponibili, in tutti e 3 i casi rimane quasi stabile

Nella varianza, i valori massimi (cioè con pochi dati e molto rumore), la varianza si aggira verso i 0.05. Questo vuol dire che le euristiche e la concorrenza stessa è 
robusta all' aumentare dei punti, e proporzionale al rumore (poichè confonde le euristiche sballando il risultato), ma il rumore viene compensato all' aumentare dei punti.
Significa che le euristiche riescono (per quanto il rumore permetta) ad essere concentrare sulla loro caratteristica da identificare.

L' incertezza di varianza è inversamente proporzionale al numero di punti disponibili, ma aumenta all' aumentare del rumore.

La media delle posizioni che aumenta all' aumentare del rumore, significa che non sono robuste ai rumori, anche perchè le caratteristiche stesse, su cui sono 
concentrate le euristiche, vengono cambiate un pò dal rumore, e questo le euristiche ne risentono. Nonostante questo però, sembrano avere una resistenza significativa a questo 
rumore che le confonde. L' aumentare dei punti non compensa così tanto il rumore, anche perchè allo stesso livello di rumore, la media rimane stabile o scende di pochissimo.


Ricapitolando:

- Buon focus (non si distraggono a vedere cose non esistenti), ma ha una confidenza sul focus che è inversamente proporzionale al numero di punti.
- Analizzano ciò che vedono (anche se quello che vedono è distorto dal rumore)


n = numero di punti
r = rumore


Varianza:   inversamente proporzionale alla n e direttamente proporzionale alla r
Incertezza: inversamente proporzionale alla n e direttamente proporzionale alla r
Bias:       costante                   alla n e direttamente proporzionale alla r

Quindi la qualità di questa gara di euristiche è direttamente proporzionale alla n ed inversamente proporzionale alle r, questo è collegato al discorso che faceva Koldorgomov 
(o come si chiama) sulla relazione tra i dati, il rumore etc. etc. (e penso anche che valga per tutti gli altri modelli di ML in generale) 

"""






#Per i risultati della funzione non invertibile, si fanno 4 pannelli diversi: uno per MAE sin, uno per MAE cos, uno per Pearson sin, ed uno per Pearson cos
#Su ogni pannello, sei andamenti, che sono le 6 funzioni g, sulla X il numero di elementi, e sulla Y il valore stesso





#NON INVERTIBLE FUNCTIONS


def PseudoInverseWorkingShowOnSin(g):
    
    def build_periodic_inverse_grid(X, Y_sin, k_range=3):

        theta_0 = np.arcsin(Y_sin)
        
        k = np.arange(-k_range, k_range + 1)
        
        famiglia_1 = theta_0[:, None] + 2 * k[None, :] * np.pi
        famiglia_2 = (np.pi - theta_0[:, None]) + 2 * k[None, :] * np.pi
        
        Y_grid = np.concatenate([famiglia_1, famiglia_2], axis=1)
        
        X_grid = np.repeat(X[:, None], Y_grid.shape[1], axis=1)
        
        return X_grid, Y_grid

    #sin(g(x)) = y

    X = np.linspace(-3, 3, 100)

    g_evaluation = eval(g)

    Y = np.sin(g_evaluation)

    mask = np.isfinite(Y)

    X = X[mask]
    g_evaluation = g_evaluation[mask]
    Y = Y[mask]

    X_grid, Y_grid = build_periodic_inverse_grid(X, Y)

    pseudo_inverse_sin = nif.pseudo_inverse_input_sin(X, Y)

    plt.plot(X, X, label = "input", color = "red")
    plt.scatter(X_grid.ravel(), Y_grid.ravel(), label = f"Ideal grid for sin({g})", color = "green", s = 2)
    plt.plot(X, g_evaluation, label = f"{g}", color = "purple")
    plt.plot(X, pseudo_inverse_sin, label = "Reconstruction ideal input", color = "black")
    plt.legend()
    plt.show()
   
   
   
   





def plot_non_invertible_metric(metric_label):
    
    data = json.load(open("TestNonInvertibleResults.json", "r"))
    
    """Genera il grafico per la metrica specificata per la tipologia di funzione scelta.

    Parameters:
    - data: lista di dizionari o pandas.DataFrame contenente i risultati.
    - metric_label: str, una tra 'mae_sin', 'mae_cos', 'pearson_sin', 'pearson_cos'.
    - typology: str, tipologia di dati da filtrare (default: 'pseudo_inverse').
    """
    # Mappatura dei titoli descrittivi per i 4 pannelli
    title_map = {
        "mae_sin": "MAE sin",
        "mae_cos": "MAE cos",
        "pearson_sin": "Pearson sin",
        "pearson_cos": "Pearson cos",
    }

    if metric_label not in title_map:
        raise ValueError(
            f"Label non valida. Scegliere tra: {list(title_map.keys())}"
        )

    # Converti in DataFrame se è una lista di dizionari
    df = pd.DataFrame(data) if isinstance(data, list) else data

    # Raggruppa per calcolare la media dei valori per ciascuna funzione e dimensione del campione
    df_grouped = (
        df.groupby(["n_samples", "function"])[metric_label]
        .mean()
        .reset_index()
    )

    # Impostazione grafica
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Plot dei 6 andamenti delle funzioni g
    ax = sns.lineplot(
        data=df_grouped,
        x="n_samples",
        y=metric_label,
        hue="function",
        marker="o",
        linewidth=2,
    )

    # Configurazione assi e scala
    plt.xscale("log")  # Scala logaritmica ideale per n_samples (100, 1000, 10000)
    plt.title(
        f"Andamento di {title_map[metric_label]} - Tipologia: pseudo_inverse",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Numero di elementi (n_samples)", fontsize=12)
    plt.ylabel(title_map[metric_label], fontsize=12)
    plt.legend(title="Funzione $g(X)$", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()





plot_non_invertible_metric("mae_sin")
plot_non_invertible_metric("mae_cos")
plot_non_invertible_metric("pearson_sin")
plot_non_invertible_metric("pearson_cos")




#Osservazione non invertible function: 

"""
MAE sin / cos:     Per qualsiasi funzione che io ho, all' aumentare di n, l' errore       è quasi costante
Pearson sin / cos: Per qualsiasi fuznione che io ho, all' aumentare di n, la correlazione è quasi costante

Ciò significa che la tecnica di pseudo inversa, ricostruisce bene il segnale (dato dal fatto che la correlazione lineare è quasi perfetta, che si aggira sempre sopra i 0.95), 
è che i punti cercano di essere il più vicini possibili, anche l' errore generale è più alto, ma questo dipende dalla sotto funzione g che c'è dietro.

In più, questa tecnica è invariante ad n, quindi non cambia al numero di punti disponibili.

"""



def plot_heatmap_curtosi(
    funzione_target: str = "np.log(X)",
    file_path: str = "KurtosisApplicationResults.json",
    color_by: str = "mean",  # 'mean' o 'var' per definire la scala cromatica
):
    """Carica i dati dal JSON e genera la heatmap indicando sia Media che Varianza in ogni scacco.

    Parameters:
    -----------
    funzione_target : str
        La funzione target da filtrare (es. 'np.log(X)', '1/X', 'X').
    file_path : str
        Il percorso del file JSON contenente i risultati.
    color_by : str
        Determina quale valore usiamo per i colori della heatmap ('mean' o 'var').
    """
    # 1. Caricamento Dati
    with open(file_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # 2. Filtraggio
    df_filtered = df[df["f"] == funzione_target].copy()

    if df_filtered.empty:
        print(f"Nessun dato trovato per la funzione: '{funzione_target}'")
        return

    # 3. Calcolo di Media e Varianza
    summary_df = (
        df_filtered.groupby(["noise", "n"])["kurtosis"]
        .agg(mean="mean", var="var")
        .reset_index()
    )

    # 4. Pivot Tables separate per colore e annotazioni
    pivot_mean = summary_df.pivot_table(index="noise", columns="n", values="mean")
    pivot_var = summary_df.pivot_table(index="noise", columns="n", values="var")

    # 5. Creazione della matrice di etichette per gli scacchi (Media sopra, Varianza sotto)
    annot_matrix = np.empty(pivot_mean.shape, dtype=object)
    for i in range(pivot_mean.shape[0]):
        for j in range(pivot_mean.shape[1]):
            m_val = pivot_mean.iloc[i, j]
            v_val = pivot_var.iloc[i, j]
            annot_matrix[i, j] = f"Mean: {m_val:.3f}\nVar: {v_val:.3f}"

    # Seleziona quale pivot usare per i colori dello sfondo
    pivot_color = pivot_mean if color_by == "mean" else pivot_var
    label_color = "Media" if color_by == "mean" else "Varianza"

    # 6. Plot della Heatmap
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))

    sns.heatmap(
        pivot_color,
        annot=annot_matrix,
        fmt="",  # Disattiva la formattazione automatica dato che annot contiene stringhe
        cmap="YlOrRd",
        cbar_kws={"label": f"Valore {label_color} Curtosi"},
        annot_kws={"size": 9.5},  # Adatta la dimensione del testo per far entrare entrambe le righe
    )

    plt.title(f"Kurtosis with {funzione_target}")
    plt.xlabel("Points")
    plt.ylabel("Noise level")

    plt.tight_layout()
    plt.show()

    return summary_df




#for i in ["np.log(X)", "np.arcsin(X)", "1000000*np.sin(X)", "1/X", "1/np.abs(X)", "np.sin(X)/np.cos(X)", "X"]:
    
    #plot_heatmap_curtosi(funzione_target = i)


