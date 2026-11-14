import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sympy as sp



import NonInvertibleFunctions as nif
import Operations as op
from Model import SymbolicRegression



#NON INVERTIBLE FUNCTIONS SHOW
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







def plot_non_invertible_metric(noise):
    data = json.load(open("TestNonInvertibleResults.json", "r"))

    # Converti in DataFrame se è una lista di dizionari
    df = pd.DataFrame(data) if isinstance(data, list) else data

    df_filtered = df[df["noise"] == noise]

    if df_filtered.empty:
        raise ValueError(f"Nessun dato trovato per noise = {noise}")

    # Definizione delle metriche da tracciare e dei relativi titoli
    metrics_map = {
        "pearson_sin": "Pearson sin",
        "pearson_cos": "Pearson cos",
    }

    # Impostazione grafica della figura (1 riga, 2 colonne)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Disegna i due subplot
    for i, (metric_label, title) in enumerate(metrics_map.items()):
        ax = axes[i]

        df_grouped = (
            df_filtered.groupby(["n_samples", "function"])[metric_label]
            .mean()
            .reset_index()
        )

        # Mostriamo la legenda SOLO sul secondo subplot (i == 1)
        show_legend = i == 1

        sns.lineplot(
            data=df_grouped,
            x="n_samples",
            y=metric_label,
            hue="function",
            marker="o",
            linewidth=2,
            ax=ax,
            legend=show_legend,  # Genera la legenda nativa di Seaborn solo nel secondo grafico
        )

        # Configurazione assi e titoli
        ax.set_xscale("log")
        ax.set_title(
            f"{title}", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("n samples", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_ylim(0.95, 1.005)
        
    fig.suptitle(
        f"Non-Invertible Metric Comparition (Noise: {noise})",
        fontsize=16,
        fontweight="bold",
        #y=1,  # Regola l'altezza verticale per non sovrapporsi ai grafici
    )

    # Sposta la legenda generata sul secondo grafico all'esterno della figura sulla destra
    sns.move_legend(
        axes[1],
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        title="Function $g(X)$",
        frameon=True,
    )

    plt.tight_layout()
    plt.show()


    
#plot_non_invertible_metric(False)
#plot_non_invertible_metric(True)



def distribution_difference_non_invertible(g):
    
    data = [x for x in json.load(open("TestNonInvertibleResults.json", "r")) if x["function"] == g and x["noise"]]
    
    clean_data_sin = []
    noise_data_sin = []
    
    clean_data_cos = []
    noise_data_cos = []
    
    for x in data:
        
        generator = np.random.default_rng(x["seed"])
        
        n = x["n_samples"]
        
        X = generator.uniform(low = -10, high = 10, size = n)
        noise = generator.uniform(low = -1, high = 1, size = n)
        
        g_function = eval(g)
        
        sin_g = np.sin(g_function)
        cos_g = np.cos(g_function)
        
        clean_data_sin += sin_g.tolist()
        clean_data_cos += cos_g.tolist()
        
        noise_data_sin += (sin_g * noise).tolist()
        noise_data_cos += (cos_g * noise).tolist()
    
    clean_data_sin = np.array(clean_data_sin).flatten()
    noise_data_sin = np.array(noise_data_sin).flatten()
    clean_data_cos = np.array(clean_data_cos).flatten()
    noise_data_cos = np.array(noise_data_cos).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    sns.set_theme(style="whitegrid")

    # histplot con kde=True genera sia le colonne sia la linea morbida di densità
    sns.histplot(clean_data_sin, bins=100, stat="density", kde=True, color="blue", element="step", label = "Clean output", ax = axes[0])
    sns.histplot(noise_data_sin, bins=100, stat="density", kde=True, color="red", element="step", label = "Noise output", ax = axes[0])
    
    sns.histplot(clean_data_cos, bins=100, stat="density", kde=True, color="blue", element="step", label = "Clean output", ax = axes[1])
    sns.histplot(noise_data_cos, bins=100, stat="density", kde=True, color="red", element="step", label = "Noise output", ax = axes[1])
    
    axes[0].set_xlim(-1, 1)
    axes[0].set_title(r"$\sin(g(X))$ distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel(r"Values of $\sin(g(X))$", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].legend(title="Data Type")
    
    axes[1].set_xlim(-1, 1)
    axes[1].set_title(r"$\cos(g(X))$ distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel(r"Values of $\cos(g(X))$", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].legend(title="Data Type")
    
    # Fissa il range dell'asse X esattamente tra -1 e 1
    fig.suptitle(f"Codomain Distribution Shift for $g(X) = {g}$", fontsize=16, fontweight="bold")
    
    plt.tight_layout()
    plt.show()
    
    
    
#for i in ["X",
#          "(2*X) + (np.pi/4)",
#          "np.exp(0.5*X)",
#          "(X**3) - (3*X)",
#          "1/X",
#          "np.abs(X)",
#          "X + (0.5*np.sin(3*X))"]:
#
#    distribution_difference_non_invertible(i)




#All tools for show results about training on benchmark datasets



informations = json.load(open("BenchmarkComparitions.json"))



def convert_floats_to_fractions(expr_str, tolerance=1e-5):

    expr_parsed = sp.sympify(expr_str)
    frac = sp.nsimplify(expr_parsed, rational=True, tolerance=tolerance)
    
    return "\\frac{" + str(frac.p) + "}{" + str(frac.q) + "}" if frac.q != 1 else str(frac.p)


def overleaf_string(node):
    
    if(isinstance(node, op.Leaf)): return str(node if not node.is_constant() else convert_floats_to_fractions(node.get_name()))
    else:                          return dict_overleaf_string[str(node.__class__.__name__)](node.get_childrens())



dict_overleaf_string = {
    "add": lambda x: " + ".join(overleaf_string(i) for i in x),
    "mul": lambda x: " ".join(overleaf_string(i) for i in x),
    "ln": lambda x: "\\ln( " + overleaf_string(x[0]) + " )",
    "exp": lambda x: "\\exp( " + overleaf_string(x[0]) + " )",
    "div": lambda x: "\\frac{1}{" + overleaf_string(x[0]) + "}",
    "sqrt": lambda x: "\\sqrt{" + overleaf_string(x[0]) + "}",
    "sin": lambda x: "\\sin( " + overleaf_string(x[0]) + " )",
    "cos": lambda x: "\\cos( " + overleaf_string(x[0]) + " )",
    "arcsin": lambda x: "\\arcsin( " + overleaf_string(x[0]) + " )",
    "arccos": lambda x: "\\arccos( " + overleaf_string(x[0]) + " )"
    }


def convert_in_overleaf_standard(string):
    return overleaf_string(SymbolicRegression.string_to_tree(string).get_root())


def approx(x, level):
    try:
        shift = 10**level
        return int(x*shift)/shift
    except: return x


def creation_performance_table():
    
    #Create the extended table comparation Last - Best
    
    string = r"""\footnotesize
\setlength{\tabcolsep}{2.5pt}

\binoppenalty=0
\relpenalty=0

\begin{longtable}{|p{0.12\textwidth}|p{0.10\textwidth}|c|c|c|p{0.35\textwidth}|c|c|c|}
    
    \caption{Performance's model} \label{tab:performances_models} \\
    \hline
    \textbf{Dataset} & \textbf{Ground-truth} & \textbf{Iter} & \textbf{Time (s)} & \textbf{Type} & \textbf{Model} & \textbf{$R^2$} & \textbf{MAE} & \textbf{Complexity} \\
    \hline
    \endhead
"""

    values_PMLB = {}
    
    for x in ["R2", "MAE", "complexity"]:
        values_PMLB |= {f"best {x}": [], f"last {x}": []}

    values_strogatz = values_PMLB.copy()
    
    
    def create_dict():
        Dict = {}
        
        for x in ["R2", "MAE", "complexity"]:
            Dict |= {f"best {x}": [], f"last {x}": []}
        
        return Dict
    
    
            
    values_PMLB = create_dict()
    values_strogatz = create_dict()
    

    for k in list(informations.keys()):
        row = informations[k]["GSM"]

        k_clean = str(k).replace("_", r" ")
        gt_clean = str(informations[k]["ground truth"]).replace("_", r"\_")
        iter_val = row["total iterations"]
        time_val = approx(row["total seconds"], 4)

        last_m = convert_in_overleaf_standard(row["last model"])
        r2_last = approx(row["R2 TS last"], 3)
        mae_last = approx(row["MAE TS last"], 4)
        c_last = row["complexity last"]

        best_m = convert_in_overleaf_standard(row["best model"])
        r2_best = approx(row["R2 TS best"], 3)
        mae_best = approx(row["MAE TS best"], 4)
        c_best = row["complexity best"]
        
        values = values_strogatz if k.startswith("strogatz") else values_PMLB
        
        values["last R2"]        .append(r2_last)
        values["last MAE"]       .append(mae_last)
        values["last complexity"].append(int(c_last))
        values["best R2"]        .append(r2_best)
        values["best MAE"]       .append(mae_best)
        values["best complexity"].append(int(c_best))
        

        string += f""" {k_clean} & {gt_clean} & {iter_val} & {time_val} & Last & ${last_m}$ & {r2_last} & {mae_last} & {c_last} \\\\\n"""
        string += f"""           &            &            &            & Best & ${best_m}$ & {r2_best} & {mae_best} & {c_best} \\\\\n"""
        string += r"    \hline" + "\n"

    string += r"""\end{longtable}"""
    
    string = string.replace("\_frac", "\\frac")
    string = string.replace(" ) ", " ) \\allowbreak ")
    
    R2_delta_PMLB =             np.array(values_PMLB["last R2"]        )     - np.array(values_PMLB["best R2"]        )
    MAE_delta_PMLB =            np.array(values_PMLB["last MAE"]       )     - np.array(values_PMLB["best MAE"]       )
    Complexity_delta_PMLB =     np.array(values_PMLB["last complexity"])     - np.array(values_PMLB["best complexity"])
    
    R2_delta_Strogatz =         np.array(values_strogatz["last R2"]        ) - np.array(values_strogatz["best R2"]        )
    MAE_delta_Strogatz =        np.array(values_strogatz["last MAE"]       ) - np.array(values_strogatz["best MAE"]       )
    Complexity_delta_Strogatz = np.array(values_strogatz["last complexity"]) - np.array(values_strogatz["best complexity"])
    

    mask_R2_delta_PMLB =             np.isfinite(R2_delta_PMLB            )
    mask_MAE_delta_PMLB =            np.isfinite(MAE_delta_PMLB           )
    mask_Complexity_delta_PMLB =     np.isfinite(Complexity_delta_PMLB    )
    mask_R2_delta_Strogatz =         np.isfinite(R2_delta_Strogatz        )
    mask_MAE_delta_Strogatz =        np.isfinite(MAE_delta_Strogatz       )
    mask_Complexity_delta_Strogatz = np.isfinite(Complexity_delta_Strogatz)
    
    
    #Create the resume table comparation delta: Last - Best
        
        
    string += r"""\footnotesize
\setlength{\tabcolsep}{2.5pt}

\binoppenalty=0
\relpenalty=0

\begin{longtable}{|p{0.12\textwidth}|p{0.12\textwidth}|c|c|c|c|c|}

    \caption{Internal performance resume Delta: Last - Best} \label{tab:internal_performance_resume} \\
    \hline
    \textbf{Dataset} & \textbf{Metric} & \textbf{$R^2$} & \textbf{MAE} & \textbf{Complexity} \\
    \hline
    \endhead
"""

    string += f"""
    PMLB     & mean & {approx(np.mean(R2_delta_PMLB[mask_R2_delta_PMLB]), 4)} (with {np.sum(mask_R2_delta_PMLB)} / {len(mask_R2_delta_PMLB)} values) & {approx(np.mean(MAE_delta_PMLB[mask_MAE_delta_PMLB]), 4)} (with {np.sum(mask_MAE_delta_PMLB)} / {len(mask_MAE_delta_PMLB)} values) & {approx(np.mean(Complexity_delta_PMLB[mask_Complexity_delta_PMLB]), 4)}  \\\\
             & std  & {approx(np.std (R2_delta_PMLB[mask_R2_delta_PMLB]), 4)} (with {np.sum(mask_R2_delta_PMLB)} / {len(mask_R2_delta_PMLB)} values) & {approx(np.std (MAE_delta_PMLB[mask_MAE_delta_PMLB]), 4)} (with {np.sum(mask_MAE_delta_PMLB)} / {len(mask_MAE_delta_PMLB)} values) & {approx(np.std (Complexity_delta_PMLB[mask_Complexity_delta_PMLB]), 4)}  \\\\
             & min  & {approx(np.min (R2_delta_PMLB[mask_R2_delta_PMLB]), 4)} (with {np.sum(mask_R2_delta_PMLB)} / {len(mask_R2_delta_PMLB)} values) & {approx(np.min (MAE_delta_PMLB[mask_MAE_delta_PMLB]), 4)} (with {np.sum(mask_MAE_delta_PMLB)} / {len(mask_MAE_delta_PMLB)} values) & {approx(np.min (Complexity_delta_PMLB[mask_Complexity_delta_PMLB]), 4)}  \\\\
             & max  & {approx(np.max (R2_delta_PMLB[mask_R2_delta_PMLB]), 4)} (with {np.sum(mask_R2_delta_PMLB)} / {len(mask_R2_delta_PMLB)} values) & {approx(np.max (MAE_delta_PMLB[mask_MAE_delta_PMLB]), 4)} (with {np.sum(mask_MAE_delta_PMLB)} / {len(mask_MAE_delta_PMLB)} values) & {approx(np.max (Complexity_delta_PMLB[mask_Complexity_delta_PMLB]), 4)}  \\\\
    \\hline
    Strogatz & mean & {approx(np.mean(R2_delta_Strogatz[mask_R2_delta_Strogatz]), 4)} (with {np.sum(mask_R2_delta_Strogatz)} / {len(mask_R2_delta_Strogatz)} values) & {approx(np.mean(MAE_delta_Strogatz[mask_MAE_delta_Strogatz]), 4)} (with {np.sum(mask_MAE_delta_Strogatz)} / {len(mask_MAE_delta_Strogatz)} values) & {approx(np.mean(Complexity_delta_Strogatz[mask_Complexity_delta_Strogatz]), 4)}  \\\\
             & std  & {approx(np.std (R2_delta_Strogatz[mask_R2_delta_Strogatz]), 4)} (with {np.sum(mask_R2_delta_Strogatz)} / {len(mask_R2_delta_Strogatz)} values) & {approx(np.std (MAE_delta_Strogatz[mask_MAE_delta_Strogatz]), 4)} (with {np.sum(mask_MAE_delta_Strogatz)} / {len(mask_MAE_delta_Strogatz)} values) & {approx(np.std (Complexity_delta_Strogatz[mask_Complexity_delta_Strogatz]), 4)}  \\\\
             & min  & {approx(np.min (R2_delta_Strogatz[mask_R2_delta_Strogatz]), 4)} (with {np.sum(mask_R2_delta_Strogatz)} / {len(mask_R2_delta_Strogatz)} values) & {approx(np.min (MAE_delta_Strogatz[mask_MAE_delta_Strogatz]), 4)} (with {np.sum(mask_MAE_delta_Strogatz)} / {len(mask_MAE_delta_Strogatz)} values) & {approx(np.min (Complexity_delta_Strogatz[mask_Complexity_delta_Strogatz]), 4)}  \\\\
             & max  & {approx(np.max (R2_delta_Strogatz[mask_R2_delta_Strogatz]), 4)} (with {np.sum(mask_R2_delta_Strogatz)} / {len(mask_R2_delta_Strogatz)} values) & {approx(np.max (MAE_delta_Strogatz[mask_MAE_delta_Strogatz]), 4)} (with {np.sum(mask_MAE_delta_Strogatz)} / {len(mask_MAE_delta_Strogatz)} values) & {approx(np.max (Complexity_delta_Strogatz[mask_Complexity_delta_Strogatz]), 4)}  \\\\
    \\hline
"""

    string += r"""\end{longtable}
"""

    with open("Performance's model.tex", "w", encoding="utf-8") as f:
        f.write(string)



def creation_comparation_table():
    
    def add_resume_table(values, caption, label):
        
        string = r"""\footnotesize
\setlength{\tabcolsep}{2.5pt}

\binoppenalty=0
\relpenalty=0

\begin{longtable}{|p{0.12\textwidth}|p{0.12\textwidth}|c|c|c|c|c|}

    \caption{""" + caption + """} \label{tab:""" + label + r"""} \\
    \hline
    \textbf{Model} & \textbf{Metric} & \textbf{$R^2$} & \textbf{MAE} & \textbf{Time (s)} & \textbf{Complexity} & \textbf{GSM $R^2$ rank \%} \\
    \hline
    \endhead
"""

        string += f"""
    & mean & {np.mean(values["GSM R2"])} & {np.mean(values["GSM MAE"])} & {np.mean(values["GSM seconds"])} & {np.mean(values["GSM complexity"])} & {np.mean(values["rank"])}    \\\\
GSM & std  & {np.std (values["GSM R2"])} & {np.std (values["GSM MAE"])} & {np.std (values["GSM seconds"])} & {np.std (values["GSM complexity"])} & {np.std (values["rank"])}    \\\\
    & min  & {np.min (values["GSM R2"])} & {np.min (values["GSM MAE"])} & {np.min (values["GSM seconds"])} & {np.min (values["GSM complexity"])} & {np.min (values["rank"])}    \\\\
    & max  & {np.max (values["GSM R2"])} & {np.max (values["GSM MAE"])} & {np.max (values["GSM seconds"])} & {np.max (values["GSM complexity"])} & {np.max (values["rank"])}    \\\\
    &      &                             &                              &                                  &                                     & Total wins: {values["wins"]} \\\\                             
\\hline

                     & mean & {np.mean(values["min R2"])} & {np.mean(values["min MAE"])} & {np.mean(values["min seconds"])} & {np.mean(values["min complexity"])} & - \\\\
Min $R^2$ Competitor & std  & {np.std (values["min R2"])} & {np.std (values["min MAE"])} & {np.std (values["min seconds"])} & {np.std (values["min complexity"])} & - \\\\
                     & min  & {np.min (values["min R2"])} & {np.min (values["min MAE"])} & {np.min (values["min seconds"])} & {np.min (values["min complexity"])} & - \\\\
                     & max  & {np.max (values["min R2"])} & {np.max (values["min MAE"])} & {np.max (values["min seconds"])} & {np.max (values["min complexity"])} & - \\\\
\\hline
                
                     & mean & {np.mean(values["max R2"])} & {np.mean(values["max MAE"])} & {np.mean(values["max seconds"])} & {np.mean(values["max complexity"])} & - \\\\
Max $R^2$ Competitor & std  & {np.std (values["max R2"])} & {np.std (values["max MAE"])} & {np.std (values["max seconds"])} & {np.std (values["max complexity"])} & - \\\\
                     & min  & {np.min (values["max R2"])} & {np.min (values["max MAE"])} & {np.min (values["max seconds"])} & {np.min (values["max complexity"])} & - \\\\
                     & max  & {np.max (values["max R2"])} & {np.max (values["max MAE"])} & {np.max (values["max seconds"])} & {np.max (values["max complexity"])} & - \\\\
\\hline
"""

        string += r"""\end{longtable}
        """
        
        return string
        
      
        
    #Create the extended table comparation GSM, min and max R^2 model   
    
    string = r"""\footnotesize
\setlength{\tabcolsep}{2.5pt}

\binoppenalty=0
\relpenalty=0

\begin{longtable}{|p{0.12\textwidth}|p{0.30\textwidth}|c|c|c|c|c|p{0.10\textwidth}|}
    
    \caption{Benchmark comparition's models} \label{tab:benchmark_comparition_models} \\
    \hline
    \textbf{Dataset} & \textbf{Best model} & \textbf{$R^2$} & \textbf{MAE} & \textbf{Time (s)} & \textbf{Complexity} & \textbf{GSM $R^2$ rank} \\
    \hline
    \endhead
"""

    def create_dict():
        Dict = {}
        
        for x in ["R2", "MAE", "seconds", "complexity"]:
            Dict |= {f"GSM {x}": [], f"min {x}": [], f"max {x}": []}
        
        Dict["rank"] = []
        Dict["wins"] = 0
        
        return Dict
        
    values_PMLB = create_dict()
    values_strogatz = create_dict()

    for k in list(informations.keys()):
        
        row = informations[k]
        
        # From minor to major
        sort_best_competitors = sorted([x for x in row["competitors"].keys()], key = lambda d: row["competitors"][d]["R2 TS best"])
        sort_worst_competitors = sorted([x for x in row["competitors"].keys()], key = lambda d: row["competitors"][d]["R2 TS worst"])
        
        best_rank = [(x, row["competitors"][x]["R2 TS best"]) for x in row["competitors"].keys()] + [("GSM", row["GSM"]["R2 TS best"])]
        best_rank = sorted(best_rank, key = lambda k: -k[1])
        
        total_competitors = len(best_rank)
        # +1 per convertire da indice 0-based a posizione 1-based (es. 1° posto)
        gsm_rank_pos = [x[0] for x in best_rank].index("GSM") + 1
        
        min_R2_model = sort_best_competitors[0]
        max_R2_model = sort_best_competitors[-1]
        
        dataset = str(k).replace("_", r" ")
        
        # Nomi dei modelli formattati per LaTeX (escape degli underscore)
        min_model_name = min_R2_model.replace("_", r" ")
        max_model_name = max_R2_model.replace("_", r" ")
        
        # Stringa del rank centrato (inserita solo nella riga centrale)
        rank_str = f"{gsm_rank_pos} / {total_competitors}"
        
        GSM_R2 = approx(row["GSM"]["R2 TS best"], 3)
        GSM_MAE = approx(row["GSM"]["MAE TS best"], 4)
        GSM_seconds = approx(row["GSM"]["total seconds"], 4)
        GSM_complexity = int(row["GSM"]["complexity best"])
        
        min_R2 = approx(row["competitors"][min_R2_model]["R2 TS best"], 3)
        min_MAE = approx(row["competitors"][min_R2_model]["MAE TS best"], 4)
        min_seconds = approx(row["competitors"][min_R2_model]["best total seconds"], 4)
        min_complexity = int(row["competitors"][min_R2_model]["complexity best"])
        
        max_R2 = approx(row["competitors"]     [max_R2_model]["R2 TS best"], 3)
        max_MAE = approx(row["competitors"]    [max_R2_model]["MAE TS best"], 4)
        max_seconds = approx(row["competitors"][max_R2_model]["best total seconds"], 4)
        max_complexity = int(row["competitors"]    [max_R2_model]["complexity best"])
        
        
        values = values_strogatz if k.startswith("strogatz") else values_PMLB
        
        
        values["GSM R2"]        .append(GSM_R2)
        values["GSM MAE"]       .append(GSM_MAE)
        values["GSM seconds"]   .append(GSM_seconds)
        values["GSM complexity"].append(GSM_complexity)
        values["min R2"]        .append(min_R2)
        values["min MAE"]       .append(min_MAE)
        values["min seconds"]   .append(min_seconds)
        values["min complexity"].append(min_complexity)
        values["max R2"]        .append(max_R2)
        values["max MAE"]       .append(max_MAE)
        values["max seconds"]   .append(max_seconds)
        values["max complexity"].append(max_complexity)
        values["rank"].append(approx((1 - ((gsm_rank_pos-1) / total_competitors)) * 100, 4))
        
        if(gsm_rank_pos == 1): values["wins"] += 1
        
        
        string += f"""          & GSM &                                  ${GSM_R2}$ & ${GSM_MAE}$ & ${GSM_seconds}$ & ${GSM_complexity}$ &            \\\\\n"""
        string += f"""{dataset} & ${min_model_name}$ (min $R^2$ model) & ${min_R2}$ & ${min_MAE}$ & ${min_seconds}$ & ${min_complexity}$ & {rank_str} \\\\\n"""
        string += f"""          & ${max_model_name}$ (max $R^2$ model) & ${max_R2}$ & ${max_MAE}$ & ${max_seconds}$ & ${max_complexity}$ &            \\\\\n"""
        string += r"    \hline" + "\n"
        
    string += r"""\end{longtable}
"""

    string = string.replace("\_frac", "\\frac")
    string = string.replace(" ) ", " ) \\allowbreak ")
    
    #Create the resume tables comparation GSM, min and max R^2 model, on PMLB and Stogatz datasets
    string += add_resume_table(values_PMLB, "PMLB resume table", "PMLB resume")
    string += add_resume_table(values_strogatz, "Strogatz resume table", "Strogatz resume")

    
    with open("Comparation's model.tex", "w", encoding="utf-8") as f:
        f.write(string)
        
        

creation_performance_table()
creation_comparation_table()

