import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sympy as sp
import os

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
        "Absolute_Pearson_sin": "Absolute_Pearson sin",
        "Absolute_Pearson_cos": "Absolute_Pearson cos",
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



def plot_non_invertible_metric2():
    with open("TestNonInvertibleResults2.json", "r") as f:
        data = json.load(f)

    # Converti in DataFrame se è una lista di dizionari
    df = pd.DataFrame(data) if isinstance(data, list) else data

    if df.empty:
        raise ValueError("Nessun dato trovato nel file JSON")

    # Raggruppa per n_samples e function calcolando la media su tutti i seed
    df_grouped = (
        df.groupby(["n_samples", "function"])["Absolute_Pearson"]
        .mean()
        .reset_index()
    )

    # Impostazione grafica della figura (singolo grafico)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.lineplot(
        data=df_grouped,
        x="n_samples",
        y="Absolute_Pearson",
        hue="function",
        marker="o",
        linewidth=2,
        ax=ax,
    )

    # Configurazione assi e titoli
    ax.set_xscale("log")
    ax.set_title("Absolute_Pearson Correlation", fontsize=14, fontweight="bold")
    ax.set_xlabel("n samples", fontsize=12)
    ax.set_ylabel("Absolute_Pearson", fontsize=12)
    ax.set_ylim(-0.05, 1.05)  # Range adattato ai valori [0, 1] del nuovo dataset

    fig.suptitle(
        "Non-Invertible Metric Comparison",
        fontsize=16,
        fontweight="bold",
    )

    # Sposta la legenda all'esterno della figura sulla destra
    sns.move_legend(
        ax,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        title="Function $g(X)$",
        frameon=True,
    )

    plt.tight_layout()
    plt.show()



#plot_non_invertible_metric2()

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





informations = json.load(open("BenchmarkComparitions.json"))
dataset_informations = json.load(open("DatasetDimension.json"))


#intersection_competitors_presence = set()
#for k in informations.keys():
#    intersection_competitors_presence = intersection_competitors_presence.union(set(informations[k]["competitors"].keys()))
#    
#for k in informations.keys():
#    intersection_competitors_presence = intersection_competitors_presence.intersection(set(informations[k]["competitors"].keys()))
#
#competitors_list = list(intersection_competitors_presence) + ["AIFeynman"]




#competitors_list = ["AIFeynman", "DSR", "gplearn", "Operon", "GP-GOMEA"]
competitors_list = []





def approx(x, l):
    
    threshold =       1e+10
    str_threshold = f"1e+10"
    
    result = round(x, l)
    
    return str(result)
    
    #if(result > threshold): return f"$>{str_threshold}$"
    #elif(result < -threshold and result < 0): return f"$<-{str_threshold}$"
    #else: return str(result)



def analysis_comparation_benchmark():
    
    def add_resume_table(values, caption, label):
        
        string = r"""\footnotesize
\setlength{\tabcolsep}{2.5pt}

\binoppenalty=0
\relpenalty=0

\begin{longtable}{|p{0.08\textwidth}|p{0.28\textwidth}|p{0.28\textwidth}|c|c|c|p{0.08\textwidth}||}

    \caption{""" + caption + """} \label{tab:""" + label + r"""} \\
    \hline
    \textbf{Model} & \textbf{$R^2$} & \textbf{MAE} & \textbf{Time (s)} & \textbf{Complexity} & \textbf{Absolute Pearson}\\
    \hline
    \endhead
"""
        for c in ["GSM"] + competitors_list:
            
            R2 =         np.array(values[f"{c} R2"]        )
            MAE =        np.array(values[f"{c} MAE"]       )
            seconds =    np.array(values[f"{c} seconds"]   )
            complexity = np.array(values[f"{c} complexity"])   
            Absolute_Pearson =    np.array(values[f"{c} Absolute_Pearson"]   )
            
            mask_R2 =         np.isfinite(values[f"{c} R2"]        )
            mask_MAE =        np.isfinite(values[f"{c} MAE"]       )
            mask_seconds =    np.isfinite(values[f"{c} seconds"]   )
            mask_complexity = np.isfinite(values[f"{c} complexity"])
            mask_Absolute_Pearson =    np.isfinite(values[f"{c} Absolute_Pearson"]   )
            
            
            l = len(values[f"{c} R2"])

            string += f"""            
{c} & $\mu$ {approx(np.mean(R2[mask_R2]), 3)} $\sigma$ {approx(np.std(R2[mask_R2]), 3)} ({np.sum(mask_R2)} / {l}) &
      $\mu$ {approx(np.mean(MAE[mask_MAE]), 3)} $\sigma$ {approx(np.std(MAE[mask_MAE]), 3)} ({np.sum(mask_MAE)} / {l}) & 
      $\mu$ {approx(np.mean(seconds[mask_seconds]), 3)} $\sigma$ {approx(np.std(seconds[mask_seconds]), 3)} ({np.sum(mask_seconds)} / {l}) & 
      $\mu$ {approx(np.mean(complexity[mask_complexity]), 3)} $\sigma$ {approx(np.std(complexity[mask_complexity]), 3)} ({np.sum(mask_complexity)} / {l}) & 
      $\mu$ {approx(np.mean(Absolute_Pearson[mask_Absolute_Pearson]), 3)} $\sigma$ {approx(np.std(Absolute_Pearson[mask_Absolute_Pearson]), 3)} ({np.sum(mask_Absolute_Pearson)} / {l}) \\\\
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

\begin{longtable}{|p{0.12\textwidth}|p{0.30\textwidth}|c|c|c|c|p{0.16\textwidth}|}
    
    \caption{Benchmark comparition's models} \label{tab:benchmark_comparition_models} \\
    \hline
    \textbf{Dataset} & \textbf{Model} & \textbf{$R^2$} & \textbf{MAE} & \textbf{Time (s)} & \textbf{Complexity} & \textbf{Absolute Pearson} \\
    \hline
    \endhead
"""

    

    
    
    def create_competitors_dict():
        Dict = {}
        
        for x in ["R2", "MAE", "seconds", "complexity", "Absolute_Pearson"]:
            Dict |= {f"{c} {x}": [] for c in competitors_list + ["GSM"]}
            
        Dict["valid"] = []
        
        return Dict 
    
    competitors_PMLB =     create_competitors_dict()
    competitors_Strogatz = create_competitors_dict()
    
    
    
    

    
    for k in list(informations.keys()):
        
        #Apply filter if dataset is a PMLB dataset
        if(not k.startswith("strogatz")):
            
            dataset_inf = [x for x in dataset_informations if x["dataset"] == k][0]
            
            #if(dataset_inf["features"]                           < 5):   continue
            #if(dataset_inf["TR points"]                          < 200): continue
            
            
        
        row = informations[k]
        
        dataset = str(k).replace("_", r" ")
        
        competitors = competitors_Strogatz if k.startswith("strogatz") else competitors_PMLB
        
        if(set(competitors_list).issubset(set(row["competitors"].keys()))):
            
            for comp in competitors_list:
                
                data = row["competitors"][comp]
                
                competitors[f"{comp} R2"]        .append(data["R2 TS best"])
                competitors[f"{comp} MAE"]       .append(data["MAE TS best"])
                competitors[f"{comp} seconds"]   .append(data["best total seconds"])
                competitors[f"{comp} complexity"].append(int(data["complexity best"]))
                competitors[f"{comp} Absolute_Pearson"]   .append(data["Absolute_Pearson best"])
                

            competitors["GSM R2"]        .append(row["GSM"]["R2 TS best"])
            competitors["GSM MAE"]       .append(row["GSM"]["MAE TS best"])
            competitors["GSM seconds"]   .append(row["GSM"]["total seconds"])
            competitors["GSM complexity"].append(int(row["GSM"]["complexity best"]))
            competitors["GSM Absolute_Pearson"]   .append(row["GSM"]["Absolute_Pearson best"])
            
            competitors["valid"].append(True)
        
        else:
            competitors["valid"].append(False)
            print(f"SKIP {k} dataset")
            continue
        
        
        max_indexes = dict()
        
        for metric in ["R2", "MAE", "seconds", "complexity", "Absolute_Pearson"]:
            
            values = [competitors[f"{comp} {metric}"][-1] for comp in ["GSM"] + competitors_list]
            
            winner_value = np.min(values) if metric != "R2" else np.max(values)
            
            max_indexes[metric] = [i for i, x in enumerate(values) if approx(winner_value, 3) == approx(x, 3)]
            
        
        for i, c in enumerate(["GSM"] + competitors_list):
            
            R2_value                  = approx(competitors[f"{c} R2"][-1],         3)
            MAE_value                 = approx(competitors[f"{c} MAE"][-1],        3)
            seconds_value             = approx(competitors[f"{c} seconds"][-1],    3)
            complexity_value          = approx(competitors[f"{c} complexity"][-1], 3)
            Absolute_Pearson_value    = approx(competitors[f"{c} Absolute_Pearson"][-1],    3)
            
            b = r"\textbf{"
            
            string += f"""{dataset if c == 'GSM' else ''} & {c} & 
            {b + str(R2_value        ) + "}" if i in max_indexes["R2"]         else R2_value        } &
            {b + str(MAE_value       ) + "}" if i in max_indexes["MAE"]        else MAE_value       } &
            {b + str(seconds_value   ) + "}" if i in max_indexes["seconds"]    else seconds_value   } &
            {b + str(complexity_value) + "}" if i in max_indexes["complexity"] else complexity_value} &
            {b + str(Absolute_Pearson_value   ) + "}" if i in max_indexes["Absolute_Pearson"]    else Absolute_Pearson_value   }\\\\
"""
        
        string += f"""\\hline
"""
        
        
    string += r"""\end{longtable}
"""

    string = string.replace("\_frac", "\\frac")
    string = string.replace(" ) ", " ) \\allowbreak ")
    
    #Create the resume tables comparation GSM, min and max R^2 model, on PMLB and Stogatz datasets
    string += add_resume_table(competitors_PMLB, "PMLB resume table", "PMLB resume")
    string += add_resume_table(competitors_Strogatz, "Strogatz resume table", "Strogatz resume")
    
    with open("Comparation's model.tex", "w", encoding="utf-8") as f:
        f.write(string)


        
        
def computational_empirical_cost():
    
    dataset_information = json.load(open("DatasetDimension.json"))
    performance_informations = {d.split(".")[0]: json.load(open(f"BenchmarkCompleteResults/{d}")) for d in os.listdir("BenchmarkCompleteResults")}
    
    X = []
    Y = []
    
    for d in performance_informations.keys():
    
        #Calculate delta from 2 timestamp
        delta_timestamp = np.diff(performance_informations[d]["results"]["timestamp"])
        
        #Take all starting tree: it is necessary FROM which tree algorithm starts (for compare with computational cost), not TO
        nodes = np.array(performance_informations[d]["results"]["Nodes"][:-1])
        
        dataset = [x for x in dataset_information if x["dataset"] == d]
        
        features = np.array([dataset[0]["features"]]*len(nodes))
        points = np.array([dataset[0]["TR points"]]*len(nodes))
    
        X += (nodes * features * points).tolist()
        Y += (delta_timestamp / np.log2(points)).tolist()
        
    
    X = np.array(X)
    Y = np.array(Y)
    
    points = [(x, y) for x, y in zip(X, Y)]
    
    distinct_point = []
    
    for x_d in set(X):
        
        total_y = [y for x, y in points if x == x_d]
        distinct_point.append((x_d, np.max(total_y)))
        
        
    distinct_point = sorted(distinct_point, key = lambda p: p[0])
    
    
    i = 1
    
    while(i < len(distinct_point)):
        
        if(distinct_point[i][1] < distinct_point[i-1][1]): del distinct_point[i]
        else: i += 1
        

    x_bound = [x for x, y in distinct_point]
    y_bound = [y for x, y in distinct_point]
    
    
    plt.scatter(X, Y, label="Single iteration cost", color="black", alpha=0.35, s=4)
    plt.plot(x_bound, y_bound, label="Upper bound", color="red")
    

    plt.xlabel(r"Iteration workload $T \cdot k \cdot n$")
    plt.ylabel("Execution time per iteration (s) / log(n)")
    plt.title("Empirical Computational Cost of GSM")
    plt.legend()
    plt.show()
        

  


#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
  





Correlations = {"features": [],
                "TR points": [],
                "TS R2 best GSM" :[]}


bad = 0
good = 0
neutral_positive = 0
neutral_negative = 0
neutral = 0


for d in informations.keys():
    
    if(d.startswith("strogatz")): continue
    
    dataset_inf = [x for x in dataset_informations if x["dataset"] == d][0]
    
    GSM_R2 = informations[d]["GSM"]["R2 TS best"]
    
    Correlations["features"].append(dataset_inf["features"])
    Correlations["TR points"].append(dataset_inf["TR points"])
    Correlations["TS R2 best GSM"].append(GSM_R2)
    
    if(GSM_R2 > 0.1): good += 1
    elif(GSM_R2 < -0.1): bad += 1
    else:
        
        if(GSM_R2 > 0): neutral_positive += 1
        if(GSM_R2 < 0): neutral_negative += 1
        else: neutral = 0
    
    
    
    
  
#print("On dataset PMLB")  
#print(f"bad (R2 < -0.1): {bad}")
#print(f"neutral negative (-0.1 <= R2 < 0): {neutral_negative}")
#print(f"neutral (R2 = 0): {neutral}")
#print(f"neutral positive (0 < R2 <= 0.1): {neutral_positive}")
#print(f"good (R2 > 0.1): {good}")


    
Correlations["features"] = np.array(Correlations["features"])
Correlations["TR points"] = np.array(Correlations["TR points"])
Correlations["TS R2 best GSM"] = np.array(Correlations["TS R2 best GSM"])


    
def fast_plot(X, Y, title, X_label, Y_label):
    
    index = np.argsort(X)
    
    plt.plot(X[index], Y[index])
    
    plt.xlabel(X_label)
    plt.ylabel(Y_label)

    plt.title(title)
    plt.legend()
    plt.show()
    

#fast_plot(Correlations["features"], Correlations["TS R2 best GSM"], "(PMLB) Features vs R2", "Features", "R2")
#fast_plot(Correlations["TR points"], Correlations["TS R2 best GSM"], "(PMLB) TR points vs R2", " TR points", "R2")
#fast_plot(Correlations["features"] * Correlations["TR points"], Correlations["TS R2 best GSM"], "(PMLB) Features * TR points vs R2", "Features * TR points", "R2")

analysis_comparation_benchmark()
#computational_empirical_cost()

