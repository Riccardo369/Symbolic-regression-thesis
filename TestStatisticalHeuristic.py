import numpy as np
import json
from itertools import permutations
import os
from dotenv import load_dotenv



import StatisticalHeuristics as sh
from Model import SymbolicRegression
import Operations as op

a = 10

domain = {
    "sin": (-a, a),
    "cos": (-a, a),
    "arcsin": (-1, 1),
    "arccos": (-1, 1),
    "exp": (-a, a),
    "ln": (0, 2*a),
    "div": (-a, a),
    "sqrt": (0, 2*a)
    }






acceptable_brothers = {"sin": ["cos"],
                       "cos": ["sin"],
                       "arcsin": ["arccos"],
                       "arccos": ["arcsin"]}



n = [100, 1000, 10000]



def internal_test(X_distributions, Y_tree, variable_operation_couple, noise, seed, n):
    
    def X_generation(X_distributions, generator, n):
        return {k: generator.uniform(X_distributions[k]["domain"][0], X_distributions[k]["domain"][1], n) for k in X_distributions.keys()}

    generator = np.random.default_rng(seed)
    
    #Calculate inputs values
    X = X_generation(X_distributions, generator, n)

    #Calculate result
    Y = Y_tree.get_root().forward_calculation_memory_sub_tree(X).get_memory("forward")
        
    #Remove nan points
    tot = len(Y)
    mask = np.isfinite(Y)
    diff = tot - np.sum(mask)
        
    X = {k: X[k][mask] for k in X.keys()}
    Y = Y[mask]
        
    #While the valid points are minus of minimum points required, add new random points
    while(diff > 0):
            
        #Add new X random points
        additional_X = X_generation(X_distributions, generator, n)
        additional_Y = Y_tree.get_root().forward_calculation_memory_sub_tree(additional_X).get_memory("forward")
            
        #Apply mask
        additional_mask = np.isfinite(additional_Y)
        additional_X = {k: additional_X[k][additional_mask] for k in additional_X.keys()}
        additional_Y = additional_Y[additional_mask]
            
        #Concatenate with actual and new valid points
        for k in X.keys(): X[k] = np.concatenate((X[k], additional_X[k]))
        Y = np.concatenate((Y, additional_Y))
  
        #Ricalculat diff
        diff = tot - len(Y)
        
    #Remove the residual valid points 
    X = {k: X[k][:tot] for k in X.keys()}
    Y = Y[:tot]
   
    #Add noise if requested
    if(noise): Y += generator.normal(0, noise * np.std(Y), size = Y.shape)
        
    #Results
    coefficients_results = []
        
    #For each input x
    for x in X.keys(): coefficients_results += sh.all_heuristic_statisticals_complete(X[x], Y, x)
                 
    #Sort results
    classification_coefficients = sorted([x for x in coefficients_results if np.isfinite(x["value"])], key = lambda x: -x["value"])

    couple_var_oper = [(x["variable"], x["operation"]) for x in classification_coefficients]
        
    indexes = []
        
    for x in variable_operation_couple:
            
        try:
                
            index = couple_var_oper.index(x)
            indexes.append(index)
                
        except ValueError: pass
            
            
    try: result = sorted(indexes)[0]
    except IndexError: result = None

    return result








#Per il riconoscimento di funzioni f(x) = y, c'è il coefficiente di pearson
#Per il riconoscimento di funzioni a*f(x) + b = y: c'è il coefficiente di pearson
#Per il riconoscimento di funzioni f(a*x + b) = y: Fai l' inversa, e fai poi pearson dentro, se la funzione non è suriettiva sinusoidale, altrimenti fai tecniche particolari per ricavare a e b
#Per il riconoscimento di funzioni a*f(c*x + d) + b = y, 
#Per il riconoscimento di funzioni annidate: bisogna vedere
#Per il riconoscimento di funzioni annidate con costanti: bisogna vedere
#Extract all couples (var, oper) in the tree
def extract_variable_operation_couples(tree):
    
    result = set()
    def recursion(node, variables):
        
        for c in node.get_childrens():
            variables += recursion(c, variables)
        
        if(isinstance(node, op.Leaf)):
            if(node.is_constant()): return []
            else: return [node.get_name()]
            
        else:
            for x in variables: result.add((x, node.__class__.__name__))
            return variables
        
    recursion(tree.get_root(), [])
    
    return result


def extract_valid_domain(tree, var):
    
    result = set()
    
    def recursion(node, operations):
            
        for c in node.get_childrens(): recursion(c, operations)
            
        if(isinstance(node, op.Leaf) and node.get_name() == var):
            
            father = node.get_father()
            if(father != None and father.__class__.__name__ not in ["add", "mul"]): operations.add(father.__class__.__name__)
    
    
    recursion(tree.get_root(), result)
    
    domains = [domain[x] for x in result] + [(-a, a)]
    
    min_domain = max([x[0] for x in domains])
    max_domain = min([x[1] for x in domains])
    return (min_domain, max_domain)
    
    
    
    
  

#Dentro il metodo test fai la funzione una sola volta, con tutto l' albero parsato e l' analisi con associazione variabile - operazione
#Come stringa di Y_function ci passi la stringa valida per il modello SymbolicRegression
def test(Y_string_tree, result, n, noise, seeds):
    
    print(f"Statistical heuristics: tests {len(seeds)}, expression = {Y_string_tree}, with {n} points, noise = {noise}")
    
    #Create tree
    tree = SymbolicRegression.string_to_tree(Y_string_tree)
    
    #Extract all situation that can be predicted
    variable_operation_couples = extract_variable_operation_couples(tree)
        
    #Add all situation considering that some operations can be confused with other operations (like sin with cos and viceversa)
    new_variable_operation_couple = set()
        
    for var, oper in variable_operation_couples:
            
        try:
            for brother in acceptable_brothers[oper]:
                new_variable_operation_couple.add((var, brother))
        except KeyError: pass
                
    variable_operation_couples = variable_operation_couples.union(new_variable_operation_couple)
    
    variables = set([x[0] for x in variable_operation_couples])
    
    Inputs_string = {}
    
    for var in variables:
        
        domain = extract_valid_domain(tree, var)
        Inputs_string[var] = {'domain': (domain[0], domain[1])}


    #Create label
    X_labels = [f"{k}: " + f'{str(Inputs_string[var]["domain"])}' for k in Inputs_string.keys()]
    label = ", ".join(X_labels) +  f", Y = {Y_string_tree}" + f" [n = {n}] [noise = {noise}]"
    
    
    if(label not in result.keys()): actual_tests = 0
    else: actual_tests = len(result[label])
    
    if(label not in result.keys()): result[label] = []
    
    #Repeat test
    for i in range(max(len(seeds) - actual_tests, 0)):
        
        result[label].append(internal_test(Inputs_string, tree, variable_operation_couples, noise, seeds[i], n))
            
    return result


forbidden_concatenation = [" - ".join(x) for x in [("ln", "exp"),
                                                   ("exp", "ln"),
                                                   ("sin", "arcsin"),
                                                   ("cos", "arccos")]]




def depth_test(i, result, n, noise, seeds):
    
    for c in list(permutations(domain.keys(), i)):
        
        if(" - ".join(c) in forbidden_concatenation): continue
        
        Y_string = "(".join([x for x in c]) + "(X" + (")"*i)
        
        result = test(Y_string, result, n, noise, seeds)
        
        
def fast_test(Y_string, result, n, noise, seeds):
    result = test(Y_string, result, n, noise, seeds)





result = {}


load_dotenv()
seeds_str = os.getenv("SEEDS_LIST")
seeds = [int(seed.strip()) for seed in seeds_str.split(",") if seed.strip()]

for noise in [0, 0.5, 1]:

    for n in [100, 1000, 10000]:
        
        depth_test(1, result, n, noise, seeds)
        depth_test(2, result, n, noise, seeds)
        
        fast_test("add(exp(X1), sqrt(X2))", result, n, noise, seeds)
        fast_test("add(sin(X1), exp(X2))", result, n, noise, seeds)
        fast_test("mul(X1, sin(X2))", result, n, noise, seeds)
        fast_test("sin(mul(X1, X2))", result, n, noise, seeds)
        fast_test("add(exp(X1), sqrt(X2), arcsin(X3))", result, n, noise, seeds)
        fast_test("add(sin(X1), mul(exp(X1), cos(X1)))", result, n, noise, seeds)
        fast_test("mul(X1, sin(add(X2, X3)))", result, n, noise, seeds)
        fast_test("add(sin(mul(X1, X2)), X3)", result, n, noise, seeds)
    


print("Save results...")
json.dump(result, open("TestStatisticalHeuristicResults.json", "w", encoding="utf-8"), indent=4)
print("Statistical tests done!")


