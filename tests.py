import numpy as np
import matplotlib.pyplot as plt

import LearningStrategy as ls
from Model import SymbolicRegression
import Operations as op
import Errors as er

import Indicators as ind

def chaterjee_metric(Y_output, Y_target):
    
    if(np.isscalar(Y_output) and np.isscalar(Y_target)): return -1
    
    if(np.isscalar(Y_output)): Y_output = np.array([Y_output for _ in range(len(Y_target))])
    if(np.isscalar(Y_target)): Y_target = np.array([Y_target for _ in range(len(Y_output))])
    
    return -ind.chatterjee_correlation(Y_output, Y_target)


def get_package_importance_nodes(nodes, unary_operations, TR_inputs_dict, TR_output):
        
    combinations = []
        
    for n in nodes:
        
        if(isinstance(n, op.Leaf)): combinations.append({"old node": n, "new node": op.Leaf(np.mean(n.get_memory("backward")))})
            
        for f in TR_inputs_dict.keys():
                
            if(isinstance(n, op.Leaf)): combinations.append({"old node": n, "new node": op.Leaf(f)})
                
            for o in unary_operations:
                    
                oper = o()
                
                if(not oper.check_input_domain(0, TR_inputs_dict[f] if isinstance(n, op.Leaf) else n.get_memory("forward"))): continue
                if(n.get_father() != None and not oper.check_output_domain(n.get_memory("backward") if n.get_father() != None else TR_output)): continue
                                
                oper.new_connection(op.Leaf(f) if(isinstance(n, op.Leaf)) else n.copy_sub_tree())
                            
                combinations.append({"old node": n, "new node": oper})

    return [combinations]

show_plot = True

test_functions = ["3",
                  "X",
                  "np.cos(X)",
                  "3 * np.cos(X) + 5",
                  "X**2 + 2*X",
                  "2*(X**2) + 5*X + 6",
                  "X**2 + np.cos(X)",
                  "X**2 * np.cos(X)",
                  "X**2 * np.cos(X**2)",
                  "2**X + np.cos(X)"
                  ]

X = np.array([x for x in range(1, 100000, 1)], dtype = np.float64) / 100

for test in test_functions:
    
    model = SymbolicRegression.string_to_tree("0")
    
    Y = eval(f"lambda X: {test}")(X)
    
    metric_results = model.learn({"X": X}, Y,
                            ls.HeuristicLearning([op.neg, op.div, op.sin, op.cos, op.abs, op.exp, op.ln],
                                                    lambda x: (np.isfinite(x["Chaterjee"]) and x["Chaterjee"] > 0) or (np.isfinite(x["MAE"]) and x["MAE"] > 0),
                                                    lambda x: (-x["Chaterjee"], -x["MAE"]),
                                                    get_package_importance_nodes,
                                                    {"MAE": lambda model: er.MAE(model.get_root().get_memory("forward"), Y),
                                                     "Chaterjee": lambda model: chaterjee_metric(model.get_root().get_memory("forward"), Y)}),
                            
                            
                            metric_functions = {"tree": lambda model: str(model.get_root()),
                                                #"MAE error": lambda model: er.MAE(model.get_root().get_memory("forward"), Y),
                                                #"Chaterjee": lambda model: ind.chatterjee_correlation(model.get_root().get_memory("forward"), Y)
                                                }
                            )
    
    print("*"*100)
    print("Ideal:   ", test)
    print("inferred:", metric_results["results"]["tree"][-1])
    print("*"*100)
    
    if(show_plot):
        
        inferred_Y = model.get_calculation_function()({"X": X})
        
        if(type(Y) in [int, float]): Y = np.array([Y] * len(X))
        if(type(inferred_Y) in [int, float]): inferred_Y = np.array([inferred_Y] * len(X))
        
        fig = plt.figure(facecolor='black')
        ax = plt.axes()

        ax.set_facecolor('black')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        plt.plot(X, Y, color = "red", label = f"ideal: {test}")
        plt.plot(X, inferred_Y, color = "blue", label = f"inferred: {model.get_root().get_string_calculation_function()['body']}")

        plt.legend()
        plt.show(block = False)
    
if(show_plot): plt.show(block = True)
