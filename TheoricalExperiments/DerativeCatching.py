import numpy as np

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path: sys.path.append(parent_dir)

import Indicators as ind
import LearningStrategy as ls

def estimate_gradient(inputs_dict, output):
    
    def incremental(sp, r1, r2, i):
        delta_y = sp[r2].y - sp[r1].y
        delta_x =  sp[r2].X[i] - sp[r1].X[i]
        return delta_y / delta_x  if np.abs(delta_x) >= 1e-40 else 0
    
    points = [ls.AnalysisRamification.GradPoint([inputs_dict[k][i] for k in inputs_dict.keys()], output[i]) for i in range(len(output))]
    
    for i in range(len(inputs_dict.keys())):
        
        #Order point considering the i dimension
        sp = sorted(points, key = lambda x: x.X[i])
        
        #For each point
        for r in range(len(sp)):
            
            if(r != 0):           left =  incremental(sp, r, r-1, i)
            if(r != len(sp) - 1): right = incremental(sp, r+1, r, i)
            
            #If point is the first of dimension
            if(r == 0): sp[r].X_grad[i] = right
              
            #If point is the last of dimensione  
            elif(r == len(sp) - 1): sp[r].X_grad[i] = left
            
            #If point is in the midterm
            else: sp[r].X_grad[i] = (left + right) / 2
    
    
    return np.array([x.X_grad for x in points])

def estimate_hessian(inputs_dict, output):
    
    hessian = []
    
    for d in ls.AnalysisRamification.estimate_gradient(inputs_dict, output).T:
        
        hessian.append(ls.AnalysisRamification.estimate_gradient(inputs_dict, d))
        
    return np.array(hessian)
    


rng = np.random.default_rng()

def constant(x):
    return np.std(x) / (np.abs(np.mean(x)) + 1e-15)
    
def proportional(A, B):
    r = A / (B + 1e-15)
    return np.std(r) / np.abs(np.mean(r))

def calculate_track(inputs_dict, Y, label):
    
    print("FUNCTION: ", label)
    
    inputs = [k for k in inputs_dict.keys()]
    
    gradient = estimate_gradient(inputs_dict, Y).T
    hessian = estimate_hessian(inputs_dict, Y).T
    
    for i1, x1 in enumerate(inputs):
        
        print("variables: (", inputs[i1], ")")
        
        mask_hessian = np.isfinite(hessian[i1][i1])
        
        print("Test per abs (media e varianza devono essere 0), hessian mean:", np.mean(hessian[mask_hessian]), "hessian variance:", np.var(hessian[mask_hessian]))
        
        for i2, x2 in enumerate(inputs):
            
            print("variables: (", inputs[i1], inputs[i2], ")")
            
            mask_hessian = np.isfinite(hessian[i1][i2])
            print("hessian mean:", np.mean(hessian[mask_hessian]), "hessian variance:", np.var(hessian[mask_hessian]))
            
            print("Test mul (più è basso meglio è per mul): ", constant(gradient[i1] / inputs_dict[x2]))
            
            print(f"Test div {inputs[i1]} / {inputs[i2]} per {inputs[i1]}: ", proportional(gradient[i1], 1 / (inputs_dict[x1] + 1e-15)))
            print(f"Test div {inputs[i1]} / {inputs[i2]} per {inputs[i2]}: ", proportional(gradient[i2], -inputs_dict[x1] / (inputs_dict[x2]**2 + 1e-15)))
            
            with np.errstate(divide='ignore', invalid='ignore'):  
                valid_ln = np.log(gradient[i1])
            
            mask_ln = np.isfinite(valid_ln)
            
            pow_index = ind.pearson_correlation(valid_ln[mask_ln], inputs_dict[x2][mask_ln]) if np.sum(mask_ln) > 0 else 0
                
            print(f"Test pow {inputs[i1]} ** {inputs[i2]} per {inputs[i1]}: ", pow_index)
            
       
       #MANCA: sin, cos, tan,      

n = 10000

X = [rng.permutation(rng.standard_normal(n)),
     rng.permutation(rng.standard_normal(n))]

test_functions = [#"3",
                  #"X1",
                  #"3*X1 + 4",
                  #"X1**2",
                  #"X1**2 + 2*X1 + 5",
                  #"2*X1**3 + X1**2 + 2*X1 + 5",
                  #"8*X1**4 + 2*X1**3 + X1**2 + 2*X1 + 5",
                  #"np.log(X1)",
                  #"4**X1",
                  #"np.cos(X1)",
                  #"np.sin(X1)",
                  #"np.tan(X1)",
                  #"rng.permutation(rng.standard_normal(len(X1)))",
                  #
                  #"np.sin(X1**2)",
                  #"2**np.cos(X1)",
                  #"X1**2 + np.cos(X1)",
                  #"X1**2 * np.cos(X1)",
                  #"X1**2 * np.cos(X1**2)",
                  #"2**X1 + np.cos(X1)"
                  #"X[0] + X[1]",
                  #"X[0] - X[1]",
                  "X[0] * X[1]",
                  #"X[0] / X[1]"
                  ]

for test in test_functions:
    
    Y = eval(test)
  
    noise = rng.normal(loc = 0, scale = 1, size=len(Y))
        
    label = test
        
    calculate_track({f"X{i}": X[i] for i in range(len(X))}, Y, label)
    
    
    
    #calculate_track({f"X{i}": X[i] for i in range(len(X))}, Y + noise, label + " (with gaussian noise N(0, 1))")
        
    #calculate_indicators(x, Y,         label)
    #calculate_indicators(x, Y + noise, label + " (with gaussian noise N(0, 1))")
    
    