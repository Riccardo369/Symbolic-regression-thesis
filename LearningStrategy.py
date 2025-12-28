import numpy as np

import matplotlib.pyplot as plt

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def projection(ideal, features, L):
        const = np.mean(ideal)
        proposals = [{"label": x, "value": features[x]} for x in features.keys()] + [{"label": const, "value": np.repeat(const, ideal.shape[0])}]
        return sorted(proposals, key = lambda x: L(ideal, x["value"]))[0]["label"]


class SymbolicRegressionLearning:
    
    def __init__(self):
        
        self._model = None
        self._solution = None
        self._leafs = None
        self._metric_results = None  
        self._metric_functions = None
    
    def __pre_learning(self, model, metrics):
        self._model = model
        self._leafs = model.get_leafs()
        self._solution = [x.get_name() for x in self._leafs]
        self._metric_functions = metrics
        self._metric_results = {key: [] for key in metrics.keys()}
        
    def __post_learning(self):
        self._model.get_root().clear_all_memory_sub_tree()
        self._model.fix_leafs(self._solution)
        
    def checkpoint_metrics(self):
        for key in self._metric_results: self._metric_results[key].append(self._metric_functions[key](self._model))
        
    def body_learning(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, stop_criteria):
        return NotImplemented
        
    def __call__(self, model, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, stop_criteria = lambda model: False, metric_functions = {}):
        self.__pre_learning(model, metric_functions)
        self.body_learning(TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, stop_criteria)
        self.__post_learning()
        
        return self._metric_results.copy()
        
        
        
class ExpectationMaximization(SymbolicRegressionLearning):
    
    def __init__(self):
        super().__init__()       
        
    def body_learning(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, stop_criteria):
        
        prec_solution = None
        
        self.checkpoint_metrics()
        
        while(prec_solution != self._solution or stop_criteria(self._model)):
                
            self._model.fix_leafs(self._solution)
            
            self._model.forward_calculation(TR_inputs_dict).backward_calculation(TR_output)
                
            prec_solution = self._solution
                
            self._solution = [projection(leaf.get_backward_calculation_memory(), TR_inputs_dict, lambda a, b: -cosine(a, b)) for leaf in self._leafs]

            self.checkpoint_metrics()
        
        
   
class SingleExpectationMaximization(SymbolicRegressionLearning):
    
    def __init__(self):
        super().__init__()       
        
    def body_learning(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, stop_criteria):
        
        prec_solution = None
        self.checkpoint_metrics()
        
        r = 0
        
        while(prec_solution != self._solution or stop_criteria(self._model)):
                
            self._model.fix_leafs(self._solution)
            
            self._model.forward_calculation(TR_inputs_dict).backward_calculation(TR_output)
                
            prec_solution = self._solution
            
            self._solution = [projection(leaf.get_backward_calculation_memory(), TR_inputs_dict, lambda a, b: -cosine(a, b)) if i == r else leaf.get_name() for i, leaf in enumerate(self._leafs)]

            r = (r + 1) % len(self._leafs)
            
            self.checkpoint_metrics()
       
   
          
       
