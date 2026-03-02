import numpy as np
import matplotlib.pyplot as plt

from Model import SymbolicRegression
import Operations as op
import Errors as er

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def projection(ideal, features, L):
        const = np.mean(ideal)
        proposals = [{"label": x, "value": features[x]} for x in features.keys()] + [{"label": const, "value": np.repeat(const, ideal.shape[0])}]
        return sorted(proposals, key = lambda x: L(ideal, x["value"]))[0]["label"]

def durbin_watson(residuals):
    numerator = np.sum(np.diff(residuals)**2)
    denominator = np.sum(residuals**2)
    
    if denominator < 1e-15: return 2.0
    return numerator / denominator


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
        self._model.get_root().clear_all_calculation_memory_sub_tree()
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
 
 
 
 
 
    

class IndipendentRamification(SymbolicRegressionLearning):
    
    __families = [{"family": "linear", "operations": [op.add, op.sub, op.mul, op.div], "transformation": lambda x: x, "residual": lambda x, a, b: (a * x) + b}]
    
    def __init__(self, operations):
        super().__init__()
        
        for oper in operations:
            if(not issubclass(oper, op.OperationNode)): raise ValueError(f"The operation {oper} is not a OperationNode type")
        
        self._operations = list(operations)
        
        
    #Ho il mio TR_output / residuo:
    #
    #1) applico ad ogni feature delle trasformazioni (ogni trasformazione per ogni famiglia)
    #2) da ogni trasformazione - feature, ricavo poi il nuovo residuo
    #3) scelta la famiglia e la feature, scelgo la seconda feature / costante e l' operazione, che migliorano sempre la trasparenza del residuo
    #4) Scelta l' operazione con le feature, si fa lo split calcolando il peso interno e lo split del residuo / output atteso
    #
    # Sul punto 2 e 3, quando scelgo la famiglia / feature e l' operazione, scelgo sempre l' opzione che rende il residuo successivo più trasparente
    #
    def extend(self, TR_inputs_dict, TR_output):
        
        #Choose in a greedy way a family of functions and a feature
        choices = []
        
        for feature in TR_inputs_dict.keys():
            for family in IndipendentRamification.__families:
                
                #Calculate transformation
                s = family["transformation"](TR_inputs_dict[feature])
                
                #Extract future residual
                mean_R = np.mean(TR_output)
                mean_s = np.mean(s)
                
                a = np.sum((s - mean_s)*(TR_output - mean_R)) / ((np.sum(s - mean_s)**2)+1e-25)
                b = mean_R - (a * mean_s)
                
                #Calculate of transparency of new residual (greedy transparency)
                DW = durbin_watson(TR_output - family["residual"](TR_inputs_dict[feature], a, b))
                
                choices.append({"feature": feature, "family": family["family"], "transparency": np.abs(DW - 2)})
                
        #Choose the best combination feature / family
        choices = sorted(choices, key = lambda x: -x["transparency"])
        
        feature = choices[0]["feature"]
        operations = list(filter(lambda x: x["family"] == choices[0]["family"], IndipendentRamification.__families))[0]["operations"]
        
        choices = []

        #For each operations
        for oper in operations:
                
            oper_node = oper()
            for i in range(oper_node.get_channels()):
                op.OperationNode.new_connection(oper_node, op.Leaf(f"l{i}"), i)
                    
            childrens = oper_node.get_childrens()
                    
            oper_node.set_backward_calculation_memory(TR_output)
                    
            if(oper_node.get_channels() == 2):
                    
                base = oper_node.get_default_inputs()
                        
                #Doing the projection to 1° slot / child
                childrens[0].set_name(feature)
                childrens[0].set_forward_calculation_memory(TR_inputs_dict[feature])
                childrens[1].set_forward_calculation_memory(base[1])
                childrens[1].set_name(projection(oper_node.backward_calculation()[1], TR_inputs_dict, lambda a, b: -cosine(a, b)))
                choices.append(oper_node.copy_sub_tree())
                
                #Doing the projection to 0° slot / child if operation is not commutative
                if(not oper_node.is_commutative()):
                                            
                    childrens[1].set_name(feature)
                    childrens[0].set_forward_calculation_memory(base[0])
                    childrens[1].set_forward_calculation_memory(TR_inputs_dict[feature])
                    childrens[0].set_name(projection(oper_node.backward_calculation()[0], TR_inputs_dict, lambda a, b: -cosine(a, b)))
                    choices.append(oper_node.copy_sub_tree())
                
            elif(oper_node.get_channels() == 1):
      
                childrens[0].set_name(feature)
                choices.append(oper_node.copy_sub_tree())
                    
            else: raise TypeError(f"{oper} has no 1 or 2 channels, it is not acceptable for IndipendentRamification learning strategy")
            
        #Choice the best operation / features - constant
        choices = [{"structure": choice, "transparency": durbin_watson(TR_output - choice.forward_calculation_memory_sub_tree(TR_inputs_dict).get_forward_calculation_memory())} for choice in choices]
        
        choices = sorted(choices, key = lambda x: -x["transparency"])
        
        #Calculate all memory and weights
        choice = choices[0]["structure"].backward_calculation_memory_sub_tree(TR_output)
        choice.set_weight_memory(choice.factorization_backward_calculation())
        choice.factorization_backward_calculation_memory_sub_tree()
        
        return choice
    
        
    def body_learning(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, stop_criteria):
        
        if(not TR_inputs_dict.keys() == set(VL_inputs_dict.keys())): raise ValueError("TR and VL input sets has not same features")
        
        #Checkpoint
        self.checkpoint_metrics()
        
        #Calculate all memory
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        
        #Calculate first candidates
        working_leafs = [{"leaf": leaf, "extension": None} for leaf in self._model.get_leafs()]

        
        #While cycle
        # #DA FINIRE lo stop del ciclo
        while(stop_criteria(self) or len(working_leafs) > 0):
            
            working_leafs[0]["extension"] = self.extend(TR_inputs_dict, TR_output)
            
            break
            
            #If best extension is worst than leaf, doing nothing
            if(choices[0]["candidate"] == str(leaf)): continue
            
            #Do the extension
            leaf_candidates += extension.get_leafs()
        
            father_leaf = leaf.get_father()

            op.OperationNode.free_connection(father_leaf, leaf)
            op.OperationNode.new_connection(father_leaf, extension)
            
            del working_leafs[0]
            





