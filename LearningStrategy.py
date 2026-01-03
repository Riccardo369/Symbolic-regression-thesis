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
    

class IndipendentRamification(SymbolicRegressionLearning):
    
    def __init__(self, operations):
        super().__init__()
        
        for oper in operations:
            if(not issubclass(oper, op.OperationNode)): raise ValueError(f"The operation {oper} is not a OperationNode type")
        
        self._operations = list(operations)
        
    def __check_one_node(node):
        leafs = node.get_leafs()
        return len(leafs) == 1 and leafs[0] and leafs[0] == node
        
    def body_learning(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, stop_criteria):
        
        if(not set(TR_inputs_dict.keys()).equals(set(VL_inputs_dict.keys()))): raise ValueError("TR and VL input sets has not same features")
        
        self.checkpoint_metrics()
        
        #Pre - conditioning if is necessary
        if(IndipendentRamification.__check_one_node(self._model.get_root())): self._model.get_root().set_name(projection(TR_output, TR_inputs_dict, lambda a, b: -cosine(a, b)))
        
        self._model.forward_calculation(TR_inputs_dict).backward_calculation(TR_output)
        
        leaf_candidates = self._model.get_leafs()
        
        while(stop_criteria(self) or len(leaf_candidates) > 0):
            
            leaf = leaf_candidates[0]
            del leaf_candidates[0]
            
            ramification_function = lambda x: x
            
            operation_candidates = []
            
            #Add all possible tree (1° sub problem: search for each operation the best inputs combination)
            for oper in self._operations:
                
                oper_node = oper()
                for i, in range(oper_node.get_channels()):
                    op.OperationNode.new_connection(oper_node, op.Leaf(f"l{i}"))
                    
                childrens = oper_node.get_childrens()
                    
                oper_node.set_backward_calculation_memory(leaf.get_backward_calculation_memory())
                    
                if(oper_node.get_channels() == 2):
                    
                    base = oper_node.get_default_inputs()

                    for feature in TR_inputs_dict.keys():
                        
                        #Doing the projection to 1° slot / child
                        childrens[0].set_forward_calculation_memory(feature)
                        childrens[1].set_forward_calculation_memory(base[1])
                        childrens[1].set_name(projection(oper_node.backward_calculation()[1], TR_inputs_dict, lambda a, b: -cosine(a, b)))
                        operation_candidates.append(oper_node.copy_sub_tree())
                                        
                        #Doing the projection to 0° slot / child
                        childrens[0].set_forward_calculation_memory(base[0])
                        childrens[1].set_forward_calculation_memory(feature)
                        childrens[0].set_name(projection(oper_node.backward_calculation()[0], TR_inputs_dict, lambda a, b: -cosine(a, b)))
                        operation_candidates.append(oper_node.copy_sub_tree())

                elif(oper_node.get_channels() == 1):

                    for feature in TR_inputs_dict.keys():
                        
                        childrens[0].set_name(feature)
                        operation_candidates.append(oper_node.copy_sub_tree())
                    
                else: raise TypeError(f"{oper} has no 1 or 2 channels, it is not acceptable for IndipendentRamification learning strategy")



            inputs_dict = {key: np.concatenate([TR_inputs_dict[key], VL_inputs_dict[key]]) for key in TR_inputs_dict.keys()}
            output = np.concatenate([TR_output, VL_output])
               
            #2° sub problem: choose the best extension 
            operation_candidates = sorted(operation_candidates, keys = lambda x: er.MAE(ramification_function(SymbolicRegression(x).forward_calculation(inputs_dict).get_root().get_forward_calculation_memory()), output + [leaf]))
            
            #If best extension is worst than leaf, doing nothing
            if(operation_candidates[0] == leaf): continue
            
            #Do the extension
            extension = SymbolicRegression(operation_candidates[0])
            leaf_candidates += extension.get_leafs()
            extension = extension.get_root()
        
            father_leaf = leaf.get_father()

            op.OperationNode.free_connection(father_leaf, leaf)
            op.OperationNode.new_connection(father_leaf, extension)
            
            #3° sub problem: only for binary operations, split in right way the backwards calculation
            if(extension.get_channels() == 2):
                
                #Solve sub problem of indipendent weights
                extension.set_weight_vector_memory(extension.weight_vector_calculation())
                
                #Set final indipendent backwards ideal values
                for child, back in zip(extension.get_childrens(), extension.backward_indipendent_calculation()): child.set_backward_calculation_memory(back)
            






