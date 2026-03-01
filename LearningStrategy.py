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
    
    def __init__(self, operations):
        super().__init__()
        
        for oper in operations:
            if(not issubclass(oper, op.OperationNode)): raise ValueError(f"The operation {oper} is not a OperationNode type")
        
        self._operations = list(operations)
        
        
    
    def extend(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, f):
        
        #Add all possible tree (1° sub problem: search for each operation the best inputs combination)
        for oper in self._operations:
            
            return None
                
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

        return None

        inputs_dict = {key: np.concatenate([TR_inputs_dict[key], VL_inputs_dict[key]]) for key in TR_inputs_dict.keys()}
        output = np.concatenate([TR_output, VL_output])
               
        #2° sub problem: choose the best extension
        operation_candidates = sorted([{"candidate": str(x), "error": er.MAE(ramification_function(SymbolicRegression(x).forward_calculation(inputs_dict).get_root().get_forward_calculation_memory()), output)} for x in operation_candidates + [leaf]], lambda x: x["error"])

        
        return None
        
    def body_learning(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, stop_criteria):
        
        if(not TR_inputs_dict.keys() == set(VL_inputs_dict.keys())): raise ValueError("TR and VL input sets has not same features")
        
        #Checkpoint
        self.checkpoint_metrics()
        
        #Calculate all memory
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        
        #Calculate first candidates
        working_leafs = [{"leaf": leaf, "extension": None} for leaf in self._model.get_leafs()]
        
        #NON ESISTE CHE SI PARTA DA UN MODELLO GIà FATTO, SI PARTE SEMPRE DALL' INIZIO, COSE SEMPLICI, SENZA COMPLICARCI L' ESISTENZA
        
        #Per la parallelizzazione pattern multi writer - single reader:
        # I writer creano le extension dei nodi che non hanno la propria extension, alla fine di ogni lavoro di writer, quel writer 
        # aggiunge altri nodi, per non far fermare il lavoro
        # il reader, in sequenza, controlla i nodi, vede se andare avanti con le estensioni, altrimenti, si ferma, e fermandosi, i
        # lavori dei writer già fatti verranno eliminati, ma intanto in questo modo, lo scopo è quello di ridurre il più possibile il 
        # tempo che il reader ci mette a: estendere l' albero e fare i vari calcoli di regolarizzazione
    
        
        
        #While cycle
        while(stop_criteria(self) or len(working_leafs) > 0):
            
            #Parallelism phase without synchronization
            
            ramification_function = lambda x: x
            working_leafs[0]["extension"] = self.extend(TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, ramification_function)
            
            break
        
            #Barrier synchronization
            
            #If best extension is worst than leaf, doing nothing
            if(operation_candidates[0]["candidate"] == str(leaf)): continue
            
            #Do the extension
            leaf_candidates += extension.get_leafs()
        
            father_leaf = leaf.get_father()

            op.OperationNode.free_connection(father_leaf, leaf)
            op.OperationNode.new_connection(father_leaf, extension)
            





