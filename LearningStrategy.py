import numpy as np
import matplotlib.pyplot as plt

from Model import SymbolicRegression
import Operations as op
import Errors as er
import Indicators as In
import Exceptions as ex
import utils
import TheoricalExperiments.ConstantOptimization as co

class SymbolicRegressionLearning:
    
    def __init__(self):
        
        self._model = None
        self._metric_results = {}
        self._metric_functions = []
    
    def _pre_learning(self, model, metrics, TR_inputs_dict, TR_output):
        self._model = model
        self._metric_functions = metrics
        self._metric_results = {key: [] for key in metrics.keys()}
        
    def _post_learning(self):
        pass
        
    def checkpoint_metrics(self):
        for key in self._metric_results:
            result = self._metric_functions[key](self._model)
            self._metric_results[key].append(result)   
            print(key, ":", result)
        
    def _body_learning(self, TR_inputs_dict, TR_output, stop_criteria):
        return NotImplemented
        
    def __call__(self, model, TR_inputs_dict, TR_output, stop_criteria, metric_functions = {}):
        self._pre_learning(model, metric_functions, TR_inputs_dict, TR_output)
        self._body_learning(TR_inputs_dict, TR_output, stop_criteria)
        self._post_learning()
        
        result = self._metric_results.copy()
        self._metric_results = {}
        return result

class AnalysisRamification(SymbolicRegressionLearning):
    
    #The order of binary operations is the order of construction every time a node is extended (every node will be extended with all binary operations, and the order is defined by
    # the order of binary_operations list)
    
    #get_package_importance_nodes analyze nodes and returns a list of packages (for importance), where every package contains a list of triples node - feature- operation. 
    # If it is possible to choose a valid extension with first package, the algorithm will stop, otherwise it will search the extension on next package
    
    def __init__(self, unary_operations, filtering_criterion, sorting_criterion, get_package_importance_nodes, heuristic_metrics):
        
        super().__init__()
        
        for oper in unary_operations:
            if(not issubclass(oper, op.OperationNode)): raise ValueError(f"The operation {oper} is not a OperationNode type")
            if(oper().get_channels() != 1): raise TypeError(f"{oper} has no 1 channel, it is not acceptable an unary operation like that")
        
        self._unary_operations = list(unary_operations)
        
        self.__filtering_criterion = filtering_criterion
        self.__sorting_criterion = sorting_criterion

        self._heuristic_metrics = heuristic_metrics
        
        self.__get_package_importance_nodes = get_package_importance_nodes
        
        self._ranges_iterations = []
        
        #Every values of dict = (index of package, index of extension to start inside the package)
        self._dict_extension_history = dict()
        self._dict_pruning_history = dict()
        
    def __call__(self, model, TR_inputs_dict, TR_output, stop_criteria, metric_functions = {}):
        result = super().__call__(model, TR_inputs_dict, TR_output, stop_criteria, metric_functions)
        result = {"results": result, "ranges_iterations": list(self._ranges_iterations), "history_dicts": {"extension": self._dict_extension_history,
                                                                                                        "pruning": self._dict_pruning_history}}
        
        self._dict_extension_history = dict()
        self._dict_pruning_history = dict()
        
        return result
    
    def _pre_learning(self, model, metrics, TR_inputs_dict, TR_output):
        super()._pre_learning(model, metrics, TR_inputs_dict, TR_output)
        self._ranges_iterations = []
         
    def __structural_domain_check(model, TR_inputs_dict, TR_output):
        model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        return np.all(np.isfinite(model.get_root().get_memory("forward"))) & np.all([np.all(np.isfinite(x.get_memory("backward"))) for x in model.get_root().get_all_subtree_nodes() if isinstance(x, op.Leaf)])

    def __checkpoint_heuristic_metrics(self):
        return {key: self._heuristic_metrics[key](self._model) for key in self._heuristic_metrics.keys()}

    def __valid_dict_history_form(Dict):
        if(not all([type(x) == str for x in Dict.keys()])): return False
        if(not all([type(x) == tuple and 
                    len(x) == 2 and 
                    type(x[0]) == int and 
                    type(x[1]) == int for x in Dict.values()])): return False
        
        for key in Dict.keys():
            try: SymbolicRegression.string_to_tree(key)
            except: return False
            
        return True
 
    def set_dict_extension_history(self, Dict):
        valid = AnalysisRamification.__valid_dict_history_form(Dict)
        if(valid): self._dict_extension_history = Dict.copy()
        return valid
    
    def set_dict_pruning_history(self, Dict):
        valid = AnalysisRamification.__valid_dict_history_form(Dict)
        if(valid): self._dict_pruning_history = Dict.copy()
        return valid

    def linear_extend_model(model):
        
        def recursion(node):
            
            if(node.is_constant()): return
            
            for c in node.get_childrens(): recursion(c)
            
            model.replace_subtree(node, SymbolicRegression.string_to_tree(f"add(mul({node}, 1), 0)").get_root())
            
        recursion(model.get_root())

    #Get (if possible) the extension (with delta metric) of this combination
    def __analyze_extension(self, TR_inputs_dict, TR_output, config):
                
        node = config["old node"]
        oper_node = config["new node"]
        
        linear_ext = utils.linear_extension(oper_node.get_memory("forward"), oper_node.get_memory("backward")) 
        m = linear_ext["m"] if np.isfinite(linear_ext["m"]) else 1
        q = linear_ext["q"] if np.isfinite(linear_ext["q"]) else 0

        for try_node in [oper_node, SymbolicRegression.string_to_tree(f"add(mul({oper_node}, {m}), {q})").get_root()]:
        
            #Try the substition
            self._model.replace_subtree(node, try_node)
                
            #Check the structural domain
            valid_domain = AnalysisRamification.__structural_domain_check(self._model, TR_inputs_dict, TR_output)
                
            #Calculate the new metrics (if domain permitted)
            if(valid_domain): new_metrics = self.__checkpoint_heuristic_metrics()

            #Do the substition
            self._model.replace_subtree(try_node, node)
            
            config["new node"] = try_node
            
            #print(try_node, valid_domain)

            #Check structure domain
            if(valid_domain): return {key: new_metrics[key] for key in self._heuristic_metrics.keys()}
        
        return None

    #Extension phase (tree already wet with TR set)
    def __extension_step(self, TR_inputs_dict, TR_output):
        
        original_metrics = self.__checkpoint_heuristic_metrics()
        
        model_firm = str(self._model.copy().simplify_model())
        k, i = self._dict_extension_history[model_firm] if model_firm in self._dict_extension_history.keys() else (0, 0)
        
        choices = []
        
        AnalysisRamification.linear_extend_model(self._model)
        
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        
        #Package the choices by importance
        for package in self.__get_package_importance_nodes(self._model.get_root().get_all_subtree_nodes(), self._unary_operations, TR_inputs_dict, TR_output)[k:]:

            choices = []
            
            #For each choice inside the package, calculate the metrics and the possible extension
            for choice in package:

                #Analyze the extension
                result = self.__analyze_extension(TR_inputs_dict, TR_output, choice)
                
                #If extension is aviable, calculate all data about it
                if(result != None):

                    choices.append(choice | {"delta heuristic metrics": {key: original_metrics[key] - result[key] for key in self._heuristic_metrics.keys()}})
            
            choices = self.__filtering_criterion(choices)

            choices = self.__sorting_criterion(choices)
        
            #print("Survival choices")
            #for x in choices: print(x)

            #If there is no any choices, go ahead  
            if(len(choices) == 0): continue

            #Do the extension
            self._model.replace_subtree(choices[i]["old node"], choices[i]["new node"])
            
            #print("Extension choosed", choices[i])
            
            #Go to one step with i index
            if(i < len(choices)): i += 1
            
            #Update the indices in the case i reach the next package
            if(i == len(choices)):
                i = 0
                k += 1
                
            #Save the new position
            self._dict_extension_history[model_firm] = (k, i)
                
            break

        return len(choices) > 0

    #Pruning phase
    def __pruning_step(self, TR_inputs_dict, TR_output):
        
        choices = []
        
        original_metrics = self.__checkpoint_heuristic_metrics()
        
        #Calculate features
        features = list(TR_inputs_dict.keys())
        
        #For every node considered valid for pruning part
        for node in self._model.get_root().get_all_subtree_nodes():

            choices = []
            
            #Save the original node
            original_node = node
            
            #Ricalculate model (for calculate the constant)
            self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)

            #For each feature + the constant
            for f in features + [np.mean(original_node.get_memory("backward"))]:
                
                #Replace the subtree
                node = op.Leaf(f)
                self._model.replace_subtree(original_node, node)
                
                valid_domain = AnalysisRamification.__structural_domain_check(self._model, TR_inputs_dict, TR_output)
                
                #Calculate the new metric
                new_metrics = self.__checkpoint_heuristic_metrics()
                
                #Return to normal state
                self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
                
                self._model.replace_subtree(node, original_node)
                
                #If domain is not valid, do not consider the combination
                if(not valid_domain): continue
                
                choices.append({"original node": original_node, "new node": node, "delta": {key: original_metrics[key] - new_metrics[key] for key in self._heuristic_metrics.keys()}})
                
        choices = self.__filtering_criterion(choices)

        if(len(choices) == 0): return False
        
        choices = self.__sorting_criterion(choices)
        
        self._model.replace_subtree(choices[0]["original node"], choices[0]["new node"])

        return True
   
        
        
    #TR_inputs_dict: numpy array's dict
    #VL_inputs_dict: numpy array's dict
    #TR_output: constant or numpy array
    #VL_output: constant or numpy array
    def _body_learning(self, TR_inputs_dict, TR_output, stop_criteria):
        
        if(np.var([len(TR_inputs_dict[x]) for x in TR_inputs_dict.keys()]) > 0): raise ValueError("All TR features has to have the same value numbers")

        if(type(TR_output) not in [int, float] and len(TR_output) != len(TR_inputs_dict[list(TR_inputs_dict.keys())[0]])): raise ValueError("TR output has no the same points of TR input")

        #Set the constant in numpy array (so everything is in array)
        if(type(TR_output) in [int, float]): TR_output = np.array([TR_output for _ in range(len(TR_inputs_dict[list(TR_inputs_dict.keys())[0]]))])

        #Calculate all memory
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)

        #Checkpoint
        self.checkpoint_metrics()
        
        extension_memories = []
        pruning_memories = []
        
        new_extension_memory = None
        new_pruning_memory = None
        
        i = 0
        r = 0
        
        print(f"Start training")
        
        #In a cycle, bring the model in a equilibrium between underfitting and overfitting, with a zig zag strategy
        while(True):
        
            #If model is the same of last cycle (considering the last time you start the last extension), break
            new_extension_memory = str(self._model)
            if(new_extension_memory in extension_memories): break
            extension_memories.append(new_extension_memory)

            r = i

            self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
            
            #Do the extension phase (bring the model to overfitting direction)
            while(not stop_criteria(self._metric_results) and self.__extension_step(TR_inputs_dict, TR_output)):
                
                print(f"Extension phase, iteration {i+1} done {'-'*50}")
                
                self._model.simplify_model()
                #co.linear_extension_optimization(self._model, TR_inputs_dict, TR_output)
                self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
                self.checkpoint_metrics()
                
                i += 1
                
            if(r != i): self._ranges_iterations.append({"period": "extension", "range": (r, i)})
                   
            r = i
            
            break
         
            #If model is the same of last cycle (considering the last time you start the last pruning), break
            new_pruning_memory = str(self._model)
            if(new_pruning_memory in pruning_memories): break
            pruning_memories.append(new_pruning_memory)
            
            internal_history = []
            
            #Do the pruning phase (bring the model to underfitting direction)
            while(not stop_criteria(self._metric_results) and self.__pruning_step(TR_inputs_dict, TR_output)): 
                
                print(f"Pruning phase, iteration {i+1} done {'-'*50}")
                self._model.simplify_model()
                self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
                self.checkpoint_metrics()
                i += 1
                
                if(str(self._model) in internal_history): break
                
                internal_history.append(str(self._model))
                
            if(r != i): self._ranges_iterations.append({"period": "pruning", "range": (r, i)})
            
        print("Training done!")
        
        #Last check
        self._model.get_root().clear_all_subtree_memory("forward").clear_all_subtree_memory("backward")
        
        self._model.simplify_model()
        
        print("Model prepared to use!")
        
