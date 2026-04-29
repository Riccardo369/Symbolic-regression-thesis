import numpy as np
import matplotlib.pyplot as plt

from Model import SymbolicRegression
import Operations as op
import Errors as er
import Indicators as In
import Exceptions as ex

class SymbolicRegressionLearning:
    
    def __init__(self):
        
        self._model = None
        self._metric_results = {}
        self._metric_functions = []
    
    def _pre_learning(self, model, metrics, TR_inputs_dict, TR_output):
        self._model = model
        self._metric_functions = metrics
        self._metric_results = {key: [] for key in metrics.keys()}
        
        #If model has one node (leaf and the root), it is necessary to set the node to best feature or best constant
        if(isinstance(self._model.get_root(), op.Leaf)):
            
            #Order in base of best candidates (from features and best constant)
            choices = sorted(list(TR_inputs_dict.keys()) + [float(np.mean(TR_output))], key = lambda x: er.MAE(TR_inputs_dict[x] if type(x) == str else x, TR_output))
            self._model.get_root().set_name(choices[0])
        
    def _post_learning(self):
        pass
        
    def checkpoint_metrics(self):
        for key in self._metric_results:
            result = self._metric_functions[key](self._model)
            self._metric_results[key].append(result)   
            print(key, ":", result)
        
    def _body_learning(self, TR_inputs_dict, TR_output):
        return NotImplemented
        
    def __call__(self, model, TR_inputs_dict, TR_output,  metric_functions = {}):
        self._pre_learning(model, metric_functions, TR_inputs_dict, TR_output)
        self._body_learning(TR_inputs_dict, TR_output)
        self._post_learning()
        
        result = self._metric_results.copy()
        self._metric_results = {}
        return result

class AnalysisRamification(SymbolicRegressionLearning):
    
    #The order of binary operations is the order of construction every time a node is extended (every node will be extended with all binary operations, and the order is defined by
    # the order of binary_operations list)
    
    #get_package_importance_nodes analyze nodes and returns a list of packages (for importance), where every package contains a list of triples node - feature- operation. 
    # If it is possible to choose a valid extension with first package, the algorithm will stop, otherwise it will search the extension on next package
    
    def __init__(self, unary_operations, binary_operations, get_package_importance_nodes, heuristic_metrics):
        
        super().__init__()
        
        for oper in unary_operations:
            if(not issubclass(oper, op.OperationNode)): raise ValueError(f"The operation {oper} is not a OperationNode type")
            if(oper().get_channels() != 1): raise TypeError(f"{oper} has no 1 channel, it is not acceptable an unary operation like that")
        
        for oper, i in binary_operations:
            if(not issubclass(oper, op.OperationNode)): raise ValueError(f"The operation {oper} is not a OperationNode type")
            if(oper().get_channels() != 2): raise TypeError(f"{oper} has no 2 channels, it is not acceptable an binary operation like that")
        
        self._unary_operations = list(unary_operations)
        self._binary_operations = [{"neutral": i, "oper": o} for o, i in binary_operations]
        
        self._heuristic_metrics = heuristic_metrics
        
        self.__get_package_importance_nodes = get_package_importance_nodes
        
        self._ranges_iterations = []
        
    def __call__(self, model, TR_inputs_dict, TR_output, metric_functions = {}):
        result = super().__call__(model, TR_inputs_dict, TR_output,   metric_functions)
        result_ranges_iterations = list(self._ranges_iterations)
        self._ranges_iterations = []
        return {"result": result, "ranges_iterations": result_ranges_iterations}
    
    def _pre_learning(self, model, metrics, TR_inputs_dict, TR_output):
        super()._pre_learning(model, metrics, TR_inputs_dict, TR_output)
        self._ranges_iterations = []
                
    #Replace the node1 subtree with node2 subtree                            
    def __replace_subtree(self, node1, node2):
        
        #Get the father
        father_leaf = node1.get_father()
                
        #The leaf is the root
        if(father_leaf == None): self._model.set_root(node2)
                   
        else:
                
            i = father_leaf.get_childrens().index(node1)
  
            op.OperationNode.free_connection(father_leaf, node1)
            op.OperationNode.new_connection(father_leaf, node2, i)

    def __structural_domain_check(model, TR_inputs_dict, TR_output):
        
        model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        return np.all(np.isfinite(model.get_root().get_memory("forward"))) & np.all([np.all(np.isfinite(x.get_memory("backward"))) for x in model.get_root().get_all_subtree_nodes() if isinstance(x, op.Leaf)])

    #Get (if possible) the extension (with delta metric) of this combination
    def __get_extensions(self, TR_inputs_dict, TR_output, config):
        
        node = config["node"]
        feature = config["feature"]
        unary_oper = config["unary operation"]
        
        if(config["typology"] in ["constant", "feature"]): oper_node = op.Leaf(feature) if feature != None else node
            
        if(config["typology"] == "feature - operation"):

            #Create node
            oper_node = unary_oper()
            
            #Domain input check
            if(not oper_node.check_input_domain(0, TR_inputs_dict[feature] if isinstance(node, op.Leaf) else node.get_memory("forward"))): return None
            
            #Domain output check
            if(not oper_node.check_output_domain(node.get_memory("backward") if node.get_father() != None else TR_output)): return None
                
            #Create children
            op.OperationNode.new_connection(oper_node, op.Leaf("l0"), 0)

            #If node is a leaf
            if(isinstance(node, op.Leaf)): f1 = op.Leaf(feature)
                
            #If node is a midterm node or the root, the f1 became che node itself   
            else: f1 = node.copy_sub_tree()
            
            #Attach f1 to the unique children of operation
            self.__replace_subtree(oper_node.get_childrens()[0], f1)
           
        results = []
        
        try_nodes = [oper_node]
        
        #If it is possible to try also with the scaled and trasleted version of oper_node
        #output_oper_node = oper_node.forward_calculation_memory_sub_tree(TR_inputs_dict).get_memory("forward")
        #linear_convertion = AnalysisRamification.__scaling_and_traslating(output_oper_node, node.get_memory("backward"))

        #if(np.isfinite(linear_convertion["m"]) and np.isfinite(linear_convertion["q"])):
            #try_nodes.append(SymbolicRegression.string_to_tree(f"add(mul({oper_node}, {linear_convertion['m']}), {linear_convertion['q']})").get_root())
            
        #Try for standard and scaling / translated form
        for try_oper_node in try_nodes:
        
            #Try the substition
            self.__replace_subtree(node, oper_node)
            
            #Check the structural domain
            valid_domain = AnalysisRamification.__structural_domain_check(self._model, TR_inputs_dict, TR_output)
            
            #Calculate the new metrics
            new_metrics = {key: self._heuristic_metrics[key](self._model) for key in self._heuristic_metrics.keys()}
            
            #Do the substition
            self.__replace_subtree(oper_node, node)
            
            #Doing functional calculation
            self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
            
            #Check structure domain
            if(not valid_domain): continue
            
            #Collect the parameters if the version is aviable
            results.append({"new node": try_oper_node} | {key: new_metrics[key] for key in self._heuristic_metrics.keys()})
        
        return results
      
    def __scaling_and_traslating(X, Y):

        mean_X = np.mean(X)
        mean_Y = np.mean(Y)
        
        m = np.sum((X - mean_X) * (Y - mean_Y)) / np.sum((X - mean_X)**2)
        q = mean_Y - (m*mean_X)
        
        return {"m": m, "q": q}

    #Extension phase (tree already wet with TR set)
    def __extension_step(self, TR_inputs_dict, TR_output):
        
        original_metrics = {key: self._heuristic_metrics[key](self._model) for key in self._heuristic_metrics.keys()}
        
        #Package the choices by importance
        for package in self.__get_package_importance_nodes(self._model.get_root().get_all_subtree_nodes(), self._unary_operations, TR_inputs_dict, TR_output):
            
            choices = []
            
           #For each choice inside the package, calculate the metrics and the possible extension
            for choice in package:

                #Calculate the extension
                results = self.__get_extensions(TR_inputs_dict, TR_output, choice)
                
                #If extension is aviable, calculate all data about it
                if(results != None):
                    
                    for result in results:
                        
                        choices.append({"new node": result["new node"], "original node": choice["node"]} | 
                                       {f"{key} delta": original_metrics[key] - result[key] for key in self._heuristic_metrics.keys()})

            if(len(choices) > 0): break
            
        for i in choices: print(i)
            
        choices = [x for x in choices if np.isfinite(x["Chaterjee delta"]) and x["Chaterjee delta"] > 0]
        
        if(len(choices) == 0): return False
        
        choices = sorted(choices, key = lambda x: (-x["Chaterjee delta"], -x["MAE delta"]))

        new_node = choices[0]["new node"]
        
        for bo in self._binary_operations:
                
            oper = bo["oper"]()
            oper.new_connection(new_node.copy_sub_tree(), 0)
            oper.new_connection(op.Leaf(bo["neutral"]), 1)
                
            new_node = oper

        #Do the extension
        self.__replace_subtree(choices[0]["original node"], new_node)
        
        return True

    #Pruning phase         
    def __pruning_step(self, TR_inputs_dict, TR_output):
    
        choices = []
        
        original_metrics = {key: self._heuristic_metrics[key](self._model) for key in self._heuristic_metrics.keys()}
        
        #Calculate features
        features = list(TR_inputs_dict.keys())
        
        #For every node considered valid for pruning part
        for node in self._model.get_root().get_all_subtree_nodes():

            choices = []
            
            #Save the original node
            original_node = node
            old_node = node
            
            #Ricalculate model (for calculate the constant)
            self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)

            #For each feature + the constant
            for f in features + [np.mean(original_node.get_memory("backward"))]:
                
                #Replace the subtree
                node = op.Leaf(f)
                self.__replace_subtree(old_node, node)
                old_node = node
                
                valid_domain = AnalysisRamification.__structural_domain_check(self._model, TR_inputs_dict, TR_output)
                
                #Calculate the new metric
                new_metrics = {key: self._heuristic_metrics[key](self._model) for key in self._heuristic_metrics.keys()}
                
                #Return to normal state
                self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
                
                #If domain is not valid, do not consider the combination
                if(not valid_domain): continue
                
                choices.append({"new node": node} | {f"{key} delta": original_metrics[key] - new_metrics[key] for key in self._heuristic_metrics.keys()})
                
                self.__replace_subtree(old_node, original_node)
            
        #Filter choices
        choices = [x for x in choices if x["Chaterjee delta"] > 0]
 
        #If there is no any choices to make, there is no any valid pruning to do
        if(len(choices) == 0): return False
        
        #In choice the best extension, choose the extension that minimize the VL and after the TR
        choices = sorted(choices, key = lambda x: (-x["Chaterjee delta"], -x["MAE delta"]))
        
        self.__replace_subtree(choices[0]["original node"], choices[0]["new node"])

        return True
    
    
    
  
    #TR_inputs_dict: numpy array's dict
    #VL_inputs_dict: numpy array's dict
    #TR_output: constant or numpy array
    #VL_output: constant or numpy array
    def _body_learning(self, TR_inputs_dict, TR_output):
        
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
            while(self.__extension_step(TR_inputs_dict, TR_output)):
                            
                print(f"Extension phase, iteration {i+1} done {'-'*50}")
                self._model.simplify_model()
                self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
                self.checkpoint_metrics()
                i += 1
                
            self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
                
            if(r != i): self._ranges_iterations.append({"period": "extension", "range": (r, i)})
                   
            r = i
         
            #If model is the same of last cycle (considering the last time you start the last pruning), break
            new_pruning_memory = str(self._model)
            if(new_pruning_memory in pruning_memories): break
            pruning_memories.append(new_pruning_memory)
            
            #Do the pruning phase (bring the model to underfitting direction)
            while(self.__pruning_step(TR_inputs_dict, TR_output)): 
                
                print(f"Pruning phase, iteration {i+1} done {'-'*50}")
                self._model.simplify_model()
                self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
                self.checkpoint_metrics()
                i += 1
                
            self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
                
            if(r != i): self._ranges_iterations.append({"period": "pruning", "range": (r, i)})
            
        print("Training done!")
        
        #Last check
        self._model.get_root().clear_all_subtree_memory("forward").clear_all_subtree_memory("backward")
        
        print("Model prepared to use!")
        
