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
        
    def _body_learning(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output):
        return NotImplemented
        
    def __call__(self, model, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, metric_functions = {}):
        self._pre_learning(model, metric_functions, TR_inputs_dict, TR_output)
        self._body_learning(TR_inputs_dict, TR_output, VL_inputs_dict, VL_output)
        self._post_learning()
        
        result = self._metric_results.copy()
        self._metric_results = {}
        return result



class AnalysisRamification(SymbolicRegressionLearning):

    class GradPoint:
        
        def __init__(self, X, y):
            self.X = X
            self.y = y
            self.X_grad = [None for _ in range(len(X))]
    
    def __init__(self, operations):
        super().__init__()
        
        for oper in operations:
            if(not issubclass(oper, op.OperationNode)): raise ValueError(f"The operation {oper} is not a OperationNode type")
        
        self._operations = list(operations)
        
        self._ranges_iterations = []
        
    def __call__(self, model, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, metric_functions = {}):
        result = super().__call__(model, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, metric_functions)
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
    
    #Tell if candidate c is accepted for dataset (c["tree"] must be already inserted on tree)
    def __try_candidate_extension_on_dataset(self, c, label, inputs_dict, output):
        
        root = self._model.get_root()
        
        #Calculate forward and backward
        try: root.forward_calculation_memory_sub_tree(inputs_dict).backward_calculation_memory_sub_tree(output)   
        except (ex.ForwardCalculationError, ex.BackwardCalculationError): return False
        
        #Calculate domain validation (considering root and all leafs)
        domain_valid = all(map(lambda x: np.all(np.isfinite(x)), [root.get_memory("forward")] + [x.get_memory("backward") for x in self._model.get_leafs()]))

        #Calculate error on dataset
        c[f"{label}_error_delta"] = er.MAE(output, root.get_memory("forward"))
        
        #Return the validation of domain on this dataset
        return domain_valid


     #Process of adding the candidates considering 
    
    #Process of adding the candidates
    def __process_2_input_node(self, i, choices, local_output, childrens, oper_node, node1, node2):
 
        #If domain si not valid in first or second inputs, stop to process
        if((not oper_node.check_input_domain(i, node1.get_memory("forward"))) or 
           (not oper_node.check_input_domain(1-i, node2.get_memory("forward")))): return
        
        #Set memory backward
        oper_node.set_memory("backward", local_output)
        
        self.__replace_subtree(childrens[i], node1)
        self.__replace_subtree(childrens[1-i], node2)
        choices.add(str(oper_node))
        
        #Calculate constant
        try: c = oper_node.best_constant(1-i)
        except (ex.ForwardCalculationError, ex.BackwardCalculationError): return
        
        #If constant is not valid in 1-i position, stop to process
        if(not oper_node.check_input_domain(1-i, c)): return
        
        self.__replace_subtree(oper_node.get_childrens()[1-i], op.Leaf(c))
        choices.add(str(oper_node))
        
        new_childrens = oper_node.get_childrens()
        
        self.__replace_subtree(new_childrens[i], childrens[i])
        self.__replace_subtree(new_childrens[1-i], childrens[1-i])     

    #HEURISTIC LOGIC FOR AVOID TO DO ALL OPERATIONS
    def heuristic_analysis(self, node, feature, choices, input, output):
        
        #Calculate all indicators
                
        #If expectated_output is a constant, or an array of same constant
        if(type(output) in [float, int] or np.var(output) <= 1e-10):
                    
            pearson = 1
            spearman = 0
            chaterjee = 0
                 
        else:
                    
            pearson = In.pearson_correlation(input, output)
            spearman = In.spearman_correlation(input, output)
            chaterjee = In.chatterjee_correlation(input, output)
                
        #Calculate all strategies from indicators for family identification
        choices.append({"node": node, "feature": feature, "type": "linear",      "value": pearson})
        choices.append({"node": node, "feature": feature, "type": "monoton",     "value": spearman})
        choices.append({"node": node, "feature": feature, "type": "sinusoidal",  "value": chaterjee})

    #Find best extension's leaf by operations, node, first feature and TR / VL datasets (tree already wet from TR set)
    def __extend(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, node, operations, feature1, output):
        
        #Create candidates

        #Start with start node
        original_node_string = str(node)
        choices = set([original_node_string])
        
        #Considering to add the expectated output constant if necessary
        choices.add(str(np.mean(output)))

        #For each operations
        for oper in operations:
            
            #Create node
            oper_node = oper()
            
            #If considering the expectated output, this operation cannot be valid
            if(not oper_node.check_output_domain(output)): continue
                
            if(oper_node.get_channels() not in [1, 2]): raise TypeError(f"{oper} has no 1 or 2 channels, it is not acceptable for IndipendentRamification learning strategy")
                
            #Create childrens
            for i in range(oper_node.get_channels()): op.OperationNode.new_connection(oper_node, op.Leaf(f"l{i}"), i)
                
            channels = oper_node.get_channels()
            childrens = oper_node.get_childrens()
            
            if(not isinstance(node, op.Leaf)):
                
                copy_original_node = node.copy_sub_tree()
                
                if(channels == 1):
                    
                    if(not oper_node.check_input_domain(0, copy_original_node.get_memory("forward"))): continue
                    
                    old_node = childrens[0]
                    
                    self.__replace_subtree(old_node, copy_original_node)
                    choices.add(str(oper_node))
                    self.__replace_subtree(copy_original_node, old_node)
                
                else:
                    
                    for feature2 in TR_inputs_dict.keys():
                        
                        if(channels == 2):
                            
                            f2 = op.Leaf(feature2)
                            f2.set_memory("forward", TR_inputs_dict[feature2])
                            
                            self.__process_2_input_node(0, choices, node.get_memory("backward"), childrens, oper_node, copy_original_node, f2)
                
                            if(not oper_node.is_commutative()):
                                self.__process_2_input_node(1, choices, node.get_memory("backward"), childrens, oper_node, copy_original_node, f2)
                    
            else:
                
                #For each feature
                for feature2 in TR_inputs_dict.keys():
                    
                    if(channels == 1):
                        
                        if(not oper_node.check_input_domain(0, TR_inputs_dict[feature1])): continue
                        
                        self.__replace_subtree(childrens[0], op.Leaf(feature1))
                        choices.add(str(oper_node))
                        
                    elif(channels == 2):
                        
                        f1 = op.Leaf(feature1)
                        f1.set_memory("forward", TR_inputs_dict[feature1])
                        
                        f2 = op.Leaf(feature2)
                        f2.set_memory("forward", TR_inputs_dict[feature2])
                        
                        self.__process_2_input_node(0, choices, node.get_memory("backward"), childrens, oper_node, f1, f2)
                        
                        if(not oper_node.is_commutative()):
                            self.__process_2_input_node(1, choices, node.get_memory("backward"), childrens, oper_node, f1, f2)            

        #For each candidates, calculate the check domain and the error (in TR and VL set)
        choices = [{"tree": x} for x in choices]

        print("Enter node", node, operations)
        
        print("Phase 1")
        for i in choices: print(i)
        
        original_node = node
        
        for c in choices:
            
            new_node = SymbolicRegression.string_to_tree(c["tree"]).get_root()
            
            #Replace the old node with new node
            self.__replace_subtree(node, new_node)
            
            node = new_node
            
            #Try on VL dataset
            if(not self.__try_candidate_extension_on_dataset(c, "VL", VL_inputs_dict, VL_output)): continue
            
            #Try on TR dataset
            if(not self.__try_candidate_extension_on_dataset(c, "TR", TR_inputs_dict, TR_output)): continue
                
        #Reload original node
        self.__replace_subtree(node, original_node)
      
        #Delete che not valid choices               
        choices = [x for x in choices if set(x.keys()) == set(["tree", "TR_error_delta", "VL_error_delta"])]
        
        if(len(choices) == 0): return None

        #Take the original node and extract the start TR and VL errors and delete the original node from choices
        node_reference = [x for x in choices if x["tree"] == original_node_string][0]
        original_TR_error = node_reference["TR_error_delta"]
        original_VL_error = node_reference["VL_error_delta"]
        
        choices = [x for x in choices if x["tree"] != original_node_string]
        
        #Update the delta about the TR and VL errors
        for c in choices:
            
            c["TR_error_delta"] = original_TR_error - c["TR_error_delta"]
            c["VL_error_delta"] = original_VL_error - c["VL_error_delta"]
            
        print("Phase 2")
        for i in choices: print(i)

        #In extension, delete all extension that increase TR or VL
        choices = [x for x in choices if x["TR_error_delta"] >= 0 and x["VL_error_delta"] > 0]
        
        print("Phase 3")
        for i in choices: print(i)

        #If there is no valid extension
        if(len(choices) == 0): return None
        
        #In choice the best extension, choose the extension that minimize the VL and after the TR
        choices = sorted(choices, key = lambda x: (-x["VL_error_delta"], -x["TR_error_delta"]))
        
        #Choose best extension
        return SymbolicRegression.string_to_tree(choices[0]["tree"]).get_root()

    #Extension phase (tree already wet with TR set)
    def __extension_phase(self, TR_inputs_dict, VL_inputs_dict, TR_output, VL_output):

        choices = []
        
        result_extension = None
        
        #Actual extension to do
        residual_operations = []
        operations_dict = {"linear": [op.add, op.mul, op.div], "monoton": [], "sinusoidal": [op.sin, op.cos]}
   
        #For all nodes to consider
        
        #for n in [x for x in self._model.get_root().get_all_subtree_nodes() if isinstance(x, op.Leaf)]:
        for n in [x for x in self._model.get_root().get_all_subtree_nodes()]:

            #If node is a leaf
            if(isinstance(n, op.Leaf)):
                
                #Expectated output
                expectated_output = n.get_memory("backward")
                
                #For all features
                for f in TR_inputs_dict.keys():
                    self.heuristic_analysis(n, f, choices, TR_inputs_dict[f], expectated_output)
                
            #If node is a root or a internal node
            else: self.heuristic_analysis(n, None, choices, n.get_memory("forward"), n.get_memory("backward"))      
                
        #Order choices by best value (all indicators fall in [-1, 1] range)
        choices = sorted(choices , key = lambda x: np.clip(abs(x["value"]), 0, 1))
        
        node_to_replace = None
        
        operations_cache = []     
        
        #Repeat until find the best extension or choices are finished
        while(result_extension == None and len(choices) > 0):
            
            operations = operations_dict[choices[0]["type"]]
            
            #Add new operations to cache
            if(choices[0]["node"] not in [x["node"] for x in operations_cache]):
                operations_cache.append({"node": choices[0]["node"], "operations": [x for x in operations_dict[choices[0]["type"]]]})
                
            #Update and add the operations used
            else:
                
                operations_used = list(filter(lambda x: x["node"] == choices[0]["node"], operations_cache))[0]
                operations = [x for x in operations if x not in operations_used["operations"]]
                operations_used = [x for x in operations + operations_used["operations"] if x not in operations_used["operations"]]
                
            
                
            #Calculate the best extension   
            result_extension = self.__extend(TR_inputs_dict,
                                             TR_output,
                                             VL_inputs_dict,
                                             VL_output,
                                             choices[0]["node"],
                                             operations,
                                             choices[0]["feature"],
                                             choices[0]["node"].get_memory("backward") if choices[0]["node"].get_father() != None else TR_output)
            
            
            self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
            
            if(result_extension != None): node_to_replace = choices[0]["node"]
            
            print("result_extension: ", result_extension)
            
            #Delete the used choice      
            del choices[0]
            
        #No extension is possible
        if(result_extension == None): return False

        #Do the extension
        self.__replace_subtree(node_to_replace, result_extension)
            
        return True
     
    #Pruning phase         
    def __pruning_phase(self, TR_inputs_dict, VL_inputs_dict, TR_output, VL_output):
        
        #Calculate base errors (on TR and VL dataset) 
        original_VL_error = er.MAE(VL_output, self._model.get_root().forward_calculation_memory_sub_tree(VL_inputs_dict).get_memory("forward"))
        original_TR_error = er.MAE(TR_output, self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).get_memory("forward"))
        
        #Calculate features
        features = list(TR_inputs_dict.keys())
        
        choices = []
        
        #For every node considered valid for pruning part
        for node in  [x for x in self._model.get_root().get_all_subtree_nodes() if not isinstance(x, op.Leaf)]:
            
            #Save the original node
            original_node = node
            old_node = node
            
            #Ricalculate model (for calculate the constant)
            self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)

            #For each feature + the constant
            for f in features + [np.mean(original_node.get_memory("backward"))]:
                
                #Replace the subtree
                node = SymbolicRegression.string_to_tree(str(f)).get_root()
                self.__replace_subtree(old_node, node)
                old_node = node
                
                #Candidate
                c = {"node": str(node), "original_node": original_node}
                
                #Try with TR and VL dataset the domain and calculate the error
                self.__try_candidate_extension_on_dataset(c, "VL", VL_inputs_dict, VL_output)
                self.__try_candidate_extension_on_dataset(c, "TR", TR_inputs_dict, TR_output)
                
                #If candidate c pruning is valid, accept that
                if(set(c.keys()) == set(["node", "original_node", "TR_error_delta", "VL_error_delta"])): choices.append(c)
             
            #Return to the original situation (for doing the correct calculation on next leafs)   
            self.__replace_subtree(old_node, original_node)
                

        #Finish the TR and VL delta error with original TR and errors
        for c in choices:
            
            c["TR_error_delta"] = original_TR_error - c["TR_error_delta"]
            c["VL_error_delta"] = original_VL_error - c["VL_error_delta"]

        #Filter choices
        choices = [x for x in choices if x["TR_error_delta"] >= 0 and x["VL_error_delta"] > 0]
        
        #If there is no any valid choices, return false
        if(len(choices) == 0): return False
        
        #In choice the best extension, choose the extension that minimize the VL and after the TR
        choices = sorted(choices, key = lambda x: (-x["VL_error_delta"], -x["TR_error_delta"]))
        
        #Replace the subtree with the feature / constant best find (pruning part)
        self.__replace_subtree(choices["node"], op.Leaf(choices["feature"]))

        return True

    #Estimate gradient taking a dataset of points
    def estimate_gradient(inputs_dict, output):
        
        def incremental(sp, r1, r2, i):
            delta_y = sp[r2].y - sp[r1].y
            delta_x =  sp[r2].X[i] - sp[r1].X[i]
            return delta_y / delta_x  if np.abs(delta_x) >= 1e-40 else 0
        
        points = [AnalysisRamification.GradPoint([inputs_dict[k][i] for k in inputs_dict.keys()], output[i]) for i in range(len(output))]
        
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
        
        for d in AnalysisRamification.estimate_gradient(inputs_dict, output).T:
            
            hessian.append(AnalysisRamification.estimate_gradient(inputs_dict, d))
            
        return np.array(hessian)
        
    #TR_inputs_dict: numpy array's dict
    #VL_inputs_dict: numpy array's dict
    #TR_output: constant or numpy array
    #VL_output: constant or numpy array
    def _body_learning(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output):
        
        if(set(TR_inputs_dict.keys()) != set(VL_inputs_dict.keys())): raise ValueError("TR and VL input sets has not same features")
        
        if(np.var([len(TR_inputs_dict[x]) for x in TR_inputs_dict.keys()]) > 0): raise ValueError("All TR features has to have the same value numbers")
        if(np.var([len(VL_inputs_dict[x]) for x in VL_inputs_dict.keys()]) > 0): raise ValueError("All VL features has to have the same value numbers")
        
        if(type(TR_output) not in [int, float] and len(TR_output) != len(TR_inputs_dict[list(TR_inputs_dict.keys())[0]])): raise ValueError("TR output has no the same points of TR input")
        if(type(VL_output) not in [int, float] and len(VL_output) != len(VL_inputs_dict[list(VL_inputs_dict.keys())[0]])): raise ValueError("VL output has no the same points of VL input")

        #Set the constant in numpy array (so everything is in array)
        if(type(TR_output) in [int, float]): TR_output = np.array([TR_output for _ in range(len(TR_inputs_dict[list(TR_inputs_dict.keys())[0]]))])
        if(type(VL_output) in [int, float]): VL_output = np.array([VL_output for _ in range(len(VL_inputs_dict[list(VL_inputs_dict.keys())[0]]))])

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
            while(self.__extension_phase(TR_inputs_dict, VL_inputs_dict, TR_output, VL_output)):
                            
                print(f"Extension phase, iteration {i+1} done {'-'*50}")
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
            while(self.__pruning_phase(TR_inputs_dict, VL_inputs_dict, TR_output, VL_output)): 
                
                print(f"Pruning phase, iteration {i+1} done {'-'*50}")
                self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
                self.checkpoint_metrics()
                i += 1
                
            self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
                
            if(r != i): self._ranges_iterations.append({"period": "pruning", "range": (r, i)})
            
        print("Training done!")
        
        #Last check
        self._model.get_root().clear_all_subtree_memory("forward").clear_all_subtree_memory("backward")
        
        print("Model prepared to use!")
        
        
        