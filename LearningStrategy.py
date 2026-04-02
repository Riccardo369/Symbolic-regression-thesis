import numpy as np
import matplotlib.pyplot as plt

from Model import SymbolicRegression
import Operations as op
import Errors as er
import Indicators as In
import Exceptions as ex

#CAPIRE UN ATTIMO COME FARE CON LE OPERAZIONI DI BASE NON IDENTIFICATE IN STRATEGIE:

#DA FARE ANCORA IL DOMINIO: Sistemare tutto il fatto del dominio (e sistemare tutto il fatto dei controlli dei domini in TR e VL)
#Semplicemente, far partire i calcoli, e vedere se in radice forward e foglie backward non ci siano NaN od inf, e vedere se niente
#va in errore, questo perchè: errore in forward, non si può fare per dominio, errore in backward, non si può fare per come è 
# strutturato l' algoritmo. In questo modo l' algoritmo è capace di fare delle variazioni di valori senza fissarsi troppo 
# sulle supposizioni (che possono causare lo scartare delle operazioni che magari erano proprio quelle che servivano).

#COME GESTISCO IL FATTO CHE IL DOMINO POSSA PASSARE PER LA FEATURE IN TR MA NON PER LA STESSA FEATURE IN VL? Considero anche la 
# VL per decidere le feature accettate in un dominio, questo perchè il significato di VL dipende dalla struttura del modello
# (in questo caso, usare la VL solo per vedere il modello migliore non ha senso, non è una NN, e questa symbolic regression ha 
# una model selection spalmata lungo tutto l' algoritmo di training).

class SymbolicRegressionLearning:
    
    def __init__(self):
        
        self._model = None
        self._metric_results = {}
        self._metric_functions = []
    
    def _pre_learning(self, model, metrics, TR_inputs_dict, TR_output):
        self._model = model
        self._metric_functions = metrics
        self._metric_results = {key: [] for key in metrics.keys()}
        
        if(isinstance(self._model.get_root(), op.Leaf)):
            
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
    

    def __get_model_function_enter_node(self, inputs_dict, enter_node):
                   
        #Create name of entering node
        exists_inputs = inputs_dict.keys()
        input_name = "x"
        while(input_name in exists_inputs): input_name += "x"
        
        #Take all parameters
        leaf_names = []
        
        def recursion(node):
            
            #Enter's node, all subtree of node has no importance
            if(node == enter_node):
                name = input_name
                leaf_names.append(name)
                return name
            
            #Leaf node
            elif(issubclass(type(node), op.Leaf)):
                name = str(node)
                if(name not in leaf_names and name in exists_inputs): leaf_names.append(name)
                return name
                    
            #Internal node
            else: return node.get_string_structure([f"({recursion(x)})" for x in node.get_childrens()])
            
        #Create the partial function
        body = recursion(self._model.get_root())
        string_function = f"lambda {', '.join(leaf_names)}: {body}"        
        function = eval(string_function)
        
        #Create the function
        return lambda x: function(**({key: inputs_dict[key] for key in inputs_dict.keys() if key in leaf_names} | {input_name: x}))

    def __double_inputs_leaf_procedure(self, i, choices, oper_node, childrens, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, feature1, feature2, TR_f_error, VL_f_error, actual_TR_error, actual_VL_error):
        
        if(i not in [0, 1]): raise ValueError("i must be 0 or 1")
        
        #If all conditions are respected
        if(oper_node.get_channels() == 2 and all([oper_node.check_input_domain(i, TR_inputs_dict[feature1]),
                                                  oper_node.check_input_domain(i, VL_inputs_dict[feature1]),
                                                  oper_node.check_input_domain(1-i, TR_inputs_dict[feature2]),
                                                  oper_node.check_input_domain(1-i, VL_inputs_dict[feature2])])):
              
            #Set features for the transition
            childrens[i].set_name(feature1)
            childrens[1-i].set_name(feature2)
            
            try:

                TR_output_oper_node = oper_node.forward_calculation_memory_sub_tree(TR_inputs_dict).get_memory("forward")
                VL_output_oper_node = oper_node.forward_calculation_memory_sub_tree(VL_inputs_dict).get_memory("forward")
                
                if(self.__domain_check(VL_inputs_dict, VL_output, VL_inputs_dict, VL_output, oper_node, TR_output_oper_node) and 
                self.__domain_check(TR_inputs_dict, TR_output, TR_inputs_dict, TR_output, oper_node, VL_output_oper_node)):
                    
                    choices.append({"tree": str(oper_node),
                                "output": TR_output_oper_node,
                                "TR_error_delta": actual_TR_error - TR_f_error(TR_output_oper_node),
                                "VL_error_delta": actual_VL_error - VL_f_error(VL_output_oper_node)})
                        
            except ex.ForwardCalculationError: pass
                    
            
     
            #Aggiungere anche la costante come seconda feature se possibile
            mean_ideal_constant = np.mean(oper_node.backward_calculation()[i])
                            
            if(oper_node.check_input_domain(1-i, mean_ideal_constant)):
                
                #Set constant
                childrens[1-i].set_name(mean_ideal_constant)
                childrens[1-i].set_memory("forward", mean_ideal_constant)
                
                try:

                    TR_output_oper_node = oper_node.forward_calculation_memory_sub_tree(TR_inputs_dict).get_memory("forward")
                    VL_output_oper_node = oper_node.forward_calculation_memory_sub_tree(VL_inputs_dict).get_memory("forward")
                    
                    if(self.__domain_check(VL_inputs_dict, VL_output, VL_inputs_dict, VL_output, oper_node, TR_output_oper_node) and 
                        self.__domain_check(TR_inputs_dict, TR_output, TR_inputs_dict, TR_output, oper_node, VL_output_oper_node)):
                            
                        choices.append({"tree": str(oper_node),
                                        "output": TR_output_oper_node,
                                        "TR_error_delta": actual_TR_error - TR_f_error(TR_output_oper_node),
                                        "VL_error_delta": actual_VL_error - VL_f_error(VL_output_oper_node)})
                        
                except ex.ForwardCalculationError: pass
                                  
    def __replace_subtree(self, node1, node2):
        
        #Get the father
        father_leaf = node1.get_father()
                
        #The leaf is the root
        if(father_leaf == None): self._model.set_root(node2)
                    
        else:
                
            i = father_leaf.get_childrens().index(node1)
  
            op.OperationNode.free_connection(father_leaf, node1)
            op.OperationNode.new_connection(father_leaf, node2, i)
      
    #Check if the subtree has no affected to the domain in forward (main tree function) and backward (for learning functionality)    
    def __domain_check(self, original_inputs_dict, original_output, inputs_dict, excepted_output, leaf_start, x):
        
        if(not isinstance(leaf_start, op.Leaf)): raise TypeError("leaf_start must be a leaf")
        
        root = self._model.get_root()

        #A forward calculation considering to insert x in leaf_start leaf like forward memory (and do forward and backward passes)
        def recursion(node):
            for child in node.get_childrens(): recursion(child)
            if(isinstance(node, op.Leaf)):
                if(leaf_start == node): node.set_memory("forward", x)
                else: node.set_memory("forward", inputs_dict[node.get_name()] if type(node.get_name()) == str else node.get_name())
            else: node.set_memory("forward", node.forward_calculation())
                              
        result = True
        
        try:
            
            recursion(root)
            root.backward_calculation_memory_sub_tree(excepted_output)
            
            #Check if every situation is fine
            result = all(map(lambda x: np.all(np.isfinite(x)), [root.get_memory("forward")] + [x.get_memory("backward") for x in self._model.get_leafs()]))
            
        except (ex.ForwardCalculationError, ex.BackwardCalculationError): result = False
        
        #Restore value
        root.forward_calculation_memory_sub_tree(original_inputs_dict).backward_calculation_memory_sub_tree(original_output)
        
        return result
     
    #Return none if it is not possible to do any extension
    def __extend(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, node, TR_f_error, VL_f_error, actual_TR_error, actual_VL_error, operations, feature1):
                
        choices = []

        #For each operations
        for oper in operations:
            
            #For each feature
            for feature2 in TR_inputs_dict.keys():
                
                #Create node
                oper_node = oper()
            
                #Set backward output
                oper_node.set_memory("backward", VL_output)
                
                #Create children
                for i in range(oper_node.get_channels()): op.OperationNode.new_connection(oper_node, op.Leaf(f"l{i}"), i)
                    
                childrens = oper_node.get_childrens()
                
                #1 channel operation
                if(oper_node.get_channels() == 1 and all([oper_node.check_input_domain(0, TR_inputs_dict[feature1]),
                                                          oper_node.check_input_domain(0, VL_inputs_dict[feature1])])):
                    
                    childrens[0].set_name(feature1)
                    
                    try:

                        TR_output_oper_node = oper_node.forward_calculation_memory_sub_tree(TR_inputs_dict).get_memory("forward")
                        VL_output_oper_node = oper_node.forward_calculation_memory_sub_tree(VL_inputs_dict).get_memory("forward")
                        
                        if(self.__domain_check(VL_inputs_dict, VL_output, VL_inputs_dict, VL_output, node, TR_output_oper_node) and 
                            self.__domain_check(TR_inputs_dict, TR_output, TR_inputs_dict, TR_output, node, VL_output_oper_node)):
                            
                                choices.append({"tree": str(oper_node),
                                                "output": TR_output_oper_node,
                                                "TR_error_delta": actual_TR_error - TR_f_error(TR_output_oper_node),
                                                "VL_error_delta": actual_VL_error - VL_f_error(VL_output_oper_node)})
                        
                    except ex.ForwardCalculationError: continue
                    
                    
                
                #2 channels operation
                self.__double_inputs_leaf_procedure(0, choices, oper_node, childrens, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, feature1, feature2, TR_f_error, VL_f_error, actual_TR_error, actual_VL_error)   
                 
                #2 channels operation (other parts if operation is not commutative)
                if(not oper_node.is_commutative()):
                    self.__double_inputs_leaf_procedure(1, choices, oper_node, childrens, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, feature1, feature2, TR_f_error, VL_f_error, actual_TR_error, actual_VL_error)   
                   
                #Otherwise case (error)
                if(oper_node.get_channels() not in [1, 2]): raise TypeError(f"{oper} has no 1 or 2 channels, it is not acceptable for IndipendentRamification learning strategy")

        #In extension, delete all extension that increase TR or VL
        choices = [x for x in choices if x["TR_error_delta"] >= 0 and x["VL_error_delta"] >= 0]
        
        if(len(choices) == 0): return None
        
        #In choice the best extension, choose the extension that minimize the VL and after the TR
        choices = sorted(choices, key = lambda x: (-x["VL_error_delta"], -x["TR_error_delta"]))
        
        #Choose best extension
        return SymbolicRegression.string_to_tree(choices[0]["tree"]).get_root()
    
    def __extension_phase(self, TR_inputs_dict, VL_inputs_dict, TR_output, VL_output):

        choices = []
        
        result_extension = None
        
        #Actual extension to do
        residual_operations = []
        operations_dict = {"linear": [op.add, op.mul, op.div], "monoton": [], "sinusoidal": [op.sin, op.cos]}
   
        #Analyze all leafs

        #For all leafs
        for l in self._model.get_leafs():
                
            #For all features
            for f in TR_inputs_dict.keys():
                    
                #Expectated output
                expectated_output = l.get_memory("backward")
                
                #If expectated_output is a constant
                if(not isinstance(expectated_output, np.ndarray)):
                    expectated_output = np.array([expectated_output for _ in range(len(TR_inputs_dict[list(TR_inputs_dict.keys())[0]]))])
                
                #HEURISTIC LOGIC FOR AVOID TO DO ALL OPERATIONS
                
                #Calculate all indicators
                pearson = In.pearson_correlation(TR_inputs_dict[f], expectated_output)
                spearman = In.spearman_correlation(TR_inputs_dict[f], expectated_output)
                chaterjee = In.chatterjee_correlation(TR_inputs_dict[f], expectated_output)
                
                #Calculate all strategies from indicators for family identification
                choices.append({"leaf": l, "feature": f, "type": "linear",      "value": pearson})
                choices.append({"leaf": l, "feature": f, "type": "monoton",     "value": spearman})
                choices.append({"leaf": l, "feature": f, "type": "sinusoidal",  "value": chaterjee})
                
        
        #Repeat until find the best extension
        while(result_extension == None):
            
            #If choices are finished, exit the extension cycle
            if(len(choices) == 0): break
            
            #Order choices by best value (all indicators fall in [-1, 1] range)
            choices = sorted(choices , key = lambda x: np.clip(abs(x["value"]), 0, 1))
            choice = choices[0]
            
            #Take the operations to try
            operations = operations_dict[choice["type"]]

            #Extract valid domain for this node
            #Filter all on_operations considering domain extracted
            
            
            
            #If it is not possible to do the operation, next with the next best choice
            if(len(operations) == 0):
                del choices[0]
                continue
                
            TR_f_error1 = self.__get_model_function_enter_node(TR_inputs_dict, choice["leaf"])
            VL_f_error1 = self.__get_model_function_enter_node(VL_inputs_dict, choice["leaf"])
                
            TR_f_error = lambda x: er.MAE(TR_f_error1(x), TR_output)
            VL_f_error = lambda x: er.MAE(VL_f_error1(x), VL_output)
          
            #Calculate the best extension   
            result_extension = self.__extend(TR_inputs_dict,
                                           TR_output,
                                           VL_inputs_dict,
                                           VL_output,
                                           choice["leaf"],
                                           TR_f_error,
                                           VL_f_error,
                                           TR_f_error(choice["leaf"].get_memory("forward")),
                                           VL_f_error(choice["leaf"].forward_calculation_memory_sub_tree(VL_inputs_dict).backward_calculation_memory_sub_tree(VL_output).get_memory("forward")),
                                           operations,
                                           choice["feature"])
                   
            del choices[0]
            
            #No extension is possible
            if(result_extension == None): return False

            #Do the extension
            self.__replace_subtree(choice["leaf"], result_extension)
            
            return True
              
    def __pruning_phase(self, TR_inputs_dict, VL_inputs_dict, TR_output, VL_output):
        
        #Take all nodes that can be replaced
        nodes_list = []
        
        #For each node, calculate teh father of every node
        for leaf in self._model.get_leafs():
            father = leaf.get_father()
            if(father != None and father not in nodes_list): nodes_list.append(father)
        
        #Calculate features
        features = TR_inputs_dict.keys()
        
        choices = []
        
        #For every node considered valid for pruning part
        for node in nodes_list:
            
            #Extract domain check about output node
            
            #Calculate TR actual forward from node
            actual_forward_TR = node.forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output).get_memory("forward")
            
            #Calculate a constant
            constant = np.mean(node.get_memory("backward"))
            
            #Calculate VL actual forward from node
            actual_forward_VL = node.forward_calculation_memory_sub_tree(VL_inputs_dict).backward_calculation_memory_sub_tree(VL_output).get_memory("forward")

            #Calculate functions for error calculation
            TR_f_error1 = self.__get_model_function_enter_node(TR_inputs_dict, node)
            VL_f_error1 = self.__get_model_function_enter_node(VL_inputs_dict, node)
                
            TR_f_error = lambda x: er.MAE(TR_f_error1(x), TR_output)
            VL_f_error = lambda x: er.MAE(VL_f_error1(x), VL_output)
            
            actual_TR_error =  TR_f_error(actual_forward_TR)
            actual_VL_error =  VL_f_error(actual_forward_VL)
            
            #For every features
            for f in features:
                
                #Check if feature is valid
                #Se la feature f passa il dominio del nodo (DA IMPLEMENTARE), da vedere se la feature f è valida sia in TR che VL
                if(True):
        
                    choices.append({"node": node,
                                    "feature": f,
                                    "TR_error_delta": actual_TR_error - TR_f_error(TR_inputs_dict[f]),
                                    "VL_error_delta": actual_VL_error - VL_f_error(VL_inputs_dict[f])})

            #Append the constant situation
            choices.append({"node": node,
                            "feature": constant,
                            "TR_error_delta": actual_TR_error - TR_f_error(constant),
                            "VL_error_delta": actual_VL_error - VL_f_error(constant)})
            
        #Filter choices
        choices = [x for x in choices if x["TR_error_delta"] >= 0 and x["VL_error_delta"] >= 0]
        
        #If there is no any valid choices, return false
        if(len(choices) == 0): return False
        
        #In choice the best extension, choose the extension that minimize the VL and after the TR
        choices = sorted(choices, key = lambda x: (-x["VL_error_delta"], -x["TR_error_delta"]))
        
        #Replace the subtree with the feature / constant best find (pruning part)
        self.__replace_subtree(choices["node"], op.Leaf(choices["feature"]))

        return True
    
    def _body_learning(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output):
        
        if(set(TR_inputs_dict.keys()) != set(VL_inputs_dict.keys())): raise ValueError("TR and VL input sets has not same features")
        if(np.var([len(TR_inputs_dict[x]) for x in TR_inputs_dict.keys()]) > 0): raise ValueError("All TR features has to have the same value numbers")
        if(np.var([len(VL_inputs_dict[x]) for x in VL_inputs_dict.keys()]) > 0): raise ValueError("All VL features has to have the same value numbers")
        
        
        
        if(type(TR_output) not in [int, float] and len(TR_output) != len(TR_inputs_dict[list(TR_inputs_dict.keys())[0]])): raise ValueError("TR output has no the same points of TR input")
        if(type(VL_output) not in [int, float] and len(VL_output) != len(VL_inputs_dict[list(VL_inputs_dict.keys())[0]])): raise ValueError("VL output has no the same points of VL input")
        
        
        
        #Calculate all memory
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        
        print(self._model)
        
        #Checkpoint
        self.checkpoint_metrics()
        
        extension_memories = []
        pruning_memories = []
        
        new_extension_memory = None
        new_pruning_memory = None
        
        root = self._model.get_root()

        i = 0
        r = 0
        
        print("Start training")
        
        #In a cycle, bring the model in a equilibrium between underfitting and overfitting, with a zig zag strategy
        while(True):
        
            #If model is the same of last cycle (considering the last time you start the last extension), break
            new_extension_memory = str(self._model)
            if(new_extension_memory in extension_memories): break
            extension_memories.append(new_extension_memory)
            
            r = i
            
            #Do the extension phase (bring the model to overfitting direction)
            while(self.__extension_phase(TR_inputs_dict, VL_inputs_dict, TR_output, VL_output)):
                
                print(f"Extension phase, iteration {i+1} {'-'*50}")
                root.forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
                self.checkpoint_metrics()
                i += 1
                
            if(r != i): self._ranges_iterations.append({"period": "extension", "range": (r, i)})
                   
            r = i
         
            #If model is the same of last cycle (considering the last time you start the last pruning), break
            new_pruning_memory = str(self._model)
            if(new_pruning_memory in pruning_memories): break
            pruning_memories.append(new_pruning_memory)
            
            #Do the pruning phase (bring the model to underfitting direction)
            while(self.__pruning_phase(TR_inputs_dict, VL_inputs_dict, TR_output, VL_output)): 
                
                print(f"Pruning phase, iteration {i+1} {'-'*50}")
                root.forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
                self.checkpoint_metrics()
                i += 1
                
            if(r != i): self._ranges_iterations.append({"period": "pruning", "range": (r, i)})
            
        print("Training done!")
            
        root.clear_all_subtree_memory("forward").clear_all_subtree_memory("backward")
        
        print("Model prepared to use!")
