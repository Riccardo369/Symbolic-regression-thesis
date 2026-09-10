import numpy as np
import matplotlib.pyplot as plt

from Model import SymbolicRegression
import Operations as op
import Errors as er
import Indicators as In
import Exceptions as ex
import utils
import StatisticalHeuristics as sh
import Indicators as ind



class SymbolicRegressionLearning:
    
    def __init__(self):
        
        self._model = None
        self._metric_results = {}
        self._metric_functions = []
    
    def _pre_learning(self, model, metrics):
        self._model = model
        self._metric_functions = metrics
        self._metric_results = {key: [] for key in metrics.keys()}
        
    def _post_learning(self):
        pass
        
    def checkpoint_metrics(self):
        for key in self._metric_results:
            result = self._metric_functions[key](self._model)
            self._metric_results[key].append(result)   
            #print(key, ":", result)
        
    def _body_learning(self, TR_inputs_dict, TR_output, stop_criteria):
        return NotImplemented
        
    def __call__(self, model, TR_inputs_dict, TR_output, stop_criteria, metric_functions = {}):
        self._pre_learning(model, metric_functions)
        self._body_learning(TR_inputs_dict, TR_output, stop_criteria)
        self._post_learning()
        
        result = self._metric_results.copy()
        self._metric_results = {}
        return result

    def get_model(self):
        return self._model
    
    def get_metrics(self):
        return self._metric_results.copy()
    
    def set_metrics(self, metrics):
        self._metrics = metrics.copy()

class HeuristicLearning(SymbolicRegressionLearning):
    
    #The order of binary operations is the order of construction every time a node is extended (every node will be extended with all binary operations, and the order is defined by
    # the order of binary_operations list)
    
    #get_package_importance_nodes analyze nodes and returns a list of packages (for importance), where every package contains a list of triples node - feature- operation. 
    # If it is possible to choose a valid extension with first package, the algorithm will stop, otherwise it will search the extension on next package
    def __init__(self):
        
        super().__init__()

        #Every values of dict = (index of package, index of extension to start inside the package)
        self._dict_extension_history = dict()
        
    def __call__(self, model, TR_inputs_dict, TR_output, stop_criteria, metric_functions = {}):
        result = super().__call__(model, TR_inputs_dict, TR_output, stop_criteria, metric_functions)
        result = {"results": result, "history": self._dict_extension_history}
        
        self._dict_extension_history = dict()
        
        return result
    
    def _pre_learning(self, model, metrics):
        super()._pre_learning(model, metrics)
       
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
 
    def get_dict_extension_history(self):
        return self._dict_extension_history.copy()
    
    def set_dict_extension_history(self, Dict):
        valid = HeuristicLearning.__valid_dict_history_form(Dict)
        if(valid): self._dict_extension_history = Dict.copy()
        return valid
   
    #Teletransport to a previous tree
    def teletransport(self, model_firm, TR_inputs_dict, TR_output):
      
        #Every time you do the teletransportaion, the model_firm is a trap where you cannot go further
        self._dict_extension_history[model_firm] = None
        
        #Calculate all possible transition valid transition, and sorted by value
        possible_transportations = sorted([{"model": x, "value": self._dict_extension_history[x][1]} for x in self._dict_extension_history.keys() if self._dict_extension_history[x] != None], key = lambda x: -x["value"])
        
        #If no transport is possible, the algorithm finish
        if(len(possible_transportations) == 0): return False
        
        old_model = str(self._model)

        #Go to the most promising state. The risk is to fall to a next state that is not the best considering the all next state from memory,
        #but we cannot know (otherwise the computational cost will be more heavy)
        self._model.set_root(SymbolicRegression.string_to_tree(possible_transportations[0]["model"]).get_root())
        
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict)
        
        print(f"Teletransport: {old_model} -> {self._model}")
        
        self._previous_MDL = None
        self._actual_MDL = None
        
        self.__calculate_MDL(TR_inputs_dict, TR_output)
        
        #Return the signal that algorithm must repeat the iteration
        return None
            
    #Calculate all statistical heuristic considering only the operations that has no any problem with domain 
    def valid_heuristics_analysis(self, X, Y, X_name = None):
        coefficients = sh.all_heuristic_statisticals(X, Y, X_name)
        return [x for x in coefficients if getattr(op, x['operation']).check_input_domain(0, X)]


    def _standardize_signal(self, val, target_len):
        if(isinstance(val, (int, float, np.number))): return np.full(target_len, float(val))
        val_arr = np.asarray(val)
        if val_arr.ndim == 0: return np.full(target_len, float(val_arr))
        return val_arr
      
      
      
    def __scaling_coefficients(self, coefficients, scaling_coeff):
        for c in coefficients: c["value"] *= scaling_coeff
   
    
  
    def __base_analysis(self, node, TR_inputs_dict, target_len):
        
        Y = self._standardize_signal(node.get_memory("backward"), target_len)
        
        mask = np.isfinite(Y)
        
        Y = Y[mask]
        
        results = []
        
        results.append({"variable": None, "operation": "constant", "value": 1 / (1 + np.std(Y)), "new constant": np.mean(Y)})
        
        name_var = node.get_name()
        
        for f in TR_inputs_dict.keys():
            results.append({"variable": f if f != name_var else None, "operation": "linear", "value": np.abs(ind.pearson_correlation(self._standardize_signal(TR_inputs_dict[f], target_len)[mask], Y))})

        if(not isinstance(node, op.Leaf)): results.append({"variable": None, "operation": "linear", "value": np.abs(ind.pearson_correlation(self._standardize_signal(node.get_memory("forward"), target_len)[mask], Y))})
        
        self.__scaling_coefficients(results, np.sum(mask) / target_len)
        
        return results

    def __node_non_linear_analysis(self, node, X, target_len, name):
        
        X = self._standardize_signal(X, target_len)
        
        father = node.get_father()
        father_non_invertible = father != None and not father.is_invertible()  
        
        if(father_non_invertible):
            
            results = []
                    
            Y_father = self._standardize_signal(father.get_memory("backward"), target_len)
                                                
            mask = np.isfinite(Y_father)
                                                
            #Calculate for all operations (all unary operations are needed)
            for uo in ["sin", "cos", "exp", "ln", "sqrt", "div"]:
                                                    
                #If this operation has problems alreay with domain, will create problem to tree all, so it is not a good operation
                if(not getattr(op, uo).check_input_domain(0, X)): continue
                                                    
                g = self._standardize_signal(getattr(op, uo).forward_calculation([X]), target_len)
                          
                #Calculate the linear correlation between g = ou(X) and pseudo inverse of g from Y (and apply the scaling, because it is a particular situation, the Y is not the node Y but the father Y)
                results.append({"variable": name, "operation": uo, "value": np.abs(ind.pearson_correlation(g[mask], getattr(op, uo).backward_calculation([g[mask]], Y_father[mask])[0]))})
        
        else:
            
            Y = self._standardize_signal(node.get_memory("backward"), target_len)
            
            mask = np.isfinite(Y)
            results = self.valid_heuristics_analysis(X, Y, name)
            
        self.__scaling_coefficients(results, np.sum(mask) / target_len)
            
        return results

    def __leaf_non_linear_analysis(self, node, TR_inputs_dict, target_len):
    
        Y = self._standardize_signal(node.get_memory("backward"), target_len)
        
        mask = np.isfinite(Y)
        
        variable_choices = [{"var": x, "score": np.abs(ind.chatterjee_correlation(self._standardize_signal(TR_inputs_dict[x], target_len)[mask], Y))} for x in TR_inputs_dict.keys()]
        var_choosed = sorted(variable_choices, key = lambda x: -x["score"])[0]["var"]
        
        return self.__node_non_linear_analysis(node, TR_inputs_dict[var_choosed], target_len, var_choosed)
   
    def __horizontal_complete_analysis(self, node, TR_inputs_dict, target_len, chain_type):
        
        real_node = op.Leaf(0)

        real_node.set_memory("backward", chain_type.backward_calculation([self._standardize_signal(node.get_memory("forward"), target_len)], self._standardize_signal(node.get_memory("backward"), target_len))[0])
        
        results = []
        
        results += self.__leaf_non_linear_analysis(node, TR_inputs_dict, target_len)
        results += self.__base_analysis(node, TR_inputs_dict, target_len)
        
        for c in results: c["horizontal extension"] = chain_type
        
        return results
    

    def heuristics_analysis(self, TR_inputs_dict, TR_output):
        
        target_len = len(TR_output)
        
        choices = []
  
        def all_analysis(node, choices):

            coefficients = []

            #Add constant and linear coefficients about itself (cost 1 + var)
            coefficients += self.__base_analysis(node, TR_inputs_dict, target_len)
            
            #For unary and chain operations  (cost: oper)
            if(not isinstance(node, op.Leaf)):
                
                coefficients += self.__node_non_linear_analysis(node, node.get_memory("forward"), target_len, None)
                
                X = self._standardize_signal(node.get_memory("forward"), target_len)
                Y = self._standardize_signal(node.get_memory("backward"), target_len) 
                mask = np.isfinite(Y)
                
                coefficients.append({"variable": None, "operation": "linear", "value": np.abs(ind.pearson_correlation(X[mask], Y[mask])) * (np.sum(mask) / target_len)})      

            #For leaf node (variable and constant)  (cost: oper)     
            else: coefficients += self.__leaf_non_linear_analysis(node, TR_inputs_dict, target_len)
            
            #The chain operation (cost: oper)
            if(isinstance(node, op.ChainOperationNode)):
                
                Y = node.get_memory("backward")
                mask = np.isfinite(Y)
                
                #Choose the most important features with FOCI algorithm
                matrix_features = np.array([self._standardize_signal(x.get_memory("forward"), target_len)[mask] for x in node.get_childrens()]).T
                important_childrens = ind.most_important_features(matrix_features, self._standardize_signal(Y, target_len)[mask])
                
                #Apply this addition only if there are 2 or more variables and important chilrens is at least 1
                if(np.sum([x.is_constant() for x in node.get_childrens()]) > 1 and len(important_childrens) > 0):
                
                    #Divide the important childrens to not important childrens
                    childrens = []
                    other_childrens = []
                    
                    for i, c in enumerate(node.get_childrens()):
                        
                        if(i in important_childrens): childrens.append(c)
                        else: other_childrens.append(c)

                    #Calculate the total input of important nodes and other nodes
                    total_input = node.__class__.forward_calculation([x.get_memory("forward") for x in childrens])
                    total_other_input = node.__class__.forward_calculation([x.get_memory("forward") for x in other_childrens])
                    
                    #Calculate the two backwards, considering most important nodes like one unit node, and other nodes like another unit under add chain
                    l = op.Leaf(0)
                    l.set_memory("backward", node.__class__.backward_calculation([total_input, total_other_input], Y)[0])

                    new_coefficients = self.__node_non_linear_analysis(l, total_input, target_len, None)
                    
                    #Calculate all coefficients considering only the total input of important node and backward of important node
                    for c in new_coefficients: c["childrens"] = important_childrens
                    
                    coefficients += new_coefficients
                    
            father = node.get_father()
                
            if(not isinstance(father, op.add)): coefficients += self.__horizontal_complete_analysis(node, TR_inputs_dict, target_len, op.add)
            if(not isinstance(father, op.mul)): coefficients += self.__horizontal_complete_analysis(node, TR_inputs_dict, target_len, op.mul)
                
            
            #Sort and filter the coefficients considering only the valid number and not 0 coefficients
            coefficients = sorted([x for x in coefficients if (np.isfinite(x["value"]) and x["value"] >= 1e-20)], key = lambda x: -x["value"])
            

            #Scale the coefficients considering ow many data is aviable in this node
            for coeff in coefficients: coeff["node"] = node
                
            if(len(coefficients) == 0): return False
            
            #Check if the winner is the linearity of node
            first_place = coefficients[0]
            
            #If own linearity wins, no coefficient must be considered (because in that node, it is all perfect)
            local_linearity = False
            
            if(first_place["variable"] == None and ((first_place["operation"] == "linear" and not node.is_constant()) or 
                                                    (first_place["operation"] == "constant" and node.is_constant()))): local_linearity = True
            
            choices += coefficients
            
            return local_linearity
        
        
        linearity = np.all([all_analysis(x, choices) for x in self._model.get_root().get_all_subtree_nodes()])
        
        choices = sorted(choices, key = lambda x: -x["value"])
        
        #print("CHOICES")
        #for i in choices: print(i)

        return {"choices": choices, "linearity": linearity}
    
    

    #Extension phase (tree already wet with TR set)
    def __extension_step(self, TR_inputs_dict, TR_output):
        
        print(self._model)
        print(self._dict_extension_history)
        
        #Calculate the firm
        model_firm = str(self._model)
        
        if(model_firm in self._dict_extension_history.keys() and 
           self._dict_extension_history[model_firm] == None):
            print("Algorithm falled in dangerous zone")
            return self.teletransport(model_firm, TR_inputs_dict, TR_output)
        
        i = self._dict_extension_history[model_firm][0] if model_firm in self._dict_extension_history.keys() else 0
        
        #If this new state is worse of previous state, apply teletransport
        if(self._previous_MDL != None and self._actual_MDL != None and self._previous_MDL < self._actual_MDL):
            
            print(f"Non accepted by MDL (previous {self._previous_MDL} < actual {self._actual_MDL})")
            result = self.teletransport(model_firm, TR_inputs_dict, TR_output)
            del self._dict_extension_history[model_firm]
            return result
            
        #FORWARD - BACKWARD STEPS
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        
        #CONTROL TO TELETRANSPORT
        if(np.any(~np.isfinite(self._model.get_root().get_memory("forward")))):
            print("Problem with forward")
            return self.teletransport(model_firm, TR_inputs_dict, TR_output)

        #STATISTICAL SCORES STEP
        competition = self.heuristics_analysis(TR_inputs_dict, TR_output)
        
        #If all nodes achieve the linearity situation, the algorithm stops
        if(competition["linearity"]):
            print("ALL LINEARITY")
            return False
        
        choices = competition["choices"]
        
        #CONTROL TO TELETRANSPORT
        if(choices == []):
            print("No choice aviable")
            return self.teletransport(model_firm, TR_inputs_dict, TR_output)
        
        #Delete the linearity analysis, for saving time and avoid wasting of time.
        #Apply a linear extension with same domain is equivalent to do nothing, so for this reason, all linear extension are deleted             
        choices = [x for x in choices if not (x['variable'] == None and x['operation'] == 'linear')]
        
        #EXTENSION STEP
        if(i < len(choices)):

            choice = choices[i]
            
            print("Extension choosed", choice)
        
            #Create the base
            
            #If it is required a constant
            if(choice["operation"] == "constant"): sub_domain = op.Leaf(choice["new constant"])
            
            #If it is required a sub set of a chain
            elif("childrens" in choice.keys()): 
                
                childrens = choice["node"].get_childrens() 
                sub_domain = choice["node"].__class__()
                for c in choice["childrens"]: sub_domain.new_connection(childrens[c].copy_sub_tree())
                
            #If it is required a variable
            else: sub_domain = op.Leaf(choice["variable"]) if choice["variable"] != None else choice["node"].copy_sub_tree()
            
            #Add the operation that must put up to base
            if(choice["operation"] not in ["linear", "constant"]):
                new_node = getattr(op, choice['operation'])()
                new_node.new_connection(sub_domain)
                
            else: new_node = sub_domain
            
            
                
            #Sub group chain (solo questa non ha la constant e la linear)
            if("childrens" in choice.keys()):
                
                up_chain = choice["node"].__class__()
                
                up_chain.new_connection(new_node)
                for c in range(len(childrens)):
                    if(c not in choice["childrens"]): up_chain.new_connection(childrens[c].copy_sub_tree())
                    
                new_node = up_chain
                
            #Horizontal extension
            elif("horizontal extension" in choice.keys()): 
                
                up_chain = choice["horizontal extension"]()
                
                up_chain.new_connection(new_node)
                up_chain.new_connection(choice["node"].copy_sub_tree())
                
                new_node = up_chain
                    
            #Standard situation
            else: pass
            
            self._model.replace_subtree(choice["node"], new_node)        
                    
            #SYNTAX REDUCTION STEP
            self._model.simplify_model()
            
            self._dict_extension_history[model_firm] = (i+1, choice["value"])
            

            return True
            
        #The chance are finished, so this state is dead
        else:
            print("Choices finished")
            return self.teletransport(model_firm, TR_inputs_dict, TR_output)



    def __information_cost_structure(self, k):
        
        def recursion(node):
            
            subtree_cost = np.sum([recursion(c) for c in node.get_childrens()])
            
            # Unary operation: sin, cos, arcsin, arccos, exp, ln, sqrt, div
            if isinstance(node, op.UnaryOperationNode): node_cost = np.log2(8)
                
            # Chain operation: add, mul
            elif isinstance(node, op.ChainOperationNode): node_cost = np.log2(2) + np.log2(len(node.get_childrens()))
                
            # Constant
            elif node.is_constant(): node_cost = 1.0
                
            # Variable
            else: node_cost = 1.0 + np.log2(k)
                
            return subtree_cost + node_cost
        
        return recursion(self._model.get_root())
            
            
    
    def __calculate_MDL(self, TR_inputs_dict, TR_output):
        
        leafs = self._model.get_leafs()

        n = len(TR_output)
        k = len(TR_inputs_dict.keys())

        L_struct = self.__information_cost_structure(k)
        L_const = len([x for x in leafs if x.is_constant()]) * 0.5 * np.log2(n)
        L_res = n * 0.5 * np.log2(er.MSE(self._model.get_root().get_memory("forward"), TR_output) + 1e-40)
        
        print("MDL error: ", L_struct, L_const, L_res)
        
        MDL_score = L_struct + L_const + L_res
        MDL_score = MDL_score if np.isfinite(MDL_score) else float("inf")
        
        self._previous_MDL = self._actual_MDL
        self._actual_MDL = MDL_score
        
     
       
        
    
    def __refresh(self, TR_inputs_dict, TR_output):
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        self.checkpoint_metrics()
        self.__calculate_MDL(TR_inputs_dict, TR_output)
        
        
           
 
 
 
    
    #TR_inputs_dict: numpy array's dict
    #VL_inputs_dict: numpy array's dict
    #TR_output: constant or numpy array
    #VL_output: constant or numpy array
    def _body_learning(self, TR_inputs_dict, TR_output, stop_criteria):
        
        self._previuous_MDL = None
        self._actual_MDL   = None
        
        len_inputs = [len(TR_inputs_dict[x]) for x in TR_inputs_dict.keys()]
        
        if(np.var(len_inputs) > 0): raise ValueError("All TR features has to have the same value numbers")

        if(len(len_inputs) == 0 and np.var(TR_output) >= 1e-15): raise ValueError("TR inputs are not exists and output is not a constant")
        elif(type(TR_output) not in [int, float] and len(TR_output) != np.mean(len_inputs)): raise ValueError("TR output has no the same points of TR input")

        #Set the constant in numpy array (so everything is in array)
        if(type(TR_output) in [int, float]): TR_output = np.array([TR_output for _ in range(np.var(len_inputs))])

        
        self.__refresh(TR_inputs_dict, TR_output)
        
        i = 0
        
        #The search stop if one of this requirements are true:
        #- there is no any step to choose
        #- stop_criteria require to stop
        #- the model give a perfect linear pattern with output (more stronger that chatterjee indicator)
        while((not stop_criteria(self))):
            
            #Go can be:
            #None = Teletransport
            #True = Go with next iteration
            #False = Stop the learning
 
            Go = None
            while(Go == None): Go = self.__extension_step(TR_inputs_dict, TR_output)
                
            print(f"Extension phase, iteration {i+1} done {'-'*50}")
   
            self.__refresh(TR_inputs_dict, TR_output)

            i += 1
            
            if(not Go): break

    
        print("Training done!")





#SISTEMARE LA SITUAZIONE DOVE QUANDO LA MDL TI DICE DI NON ANDARE IN UNA CERTA ZONA, NON RENDI QUELLA ZONA NONE (perchè potrebbe essere valida in futuro)
#ma vai semplicemente avanti