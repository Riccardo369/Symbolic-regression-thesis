import numpy as np

import Operations as op
import Errors as er
import StatisticalHeuristics as sh
import Indicators as ind
from Model import SymbolicRegression


class GreedyStatisticalMapping:
    
    def __init__(self):
        
        self._model = None
        self._metric_results = {}
        self._metric_functions = []

        #Every values of dict = (index of package, index of extension to start inside the package)
        self._dict_extension_history = dict()

    #Action where save every metrics
    def checkpoint_metrics(self):
        for key in self._metric_results:
            result = self._metric_functions[key](self._model)
            self._metric_results[key].append(result)
            #print(key, result)
        
    #Operation before training phase
    def _pre_learning(self, model, metrics):
        self._model = model
        self._metric_functions = metrics
        self._metric_results = {key: [] for key in metrics.keys()}
    
    
    def __call__(self, model, TR_inputs_dict, TR_output, stop_criteria, metric_functions = {}):
        
        self._pre_learning(model, metric_functions)
        self._body_learning(TR_inputs_dict, TR_output, stop_criteria)
        
        result = self._metric_results.copy()
        self._metric_results = {}
        
        result = {"results": result, "history": self._dict_extension_history}
        
        return result
    
    def get_model(self):
        return self._model
        
    def get_metrics(self):
        return self._metric_results.copy()
        
    def set_metrics(self, metrics):
        self._metrics = metrics.copy()
       
    def __valid_dict_history_form(Dict):
        if(not all([type(x) == str for x in Dict.keys()])): return False
        if(not all([type(x) in [tuple, list] and 
                    len(x) == 2 and 
                    type(x[0]) == int and 
                    type(x[1]) in [int, float] for x in Dict.values()])): return False

        for key in Dict.keys():
            try: SymbolicRegression.string_to_tree(key)
            except: return False

        return True

    def get_dict_extension_history(self):
        return self._dict_extension_history.copy()
    
    def set_dict_extension_history(self, Dict):
        valid = GreedyStatisticalMapping.__valid_dict_history_form(Dict)
        if(valid): self._dict_extension_history = Dict.copy()
        return valid

    #rollback to a previous tree
    def rollback(self, model_firm, TR_inputs_dict, TR_output): 
      
        #Every time you do the rollbackaion, the model_firm is a trap where you cannot go further
        self._dict_extension_history[model_firm] = None
        
        #Calculate all possible transition valid transition, and sorted by value
        possible_transportations = sorted([{"model": x, "score": self._dict_extension_history[x][1]} for x in self._dict_extension_history.keys() if self._dict_extension_history[x] != None], key = lambda x: -x["score"])
        
        #If no transport is possible, the algorithm finish
        if(len(possible_transportations) == 0): return False

        #Go to the most promising state. The risk is to fall to a next state that is not the best considering the all next state from memory,
        #but we cannot know (otherwise the computational cost will be more heavy)
        self._model.set_root(SymbolicRegression.string_to_tree(possible_transportations[0]["model"]).get_root())
        
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict, len(TR_output))

        self._previous_MDL = None
        self._actual_MDL = None

        self.__calculate_MDL(TR_inputs_dict, TR_output)
        
        #Return the signal that algorithm must repeat the iteration
        return None



    #Calculate all statistical heuristic considering only the operations that has no any problem with domain 
    def valid_heuristics_analysis(self, X, Y, X_name = None):
        coefficients = sh.all_heuristic_statisticals(X, Y, X_name)
        return [x for x in coefficients if getattr(op, x['unary operation']).check_input_domain(0, X)]    
      
    #Operation of scaling coefficient
    def __scaling_coefficients(self, coefficients, scaling_coeff):
        for c in coefficients: c["score"] *= scaling_coeff
   
    #Base analysis: statistical analysis atomic operation
    def __base_analysis(self, node, TR_inputs_dict, target_len):
        
        #Extract output
        Y = node.get_memory("backward")
        
        name = node.get_name()
        
        mask = np.isfinite(Y)
        
        Y = Y[mask]
        
        results = []
        
        #Candidate modification (constant replacing)
        results.append({"modification": "replace",
                        "leaf": np.mean(Y),
                        "score": 1 / (1 + np.std(Y))})
        
        #Candidates modifications (for each variable replacing)
        for f in TR_inputs_dict.keys():
            
            #If the node is a variable f, do not consider the modification (it would be a pointless modification; the tree will not be changed)
            if(f == name): continue
            
            results.append({"modification": "replace",
                            "leaf": f,
                            "score": np.abs(ind.pearson_correlation(TR_inputs_dict[f][mask], Y))})

        self.__scaling_coefficients(results, np.sum(mask) / target_len)
        
        return results

    #Node non linear analysis: statistical analysis atomic operation
    def __node_non_linear_analysis(self, node, X, target_len, name):
        
        #Extract node father
        father = node.get_father()
        father_non_invertible = father != None and not father.is_invertible()  
        
        #Non invertible father
        if(father_non_invertible):
            
            results = []

            Y_father = father.get_memory("backward")

            mask = np.isfinite(Y_father)

            #Calculate for all operations (all unary operations are needed)
            for uo in ["sin", "cos", "exp", "ln", "sqrt", "div"]:
                
                real_uo = getattr(op, uo)
                                                    
                #If this operation has problems alreay with domain, will create problem to tree all, so it is not a good operation
                if(not real_uo.check_input_domain(0, X)): continue
                                                    
                g = real_uo.forward_calculation([X])
                          
                #Calculate the linear correlation between g = ou(X) and pseudo inverse of g from Y (and apply the scaling, because it is a particular situation, the Y is not the node Y but the father Y)
                results.append({"variable": name,
                                "modification": "hat",
                                "unary operation": real_uo,
                                "score": np.abs(ind.pearson_correlation(g[mask], real_uo.backward_calculation([g[mask]], Y_father[mask])[0]))})
        
        #Invertible father
        else:
            
            Y = node.get_memory("backward")
            
            mask = np.isfinite(Y)
            results = self.valid_heuristics_analysis(X, Y, name)
            
            for x in results:
                x["modification"] = "hat"
                x["unary operation"] = getattr(op, x["unary operation"])
            
        self.__scaling_coefficients(results, np.sum(mask) / target_len)
        
        return results

    #Leaf non linear analysis: statistical analysis atomic operation
    def __leaf_non_linear_analysis(self, node, TR_inputs_dict, target_len):
    
        Y = node.get_memory("backward")
        
        mask = np.isfinite(Y)
        
        #Choose the best variable with absolute value chatterjee correlation
        variable_choices = [{"var": x, "score": np.abs(ind.chatterjee_correlation(TR_inputs_dict[x][mask], Y))} for x in TR_inputs_dict.keys()]
        var_choosed = sorted(variable_choices, key = lambda x: -x["score"])[0]["var"]
        
        #Apply Node non linear analysis
        return self.__node_non_linear_analysis(node, TR_inputs_dict[var_choosed], target_len, var_choosed)

    #Horizontal complete analysis: statistical analysis atomic operation
    def __horizontal_complete_analysis(self, node, TR_inputs_dict, target_len, chain_type):
        
        #Create a puppet node
        real_node = op.Leaf(0)
        
        #Set forward and backward vector values for analysis
        real_node.set_memory("forward", 0, target_len)
        real_node.set_memory("backward", chain_type.backward_calculation([node.get_memory("forward")], node.get_memory("backward"))[0], target_len)
        
        results = []
        
        #Apply leaf non linear and base analysis
        results += self.__leaf_non_linear_analysis(real_node, TR_inputs_dict, target_len)
        results += self.__base_analysis(real_node, TR_inputs_dict, target_len)
        
        #Add additional labels for each extensions calculated
        for c in results:
            c["modification"] = "horizontal extension"
            c["operation extension"] = chain_type
        
        return results

    #Heuristic analysis step, where from each node, extract every candidates extensions
    def heuristics_analysis(self, TR_inputs_dict, TR_output):
        
        target_len = len(TR_output)
        
        choices = []
  
        def all_analysis(node):

            coefficients = []

            #Add constant and linear coefficients about itself (cost 1 + var)
            coefficients += self.__base_analysis(node, TR_inputs_dict, target_len)
            
            #For unary and chain operations  (cost: oper)
            if(not isinstance(node, op.Leaf)):
                
                coefficients += self.__node_non_linear_analysis(node, node.get_memory("forward"), target_len, None)
                
                X = node.get_memory("forward")
                Y = node.get_memory("backward")
                mask = np.isfinite(Y)

            #For leaf node (variable and constant)  (cost: oper)     
            else: coefficients += self.__leaf_non_linear_analysis(node, TR_inputs_dict, target_len)
            
            #The chain operation (cost: oper)
            if(isinstance(node, op.ChainOperationNode)):
                
                Y = node.get_memory("backward")
                mask = np.isfinite(Y)
                
                #Choose the most important features with FOCI algorithm
                matrix_features = np.array([x.get_memory("forward")[mask] for x in node.get_childrens()]).T
                
                #Extract important childrens
                important_childrens = ind.most_important_features(matrix_features, Y[mask])
                
                #If number of important childrens variabile is at least 1
                if(len(important_childrens) >= 1):
                
                    #Divide the important childrens to not important childrens
                    childrens = []
                    other_childrens = []
                    
                    for i, c in enumerate(node.get_childrens()):
                        
                        if(i in important_childrens): childrens.append(c)
                        else: other_childrens.append(c)
   
                    #Calculate the total input of important nodes and other nodes
                    total_input = node.__class__.forward_calculation([x.get_memory("forward") for x in childrens])
                    total_other_input = node.__class__.forward_calculation([x.get_memory("forward") for x in other_childrens])
 
                    if(type(total_input) in [float, int]):
                        total_input = np.full_like((len(TR_output)), total_input)
                    
                    if(type(total_other_input) in [float, int] or total_other_input.shape == ()):
                        total_other_input = np.full_like(total_input, 0 if type(node) == op.add else 1)
                    
                    #Calculate the two backwards, considering most important nodes like one unit node, and other nodes like another unit under add chain
                    l = op.Leaf(0)
                    l.set_memory("backward", node.__class__.backward_calculation([total_input, total_other_input], Y)[0], target_len)

                    new_coefficients = self.__node_non_linear_analysis(l, total_input, target_len, None)
                    
                    #Calculate all coefficients considering only the total input of important node and backward of important node
                    for c in new_coefficients:
                        del c["variable"]
                        c["modification"] = "sub childrens"
                        c["childrens"] = important_childrens
                    
                    coefficients += new_coefficients
                    
            father = node.get_father()
            
            #Add new extension considering horizontal extension for addition and multiplication
            if(not isinstance(father, op.add)): coefficients += self.__horizontal_complete_analysis(node, TR_inputs_dict, target_len, op.add)
            if(not isinstance(father, op.mul)): coefficients += self.__horizontal_complete_analysis(node, TR_inputs_dict, target_len, op.mul)
            
            #Sort and filter the coefficients considering only the valid number and not 0 coefficients
            coefficients = sorted([x for x in coefficients if (np.isfinite(x["score"]))], key = lambda x: -x["score"])
            
            #Scale the coefficients considering ow many data is aviable in this node
            for coeff in coefficients: coeff["node"] = node
            
            return coefficients
        
        #Extract all candidates considering from nodes
        for x in self._model.get_root().get_all_subtree_nodes(): choices += all_analysis(x)

        return sorted(choices, key = lambda x: -x["score"])
       

    #Extension phase (tree already wet with TR set)
    def __extension_step(self, TR_inputs_dict, TR_output):
        
        target_len = len(TR_output)
        
        #Calculate the firm
        model_firm = str(self._model)
        
        #If this tree is blocked, apply rollback
        if(model_firm in self._dict_extension_history.keys() and 
           self._dict_extension_history[model_firm] == None): return self.rollback(model_firm, TR_inputs_dict, TR_output)
        
        #Take next index aviable modification from this tree (considering the idnex of choice)
        i = self._dict_extension_history[model_firm][0] if model_firm in self._dict_extension_history.keys() else 0
        
        #If this new state is worse of previous state, apply rollback
        if(not (self._previous_MDL is None) and not (self._actual_MDL is None) and (self._previous_MDL < self._actual_MDL)):

            result = self.rollback(model_firm, TR_inputs_dict, TR_output)
            del self._dict_extension_history[model_firm]
            return result

        #FORWARD - BACKWARD STEPS
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict, target_len).backward_calculation_memory_sub_tree(TR_output, target_len)
        
        #CONTROL TO rollback
        if(np.any(~np.isfinite(self._model.get_root().get_memory("forward")))): return self.rollback(model_firm, TR_inputs_dict, TR_output)

        #STATISTICAL SCORES STEP
        choices = self.heuristics_analysis(TR_inputs_dict, TR_output)      
        
        #MODIFICATION STEP
        
        #CONTROL TO ROLLBACK
        if(choices == []): return self.rollback(model_firm, TR_inputs_dict, TR_output)

        #EXTENSION STEP
        if(i >= len(choices)): return self.rollback(model_firm, TR_inputs_dict, TR_output)

        #Choose te next extension
        choice = choices[i]

        modification = choice["modification"]
        node = choice["node"]
        
        #Choose the path for create the new node to substitute
        
        if(modification == "replace"): new_node = op.Leaf(choice["leaf"])
            
        elif(modification == "hat"): 
            
            new_node = choice["unary operation"]()
            new_node.new_connection(node.copy_sub_tree() if choice["variable"] == None else op.Leaf(choice["variable"]))

        elif(modification == "horizontal extension"):
            
            #Add a simple leaf
            if("leaf" in choice.keys()): brother = op.Leaf(choice["leaf"])
            else: 
                
                brother = choice["unary operation"]()            
                brother.new_connection(op.Leaf(choice["variable"]))
                
            new_node = choice["operation extension"]()
            new_node.new_connection(brother)
            new_node.new_connection(node.copy_sub_tree())
        
        elif(modification == "sub childrens"): 
            
            new_node = choice["node"].__class__()
            
            sub_set = []
            others = []
            
            important_childrens = choice["childrens"]

            for a, c in enumerate(node.get_childrens()):
                                   
                if(a in important_childrens): sub_set.append(c)
                else: others.append(c)
                
            sub_set_child = choice["unary operation"]()
            chain_sub_set_child = choice["node"].__class__()
            
            sub_set_child.new_connection(chain_sub_set_child)
            for x in sub_set: chain_sub_set_child.new_connection(x)
            
            new_node.new_connection(sub_set_child)
            for x in others: new_node.new_connection(x)
        
        else: raise TypeError(f"Modification type {modification} is not allowed")

        #Update the history dict
        self._dict_extension_history[model_firm] = (i+1, choice["score"])
            
        #Apply the substitution
        self._model.replace_subtree(choice["node"], new_node)        
                    
        #SYNTAX REDUCTION STEP
        self._model.simplify_model(TR_inputs_dict)
            
        return True

    #Structure cost for MDL score
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
    
    #Calculate MDL score, and update the MDL history values
    def __calculate_MDL(self, TR_inputs_dict, TR_output):
        
        leafs = self._model.get_leafs()

        n = len(TR_output)
        k = len(TR_inputs_dict.keys())

        L_struct = self.__information_cost_structure(k)   #Discrete solution / complexity structure
        L_const = len([x for x in leafs if x.is_constant()]) * 0.5 * np.log2(n)   #Continuous solution
        L_res = n * 0.5 * np.log2(er.MSE(self._model.get_root().get_memory("forward"), TR_output) + 1e-40)   #Error model
        
        MDL_score = L_struct + L_const + L_res
        MDL_score = MDL_score if np.isfinite(MDL_score) else float("inf")
        
        self._previous_MDL = self._actual_MDL
        self._actual_MDL = MDL_score
         
    #Apply a set of refreshing actions
    def __refresh(self, TR_inputs_dict, TR_output):
        target_len = len(TR_output)
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict, target_len).backward_calculation_memory_sub_tree(TR_output, target_len)
        self.checkpoint_metrics()
        
        self.__calculate_MDL(TR_inputs_dict, TR_output)
        
         
    #Start body iterations training
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
            
            #None = rollback
            #True = Go with next iteration
            #False = Stop the learning
 
            #While algorithm reach a valid tree
            Go = None
            while(Go is None): Go = self.__extension_step(TR_inputs_dict, TR_output)
                
            self.__refresh(TR_inputs_dict, TR_output)

            i += 1
            
            #print(f"Iteration done {i} {'-'*100}")

            if(not Go): break