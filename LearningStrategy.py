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

        self._ranges_iterations = []
        
        #Every values of dict = (index of package, index of extension to start inside the package)
        self._dict_extension_history = dict()
        
    def __call__(self, model, TR_inputs_dict, TR_output, stop_criteria, metric_functions = {}):
        result = super().__call__(model, TR_inputs_dict, TR_output, stop_criteria, metric_functions)
        result = {"results": result, "ranges_iterations": list(self._ranges_iterations), "history_dicts": {"extension": self._dict_extension_history}}
        
        self._dict_extension_history = dict()
        
        return result
    
    def _pre_learning(self, model, metrics):
        super()._pre_learning(model, metrics)
        self._ranges_iterations = []
       
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

    def __add_linear_extension(node, m, q):
        
        Q = op.add()
        M = op.mul()
        
        M.new_connection(node.copy_sub_tree())
        M.new_connection(op.Leaf(m))
        Q.new_connection(M)
        Q.new_connection(op.Leaf(q))
        
        return Q
       
    #If a node is a multiplication chain, add a addition number with 0 (non ha senso aggiungere anche un altra multiplication)
    #If a node is a addition chain, add a multiplication number with 1 (non ha senso aggiungere anche un altra addition)
    #If a node is not a constant, and not a chain (* or +), apply multiplication and addition adding
    def linear_extend_model(self):
 
        def recursion(node):
            
            #Go down
            for c in node.get_childrens(): recursion(c)
            
            #If node is a constant do nothing
            if(node.is_constant()): return 
            
            substitution = None
            
            father = node.get_father()
            
            if(isinstance(father, op.mul) or isinstance(father, op.add)):
                
                if(isinstance(father, op.mul)):
                    
                    substitution = op.add()
                    substitution.new_connection(op.Leaf(0))
                    
                if(isinstance(father, op.add)):
                    
                    substitution = op.mul()
                    substitution.new_connection(op.Leaf(1))
                    
                substitution.new_connection(node.copy_sub_tree())
                
            else: substitution = HeuristicLearning.__add_linear_extension(node, 1, 0)
                
            self._model.replace_subtree(node, substitution)
            
        recursion(self._model.get_root())
     
     
    #Teletransport to a previous tree
    def teletransport(self, model_firm):
         
        #Every time you do the teletransportaion, the model_firm is a trap where you cannot go further
        self._dict_extension_history[model_firm] = None
        
        #Calculate all possible transition valid transition, and sorted by value
        possible_transportations = sorted([{"model": x, "value": self._dict_extension_history[x][1]} for x in self._dict_extension_history.keys() if self._dict_extension_history[x] != None], key = lambda x: -x["value"])
        
        #If no transport is possible, the algorithm finish
        if(len(possible_transportations) == 0): return False
        
        #Go to the most promising state. The risk is to fall to a next state that is not the best considering the all next state from memory,
        #but we cannot know (otherwise the computational cost will be more heavy)
        self._model.set_root(SymbolicRegression.string_to_tree(possible_transportations["model"]).get_root())
        
        #Return the signal that algorithm can go again
        return True
            
    #Calculate all statistical heuristic considering only the operations that has no any problem with domain 
    def valid_heuristics_analysis(self, X, Y, X_name = None):
        coefficients = sh.all_heuristic_statisticals(X, Y, X_name)
        return [x for x in coefficients if getattr(op, x['operation']).check_input_domain(0, X)]


    def _standardize_signal(self, val, target_len):
        if(isinstance(val, (int, float, np.number))): return np.full(target_len, float(val))
        val_arr = np.asarray(val)
        if val_arr.ndim == 0: return np.full(target_len, float(val_arr))
        return val_arr
      
           
    def heuristics_analysis(self, TR_inputs_dict, TR_output):
        
        target_len = len(TR_output)

        choices = []
        
        def recursion(node, choices):
            
            #Calculate if all subtree is linear
            result_linearities = all([recursion(c, choices) for c in node.get_childrens()])
            
            coefficients = []
            
            father = node.get_father()
            
            raw_X = node.get_memory("forward")
            raw_Y = node.get_memory("backward")
                        
            original_X = self._standardize_signal(raw_X, target_len)
            Y = self._standardize_signal(raw_Y, target_len)
                            
            mask = np.isfinite(Y)                
            confidence = np.sum(mask) / len(Y)
            
            X = original_X[mask]
            Y = Y[mask]
            
            #Cost: 2 + var
            
            #Add constant and linear coefficients about itself
            coefficients.append({"variable": None, "operation": "constant", "value": 1 / (1 + np.std(Y)), "new constant": np.mean(Y)})
            coefficients.append({"variable": None, "operation": "linear", "value": np.abs(ind.pearson_correlation(X, Y))})
                  
            
            #Add linear variable coefficients (cost for variable: - 1)
            name = node.get_name() if isinstance(node, op.Leaf) and not node.is_constant() else None
            coefficients += [{"variable": f, "operation": "linear", "value": np.abs(ind.pearson_correlation(TR_inputs_dict[f][mask], Y))} for f in TR_inputs_dict.keys() if f != name]
            
            
            #For unary and chain operations  (cost: oper)
            if(not isinstance(node, op.Leaf)):
                
                if(father == None): coefficients += self.valid_heuristics_analysis(X, Y)
                
                #Invertible operation, apply the heuristics
                elif(father.is_invertible()): coefficients += self.valid_heuristics_analysis(X, Y)
                    
                #Non invertible operation, apply the linear correlation for non invertible situation
                else:
                    
                    real_Y = father.get_memory("backward")
                    real_Y = self._standardize_signal(real_Y, target_len)
                    
                    new_mask = np.isfinite(real_Y)
                    
                    new_coeff = np.sum(new_mask) / len(real_Y)
                    
                    #Calculate for all operations (all unary operations are needed)
                    for uo in [op.sin, op.cos, op.exp, op.ln, op.sqrt, op.div]:
                        
                        #If this operation has problems alreay with domain, will create problem to tree all, so it is not a good operation
                        if(uo.check_input_domain(0, original_X)): continue
                        
                        g = uo.forward_calculation([original_X])

                        #Calculate the linear correlation between g = ou(X) and pseudo inverse of g from Y (and apply the scaling, because it is a particular situation, the Y is not the node Y but the father Y)
                        coefficients.append({"variable": None, "operation": uo.__class__.__name__, "value": ind.pearson_correlation(g[new_mask], uo.backward_calculation([g[new_mask]], real_Y[new_mask])[0])*new_coeff})
             
            #For leaf node (variable and constant)  (cost: oper)     
            else:
                
                #Choose the best variable
                variable_choices = [{"var": x, "score": np.abs(ind.chatterjee_correlation(TR_inputs_dict[x][mask], Y))} for x in TR_inputs_dict.keys()]
                var_choosed = sorted(variable_choices, key = lambda x: -x["score"])[0]["var"]
                
                #Calculate all coefficients with that variable choosed
                coefficients += self.valid_heuristics_analysis(TR_inputs_dict[var_choosed][mask], Y, name)



            #The chain operation (cost: oper)
            if(isinstance(node, op.ChainOperationNode)):
                
                #Choose the most important features with FOCI algorithm
                matrix_features = np.array([self._standardize_signal(x.get_memory("forward"), target_len)[mask] for x in node.get_childrens()]).T
                important_childrens = ind.most_important_features(matrix_features, Y)
                
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
                    backwards = node.__class__.backward_calculation([total_input, total_other_input], Y)
                    
                    new_coefficients = self.valid_heuristics_analysis(self._standardize_signal(total_input, target_len)[mask],
                                                                    self._standardize_signal(backwards[0], target_len)[mask], X_name = np.nan)


                    #Calculate all coefficients considering only the total input of important node and backward of important node
                    for c in new_coefficients: c["childrens"] = important_childrens
                    
                    coefficients += new_coefficients
                
                
            
            
            #Sort the coefficients
            coefficients = sorted([x for x in coefficients], key = lambda x: -x["value"])
            
            #Filter coefficients considering only the valid number and not 0 coefficients
            coefficients = [x for x in coefficients if (np.isfinite(x["value"]) and x["value"] >= 1e-20)]
            
            #Scale the coefficients considering ow many data is aviable in this node
            for coeff in coefficients:
                
                father = node.get_father()
                
                #If father is non invertible, do not apply the scaling, because the scaling is applied already
                coeff["value"] *= 1 if father != None and not father.is_invertible() else confidence
                coeff["node"] = node
                
            if(len(coefficients) == 0): return False
            
            #Check if the winner is the linearity of node
            first_place = coefficients[0]
            
        
            #If own linearity wins, no coefficient must be considered (because in that node, it is all perfect)
            local_linearity = False
            
            if(first_place["variable"] == None and ((first_place["operation"] == "linear" and not node.is_constant()) or 
                                                    (first_place["operation"] == "constant" and node.is_constant()))): local_linearity = True
            
            choices += coefficients
            
            return local_linearity and result_linearities
            
                
               
        linearity = recursion(self._model.get_root(), choices)   
        
        choices = sorted(choices, key = lambda x: -x["value"]) 

        return {"choices": choices, "linearity": linearity}
 

    #Extension phase (tree already wet with TR set)
    def __extension_step(self, TR_inputs_dict, TR_output):
        
        #Calculate the firm
        model_firm = str(self._model)
        i = self._dict_extension_history[model_firm][0] if model_firm in self._dict_extension_history.keys() else 0
        
        #PREPARATION STEP  
        self.linear_extend_model()

        #FORWARD - BACKWARD STEPS
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        
        #CONTROL TO TELETRANSPORT
        if(np.any(~np.isfinite(self._model.get_root().get_memory("forward")))): return self.teletransport(model_firm)

        #STATISTICAL SCORES STEP
        competition = self.heuristics_analysis(TR_inputs_dict, TR_output)
        
        #If all nodes achieve the linearity situation, the algorithm stops
        if(competition["linearity"]): return False
        
        choices = competition["choices"]
        
        #CONTROL TO TELETRANSPORT
        if(choices == []): return self.teletransport(model_firm)
        
        #EXTENSION STEP
        if(i < len(choices)):
        
            choice = choices[i]
            i += 1
            
            node = choice["node"]

            if(choice["operation"] == "constant"): self._model.replace_subtree(choice["node"], op.Leaf(choice["new constant"]))       
            elif(choice["operation"] == "linear"): self._model.replace_subtree(choice["node"], op.Leaf(choice["variable"]))
                            
            #Constant / variable or unary operations or chain operation (up extension)
            elif(isinstance(node, op.Leaf) or 
                 isinstance(node, op.UnaryOperationNode) or 
                 (isinstance(node, op.ChainOperationNode) and "childrens" not in choice.keys())):
                
                base = getattr(op, choice['operation'])()
                
                if(isinstance(node, op.Leaf)): base.new_connection(op.Leaf(choice["variable"]))
                else: base.new_connection(choice["node"].copy_sub_tree())
                
                self._model.replace_subtree(choice["node"], base)
            
            #Chain operation (down extension)
            else: 
                
                childrens = node.get_childrens()
                
                index_important_childrens = choice["childrens"]
                
                root_chain = node.__class__.__name__()
                
                oper = getattr(op, choice['operation'])()

                chain_under_oper = node.__class__.__name__()
                
                root_chain.new_connection(oper)
                                
                for i in range(len(childrens)):
                                    
                    if(i not in index_important_childrens): root_chain.new_connection(childrens[i])
                
                oper.new_connection(chain_under_oper)
                for i in index_important_childrens: chain_under_oper.new_connection(childrens[i].copy_sub_tree())
                
                self._model.replace_subtree(choice["node"], root_chain)

            #SYNTAX REDUCTION STEP
            self._model.simplify_model()
            
            #UPDATE THE MNEMONIC MEMORY (APPLY i and choice score)
            self._dict_extension_history[model_firm] = (i, choice["value"])
            
            return True
            
        #The chance are finished, so this state is dead
        else: return self.teletransport(model_firm)



    
    #TR_inputs_dict: numpy array's dict
    #VL_inputs_dict: numpy array's dict
    #TR_output: constant or numpy array
    #VL_output: constant or numpy array
    def _body_learning(self, TR_inputs_dict, TR_output, stop_criteria):
        
        len_inputs = [len(TR_inputs_dict[x]) for x in TR_inputs_dict.keys()]
        
        if(np.var(len_inputs) > 0): raise ValueError("All TR features has to have the same value numbers")

        if(len(len_inputs) == 0 and np.var(TR_output) >= 1e-15): raise ValueError("TR inputs are not exists and output is not a constant")
        elif(type(TR_output) not in [int, float] and len(TR_output) != np.mean(len_inputs)): raise ValueError("TR output has no the same points of TR input")

        #Set the constant in numpy array (so everything is in array)
        if(type(TR_output) in [int, float]): TR_output = np.array([TR_output for _ in range(np.var(len_inputs))])

        #Calculate all memory
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        
        #Checkpoint
        self.checkpoint_metrics()
        
        i = 0
        
        #The search stop if one of this requirements are true:
        #- there is no any step to choose
        #- stop_criteria require to stop
        #- the model give a perfect linear pattern with output (more stronger that chatterjee indicator)
        while((not stop_criteria(self)) and self.__extension_step(TR_inputs_dict, TR_output)):
                
            print(f"Extension phase, iteration {i+1} done {'-'*50}")
   
            self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
            self.checkpoint_metrics()

            i += 1
                    
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        
        self.checkpoint_metrics()
            
        print("Training done!")

