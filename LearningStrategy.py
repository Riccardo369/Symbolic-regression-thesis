import numpy as np
import matplotlib.pyplot as plt

from Model import SymbolicRegression
import Operations as op
import Errors as er
import Indicators as In
import Exceptions as ex
import utils






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
            #print(key, ":", result)
        
    def _body_learning(self, TR_inputs_dict, TR_output, stop_criteria):
        return NotImplemented
        
    def __call__(self, model, TR_inputs_dict, TR_output, stop_criteria, metric_functions = {}):
        self._pre_learning(model, metric_functions, TR_inputs_dict, TR_output)
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

class AnalysisRamification(SymbolicRegressionLearning):
    
    #The order of binary operations is the order of construction every time a node is extended (every node will be extended with all binary operations, and the order is defined by
    # the order of binary_operations list)
    
    #get_package_importance_nodes analyze nodes and returns a list of packages (for importance), where every package contains a list of triples node - feature- operation. 
    # If it is possible to choose a valid extension with first package, the algorithm will stop, otherwise it will search the extension on next package
    def __init__(self, unary_operations, get_package_importance_nodes):
        
        super().__init__()
        
        for oper in unary_operations:
            if(not issubclass(oper, op.OperationNode)): raise ValueError(f"The operation {oper} is not a OperationNode type")
            if(oper().get_max_channels() != 1): raise TypeError(f"{oper} has no 1 channel, it is not acceptable an unary operation like that")
        
        self._unary_operations = list(unary_operations)
        
        self.__get_package_importance_nodes = get_package_importance_nodes
        
        self._ranges_iterations = []
        
        #Every values of dict = (index of package, index of extension to start inside the package)
        self._dict_extension_history = dict()
        
    def __call__(self, model, TR_inputs_dict, TR_output, stop_criteria, metric_functions = {}):
        result = super().__call__(model, TR_inputs_dict, TR_output, stop_criteria, metric_functions)
        result = {"results": result, "ranges_iterations": list(self._ranges_iterations), "history_dicts": {"extension": self._dict_extension_history}}
        
        self._dict_extension_history = dict()
        
        return result
    
    def _pre_learning(self, model, metrics, TR_inputs_dict, TR_output):
        super()._pre_learning(model, metrics, TR_inputs_dict, TR_output)
        self._ranges_iterations = []
       
    #Da vedere se considerare solo il forward, per la gestione delle scale, rumori e dominio (ma con il rischio di non avere abbastanza punti in nessuna parte dell' albero)
    #Oppure tenere così, ma capendo come gestire il dominio e rumore
    def __structural_domain_check(model, TR_inputs_dict, TR_output):
        model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        return np.all(np.isfinite(model.get_root().get_memory("forward"))) & np.all([np.all(np.isfinite(x.get_memory("backward"))) for x in model.get_root().get_all_subtree_nodes() if isinstance(x, op.Leaf)])

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
        valid = AnalysisRamification.__valid_dict_history_form(Dict)
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
    def linear_extend_model(model):
        
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
                
            else: substitution = AnalysisRamification.__add_linear_extension(node, 1, 0)
                
            model.replace_subtree(node, substitution)
            
        recursion(model.get_root())
        
        

    #Get (if possible) the extension (with delta metric) of this combination
    def __analyze_extension(self, TR_inputs_dict, TR_output, config):
          
        
        old_node = config["node"]
        oper_node = config["new node"]

        choices = [oper_node]
        
        if(oper_node.is_constant()): return True

        oper_node.forward_calculation_memory_sub_tree(TR_inputs_dict)
        
        analysis = utils.total_analysis(oper_node.get_memory("forward"), old_node.get_memory("backward"), ["linear", "general"])
        
        if(np.abs(analysis["linear"]) >= np.abs(analysis["general"])):
            
            linear_ext = utils.linear_extension(oper_node.get_memory("forward"), old_node.get_memory("backward"))
            m = linear_ext["m"] if np.isfinite(linear_ext["m"]) else 1
            q = linear_ext["q"] if np.isfinite(linear_ext["q"]) else 0
            
            choices.insert(0, AnalysisRamification.__add_linear_extension(oper_node, m, q))
            
        for try_node in choices:
            
            #Try the substition
            self._model.replace_subtree(old_node, try_node)
  
            #Check the structural domain
            valid_domain = AnalysisRamification.__structural_domain_check(self._model, TR_inputs_dict, TR_output)

            #Do the substition
            self._model.replace_subtree(try_node, old_node)

            config["new node"] = try_node
            
            #Check structure domain
            if(valid_domain): return True
        
        return False

    #Extension phase (tree already wet with TR set)
    def __extension_step(self, TR_inputs_dict, TR_output):
        
        model_firm = str(self._model)
        i = self._dict_extension_history[model_firm] if model_firm in self._dict_extension_history.keys() else 0
        
        choices = []
        
        AnalysisRamification.linear_extend_model(self._model)

        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        
        choices = self.__get_package_importance_nodes(self._model.get_root().get_all_subtree_nodes(), self._unary_operations, TR_inputs_dict)
        
        #print("All remaining choices to test")
        #for x in choices[i:]: print(x)
        
        #Package the choices by importance
        for choice in choices[i:]:

            #Analyze the extension
            acceptable = self.__analyze_extension(TR_inputs_dict, TR_output, choice)

            if(not acceptable):
                i += 1
                continue
                        
            #print("Extension choosed", choice)

            #Do the extension
            self._model.replace_subtree(choice["node"], choice["new node"])
            
            #Go to one step with i index
            if(i < len(choices)): i += 1
            else: return False

            #Save the new position
            self._dict_extension_history[model_firm] = i
                
            break

        return True

    
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
        
        print(f"Start training")
        
        #The search stop if one of this requirements are true:
        #- there is no any step to choose
        #- stop_criteria require to stop
        #- the model give a perfect linear pattern with output (more stronger that chatterjee indicator)
        while((not stop_criteria(self)) and self.__extension_step(TR_inputs_dict, TR_output)):
                
            print(f"Extension phase, iteration {i+1} done {'-'*50}")
   
            self._model.simplify_model()
            self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
            self.checkpoint_metrics()

            i += 1
                    
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        
        self.checkpoint_metrics()
            
        print("Training done!")

