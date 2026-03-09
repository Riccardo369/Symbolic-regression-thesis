import numpy as np
import matplotlib.pyplot as plt

from Model import SymbolicRegression
import Operations as op
import Errors as er

def projection(ideal, features, check_function):
        const = np.mean(ideal)
        proposals = [{"label": x, "value": features[x]} for x in features.keys()] + [{"label": const, "value": np.repeat(const, ideal.shape[0])}]

        proposals = list(filter(check_function, proposals))
        if(len(proposals) > 0): return sorted(proposals, key = lambda x: np.mean(np.abs(ideal - x["value"])))[0]["label"]
        else: None

def transparency(residuals):
    
    #Durbin watson
    numerator = np.sum(np.diff(residuals)**2)
    denominator = np.sum(residuals**2)
    
    if denominator < 1e-15: return 2.0
    return np.abs((numerator / denominator) - 2)



class SymbolicRegressionLearning:
    
    def __init__(self):
        
        self._model = None
        self._solution = None
        self._leafs = None
        self._metric_results = None  
        self._metric_functions = None
    
    def __pre_learning(self, model, metrics, TR_output):
        self._model = model
        self._leafs = model.get_leafs()
        self._solution = [x.get_name() for x in self._leafs]
        self._metric_functions = metrics
        self._metric_results = {key: [] for key in metrics.keys()}
        
        if(isinstance(self._model.get_root(), op.Leaf)):
            
            print("OLè")
            
            self._model.get_root().set_name(np.mean(TR_output))
            
            print(self._model)
        
    def __post_learning(self):
        pass
        
    def checkpoint_metrics(self):
        for key in self._metric_results:
            self._metric_results[key].append(self._metric_functions[key](self._model))   
        
    def body_learning(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, stop_criteria):
        return NotImplemented
        
    def __call__(self, model, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, stop_criteria, metric_functions = {}):
        self.__pre_learning(model, metric_functions, TR_output)
        self.body_learning(TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, stop_criteria)
        self.__post_learning()
        
        return self._metric_results.copy()
 
 
 




class AnalysisRamification(SymbolicRegressionLearning):
    
    __families = [{"family": "linear", "operations": [op.add, op.sub], "transformation": lambda x: x, "reconstruction": lambda x, a, b: (a * x) + b}]
    
    def __init__(self, operations):
        super().__init__()
        
        for oper in operations:
            if(not issubclass(oper, op.OperationNode)): raise ValueError(f"The operation {oper} is not a OperationNode type")
        
        self._operations = list(operations)
        

    def extend(self, inputs_dict, output, f):
        
        #Choose in a greedy way a family of functions and a feature
        choices = []
        
        for feature in inputs_dict.keys():
            for family in AnalysisRamification.__families:
                
                #Calculate transformation
                s = family["transformation"](inputs_dict[feature])
                
                #Extract compass
                mean_R = np.mean(output)
                mean_s = np.mean(s)
                
                a = np.sum((s - mean_s)*(output - mean_R)) / ((np.sum(s - mean_s)**2)+1e-25)
                b = mean_R - (a * mean_s)
                
                #Calculate the relevance of compasses
                choices.append({"feature": feature, "family": family["family"], "residual": np.mean(np.abs(output - family["reconstruction"](inputs_dict[feature], a, b)))})
                
        #Choose the best combination feature / family operations
        choices = sorted(choices, key = lambda x: x["residual"])

        #CONTROLLO SUL DOMINIO DI OUTPUT, CHE DEVE ESSERE COMPATIBILE CON TUTTO IL RAMO (QUINDI NON SOLO IL DOMINIO DI INPUT DEL PADRE)
        #E FARE UN CICLO SULL ANDARE NELLA PROSSIMA FAMIGLIA SE NON TROVO L' INSIEME DI OPERAZIONI POSSIBILI SU QUESTO CASO

        feature1 = choices[0]["feature"]
        operations = list(filter(lambda x: x["family"] == choices[0]["family"], AnalysisRamification.__families))[0]["operations"]
            
        choices = []

        #For each operations
        for oper in operations:
            
            #For each feature
            for feature2 in inputs_dict.keys():
                
                #Create node
                oper_node = oper()
                
                #Check output
                if(not oper_node.check_output_domain(output)): continue
                
                oper_node.set_memory("backward", output)
                
                #Create children
                op.OperationNode.new_connection(oper_node, op.Leaf(f"l0"), 0)
                op.OperationNode.new_connection(oper_node, op.Leaf(f"l1"), 1)
                    
                childrens = oper_node.get_childrens()

                #1 channel operation
                if(all([oper_node.get_channels() == 1,
                        oper_node.check_input_domain(0, inputs_dict[feature1])])):
                    
                    childrens[0].set_name(feature1)
                    childrens[0].set_forward_calculation_memory(inputs_dict[feature1])
                    
                    choices.append({"tree": str(oper_node), "output": oper_node.forward_calculation(), "error": f(oper_node.forward_calculation())})
                    
                #2 channels operation
                elif(all([oper_node.get_channels() == 2,
                          oper_node.check_input_domain(0, inputs_dict[feature1]),
                          oper_node.check_input_domain(1, inputs_dict[feature2])])):
                    
                    childrens[0].set_name(feature1)
                    childrens[0].set_memory("forward", inputs_dict[feature1])
                    
                    childrens[1].set_name(feature2)
                    childrens[1].set_memory("forward", inputs_dict[feature2])
                    
                    choices.append({"tree": str(oper_node), "output": oper_node.forward_calculation(), "error": f(oper_node.forward_calculation())})
                    
                    #Aggiungere anche la costante come seconda feature se possibile
                    mean_ideal_constant = np.mean(oper_node.backward_calculation()[0])
                    
                    if(oper_node.check_input_domain(1, mean_ideal_constant)):
                        
                        childrens[1].set_name(mean_ideal_constant)
                        childrens[1].set_memory("forward", mean_ideal_constant)
                        
                        choices.append({"tree": str(oper_node), "output": oper_node.forward_calculation(), "error": f(oper_node.forward_calculation())})
                        
                #Otherwise case (error)
                elif(oper_node.get_channels() not in [1, 2]): raise TypeError(f"{oper} has no 1 or 2 channels, it is not acceptable for IndipendentRamification learning strategy")

        #Choose best extension
        choice = SymbolicRegression.string_to_tree(sorted(choices, key = lambda x: x["error"])[0]["tree"]).get_root()

        return choice

        
    def body_learning(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, stop_criteria):
        
        if(not TR_inputs_dict.keys() == set(VL_inputs_dict.keys())): raise ValueError("TR and VL input sets has not same features")
        
        #Calculate all memory
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        
        #Checkpoint
        self.checkpoint_metrics()
        
        #Calculate first candidates
        working_leafs = self._model.get_leafs()

        #While cycle
        while(len(working_leafs) > 0 and not stop_criteria(self._model, self._metric_results)):

            #Choose the leaf with best transparency
            leaf = sorted([{"leaf": x, "transparency": transparency(x.get_memory("backward"))} for x in working_leafs], key = lambda x: x["transparency"])[0]

            #If the best leaf are not enough transparence, no leaf is automatically enough
            if(leaf["transparency"] <= 1): break
            
            def f(x):
                
                def recursion_forward(node):
                    for child in node.get_childrens(): recursion_forward(child)
                    if(type(node) != op.Leaf):
                        node.set_memory("forward", node.forward_calculation())
                
                leaf["leaf"].set_memory("forward", x)
                
                recursion_forward(self._model.get_root())
                
                Y_atteso = TR_output
                Y_ottenuto = self._model.get_root().get_memory("forward")
                
                return np.mean(np.abs(Y_atteso - Y_ottenuto))
            
            #Calculate extension
            extension = self.extend(TR_inputs_dict, leaf["leaf"].get_memory("backward"), f)

            #Do the extension
            
            #Get the father
            father_leaf = leaf["leaf"].get_father()
                
            #The leaf is the root
            if(father_leaf == None): self._model.set_root(extension)
                    
            else:
                
                i = father_leaf.get_childrens().index(leaf["leaf"])
  
                op.OperationNode.free_connection(father_leaf, leaf["leaf"])
                op.OperationNode.new_connection(father_leaf, extension, i)
            
            working_leafs = [l for l in working_leafs if l != leaf["leaf"]]
            
            working_leafs = self._model.get_leafs()
                
            self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
                    
            self.checkpoint_metrics()
            
            