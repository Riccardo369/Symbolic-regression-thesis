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
    
    def __pre_learning(self, model, metrics):
        self._model = model
        self._leafs = model.get_leafs()
        self._solution = [x.get_name() for x in self._leafs]
        self._metric_functions = metrics
        self._metric_results = {key: [] for key in metrics.keys()}
        
    def __post_learning(self):
        self._model.get_root().clear_all_calculation_memory_sub_tree()
        
    def checkpoint_metrics(self):
        for key in self._metric_results:
            self._metric_results[key].append(self._metric_functions[key](self._model))
            
        
    def body_learning(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, stop_criteria):
        return NotImplemented
        
    def __call__(self, model, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, stop_criteria, metric_functions = {}):
        self.__pre_learning(model, metric_functions)
        self.body_learning(TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, stop_criteria)
        self.__post_learning()
        
        return self._metric_results.copy()
 
 
 
 
 
    

class IndipendentRamification(SymbolicRegressionLearning):
    
    __families = [{"family": "linear", "operations": [op.add, op.sub], "transformation": lambda x: x, "reconstruction": lambda x, a, b: (a * x) + b}]
    
    def __init__(self, operations):
        super().__init__()
        
        for oper in operations:
            if(not issubclass(oper, op.OperationNode)): raise ValueError(f"The operation {oper} is not a OperationNode type")
        
        self._operations = list(operations)
        
        
    #Ho il mio TR_output / residuo:
    #
    #1) applico ad ogni feature delle trasformazioni (ogni trasformazione per ogni famiglia)
    #2) da ogni trasformazione - feature, ricavo poi il nuovo residuo
    #3) scelta la famiglia e la feature, scelgo la seconda feature / costante e l' operazione, che migliorano sempre la trasparenza del residuo
    #4) Scelta l' operazione con le feature, si fa lo split calcolando il peso interno e lo split del residuo / output atteso
    #
    # Sul punto 2 e 3, quando scelgo la famiglia / feature e l' operazione, scelgo sempre l' opzione che rende il residuo successivo più trasparente
    #
    def extend(self, TR_inputs_dict, TR_output):
        
        #Choose in a greedy way a family of functions and a feature
        choices = []
        
        for feature in TR_inputs_dict.keys():
            for family in IndipendentRamification.__families:
                
                #Calculate transformation
                s = family["transformation"](TR_inputs_dict[feature])
                
                #Extract future residual
                mean_R = np.mean(TR_output)
                mean_s = np.mean(s)
                
                a = np.sum((s - mean_s)*(TR_output - mean_R)) / ((np.sum(s - mean_s)**2)+1e-25)
                b = mean_R - (a * mean_s)
                
                #Calculate of transparency of new residual (greedy transparency)
                choices.append({"feature": feature, "family": family["family"], "transparency": transparency(TR_output - family["reconstruction"](TR_inputs_dict[feature], a, b))})
                
        #Choose the best combination feature / family
        choices = sorted(choices, key = lambda x: -x["transparency"])
        
        feature = choices[0]["feature"]
        operations = list(filter(lambda x: x["family"] == choices[0]["family"], IndipendentRamification.__families))[0]["operations"]
        
        choices = []

        #For each operations
        for oper in operations:
                
            oper_node = oper()
            for i in range(oper_node.get_channels()):
                op.OperationNode.new_connection(oper_node, op.Leaf(f"l{i}"), i)
                    
            childrens = childrens
            
            #If the residual cannot be the result of this node, it is not necessary to use this node
            if(not oper_node.check_output_domain(TR_output)): continue
                    
            oper_node.set_backward_calculation_memory(TR_output)
                    
            if(oper_node.get_channels() == 2):
                    
                base = oper_node.get_default_inputs()
                
                if(oper_node.check_input_domain(0, TR_inputs_dict[feature])):
                        
                    #Doing the projection to 1° slot / child
                    childrens[0].set_name(feature)
                    childrens[0].set_forward_calculation_memory(TR_inputs_dict[feature])
                    childrens[1].set_forward_calculation_memory(base[1])
                    
                    proj = projection(oper_node.backward_calculation()[1], TR_inputs_dict, lambda x: oper_node.check_input_domain(1, x))
            
                    if(proj != None):
                        childrens[1].set_name(proj)
                        choices.append(oper_node.copy_sub_tree())
                
                #Doing the projection to 0° slot / child if operation is not commutative
                if(not oper_node.is_commutative() and oper_node.check_input_domain(1, TR_inputs_dict[feature])):
                                            
                    childrens[1].set_name(feature)
                    childrens[0].set_forward_calculation_memory(base[0])
                    childrens[1].set_forward_calculation_memory(TR_inputs_dict[feature])
                    
                    proj = projection(oper_node.backward_calculation()[0], TR_inputs_dict, lambda x: oper_node.check_input_domain(0, x))
                    
                    if(proj != None):
                        childrens[0].set_name(proj)
                        choices.append(oper_node.copy_sub_tree())
            
            elif(oper_node.get_channels() == 1):
                
                if(oper_node.check_input_domain(0, TR_inputs_dict[feature])):
      
                    childrens[0].set_name(feature)
                    choices.append(oper_node.copy_sub_tree())
                    
            else: raise TypeError(f"{oper} has no 1 or 2 channels, it is not acceptable for IndipendentRamification learning strategy")
            
        for choice in choices:
            
            choice.forward_calculation_memory_sub_tree(TR_inputs_dict).factorization_backward_calculation_memory_sub_tree(TR_output)
            
        #Choice the best solution, considering the max transparency between childrens residual (factorization backward)
        choices = [{"structure": choice, "max transparency": np.max([transparency(child.get_factorization_backward_calculation_memory()) for child in choice.get_childrens()])} for choice in choices]
        
        choices = sorted(choices, key = lambda x: -x["max transparency"])
        
        return choices[0]

        
    def body_learning(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, stop_criteria):
        
        if(not TR_inputs_dict.keys() == set(VL_inputs_dict.keys())): raise ValueError("TR and VL input sets has not same features")
        
        #Checkpoint
        self.checkpoint_metrics()
        
        #Calculate all memory
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output).factorization_backward_calculation_memory_sub_tree(TR_output)
        
        #raise ValueError("ssssssss")
        
        #Calculate first candidates
        working_leafs = [{"leaf": leaf, "extension": None} for leaf in self._model.get_leafs()]

        #While cycle
        while(len(working_leafs) > 0):
            
            transparency_leaf = transparency(working_leafs[0]["leaf"].get_factorization_backward_calculation_memory())
            
            print(len(working_leafs))
            
            if(transparency_leaf <= 1):
                print("delete leaf")
                del working_leafs[0]
                continue
            
            working_leafs[0]["extension"] = self.extend(TR_inputs_dict, working_leafs[0]["leaf"].get_factorization_backward_calculation_memory())
            
            #If the transparency is enough to go with this extension
            if(working_leafs[0]["extension"]["max transparency"] <= 1):
                print("delete extension leaf")
                del working_leafs[0]
                continue

            #Do the extension
            father_leaf = working_leafs[0]["leaf"].get_father()
                
            #The leaf is the root
            if(father_leaf == None): self._model.set_root(working_leafs[0]["extension"]["structure"])
                    
            else:
                
                i = father_leaf.get_childrens().index(working_leafs[0]["leaf"])
  
                op.OperationNode.free_connection(father_leaf, working_leafs[0]["leaf"])
                op.OperationNode.new_connection(father_leaf, working_leafs[0]["extension"]["structure"], i)
                
            print("Add extension", str(working_leafs[0]["leaf"]), " --> ", str(working_leafs[0]["extension"]["structure"]))
                
            #Add the new leafs
            for leaf in working_leafs[0]["extension"]["structure"].get_childrens():
                working_leafs.append({"leaf": leaf, "extension": None})
                    
            self.checkpoint_metrics()
            
            print(working_leafs[0]["extension"]["max transparency"])
                
            del working_leafs[0]
            
            #break
            






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
                
                oper_node.set_backward_calculation_memory(output)
                
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
                    childrens[0].set_forward_calculation_memory(inputs_dict[feature1])
                    
                    childrens[1].set_name(feature2)
                    childrens[1].set_forward_calculation_memory(inputs_dict[feature2])
                    
                    choices.append({"tree": str(oper_node), "output": oper_node.forward_calculation(), "error": f(oper_node.forward_calculation())})
                    
                    #Aggiungere anche la costante come seconda feature se possibile
                    mean_ideal_constant = np.mean(oper_node.backward_calculation()[0])
                    
                    if(oper_node.check_input_domain(1, mean_ideal_constant)):
                        
                        childrens[1].set_name(mean_ideal_constant)
                        childrens[1].set_forward_calculation_memory(mean_ideal_constant)
                        
                        choices.append({"tree": str(oper_node), "output": oper_node.forward_calculation(), "error": f(oper_node.forward_calculation())})
                        
                #Otherwise case (error)
                elif(oper_node.get_channels() not in [1, 2]): raise TypeError(f"{oper} has no 1 or 2 channels, it is not acceptable for IndipendentRamification learning strategy")

        #Choose best extension
        choice = SymbolicRegression.string_to_tree(sorted(choices, key = lambda x: x["error"])[0]["tree"]).get_root()

        return choice

        
    def body_learning(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, stop_criteria):
        
        if(not TR_inputs_dict.keys() == set(VL_inputs_dict.keys())): raise ValueError("TR and VL input sets has not same features")
        
        #Checkpoint
        self.checkpoint_metrics()
        
        #Calculate all memory
        self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
        
        #Calculate first candidates
        working_leafs = self._model.get_leafs()

        #While cycle
        while(len(working_leafs) > 0 and not stop_criteria(self._model, self._metric_results)):

            #Choose the leaf with best transparency
            leaf = sorted([{"leaf": x, "transparency": transparency(x.get_backward_calculation_memory())} for x in working_leafs], key = lambda x: x["transparency"])[0]

            #If the best leaf are not enough transparence, no leaf is automatically enough
            if(leaf["transparency"] <= 1): break
            
            def f(x):
                
                def recursion_forward(node):
                    for child in node.get_childrens(): recursion_forward(child)
                    if(type(node) != op.Leaf):
                        node.set_forward_calculation_memory(node.forward_calculation())
                
                leaf["leaf"].set_forward_calculation_memory(x)
                
                recursion_forward(self._model.get_root())
                
                Y_atteso = TR_output
                Y_ottenuto = self._model.get_root().get_forward_calculation_memory()
                
                return np.mean(np.abs(Y_atteso - Y_ottenuto))
            
            #Calculate extension
            extension = self.extend(TR_inputs_dict, leaf["leaf"].get_backward_calculation_memory(), f)

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
                
            #Add the new leafs
            for leaf in extension.get_childrens():
                working_leafs.append(leaf)
                
            self._model.get_root().forward_calculation_memory_sub_tree(TR_inputs_dict).backward_calculation_memory_sub_tree(TR_output)
                    
            self.checkpoint_metrics()
            
            