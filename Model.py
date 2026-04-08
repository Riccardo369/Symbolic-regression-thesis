import numpy as np
import lark
import Operations as op
import matplotlib.pyplot as plt

class SymbolicRegression:
    
    __compiler = lark.Lark(f"""
                        %import common.WS
                        %ignore WS

                        ?start: expr

                        ?expr: NAME "(" [expr ("," expr)*] ")"   -> function
                            | NAME                               -> var
                            | SIGNED_NUMBER                      -> number

                        NAME: /[a-zA-Z][a-zA-Z0-9_]*/

                        %import common.SIGNED_NUMBER
                    """, start="start", parser="lalr")
    
    def __init__(self, root):
        self.set_root(root)
        
    def get_root(self):
        return self.__root
    
    def set_root(self, root):
        if(not isinstance(root, op.OperationNode)): raise ValueError(f"Root must be a operation node, not a {type(root)}")
        self.__root = root.copy_sub_tree()
    
    def __str__(self):
        return str(self.__root)
    
    def view(self, f = lambda x: " ", block = True):
        self.__root.view_sub_tree(f, block)
        
    def copy(self):
        return SymbolicRegression(self.__root)
        
    def save_state(self):
        
        def extract_weights(node, weights):
            
            weights.append(node.get_weight_memory())
            
            if(not issubclass(type(node), op.Leaf)):
                for child in node.get_childrens(): extract_weights(child, weights)
                
            return weights
        
        return {"tree": str(self.__root), "weights": extract_weights(self.__root, [])}

    def load_state(state):
        
        result = SymbolicRegression.string_to_tree(state["tree"])
        
        def assign_weights(node, weights):
            
            node.set_weight_memory(weights[0])
            del weights[0]
            
            if(not issubclass(type(node), op.Leaf)):
                for child in node.get_childrens(): assign_weights(child, weights)
        
        assign_weights(result.get_root(), state["weights"])
        
        if(len(state["weights"]) > 0): raise ValueError("Weight's number and total nodes of the tree are not matching")
        
        return result
        
    def tree_to_string(tree):
        if(not isinstance(tree, SymbolicRegression)): raise ValueError("Tree must be a SymbolicRegression type")
        return str(tree.get_root())
        
    def string_to_tree(string):
        
        parsed_tree = SymbolicRegression.__compiler.parse(string)
        
        def recursive_construction(node):
            
            first_children = node.children[0]
            
            #If it is a middle node
            if(first_children.type == "NAME" and len(node.children) > 1):
                
                result = eval(f"op.{first_children.value}()")
                childrens_result = [recursive_construction(child) for child in node.children[1:]]

                if(len(childrens_result) != result.get_channels()): raise ValueError(f"Node {first_children.value} has no {len(childrens_result)}, but {result.get_channels()}")
                
                for i, child in enumerate(childrens_result):
                    op.OperationNode.new_connection(result, child, i)
                    
                return result
               
            #If is a leaf
            else:
                return op.Leaf(first_children.value if first_children.type == "NAME" else float(first_children.value))
        
        return SymbolicRegression(recursive_construction(parsed_tree))
 
    def get_leafs(self):
        result = []
        
        def recursion(node, leafs):
            if(issubclass(type(node), op.Leaf)): leafs.append(node)
            else: 
                for child in node.get_childrens():
                    recursion(child, leafs)
                    
        recursion(self.__root, result)
                    
        return result
          
    #Generate the real forward function (to input, insert a dict, important is every parameter of generated function must be inside the input dict)
    def get_calculation_function(self):
        
        f = self.__root.get_string_calculation_function()
        f = eval(f"lambda {', '.join(f['leaf_labels'])}: {f['body']}")
        
        aviable_parameters = set(f.__code__.co_varnames[:f.__code__.co_argcount])
        
        def result_function(inputs_dict):
            
            if(not all([type(key) == str for key in inputs_dict.keys()])):
                raise ValueError(f"All keys must be strings, key getted: {[key for key in inputs_dict.keys()]}")
            
            residual_parameters = aviable_parameters.difference(set(inputs_dict.keys()))
            
            if(len(residual_parameters) > 0): raise ValueError(f"The parameters {residual_parameters} are not covered, they must be covered")
            
            return f(**{key: inputs_dict[key] for key in aviable_parameters})
        
        return result_function

    #Generate the gradient of model
    def get_gradient_function(self, variable_list):
        
        gradient_lambdas = []
        
        for f_str in self.__root.get_strings_gradient_subtree(variable_list):
            body = SymbolicRegression.string_to_tree(f_str).get_root().get_string_calculation_function()['body']
            gradient_lambdas.append(eval(f"lambda {', '.join(variable_list)}: {body}"))

        
        return lambda inputs_dict: np.stack([
            np.broadcast_to(f(**{v: inputs_dict[v] for v in variable_list}), max([len(x) for x in inputs_dict.values()])) 
            for f in gradient_lambdas])

    #Generate the hessian of model
    def get_hessian_function(self, variable_list):
        
        n = len(variable_list)
        hessian_lambdas = [[None for _ in range(n)] for _ in range(n)]
        
        derivates_first_level = self.__root.get_strings_gradient_subtree(variable_list)
        
        for i in range(n):
            current_first_deriv_tree = SymbolicRegression.string_to_tree(derivates_first_level[i])
            derivates_second_level = current_first_deriv_tree.get_root().get_strings_gradient_subtree(variable_list)
            
            for j in range(n):
                body = SymbolicRegression.string_to_tree(derivates_second_level[j]).get_root().get_string_calculation_function()['body']
                hessian_lambdas[i][j] = eval(f"lambda {', '.join(variable_list)}: {body}")
        
        def hessian_execution(inputs_dict):
            
            clean_inputs = {v: inputs_dict[v] for v in variable_list}
            # Prendiamo la lunghezza del primo array nel dizionario per sapere quanti campioni abbiamo
            n_samples = len(next(iter(inputs_dict.values())))
            n = len(variable_list)
            
            results = []
            for i in range(n):
                for j in range(n):
                    res = hessian_lambdas[i][j](**clean_inputs)
                    # Forziamo il risultato a essere un array di n_samples
                    results.append(np.broadcast_to(res, (n_samples,)))
            
            # Stack e reshape finale
            return np.stack(results).reshape(n, n, n_samples)
        
        return hessian_execution
        
    
    #Learn phase   
    def learn(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, LearningAlgorithm, metric_functions = {}):

        return LearningAlgorithm(self,
                                 TR_inputs_dict,
                                 TR_output,
                                 VL_inputs_dict,
                                 VL_output,
                                 metric_functions)


