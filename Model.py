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
        if(not isinstance(root, op.OperationNode)): raise ValueError("Root must be a operation node")
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
    
    #Generate the forward function in string form
    def get_string_function(self):
        
        leaf_labels = []
        
        def recursion(node, leaf_labels):
            if(issubclass(type(node), op.Leaf)):
                leaf_label = node.get_name()
                if(leaf_label not in leaf_labels):
                    try: float(leaf_label)
                    except: leaf_labels.append(leaf_label)
                return leaf_label
            
            return f"({node.get_string_structure([recursion(child, leaf_labels) for child in node.get_childrens()])})"
        
        body = recursion(self.__root, leaf_labels)
            
        return {"leaf_labels": leaf_labels, "body": body}
     
    #Generate the real forward function (to input, insert a dict, important is every parameter of generated function must be inside the input dict)
    def get_dict_function(self):
        
        f = self.get_string_function()
        f = eval(f"lambda {', '.join(f['leaf_labels'])}: {f['body']}")
        
        aviable_parameters = set(f.__code__.co_varnames[:f.__code__.co_argcount])
        
        def result_function(inputs_dict):
            
            if(not all([type(key) == str for key in inputs_dict.keys()])):
                raise ValueError(f"All keys must be strings, key getted: {[key for key in inputs_dict.keys()]}")
            
            residual_parameters = aviable_parameters.difference(set(inputs_dict.keys()))
            
            if(len(residual_parameters) > 0): raise ValueError(f"The parameters {residual_parameters} are not covered, they must be covered")
            
            return f(**{key: inputs_dict[key] for key in aviable_parameters})
        
        return result_function
       
    #Learn phase   
    def learn(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, LearningAlgorithm, metric_functions = {}):

        return LearningAlgorithm(self,
                                 TR_inputs_dict,
                                 TR_output,
                                 VL_inputs_dict,
                                 VL_output,
                                 metric_functions)

