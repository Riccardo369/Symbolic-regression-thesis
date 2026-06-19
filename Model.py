import numpy as np
import lark
import Operations as op
import matplotlib.pyplot as plt

import Exceptions as ex

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
        self.__root = root
    
    def replace_subtree(self, tree1, tree2):

        if(not self.__root.is_node_present_in_sub_tree(tree1)): raise ValueError(f"Sub tree 1 {tree1} must be in this tree {self}, its not")
        if(self.__root.is_node_present_in_sub_tree(tree2)): raise ValueError(f"Sub tree 2 {tree2} cannot be in this tree {self}, it is")
        
        father_leaf = tree1.get_father()
                
        #The leaf is the root
        if(father_leaf == None): self.set_root(tree2)
                    
        else:
            
            for r, c in enumerate(father_leaf.get_childrens()):
                
                if(tree1 is c):
                    i = r
                    break
            
            father_leaf.free_connection(i)
            
            father_leaf.new_connection(tree2, i)

    def __str__(self):
        return str(self.__root)
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, model):
        if(not isinstance(model, SymbolicRegression)): return False
        return self.get_root() == model.get_root()
        
    def view(self, f = lambda x: " ", block = True):
        self.__root.view_sub_tree(f, block)
        
    def copy(self):
        return SymbolicRegression(self.__root.copy_sub_tree())
        
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

                if(len(childrens_result) != result.get_channels()): raise ValueError(f"Node {first_children.value} has no {len(childrens_result)}, but {result.get_channels()} childrens")
                
                for i, child in enumerate(childrens_result): result.new_connection(child.copy_sub_tree(), i)
                    
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
        
        #print(f"f[body]:", f["body"], self)
        
        f = eval(f"lambda {', '.join(f['leaf_labels'])}: {f['body']}")
        
        aviable_parameters = set(f.__code__.co_varnames[:f.__code__.co_argcount])
        
        def result_function(inputs_dict):
            
            if(not all([type(key) == str for key in inputs_dict.keys()])):
                raise ValueError(f"All keys must be strings, key getted: {[key for key in inputs_dict.keys()]}")
            
            residual_parameters = aviable_parameters.difference(set(inputs_dict.keys()))
            
            if(len(residual_parameters) > 0): raise ValueError(f"The parameters {residual_parameters} are not covered, they must be covered")
            
            return f(**{key: inputs_dict[key] for key in aviable_parameters})
        
        return result_function

    #Get the model that is the derative on variable x of this model
    def derivate_model(self, x):
        return SymbolicRegression.string_to_tree(self.__root.get_derative_subtree(x))

    #Simplify the tree, considering to simplify every node that it is possible to simplify
    def simplify_model(self):
        
        def apply_simplification(node, filter_simply):
            
            simplification = None
            for opt in node.get_simplification_subtree():
                simply = opt()
                if(not filter_simply(simply)): continue
                try:
                    simplification = simply(node).copy_sub_tree()
                    self.replace_subtree(node, simplification)
                    break
                except ex.SimplificationError: pass
                
            return simplification
        
        def recursion(node):
            
            #Calculate the aviable top down simplification until is possible to repeat (the node can change)
            while(True):
                new_node = apply_simplification(node, lambda x: x.is_top_down())
                if(new_node == None or new_node == node): break
                node = new_node
                        
            #Go to leafs
            for c in node.get_childrens(): recursion(c)
            
            #Calculate the aviable bottom up simplification
            while(True):
                new_node = apply_simplification(node, lambda x: x.is_bottom_up())
                if(new_node == None or new_node == node): break
                node = new_node
         
            return node
        
        #history = []
        #
        #while(self.__root not in history):
        #    
        #    #Save the actual form of model
        #    history.append(self.__root.copy_sub_tree())
        #    
        #    #Apply again all simplifications
        #    recursion(self.__root)
        #    
        #best_root = sorted(history, key = lambda x: len(x.get_all_subtree_nodes()))[0]
        #
        #If the semplification is worse (more nodes), do a rollback, backing to original root
        #if(best_root != self.__root): self.replace_subtree(self.__root, best_root)
        
        best_root = recursion(self.__root.copy_sub_tree())
        if(len(best_root.get_all_subtree_nodes()) <= len(self.__root.get_all_subtree_nodes())): self.replace_subtree(self.__root, best_root)
        
        return self

    #Learn phase   
    def learn(self, TR_inputs_dict, TR_output, LearningAlgorithm, stop_criteria = lambda x: False, metric_functions = {}):

        return LearningAlgorithm(self,
                                 TR_inputs_dict,
                                 TR_output,
                                 stop_criteria,
                                 metric_functions)


