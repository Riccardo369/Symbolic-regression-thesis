import numpy as np
import lark
import Operations as op
import matplotlib.pyplot as plt

import Exceptions as ex
import Simplifications as si

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
        
        father = tree1.get_father()
                
        #The leaf is the root
        if(father == None): self.set_root(tree2)
                    
        else:
            
            father.free_connection(tree1)
            
            father.new_connection(tree2)

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

                if(not result.accept_number_childrens(len(childrens_result))): raise ValueError(f"Node {first_children.value} cannot have no {len(childrens_result)}, but {result.get_max_channels()} childrens")
                
                for child in childrens_result: result.new_connection(child)
                
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

    #Get the model that is the derative on variable x of this model
    def derivate_model(self, x):
        return SymbolicRegression.string_to_tree(self.__root.get_derative_subtree(x))
    
    
    def simplify_model(self):
        
        #Convert the function in canonic form: main add chain that hold more sub mul chains, apply
        def canonic_form(node):
            
            #TOP DOWN
            
            #Constant calculation
            try:
                
                new_node = si.constant()(node)
                self.replace_subtree(node, new_node)
                node = new_node
                
            except ex.SimplificationError: pass
               
            #Apply the distributivity mul on add property
            if(isinstance(node, op.mul)):
                
                new_node = si.mul_distributivity_on_add()(node)
                self.replace_subtree(node, new_node)
                node = new_node
                
                
            #Apply singular simplification (in base of operations)
            #...
                
            #Apply decorporation for each operation (if it is possible)
            #...
                
                
            

            # ------------------------------------------------------
            # ------------------------------------------------------
            # ------------------------------------------------------
            
            
            for c in node.get_childrens(): canonic_form(c)
            
            
            # ------------------------------------------------------
            # ------------------------------------------------------
            # ------------------------------------------------------
            
            
           
            #BOTTOM UP
            
            #Apply flat of chain
            if(isinstance(node, op.mul) or isinstance(node, op.add)):
               
               new_node = type(node)()
               for x in si.flat_chain(node): new_node.new_connection(x)
               self.replace_subtree(node, new_node)
               node = new_node
               
            
            

        
        #Apply all simplification (that change in base of node type / operation)
        def simplification(node):
            
            #TOP DOWN
            
            #Accorporate operations and repeat the formula from beginning for each accorporation
            
            
            # ------------------------------------------------------
            # ------------------------------------------------------
            # ------------------------------------------------------
            
            for c in node.get_childrens(): simplification(c)
            
            
            #BOTTOM UP
            
            
            # ------------------------------------------------------
            # ------------------------------------------------------
            # ------------------------------------------------------
            
            #Simplify chains
            if(isinstance(node, op.add) or isinstance(node, op.mul)):
                
                new_node = si.add_chain()(node) if isinstance(node, op.add) else si.mul_chain()(node)
                self.replace_subtree(node, new_node)
                node = new_node
                
        
        def cleaning(node):
            
            for c in node.get_childrens(): cleaning(c)
            
            new_node = si.cleaning()(node)
            self.replace_subtree(node, new_node)
            node = new_node 

        
        print("ORIGINAL FORM", self.__root)
        
        canonic_form(self.__root)
        
        print("CANONIC FORM ", self.__root)
        
        simplification(self.__root)
        
        print("FINAL FORM   ", self.__root)
        
        cleaning(self.__root)
        
        print("CLEANED FORM ", self.__root)

        return self
        
        
        
        


    #Learn phase   
    def learn(self, TR_inputs_dict, TR_output, LearningAlgorithm, stop_criteria = lambda x: False, metric_functions = {}):

        return LearningAlgorithm(self,
                                 TR_inputs_dict,
                                 TR_output,
                                 stop_criteria,
                                 metric_functions)


