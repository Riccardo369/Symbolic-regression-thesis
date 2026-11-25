import lark
import sympy as sp
import warnings


import Operations as op
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

        father = tree1.get_father()

        if(not self.__root.is_node_present_in_sub_tree(tree1)): raise ValueError(f"First sub tree {tree1} must be in this tree {self}, its not")
        if(self.__root.is_node_present_in_sub_tree(tree2)): raise ValueError(f"Second sub tree {tree2} cannot be in this tree {self}, it is")
        
        if(tree2.get_father() != None): raise ValueError(f"tree 2 {tree2} must has no father, has {tree2.get_father()}")
   
        #The leaf is the root
        if(father == None):
 
            self.set_root(tree2)
                    
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
     
    #View all tree model   
    def view(self, f = lambda x: " ", block = True):
        self.__root.view_sub_tree(f, block)
    
    def copy(self):
        return SymbolicRegression(self.__root.copy_sub_tree())
   
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
    
    #Apply syntax simplification con Sympy
    def simplify_model(self):
        
        def clean(tree):
            
            for c in tree.get_childrens(): clean(c)
            
            #If node is not a chain
            if((not isinstance(tree, op.ChainOperationNode)) or isinstance(tree, op.Leaf)):
                
                try:
                    
                    #Try to apply a direct calculation
                    new_node = op.Leaf(eval(f"lambda: {tree.get_string_calculation_function()['body']}")())
                    self.replace_subtree(tree, new_node) 
                    
                except: pass
                return self
            
            #Flat the chain
            childrens = si.flat_chain(tree)
            
            if(childrens[0].get_name() == 0 and type(tree) == op.mul):
                self.replace_subtree(tree, op.Leaf(0))
                return 0
            
            #Apply the substitution
            if(len(childrens) == 1):
              
                self.replace_subtree(tree, childrens[0])
                return self
            
            else:
            
                new_node = type(tree)()
                for c in childrens: new_node.new_connection(c)
                self.replace_subtree(tree, new_node)
                return self
            
            
        clean(self.get_root())
                
        return self  
        
          
        
        #Clean tree from chain node with one node
        #def clean_chain_node_one_child(tree, node):
        #    
        #    #for c in node.get_childrens(): clean_chain_node_one_child(tree, c)
        #    
        #    if(isinstance(node, op.ChainOperationNode)):
        #                        
        #        flat_childrens = []
        #        for c in node.get_childrens():
        #            if(type(c) == type(node)):
        #                flat_childrens += c.get_childrens()
        #                node.free_connection(c)
        #            
        #        for fc in flat_childrens:
        #            fc.get_father().free_connection(fc)
        #            node.new_connection(fc)
        #    
        #    if(isinstance(node, op.ChainOperationNode) and len(node.get_childrens()) == 1):
        #        
        #        child = node.get_childrens()[0]
        #        node.free_connection(child)
        #        tree.replace_subtree(node, child)
        #       
        #f = si.to_sympy(self.get_root().copy_sub_tree())
        #f = sp.expand(f)
        #f = sp.factor(f)
        #f = sp.powsimp(f)
        #f = sp.nsimplify(f)
        #f.doit()
        #
        #try:
        #
        #    f = si.from_sympy(f, TR_inputs_dict)
        #
        #if(len(f.get_all_subtree_nodes()) <= len(self.get_root().get_all_subtree_nodes())): self.replace_subtree(self.get_root(), f)
        #
        #except: return self
        #
        #clean_chain_node_one_child(self, self.get_root())
        
        
    


    #Learn phase   
    def learn(self, TR_inputs_dict, TR_output, learning_algorithm, stop_criteria = lambda x: False, metric_functions = {}):
        
        warnings.filterwarnings('ignore', category=RuntimeWarning)
                
        result = learning_algorithm(self,
                                    TR_inputs_dict,
                                    TR_output,
                                    stop_criteria,
                                    metric_functions)
                            
        warnings.filterwarnings('default', category=RuntimeWarning)
        
        return result

