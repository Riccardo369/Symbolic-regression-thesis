import numpy as np
import lark
import Operations as op
import matplotlib.pyplot as plt

class SymbolicRegression:
    
    def __projection(ideal, features, L):
        const = np.mean(ideal)
        proposals = [{"label": x, "value": features[x]} for x in features.keys()] + [{"label": const, "value": const}]
        return sorted(proposals, key = lambda x: L(ideal, x["value"]))[0]["label"]
    
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
        
        self.__root = root.copy_sub_tree()
        
    def copy(self):
        return SymbolicRegression(self.__root)
        
    def get_root(self):
        return self.__root
    
    def tree_to_string(tree):
        if(not isinstance(tree, SymbolicRegression)): raise ValueError("Tree must be a SymbolicRegression type")
        return str(tree.get_root())
        
    def string_to_tree(string):
        
        parsed_tree = SymbolicRegression.__compiler.parse(string)
        lark.tree.pydot__tree_to_png(parsed_tree, "tree.png")
        
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

    def get_general_tree(self):
        
        class Counter:
            
            def __init__(self):
                self.__i = 0
                
            def get_count(self):
                self.__i += 1
                return self.__i - 1
            
        counter = Counter()
        
        result = self.copy()
        
        def recursion(node, counter):
            
            if(issubclass(type(node), op.Leaf)):
                node.set_name(f"l{counter.get_count()}")
                
            else:
                
                for child in node.get_childrens():
                    recursion(child, counter)
        
        recursion(result.get_root(), counter)
        
        return result
    
    def fix_leafs(self, leafs_list):
        
        total_leafs = len(self.get_fixed_leaf_list())
        total_proposed_leafs = len(leafs_list)
        
        if(total_leafs != total_proposed_leafs): raise ValueError(f"This tree has {total_leafs} in total, but it is proposed to fix {total_proposed_leafs}")
        
        def recursion(node, leafs_list):
            
            if(issubclass(type(node), op.Leaf)):
                node.set_name(leafs_list[0])
                del leafs_list[0]
                
            else:
                
                for child in node.get_childrens():
                    recursion(child, leafs_list)
        
        recursion(self.__root, list(leafs_list))
        
        return self
    
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
            
        return f"lambda {', '.join(leaf_labels)}: {body}"
    
    #Generate the real forward function (to input, insert a dict, important is every parameter of generated function must be inside the input dict)
    def get_dict_function(self):
        
        f = eval(self.get_string_function())
        
        aviable_parameters = set(f.__code__.co_varnames[:f.__code__.co_argcount])
        
        def result_function(inputs_dict):
            
            if(not all([type(key) == str for key in inputs_dict.keys()])):
                raise ValueError(f"All keys must be strings, key getted: {[key for key in inputs_dict.keys()]}")
            
            residual_parameters = aviable_parameters.difference(set(inputs_dict.keys()))
            
            if(len(residual_parameters) > 0): raise ValueError(f"The parameters {residual_parameters} are not covered, they must be covered")
            
            return f(**{key: inputs_dict[key] for key in aviable_parameters})
        
        return result_function

    #Generate the solution (leaf combination assignament) of the tree
    def get_fixed_leaf_list(self):
        
        leaf_labels = []
        
        def recursion(node, leaf_labels):
            if(issubclass(type(node), op.Leaf)):
                leaf_label = node.get_name()
                try: leaf_labels.append(float(leaf_label))
                except: leaf_labels.append(leaf_label)

            else:
                
                for child in node.get_childrens():
                    recursion(child, leaf_labels)
            
        recursion(self.__root, leaf_labels)
            
        return leaf_labels

    #Calculate forward values to all nodes
    def forward_calculation(self, inputs_dict):
          
        def recursion(node):
            for child in node.get_childrens(): recursion(child)
            if(type(node) == op.Leaf): node.set_forward_calculation_memory(inputs_dict[node.get_name()] if type(node.get_name()) == str else node.get_name())
            else: node.set_forward_calculation_memory(node.forward_calculation())
                
        recursion(self.__root)
        
        return self
     
    #Calculate backward values to all nodes
    def backward_calculation(self, output):
        
        self.__root.set_backward_calculation_memory(output)
        
        def recursion(node):
            
            if(not issubclass(type(node), op.Leaf)):
                for child, backward in zip(node.get_childrens(), node.backward_calculation()):                
                    child.set_backward_calculation_memory(backward)
                    recursion(child)

        recursion(self.__root)
        
        return self
            
    def learn(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output, LearningAlgorithm):

        LearningAlgorithm(self, TR_inputs_dict, TR_output, VL_inputs_dict, VL_output)
        
        return self


X = np.array([x for x in range(10, 100, 2)])
Y = 2*X + 3

#tree1 = SymbolicRegression.string_to_tree("mul(2, x1)")
#tree1.forward_calculation({"x1": X}).backward_calculation(Y)
#tree1.get_root().view_sub_tree(lambda x: f"{np.mean(np.abs(x.get_forward_calculation_memory() - x.get_backward_calculation_memory()))}")
#tree1.get_root().view_sub_tree(lambda x: f"Forward: {x.get_forward_calculation_memory()}\nBackward: {x.get_backward_calculation_memory()}\n{np.mean(np.abs(x.get_forward_calculation_memory() - x.get_backward_calculation_memory()))}")

#tree2 = SymbolicRegression.string_to_tree("add(x1, 3)")
#tree2.forward_calculation({"x1": X}).backward_calculation(Y)
#tree2.get_root().view_sub_tree(lambda x: f"{np.mean(np.abs(x.get_forward_calculation_memory() - x.get_backward_calculation_memory()))}")
#tree2.get_root().view_sub_tree(lambda x: f"Forward: {x.get_forward_calculation_memory()}\nBackward: {x.get_backward_calculation_memory()}\n{np.mean(np.abs(x.get_forward_calculation_memory() - x.get_backward_calculation_memory()))}")

tree3 = SymbolicRegression.string_to_tree("add(mul(2, x1), 3)")
tree3.forward_calculation({"x1": X}).backward_calculation(Y)
tree3.get_root().view_sub_tree(lambda x: f"{np.mean(np.abs(x.get_forward_calculation_memory() - x.get_backward_calculation_memory()))}")
#tree3.get_root().view_sub_tree(lambda x: f"Forward: {x.get_forward_calculation_memory()}\nBackward: {x.get_backward_calculation_memory()}\n{np.mean(np.abs(x.get_forward_calculation_memory() - x.get_backward_calculation_memory()))}")

plt.show()
