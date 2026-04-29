import numpy as np

from graphviz import Digraph
import matplotlib.pyplot as plt
from PIL import Image
import io

import Domains as dm
import Exceptions as ex
import Simplifications as si

class OperationNode:
    
    _tol = 1e-40
    
    def __init__(self, channels, is_commutative = False, name = None):
        
        if(type(channels) not in [int]): raise ValueError(f"Channels number must be a int value, cannot be a {channels}")
        
        #Channel numbers
        self.__channels = channels
        
        #Special properties
        self.__is_commutative = is_commutative
        
        #Father
        self.__father = None
        
        #Set parameters
        self.set_name(name)
        
        self.__childrens = [None for _ in range(channels)]
        
        self.__memories = dict()   
        
        self.__memories["forward"] = None
        self.__memories["backward"] = None

    def __str__(self):
        if(isinstance(self, Leaf)): return self._name if type(self._name) == "str" else str(self._name)
        return f"{self.__class__.__name__}({', '.join([str(x) for x in self.__childrens])})"
    
    def __repr__(self):
        return str(self)
    
    def equals_subtree(self, node):
        
        my_childrens =    self.get_childrens()
        other_childrens = node.get_childrens()
        
        if(len(my_childrens) != len(other_childrens)): return False
        if(not isinstance(node, type(self))): return False
        return all([c.equals_subtree(x) for c, x in zip(my_childrens, other_childrens)])
     
    #Delete a connection
    def free_connection(father, child):
        if(not issubclass(type(father), OperationNode)): raise ValueError(f"Father must be a Operation node type, not a {type(father)}")
        if(not issubclass(type(child), OperationNode)): raise ValueError(f"Child must be a Operation node type, not a {type(child)}")
        
        if(child.__father != father): raise ValueError(f"Child {child} has not like a father {father}, but has {child.__father}")
        if(child not in father.__childrens): raise ValueError(f"Father {father} has not {child} in own childrens")
        
        brothers = child.__father.__childrens
        brothers[brothers.index(child)] = None
        del brothers
        
        child.__father = None
       
    #Create a connection 
    def new_connection(father, child, i):
        
        if(not issubclass(type(father), OperationNode)): raise ValueError(f"Father must be a Operation node type, not a {type(father)}")
        if(not issubclass(type(child), OperationNode)): raise ValueError(f"Child must be a Operation node type, not a {type(child)}")
        
        if(child.__father != None): raise ValueError(f"Child {child} has also a father {father}, so connection is useless")
        if(child in father.__childrens): raise ValueError(f"Father {father} has {child} in own childrens, so connection is useless")
        
        if(i >= father.__channels): raise ValueError(f"Cannot set a {i}-th channel in a node with {father.__channels} children for father")
        if(father.__childrens[i] != None): raise ValueError("Father has already a child in {i}-th position")
        
        father.__childrens[i] = child
        child.__father = father

    #GET PARAMETERS ZONE
    def is_commutative(self):
        return self.__is_commutative
     
    def is_constant(self):
        return type(self._name) in [int, float] or (np.isscalar(self._name) and type(self._name) != str)
         
    def get_channels(self):
        return self.__channels
        
    def get_name(self):
        return self._name
        
    def get_father(self):
        return self.__father
    
    def get_childrens(self):
        return list(self.__childrens)
    
    #GET MEMORY ZONE
    def get_memory(self, memory_name):
        if(memory_name not in self.__memories.keys()): return None
        value = self.__memories[memory_name]
        return value.copy() if isinstance(value, np.ndarray) else value

    #Return all node of subtree
    def get_all_subtree_nodes(self):
        
        def recursion(n):
            result = [n]
            for c in n.get_childrens(): result += recursion(c)
            return result
        
        return recursion(self)

    #SET MEMORY ZONE
    def set_memory(self, memory_name, value):  
        if(not isinstance(value, np.ndarray) and type(value) not in [float, int] and not isinstance(value, np.generic) and value != None): raise ValueError(f"Memory must contains only a numpy tensor or a number float / int, not a {type(value)}")
        self.__memories[memory_name] = value.copy() if isinstance(value, np.ndarray) else value
        return self
    
    def set_name(self, name):        
        if(not self._check_type_name(name) and name != None): raise ValueError(f"For this node {self.__class__.__name__}, {name} like name is not acceptable")
        self._name = name
        return self

    #CALCULATION AND AUTOMATICALLY SET MEMORY ALL SUB TREE ZONE
    def forward_calculation_memory_sub_tree(self, inputs_dict):
               
        def recursion(node):
            for child in node.get_childrens(): recursion(child)
            if(type(node) == Leaf): node.set_memory("forward", inputs_dict[node.get_name()] if type(node.get_name()) == str else node.get_name())
            else:                   node.set_memory("forward", node.forward_calculation())
        
        with np.errstate(divide='ignore', invalid='ignore'):  
                
            recursion(self)
        
        return self

    def backward_calculation_memory_sub_tree(self, output):
        
        self.set_memory("backward", output)
        
        def recursion(node):
            
            if(not issubclass(type(node), Leaf)):
                for child, backward in zip(node.get_childrens(), node.backward_calculation()):                
                    child.set_memory("backward", backward)
                    recursion(child)
                    
        with np.errstate(divide='ignore', invalid='ignore'):  

            recursion(self)
        
        return self 
    
    #CLEAR MEMORY OPERATIONS
    def clear_all_subtree_memory(self, memory_name):
        
        def recursion(node):
            for child in node.get_childrens(): recursion(child)
            if(type(node) == Leaf):
                node.set_memory(memory_name, None)
            else: node.set_memory(memory_name, None)
                
        recursion(self)
        
        return self
    
    #CHECK OPERATION ZONE
    def _check_type_name(self, name):
        return True
    
    def check_input_domain(self, i, input):
        if(i < 0 or i > self.__channels - 1): raise ValueError(f"Operation {type(self)} has only {self.__channels}, so i = {i} has no coherence (start count from 0)")
        return True

    def check_output_domain(self, output):
        return True
    
    #NODE METHODS (valid only for the node, not for the all sub tree). For correct functions, the require memory must be calculated already

    #Get a simplification considering the position where is the node in global tree (considering also the father etc.)
    def get_simplification_subtree(self, operations = []):
            
        for opt in operations + [si.constant]:
           
            try: return opt(self)
            except ex.SimplificationError: pass

    #Get the string function of all subtree
    def get_string_calculation_function(self):
        
        leaf_labels = []
        
        def recursion(node, leaf_labels):
            if(issubclass(type(node), Leaf)):
                leaf_label = node.get_name()
                if(leaf_label not in leaf_labels):
                    try: float(leaf_label)
                    except: leaf_labels.append(leaf_label)
                return leaf_label
            
            return f"({node.get_string_calculation_structure([recursion(child, leaf_labels) for child in node.get_childrens()])})"
        
        body = recursion(self, leaf_labels)
    
        return {"leaf_labels": leaf_labels, "body": body}
    
    def get_derative_subtree(self, x):
        
        def recursion(node, x):

            extraction = [recursion(c, x) for c in node.get_childrens()]
            str_derative = [x for x in extraction]
            return node.get_string_derative_structure(str_derative, x)
        
        return recursion(self, x)
 
    #Get the operation in string form 
    def get_string_calculation_structure(self, x):
        return NotImplemented

    #Get the derative operation in string form (for x feature)
    def get_string_derative_structure(self, sub_strings, x):
        if(len(sub_strings) != self.__channels): raise ValueError(f"sub_strings len {len(sub_strings)} must be equals to childrens number {self.__channels}")
        return NotImplemented 

    #Calculate forward value. REQUIRE forwards from childrens
    def forward_calculation(self):
        return NotImplemented

    #Calculate backward value. REQUIRE backward from father and forward from childrens
    def backward_calculation(self):
        return NotImplemented

    def best_constant(self, i):
        if(self.__channels == 1): raise ValueError("This operations has 1 channel, it has no sense this function")
        if(i > 2 or i < 0): raise ValueError(f"This operation has {self.__channels}, {i} has no sense")
        return NotImplemented

    #Copy all sub tree
    def copy_sub_tree(self):
        
        result = self.__class__(self._name) if(self.__class__.__name__ == "Leaf") else self.__class__()
        
        result.__childrens = [None for _ in self.__childrens]
        
        #Non necessita della copia perchè i metodi set fanno una copia già di default
        for key in self.__memories.keys(): result.set_memory(key, self.__memories[key])
        
        for i, child in enumerate(self.__childrens):
            if(child == None): continue
            OperationNode.new_connection(result, child.copy_sub_tree(), i)

        return result
      
    #View in a plot all sub tree structure
    def view_sub_tree(self, f = lambda x: " ", block = True, name = "figure"):
    
        class CounterGraph:
            def __init__(self):
                self.i = 0

        Printer = Digraph()
        
        title_color = 'red'
        text_color = 'blue'
        node_color = 'black'
        border_color = 'yellow'
        
        Printer.graph_attr.update(bgcolor='black', margin = '0', padding = '0')
        Printer.edge_attr.update(color='white')
        Printer.node_attr.update(fontcolor=text_color)

        count = CounterGraph()

        def recursion(node, level):

            if(type(node) in [Leaf]):
                count.i += 1
                Printer.node(f"Child {count.i}", f'<\n<FONT COLOR="{title_color}">{str(node.get_name()).upper()}</FONT><BR/>\n<FONT COLOR="{text_color}">{f(node)}</FONT>>', {'shape': 'ellipse' if node.is_variable() else 'square', 'color': border_color, 'fillcolor': node_color, 'style': 'filled'})
                return f"Child {count.i}"
            
            Printer.node(str(id(node)), f'<\n<FONT COLOR="{title_color}">{node.__class__.__name__.upper()}</FONT><BR/>\n<FONT COLOR="{text_color}">{f(node)}</FONT>>', {'shape': 'octagon', 'color': border_color, 'fillcolor': node_color, 'style': 'filled'})

            for child in [x for x in node.__childrens]: Printer.edge(str(id(node)), recursion(child, level+1))
                
            return str(id(node))
        
        recursion(self, 1)

        img_bytes = Printer.pipe(format='png')

        # Converte i byte in un'immagine
        image = Image.open(io.BytesIO(img_bytes))

        # Mostra l'immagine con matplotlib
        plt.figure(facecolor='black')
        plt.title(name, color = "white")
        plt.imshow(image)
        plt.axis('off')
        plt.show(block = block)


class Leaf(OperationNode):
    def __init__(self, label):
        super().__init__(0, False, label)
    
    def _check_type_name(self, name):
        return type(name) in [str, int, float] or isinstance(name, np.generic)
    
    def is_variable(self):
        return type(self.get_name()) in [str]
    
    def forward_calculation(self):
        raise ValueError("It is a leaf, it is not possible to do a forward calculation, it has no children")
    
    def backward_calculation(self):
        raise ValueError("It is a leaf, it is not possible to do a backward calculation, it has no children")
    
    def get_string_derative_structure(self, sub_strings, x):
        super().get_string_derative_structure(sub_strings, x)
        return "1" if x == self._name else "0"

class add(OperationNode):
    def __init__(self):
        super().__init__(2, True)
 
    def forward_calculation(self):
        inputs = [x.get_memory("forward") for x in self.get_childrens()]
        return inputs[0] + inputs[1]
    
    def backward_calculation(self):
        inputs = [x.get_memory("forward") for x in self.get_childrens()]
        output = self.get_memory("backward") 
        return [output - inputs[1], output - inputs[0]]
    
    def get_string_calculation_structure(self, x):
        return f"{x[0]} + {x[1]}"
    
    def get_string_derative_structure(self, sub_strings, x):  
        super().get_string_derative_structure(sub_strings, x)
        return f"add({sub_strings[0]}, {sub_strings[1]})"

    def best_constant(self, i):
        super().best_constant(i)
        back = self.backward_calculation()
        if(~np.all(np.isfinite(back[i]))): raise ex.BackwardCalculationError("Error not best constant calculation")
        return np.mean(back[i])
    
    #def get_simplification_subtree(self):
    #    return super().get_simplification_subtree([si.addition_identity])

class mul(OperationNode):
    def __init__(self):
        super().__init__(2, True)

    def forward_calculation(self):
        inputs = [x.get_memory("forward") for x in self.get_childrens()]
        return inputs[0] * inputs[1]
    
    def backward_calculation(self):
        inputs = [x.get_memory("forward") for x in self.get_childrens()]
        output = self.get_memory("backward")
        return [output / inputs[1], output / inputs[0]]

    def get_string_calculation_structure(self, x):
        return f"{x[0]} * {x[1]}"
    
    def get_string_derative_structure(self, sub_strings, x):
        
        super().get_string_derative_structure(sub_strings, x)
        str_function = [str(x) for x in self.get_childrens()]
        return f"add(mul({sub_strings[0]}, {str_function[1]}), mul({sub_strings[1]}, {str_function[0]}))"
    
    def best_constant(self, i):
        super().best_constant(i)
        back = self.backward_calculation()
        if(~np.all(np.isfinite(back[i]))): raise ex.BackwardCalculationError("Error not best constant calculation")
        return np.mean(back[i])
    
    #def get_simplification_subtree(self):
    #    return super().get_simplification_subtree([si.multiplication_identity, si.multiplication_null])

class neg(OperationNode):
    def __init__(self):
        super().__init__(1, False)
 
    def forward_calculation(self):
        inputs = [x.get_memory("forward") for x in self.get_childrens()]
        return -inputs[0]
    
    def backward_calculation(self):
        output = self.get_memory("backward") 
        return [-output]
    
    def get_string_calculation_structure(self, x):
        return f"-{x[0]}"
    
    def get_string_derative_structure(self, sub_strings, x):
        super().get_string_derative_structure(sub_strings, x)
        return f"neg({sub_strings[0]})"

class div(OperationNode):
    def __init__(self):
        super().__init__(1, False)
        
    def check_input_domain(self, i, input):
        super().check_input_domain(i, input)
        if(not (self._check_type_name(input) and type(input) != str)): return False
        if(i == 0 and type(input) in [float, int]): return np.abs(input) >= self._tol
        elif(i == 0): return np.all(np.abs(input) >= self._tol)

    def forward_calculation(self):
        inputs = [x.get_memory("forward") for x in self.get_childrens()]
        return 1 / inputs[0]
    
    def backward_calculation(self):
        output = self.get_memory("backward")
        return [1 / output]

    def get_string_calculation_structure(self, x):
        return f"1 / {x[0]}"
    
    def get_string_derative_structure(self, sub_strings, x):
    
        super().get_string_derative_structure(sub_strings, x)
        str_function = [str(x) for x in self.get_childrens()]
        return f"neg(div(mul({str_function[0]}, {str_function[0]})))"
    
    #def get_simplification_subtree(self):
        #return super().get_simplification_subtree([si.division_identity, si.division_null])

class sin(OperationNode):
    def __init__(self):
        super().__init__(1)
        
    def check_output_domain(self, output):
        if(not (self._check_type_name(output) and type(output) != str)): return False
        if(type(output) in [float, int]): return np.abs(output) <= 1 + self._tol
        else: return np.all(np.abs(output) <= 1 + self._tol)
    
    def forward_calculation(self):
        return np.sin(self.get_childrens()[0].get_memory("forward"))
    
    def backward_calculation(self):
        real_output = self.get_memory("backward")
        return [dm.get_ideal_input_sin(self.get_childrens()[0].get_memory("forward"), real_output)]

    def get_string_calculation_structure(self, x):
        return f"np.sin({x[0]})"
    
    def get_string_derative_structure(self, sub_strings, x):
        
        super().get_string_derative_structure(sub_strings, x)
        str_function = [str(x) for x in self.get_childrens()]
        return f"mul(cos({str_function[0]}), {sub_strings[0]})"

class cos(OperationNode):
    def __init__(self):
        super().__init__(1)
        
    def check_output_domain(self, output):
        if(not (self._check_type_name(output) and type(output) != str)): return False
        if(type(output) in [float, int]): return np.abs(output) <= 1 + self._tol
        else: return np.all(np.abs(output) <= 1 + self._tol)
    
    def forward_calculation(self):
        return np.cos(self.get_childrens()[0].get_memory("forward"))
    
    def backward_calculation(self):
        real_output = self.get_memory("backward")
        return [dm.get_ideal_input_cos(self.get_childrens()[0].get_memory("forward"), real_output)]

    def get_string_calculation_structure(self, x):
        return f"np.cos({x[0]})"
    
    def get_string_derative_structure(self, sub_strings, x):
        
        super().get_string_derative_structure(sub_strings, x)
        str_function = [str(x) for x in self.get_childrens()]
        return f"mul(neg(0, sin({str_function[0]})), {sub_strings[0]})"

class abs(OperationNode):
    def __init__(self):
        super().__init__(1)
        
    def check_output_domain(self, output):
        return np.all(output >= 0)
    
    def forward_calculation(self):
        return np.abs(self.get_childrens()[0].get_memory("forward"))
    
    def backward_calculation(self):
        return dm.get_ideal_input_abs(self.get_memory("forward"), self.get_memory("backward"))

    def get_string_calculation_structure(self, x):
        return f"np.abs({x[0]})"
    
    def get_string_derative_structure(self, sub_strings, x):
        
        super().get_string_derative_structure(sub_strings, x)
        str_function = [str(x) for x in self.get_childrens()]
        return f"mul({sub_strings[0]}, div({str_function[0]}, abs({str_function[0]})))"

class exp(OperationNode):
    def __init__(self):
        super().__init__(1, False)
        
    def check_output_domain(self, output):
        if(type(output) in [float, int]): return output > 0
        else: return np.all(output > 0)

    def forward_calculation(self):
        inputs = [x.get_memory("forward") for x in self.get_childrens()]
        return np.exp(inputs[0])

    def backward_calculation(self):
        output = self.get_memory("backward")
        return [np.log(output)]
    
    def get_string_calculation_structure(self, x):
        return f"{e} ** {x[0]}"
    
    def get_string_derative_structure(self, sub_strings, x):
        super().get_string_derative_structure(sub_strings, x)
        str_function = [str(x) for x in self.get_childrens()]
        return f"mul(exp({str_function[0]}), {sub_strings[0]})"

class ln(OperationNode):
    def __init__(self):
        super().__init__(1, False)

    def check_input_domain(self, i, input):
        
        super().check_input_domain(i, input)
        if(not (self._check_type_name(input) and type(input) != str)): return False

        #Domain argument
        if(i == 0 and type(input) in [float, int]): return input > 0
        elif(i == 0): return np.all(input > 0)
 
    def forward_calculation(self):
        inputs = [x.get_memory("forward") for x in self.get_childrens()]
        return np.log(inputs[0])
    
    def backward_calculation(self):
        return [np.exp(self.get_memory("backward"))]
        
    def get_string_calculation_structure(self, x):
        return f"np.log({x[0]})"
    
    def get_string_derative_structure(self, sub_strings, x):
        super().get_string_derative_structure(sub_strings, x)
        str_function = [str(x) for x in self.get_childrens()]
        return f"mul(div({str_function[0]}), {sub_strings[0]})"


#add   2 input  commutativa
#mul   2 input  commutativa

#neg   1 input
#div   1 input
#sin   1 input
#cos   1 input
#exp   1 input
#abs   1 input
#log   1 input


