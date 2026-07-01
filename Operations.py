import numpy as np

from graphviz import Digraph
import matplotlib.pyplot as plt
from PIL import Image
import io

import Domains as dm
import Exceptions as ex
import Simplifications as si
import Indicators as ind
import Errors as er
import utils

class OperationNode:

    _tol = 1e-40
    
    def __init__(self, max_channels, name = None):
        
        if(type(max_channels) not in [int] and max_channels != float("inf")): raise ValueError(f"Max channels must be a int number, cannot be {max_channels}")
        
        #Channel numbers
        self.__max_channels = max_channels
        
        #Father
        self.__father = None
        
        #Set parameters
        self.set_name(name)
        
        self.__childrens = []
        
        self.__memories = dict()   
        
        self.__memories["forward"] = None
        self.__memories["backward"] = None
        
    def sort_internal_nodes(self):
        
        def recursion(node):
            
            for i in node.__childrens: recursion(i)
            
            if(isinstance(node, Leaf)): return
            
            else: node.__childrens = sorted(node.__childrens, key = lambda x: str(x))
        
        recursion(self)

    def __eq__(self, x):
        
        #Check if this node is of this type
        if(not isinstance(x, type(self))): return False
        return str(self) == str(x)

    def __str__(self):
        if(isinstance(self, Leaf)): return self._name if type(self._name) == "str" else str(self._name)
        return f"{self.__class__.__name__}({', '.join([str(x) for x in self.__childrens])})"
    
    def __repr__(self):
        return str(self)
    
    def __index_child(self, node):
        
        for i, child in enumerate(self.__childrens):
            if(id(child) == id(node)): return i
            
        return None
     
    #Delete a connection
    def free_connection(self, child):
        
        if(len(self.get_childrens()) == 0): raise ValueError(f"This node has {len(self.get_childrens())}, it is not possible to remove another child")
        
        
        if(not issubclass(type(child), OperationNode)): raise ValueError(f"Child must be a Operation node type, not a {type(child)}")
        
        index = self.__index_child(child)
        if(index == None): raise ValueError(f"Child {child} are not a child of {self}")
        
        del self.__childrens[index]
        child.__father = None
        
        return self
       
    #Create a connection
    def new_connection(self, child):
        
        if(self.__max_channels == len(self.get_childrens())): raise ValueError(f"This node has {len(self.get_childrens())}, can have max {self.__max_channels}, it is not possible to add a new child")
        
        if(not issubclass(type(child), OperationNode)): raise ValueError(f"Child must be a Operation node type, not a {type(child)}")
        
        index = self.__index_child(child)
        if(index != None): raise ValueError(f"This child {child} is already like child (not form, like the real child in memory), it is not possible to add again in childrens {self.__childrens}, otherwise it will create a acycle graph (not acceptable)")

        self.__childrens.append(child)
        child.__father = self
        
        return self
     
    def is_constant(self):
        return type(self._name) in [int, float] or (np.isscalar(self._name) and type(self._name) != str)
    
    def is_node_present_in_sub_tree(self, node):
        
        if(not isinstance(node, OperationNode)): raise ValueError(f"Node {node} must be an OperationNode")
        
        def recursion(n):
            return any([recursion(c) for c in n.get_childrens()] + [id(n) == id(node)])
        
        #raise ValueError("STOP")

        return recursion(self)
    
    def get_max_channels(self):
        return self.__max_channels
           
    def get_channels(self):
        return len(self.__childrens)
    
    def accept_number_childrens(self, k):
        return k <= self.__max_channels
        
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
        if(not self._check_type_name(name)): raise ValueError(f"For this node {self.__class__.__name__}, {name} like name is not acceptable")
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
        if(i < 0 or i > len(self.__childrens) - 1): raise ValueError(f"Operation {type(self)} has {len(self.__childrens)}, so i = {i} has no coherence (start count from 0)")
        return True

    def check_output_domain(self, output):
        return True
    
    #NODE METHODS (valid only for the node, not for the all sub tree). For correct functions, the require memory must be calculated already

    #Get the list of simplification considering the position where is the node in global tree (considering also the father etc.)
    def get_simplification_subtree(self, operations = []):
        result = operations
        
        for i in result:
            if(not issubclass(i, si.Simplification)): raise TypeError(f"Simplification {i} is not a class Simplification, it must be")
        
        return result

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
            
            result = f"({node.get_string_calculation_structure([recursion(child, leaf_labels) for child in node.get_childrens()])})"
            
            return result
        
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
        if(len(sub_strings) != self.get_channels()): raise ValueError(f"sub_strings len {len(sub_strings)} must be equals to childrens number {self.get_channels()}")
        return NotImplemented 

    #Calculate forward value. REQUIRE forwards from childrens
    def forward_calculation(self):
        return NotImplemented

    #Calculate backward value. REQUIRE backward from father and forward from childrens
    def backward_calculation(self):
        return NotImplemented

    #Calculate the confidence of this node (considering the childrens). REQUIRE backward from father and forward from childrens
    def calculation_hat_confidence(self, x, y):
        return NotImplemented

    def calculate_statistical_score(total_analysis):
        return NotImplemented

    #Copy all sub tree
    def copy_sub_tree(self):
        
        result = self.__class__(self._name) if(self.__class__.__name__ == "Leaf") else self.__class__()
        
        #Non necessita della copia perchè i metodi set fanno una copia già di default
        #for key in self.__memories.keys(): result.set_memory(key, self.__memories[key])
        
        for child in self.__childrens:
            result.new_connection(child.copy_sub_tree())

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


class UnaryOperationNode(OperationNode):
    
    def __init__(self):
        super().__init__(1)
             
class ChainOperationNode(OperationNode):
    
    def __init__(self):
        super().__init__(float("inf"))
        
    def accept_number_childrens(self, k):
        return k > 1 and k < self.get_max_channels()
        

class Leaf(OperationNode):
    
    def __init__(self, label):
        super().__init__(0, label)
    
    def _check_type_name(self, name):
        return (type(name) in [str, int, float] and not isinstance(name, np.ndarray)) or np.isscalar(name)
    
    def is_variable(self):
        return type(self.get_name()) in [str]
    
    def forward_calculation(self):
        raise ValueError("It is a leaf, it is not possible to do a forward calculation, it has no children")
    
    def backward_calculation(self):
        raise ValueError("It is a leaf, it is not possible to do a backward calculation, it has no children")
    
    def calculation_hat_confidence(self, x, y):
        if(self.is_constant()): result = 1 / (np.var(y) + 1)
        else: result = ind.chatterjee_correlation(x, y)
        return result if np.isfinite(result) else 0

    def calculate_statistical_score(total_analysis):
        raise ValueError("IT is not possible to calculate statistical score of a leaf")

    def get_string_derative_structure(self, sub_strings, x):
        super().get_string_derative_structure(sub_strings, x)
        return "1" if x == self._name else "0"
    
    def get_simplification_subtree(self):
        return super().get_simplification_subtree([si.approximation])


class add(ChainOperationNode):
    
    def __init__(self):
        super().__init__()
        
    def __str__(self):
        return f"{self.__class__.__name__}({', '.join(sorted([str(x) for x in self.get_childrens()]))})"

    def forward_calculation(self):
        inputs = [x.get_memory("forward") for x in self.get_childrens()]
        result = 0
        for i in inputs: result += i
        return result
    
    def backward_calculation(self):
        inputs = [x.get_memory("forward") for x in self.get_childrens()]
        output = self.get_memory("backward") 
        return [output - i for i in inputs]
    
    def calculation_hat_confidence(self, x, y):
        raise TypeError("add is a associative and commutative operation, it not make any sense")
    
    def calculate_statistical_score(total_analysis):
        raise ValueError("IT is not possible to calculate statistical score of an add node")
    
    def get_string_calculation_structure(self, x):
        return f"{' + '.join(sorted([str(r.get_string_calculation_function()['body']) for r in self.get_childrens()]))}"
    
    def get_string_derative_structure(self, sub_strings, x):  
        super().get_string_derative_structure(sub_strings, x)
        return f"add({', '.join(sub_strings)})"
    
    def get_simplification_subtree(self):
        return super().get_simplification_subtree([])

class mul(ChainOperationNode):
    
    def __init__(self):
        super().__init__()
        
    def __str__(self):
        return f"{self.__class__.__name__}({', '.join(sorted([str(x) for x in self.get_childrens()]))})"

    def forward_calculation(self):
        inputs = [x.get_memory("forward") for x in self.get_childrens()]
        result = 0
        for i in inputs: result *= i
        return result
    
    def backward_calculation(self):
        
        inputs = [x.get_memory("forward") for x in self.get_childrens()]
        output = self.get_memory("backward")
        
        result = []
        
        for input in inputs:
            
            try: backward_output = output / input
            except ZeroDivisionError: backward_output = input
            
            if(isinstance(input, np.ndarray)):
                mask = ~np.isfinite(backward_output)
                if(not np.isscalar(input)): backward_output[mask] = input[mask]
                else: backward_output = input
            
            result.append(backward_output)
        
        return result

    def calculation_hat_confidence(self, x, y):
        raise TypeError("mul is a associative and commutative operation, it not make any sense")
    
    def calculate_statistical_score(total_analysis):
        raise ValueError("IT is not possible to calculate statistical score of an mul node")
    
    def __string_division(self, x):
        
        elements = [str(r.get_string_calculation_function()['body']) for r in self.get_childrens()]
        
        set_elements = set([x for x in elements])
        
        set_elements = [(x, elements.count(x)) for x in set_elements]
        
        #print(elements, set_elements)
        
        set_elements = [f"({x[0]} ** {x[1]})" if x[1] > 1 else f"{x[0]}" for x in set_elements]
        
        return f"{' * '.join(sorted(elements))}"
        
    
    def get_string_calculation_structure(self, x):
        
        numerator = self.__string_division([x for x in self.get_childrens() if not isinstance(x, div)])
        
        denominator = self.__string_division([x for x in self.get_childrens() if isinstance(x, div)])
        
        return f"{numerator} / {denominator}" if denominator != "" else numerator
    
    def get_string_derative_structure(self, sub_strings, x):
        
        super().get_string_derative_structure(sub_strings, x)
        str_function = [str(x) for x in self.get_childrens()]
        
        parts = []
        
        for i in range(self.get_channels()):
            
            new_part = []
            
            for j in range(self.get_channels()):
                
                if(i != j):   new_part.append(str_function[j])
                elif(i == j): new_part.append(sub_strings[j])
                
            parts.append(f"mul({', '.join(new_part)})")
        
        return f"add({', '.join(parts)})"

    def get_simplification_subtree(self):
        return super().get_simplification_subtree([])

class div(UnaryOperationNode):
    
    def __init__(self):
        super().__init__()
        
    def check_input_domain(self, i, input):
        super().check_input_domain(i, input)
        if(not (self._check_type_name(input) and type(input) != str)): return False
        if(i == 0 and type(input) in [float, int]): return np.abs(input) >= self._tol
        elif(i == 0): return np.all(np.abs(input) >= self._tol)

    def forward_calculation(self):
        inputs = [x.get_memory("forward") for x in self.get_childrens()]  
            
        try: return 1 / inputs[0] 
        except ZeroDivisionError: np.nan
    
    def backward_calculation(self):
        
        input = self.get_memory("backward")
        
        with np.errstate(divide='ignore', invalid='ignore'):

            try: output = 1 / input
            except ZeroDivisionError: output = np.nan
            
        if(isinstance(output, np.ndarray)):
            mask = ~np.isfinite(output)
            output[mask] = input[mask] if isinstance(input, np.ndarray) else input
            
        return [output]
    
    def calculation_hat_confidence(self, x, y):
    
        In, Out = ind.prepare(x, y)
    
        mask_left = In < 0
        mask_right = In > 0
        
        result_left = -ind.spearman_correlation(In[mask_left], Out[mask_left]) if np.sum(mask_left) > 1 else 1
        result_right = ind.spearman_correlation(In[mask_right], Out[mask_right]) if np.sum(mask_right) > 1 else 1
        
        result = np.abs(result_left * result_right)
        return result
    
    def calculate_statistical_score(total_analysis):
        return -total_analysis["symmetry"] * np.abs(total_analysis["vertical_asymptote"])
   
    def get_string_calculation_structure(self, x):
        return f"1 / {x[0]}"
    
    def get_string_derative_structure(self, sub_strings, x):
    
        super().get_string_derative_structure(sub_strings, x)
        str_function = [str(x) for x in self.get_childrens()]
        return f"mul(-1, (div(mul({str_function[0]}, {str_function[0]}))))"

    def get_simplification_subtree(self):
        return super().get_simplification_subtree([si.div_null,
                                                   si.div_scission])

class sin(UnaryOperationNode):
    
    def __init__(self):
        super().__init__()
        
    def check_output_domain(self, output):
        if(not (self._check_type_name(output) and type(output) != str)): return False
        if(type(output) in [float, int]): return np.abs(output) <= 1 + self._tol
        else: return np.all(np.abs(output) <= 1 + self._tol)
    
    def forward_calculation(self):
        return np.sin(self.get_childrens()[0].get_memory("forward"))
    
    def backward_calculation(self):
        real_output = self.get_memory("backward")
        return [dm.get_ideal_input_sin(self.get_childrens()[0].get_memory("forward"), real_output)]
    
    def calculation_hat_confidence(self, x, y):
        result = ind.chatterjee_correlation(x, y) * (1 - np.abs(ind.chatterjee_correlation(x, y)))
        return result
    
    def calculate_statistical_score(total_analysis):
        return np.abs(total_analysis["sinusoidal"])

    def get_string_calculation_structure(self, x):
        return f"np.sin({x[0]})"
    
    def get_string_derative_structure(self, sub_strings, x):
        
        super().get_string_derative_structure(sub_strings, x)
        str_function = [str(x) for x in self.get_childrens()]
        return f"mul(cos({str_function[0]}), {sub_strings[0]})"

class arcsin(UnaryOperationNode):
    
    def __init__(self):
        super().__init__()
        
    def check_output_domain(self, output):
        if(not (self._check_type_name(output) and type(output) != str)): return False
        if(type(output) in [float, int]): return np.abs(output) <= 1 + self._tol
        else: return np.all(np.abs(output) <= (np.pi/2) + self._tol)
        
    def check_input_domain(self, i, input):
        super().check_input_domain(i, input)
        if(not (self._check_type_name(input) and type(input) != str)): return False
        if(type(input) in [float, int]): return np.abs(input) <= 1 + self._tol
        else: return np.all(np.abs(input) <= 1 + self._tol)
    
    def forward_calculation(self):
        return np.arcsin(self.get_childrens()[0].get_memory("forward"))
    
    def backward_calculation(self):
        return [np.sin(self.get_memory("backward"))]

    def calculation_hat_confidence(self, x, y):
        result = ind.spearman_correlation(x, y)
        return result

    def calculate_statistical_score(total_analysis):
        return np.abs(total_analysis["monoton"]) * -total_analysis["symmetry"]
    
    def get_string_calculation_structure(self, x):
        return f"np.arcsin({x[0]})"
    
    def get_string_derative_structure(self, sub_strings, x):
        
        super().get_string_derative_structure(sub_strings, x)
        str_function = [str(x) for x in self.get_childrens()]
        return f"div({sub_strings[0]}, cos(arcsin({str_function[0]})))"

#Il cos non è altro che: cos(x) = sin(x + pi*1.5)

class abs(UnaryOperationNode):
    
    def __init__(self):
        super().__init__()
        
    def check_output_domain(self, output):
        return np.all(output >= 0)
    
    def forward_calculation(self):
        return np.abs(self.get_childrens()[0].get_memory("forward"))
    
    def backward_calculation(self):
        return [dm.get_ideal_input_abs(self.get_memory("forward"), self.get_memory("backward"))]

    def calculation_hat_confidence(self, x, y):
    
        In, Out = ind.prepare(x, y)
    
        mask_left = In <= 0
        mask_right = In >= 0
        
        result_left = ind.pearson_correlation(In[mask_left], Out[mask_left]) if np.sum(mask_left) > 1 else 1
        result_right = -ind.pearson_correlation(In[mask_right], Out[mask_right]) if np.sum(mask_right) > 1 else 1
        
        result = np.abs(result_left * result_right)
        return result
    
    def calculate_statistical_score(total_analysis):
        return total_analysis["symmetry"]

    def get_string_calculation_structure(self, x):
        return f"np.abs({x[0]})"
    
    def get_string_derative_structure(self, sub_strings, x):
        
        super().get_string_derative_structure(sub_strings, x)
        str_function = [str(x) for x in self.get_childrens()]
        return f"mul({sub_strings[0]}, div({str_function[0]}, abs({str_function[0]})))"
    
    def get_simplification_subtree(self):
        return super().get_simplification_subtree([si.abs_null,
                                                   si.abs_double_mul_null,
                                                   si.abs_scission,
                                                   si.abs_destroy_neg])

class exp(UnaryOperationNode):
    
    def __init__(self):
        super().__init__()
        
    def check_output_domain(self, output):
        if(type(output) in [float, int]): return output > 0
        else: return np.all(output > 0)

    def forward_calculation(self):
        with np.errstate(over='ignore'):
            result = np.exp(self.get_childrens()[0].get_memory("forward"))
        return result

    def backward_calculation(self):
        with np.errstate(over='ignore'):
            result = np.log(self.get_memory("backward"))
        return [result]
    
    def calculation_hat_confidence(self, x, y):
        result = ind.spearman_correlation(x, y)
        return result
    
    def calculate_statistical_score(total_analysis):
        return np.abs(total_analysis["monoton"]) * (1 - np.abs(total_analysis["vertical_asymptote"]))
    
    def get_string_calculation_structure(self, x):
        return f"np.exp({x[0]})"
    
    def get_string_derative_structure(self, sub_strings, x):
        super().get_string_derative_structure(sub_strings, x)
        str_function = [str(x) for x in self.get_childrens()]
        return f"mul(exp({str_function[0]}), {sub_strings[0]})"

    def get_simplification_subtree(self):
        return super().get_simplification_subtree([si.exp_ln_null])

class ln(UnaryOperationNode):
    
    def __init__(self):
        super().__init__()

    def check_input_domain(self, i, input):
        
        super().check_input_domain(i, input)
        if(not (self._check_type_name(input) and type(input) != str)): return False

        #Domain argument
        if(i == 0 and type(input) in [float, int]): return input > 0
        elif(i == 0): return np.all(input > 0)
 
    def forward_calculation(self):
        with np.errstate(over='ignore'):
            result = np.log(self.get_childrens()[0].get_memory("forward"))
        return result
    
    def backward_calculation(self):
        with np.errstate(over='ignore'):
            result = np.exp(self.get_memory("backward"))

        return [result]
    
    def calculation_hat_confidence(self, x, y):
    
        In, Out = ind.prepare(x, y)
    
        mask_right = In > 0
        
        result = ind.spearman_correlation(In[mask_right], Out[mask_right])
    
        return result if np.sum(mask_right) > 1 else 0
    
    def calculate_statistical_score(total_analysis):
        return np.abs(total_analysis["monoton"]) * np.abs(total_analysis["vertical_asymptote"])
   
    def get_string_calculation_structure(self, x):
        return f"np.log({x[0]})"
    
    def get_string_derative_structure(self, sub_strings, x):
        super().get_string_derative_structure(sub_strings, x)
        str_function = [str(x) for x in self.get_childrens()]
        return f"mul(div({str_function[0]}), {sub_strings[0]})"

    def get_simplification_subtree(self):
        return super().get_simplification_subtree([si.ln_exp_null])




