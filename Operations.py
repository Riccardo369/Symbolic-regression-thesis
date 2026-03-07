import numpy as np

import matplotlib.pyplot as plt
from graphviz import Digraph
import matplotlib.pyplot as plt
from PIL import Image
import io

class OperationNode:
    
    def __init__(self, channels, is_commutative, name = None):
        
        if(type(channels) != int): raise ValueError(f"Channels number must be a int value, cannot be a {channels}")
        
        #Channel numbers
        self.__channels = channels
        
        #Special properties
        self.__is_commutative = is_commutative
        
        #Father
        self.__father = None
        
        #Set parameters
        self.set_name(name)
        
        self.__childrens = [None for _ in range(channels)]
        
        self.__forward_calculation_memory = None
        self.__backward_calculation_memory = None
        self.__factorization_backward_calculation_memory = None
        self.__weight_memory = None     

    def __str__(self):
        if(isinstance(self, Leaf)): return self.__name if type(self.__name) == "str" else str(self.__name)
        return f"{self.__class__.__name__}({', '.join([str(x) for x in self.__childrens])})"
       
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
       
    #EQUALS SUB TREE STRUCTURE ZONE 
    def __compare_parameter(a, b, f):
            return f(a) == f(b)
        
    def structure_equals(self, node):
        return all([OperationNode.__compare_parameter(self, node, lambda x: type(x)),
                    OperationNode.__compare_parameter(self, node, lambda x: x.get_name()),
                    all([node.strucutre_equals(child) for child in node.get_childrens()])])
    
    def calculation_memory_equals(self, node):
        
        return all([OperationNode.__compare_parameter(self, node, lambda x: type(x)),
                    OperationNode.__compare_parameter(self, node, lambda x: x.get_forward_calculation_memory()),
                    OperationNode.__compare_parameter(self, node, lambda x: x.get_backward_calculation_memory()),
                    OperationNode.__compare_parameter(self, node, lambda x: x.get_factorization_backward_calculation_memory()),
                    all([node.calculation_memory_equals(child) for child in node.get_childrens()])])
        
    def weight_memory_equals(self, node):
        
        return all([OperationNode.__compare_parameter(self, node, lambda x: type(x)),
                    OperationNode.__compare_parameter(self, node, lambda x: x.get_weight_memory()),
                    all([node.calculation_memory_equals(child) for child in node.get_childrens()])])
        
    def all_equals(self, node):
        return self.structure_equals(node) and self.calculation_memory_equals(node) and self.weight_memory_equals(node)
     
    #GET PARAMETERS ZONE
    def is_commutative(self):
        return self.__is_commutative
    
    def get_default_inputs(self):
        raise ValueError(f"This operation is not aviable (operation has {self.__channels} inputs (it must be > 1))")
        
    def get_channels(self):
        return self.__channels
        
    def get_name(self):
        return self.__name
        
    def get_father(self):
        return self.__father
    
    def get_childrens(self):
        return list(self.__childrens)
    
    #GET MEMORY ZONE
    def __get_memory(self, value):
        return value.copy() if isinstance(value, np.ndarray) else value
    
    def get_forward_calculation_memory(self):
        return self.__get_memory(self.__forward_calculation_memory)
        
    def get_backward_calculation_memory(self):
        return self.__get_memory(self.__backward_calculation_memory)
        
    def get_weight_memory(self):
        return self.__get_memory(self.__weight_memory)
        
    def get_factorization_backward_calculation_memory(self):
        return self.__get_memory(self.__factorization_backward_calculation_memory)
 
    #SET MEMORY ZONE
    def __check_value_memory(self, value):
        if(not isinstance(value, np.ndarray) and type(value) not in [float, int] and not isinstance(value, np.generic) and value != None): raise ValueError("Memory must contains only a numpy tensor or a number float / int")

    def set_name(self, name):
        if(not self.check_set_name(name) and name != None): raise ValueError(f"For this node {self.__class__.__name__}, {name} like name is not acceptable")
        self.__name = name
        return self
      
    def __prepare_set_memory(self, value):
        self.__check_value_memory(value)
        return value.copy() if isinstance(value, np.ndarray) else value
    
    def set_forward_calculation_memory(self, value):
        self.__forward_calculation_memory = self.__prepare_set_memory(value)
        return self
        
    def set_backward_calculation_memory(self, value):
        self.__backward_calculation_memory = self.__prepare_set_memory(value)
        return self
        
    def set_weight_memory(self, value):
        self.__weight_memory = self.__prepare_set_memory(value)
        return self
        
    def set_factorization_backward_calculation_memory(self, value):
        self.__factorization_backward_calculation_memory = self.__prepare_set_memory(value)
        return self
   
    #CALCULATION AND AUTOMATICALLY SET MEMORY ALL SUB TREE ZONE
    def forward_calculation_memory_sub_tree(self, inputs_dict):
          
        def recursion(node):
            for child in node.get_childrens(): recursion(child)
            if(type(node) == Leaf):
                node.set_forward_calculation_memory(inputs_dict[node.get_name()] if type(node.get_name()) == str else node.get_name())
            else: node.set_forward_calculation_memory(node.forward_calculation())
                
        recursion(self)
        
        return self

    def backward_calculation_memory_sub_tree(self, output):
        
        self.set_backward_calculation_memory(output)
        
        def recursion(node):
            
            if(not issubclass(type(node), Leaf)):
                for child, backward in zip(node.get_childrens(), node.backward_calculation()):                
                    child.set_backward_calculation_memory(backward)
                    recursion(child)

        recursion(self)
        
        return self
    
    #REQUIRE: ogni nodo deve avere il suo forward
    def factorization_backward_calculation_memory_sub_tree(self, output):
        
        self.set_factorization_backward_calculation_memory(output)
        
        def recursion(node):
            
            if(not issubclass(type(node), Leaf)):
                
                node.set_weight_memory(node.weight_calculation())    
                
                for child, backward in zip(node.get_childrens(), node.factorization_backward_calculation()):   
                    child.set_factorization_backward_calculation_memory(backward)
                    recursion(child)

        recursion(self)
        
        return self

    #CLEAR ALL SUB TREE MEMORY ZONE
    def __clear_memory_sub_tree(self, set_function):
        
        def recursion(node):
            if(not issubclass(type(node), Leaf)):
                for child in node.get_childrens(): recursion(child)
                set_function(child)
        
        recursion(self)
         
    def clear_all_calculation_memory_sub_tree(self):
        self.clear_forward_calculation_memory_sub_tree()
        self.clear_backward_calculation_memory_sub_tree()
        self.clear_factorization_backward_calculation_memory_sub_tree()
        return self
        
    def clear_forward_calculation_memory_sub_tree(self):
        self.__clear_memory_sub_tree(lambda x: x.set_forward_calculation_memory(None))
        return self
        
    def clear_backward_calculation_memory_sub_tree(self):
        self.__clear_memory_sub_tree(lambda x: x.set_backward_calculation_memory(None))
        return self
        
    def clear_weight_calculation_memory_sub_tree(self):
        self.__clear_memory_sub_tree(lambda x: x.set_weight_memory(None))
        return self
    
    def clear_factorization_backward_calculation_memory_sub_tree(self):
        self.__clear_memory_sub_tree(lambda x: x.set_factorization_backward_calculation_memory(None))
        return self

    #CHECK OPERATION ZONE
    def check_set_name(self, name):
        return True
    
    def check_input_domain(self, i, input):
        return True

    def check_output_domain(self, output):
        return True
    
    #NODE METHODS (valid only for the node, not for the all sub tree). For correct functions, the require memory must be calculated already
    
    #Get the operation in string form 
    def get_string_structure(self, x):
        return NotImplemented

    #Calculate forward value. REQUIRE forwards from childrens
    def forward_calculation(self):
        return NotImplemented

    #Calculate backward value. REQUIRE backward from father and forward from childrens
    def backward_calculation(self):
        return NotImplemented
    
    #Calculate factorization backwards. REQUIRE backward from father and internal weight
    def factorization_backward_calculation(self):
        return NotImplemented
    
    #Calculate weights for factorization backward. REQUIRE backward from father and forward from childrens
    def weight_calculation(self):
        return None
    
    #Return the factorization backward inverse formula, from which i-th node you come. REQUIRE internal weight
    def get_factorization_forward_calculation_formula(self, i):
        return NotImplemented


    #Copy all sub tree
    def copy_sub_tree(self):
        
        result = self.__class__(self.__name) if(self.__class__.__name__ == "Leaf") else self.__class__()
        
        result.__childrens = [None for _ in self.__childrens]
        
        #Non necessita della copia perchè i metodi set fanno una copia già di default
        result.set_forward_calculation_memory(self.__forward_calculation_memory)
        result.set_backward_calculation_memory(self.__backward_calculation_memory)
        result.set_weight_memory(self.__weight_memory)
        result.set_factorization_backward_calculation_memory(self.__factorization_backward_calculation_memory)
        
        for i, child in enumerate(self.__childrens):
            if(child == None): continue
            OperationNode.new_connection(result, child.copy_sub_tree(), i)

        return result
    
    
    #View in a plot all sub tree structure
    def view_sub_tree(self, f = lambda x: " ", block = True):
    
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
        plt.imshow(image)
        plt.axis('off')
        plt.show(block = block)
     

class Leaf(OperationNode):
    def __init__(self, label):
        super().__init__(0, False, label)
    
    def check_set_name(self, name):
        return type(name) in [str, int, float] or isinstance(name, np.generic)
    
    def is_variable(self):
        return type(self.get_name()) in [str]
    
    def forward_calculation(self):
        raise ValueError("It is a leaf, it is not possible to do a forward calculation, it has no children")
    
    def backward_calculation(self):
        raise ValueError("It is a leaf, it is not possible to do a backward calculation, it has no children")


class neg(OperationNode):
    def __init__(self):
        super().__init__(1, False)
        
    def forward_calculation(self):
        return -self.get_childrens()[0].get_forward_calculation_memory()
    
    def backward_calculation(self):
        return [-self.get_backward_calculation_memory()]
    
    def get_string_structure(self, x):
        return f"-{x[0]}"
    
    def weight_calculation(self):
        return 0
    
    def backward_indipendent_calculation(self):
        return self.get_backward_calculation_memory()


#FACTORIZATION: a + b = c --> wc + (1-w)c = c
class add(OperationNode):
    def __init__(self):
        super().__init__(2, True)
        
    def get_default_inputs(self):
        return [0, 0]
        
    def forward_calculation(self):
        childrens = self.get_childrens()
        return childrens[0].get_forward_calculation_memory() + childrens[1].get_forward_calculation_memory()
    
    def backward_calculation(self):
        childrens = self.get_childrens()
        output = self.get_backward_calculation_memory() 
        return [output - childrens[1].get_forward_calculation_memory(), output - childrens[0].get_forward_calculation_memory()]
    
    def get_string_structure(self, x):
        return f"{x[0]} + {x[1]}"
    
    def factorization_backward_calculation(self):
        w = self.get_weight_memory()
        c = self.get_factorization_backward_calculation_memory()
        
        return [w*c, ((1 - w)*c)]
    
    def weight_calculation(self):
        a = self.get_childrens()[0].get_forward_calculation_memory()
        c = self.get_factorization_backward_calculation_memory()
        
        return np.sum(a * c) / np.sum(c**2)
     
#FACTORIZATION: a - b = c --> wc - (w-1)c = c  
class sub(OperationNode):
    def __init__(self):
        super().__init__(2, False)
        
    def get_default_inputs(self):
        return [0, 0]
        
    def forward_calculation(self):
        childrens = self.get_childrens()
        return childrens[0].get_forward_calculation_memory() - childrens[1].get_forward_calculation_memory()
    
    def backward_calculation(self):
        childrens = self.get_childrens()
        output = self.get_backward_calculation_memory() 
        return [output + childrens[1].get_forward_calculation_memory(), -output + childrens[0].get_forward_calculation_memory()]
        
    def get_string_structure(self, x):
        return f"{x[0]} - {x[1]}"
    
    def factorization_backward_calculation(self):
        w = self.get_weight_memory()
        c = self.get_factorization_backward_calculation_memory()
        
        return [w*c, ((w - 1)*c)]
    
    def weight_calculation(self):
        a = self.get_childrens()[0].get_forward_calculation_memory()
        c = self.get_factorization_backward_calculation_memory()
        
        return np.sum(a * c) / np.sum(c**2)
    
class mul(OperationNode):
    def __init__(self):
        super().__init__(2, True)
        
    def get_default_inputs(self):
        return [1, 1]
        
    def forward_calculation(self):
        childrens = self.get_childrens()
        return childrens[0].get_forward_calculation_memory() * childrens[1].get_forward_calculation_memory()
    
    def backward_calculation(self):
        childrens = self.get_childrens()
        output = self.get_backward_calculation_memory() 
        return [output / childrens[1].get_forward_calculation_memory(), output / childrens[0].get_forward_calculation_memory()]

    def get_string_structure(self, x):
        return f"{x[0]} * {x[1]}"
    
    def factorization_backward_calculation(self):
        return NotImplemented
    
    def weight_calculation(self):
        return NotImplemented


class div(OperationNode):
    def __init__(self):
        super().__init__(2, False)
        
    def get_default_inputs(self):
        return [1, 1]
        
    def check_input_domain(self, i, input):
        if(i == 1): return np.all(input != 0)
        return True
    
    def forward_calculation(self):
        childrens = self.get_childrens()
        return childrens[0].get_forward_calculation_memory() / childrens[1].get_forward_calculation_memory()
    
    def backward_calculation(self):
        childrens = self.get_childrens()
        output = self.get_backward_calculation_memory() 
        return [output * childrens[1].get_forward_calculation_memory(), childrens[0].get_forward_calculation_memory() / output]

    def get_string_structure(self, x):
        return f"{x[0]} / {x[1]}"
    
    def factorization_backward_calculation(self):
        return NotImplemented
    
    def weight_calculation(self):
        return NotImplemented




#MANCA DA FARE OPERAZIONI

#exp richiede log
#log richiede log


#neg
#add
#sub
#mul
#div
#sqrt
#sin
#cos
#tan
#cot
#sec
#csc
#Modulo
#Valore assoluto



#MANCA IL CALCOLO INVERSO GENERALE (QUINDI CREARE CLASSI APPOSITE DI INSIEMI CHE GESTISCONO PIù SOLUZIONI CONTEMPORANEAMENTE)
#PER FUNZIONI SURRIETTIVE
#MANCA LA GESTIONE DEL DOMINIO