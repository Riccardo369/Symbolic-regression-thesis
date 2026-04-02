import numpy as np

from graphviz import Digraph
import matplotlib.pyplot as plt
from PIL import Image
import io

import Domains as dm
import Exceptions as ex

class OperationNode:
    
    def __init__(self, channels, is_commutative = False, name = None):
        
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
        
        self.__memories = dict()   
        
        self.__memories["forward"] = None
        self.__memories["backward"] = None

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

    #GET PARAMETERS ZONE
    def is_commutative(self):
        return self.__is_commutative
         
    def get_channels(self):
        return self.__channels
        
    def get_name(self):
        return self.__name
        
    def get_father(self):
        return self.__father
    
    def get_childrens(self):
        return list(self.__childrens)
    
    #GET MEMORY ZONE
    def get_memory(self, memory_name):
        if(memory_name not in self.__memories.keys()): return None
        value = self.__memories[memory_name]
        return value.copy() if isinstance(value, np.ndarray) else value

    #SET MEMORY ZONE
    def set_memory(self, memory_name, value):  
        if(not isinstance(value, np.ndarray) and type(value) not in [float, int] and not isinstance(value, np.generic) and value != None): raise ValueError("Memory must contains only a numpy tensor or a number float / int")
        self.__memories[memory_name] = value.copy() if isinstance(value, np.ndarray) else value
        return self
    
    def set_name(self, name):        
        if(not self.__check_type_name(name) and name != None): raise ValueError(f"For this node {self.__class__.__name__}, {name} like name is not acceptable")
        self.__name = name
        return self

    #CALCULATION AND AUTOMATICALLY SET MEMORY ALL SUB TREE ZONE
    def forward_calculation_memory_sub_tree(self, inputs_dict):
          
        def recursion(node):
            for child in node.get_childrens(): recursion(child)
            if(type(node) == Leaf): node.set_memory("forward", inputs_dict[node.get_name()] if type(node.get_name()) == str else node.get_name())
            else:                   node.set_memory("forward", node.forward_calculation())
                
        recursion(self)
        
        return self

    def backward_calculation_memory_sub_tree(self, output):
        
        self.set_memory("backward", output)
        
        def recursion(node):
            
            if(not issubclass(type(node), Leaf)):
                for child, backward in zip(node.get_childrens(), node.backward_calculation()):                
                    child.set_memory("backward", backward)
                    recursion(child)

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
    def __check_type_name(self, name):
        return True
    
    def check_input_domain(self, i, input):
        if(i < 0 or i > self.__channels - 1): raise ValueError(f"Operation {type(self)} has only {self.__channels}, so i = {i} has no coherence (start count from 0)")
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

    #Copy all sub tree
    def copy_sub_tree(self):
        
        result = self.__class__(self.__name) if(self.__class__.__name__ == "Leaf") else self.__class__()
        
        result.__childrens = [None for _ in self.__childrens]
        
        #Non necessita della copia perchè i metodi set fanno una copia già di default
        for key in self.__memories.keys():
            result.set_memory(key, self.__memories[key])
        
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
    
    def __check_type_name(self, name):
        return type(name) in [str, int, float] or isinstance(name, np.generic)
    
    def is_variable(self):
        return type(self.get_name()) in [str]
    
    def forward_calculation(self):
        raise ValueError("It is a leaf, it is not possible to do a forward calculation, it has no children")
    
    def backward_calculation(self):
        raise ValueError("It is a leaf, it is not possible to do a backward calculation, it has no children")

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
    
    def get_string_structure(self, x):
        return f"{x[0]} + {x[1]}"

class mul(OperationNode):
    def __init__(self):
        super().__init__(2, True)

    def forward_calculation(self):
        inputs = [x.get_memory("forward") for x in self.get_childrens()]
        return inputs[0] * inputs[1]
    
    def backward_calculation(self):
        inputs = [x.get_memory("forward") for x in self.get_childrens()]
        output = self.get_memory("backward")
        
        if(np.any(inputs[0] == 0)): raise ex.BackwardCalculationError("first input = 0 in some points")
        if(np.any(inputs[1] == 0)): raise ex.BackwardCalculationError("first input = 1 in some points")
        
        return [output / inputs[1], output / inputs[0]]

    def get_string_structure(self, x):
        return f"{x[0]} * {x[1]}"

class div(OperationNode):
    def __init__(self):
        super().__init__(2, True)

    def forward_calculation(self):
        inputs = [x.get_memory("forward") for x in self.get_childrens()]
        denominator = inputs[1].get_memory("forward")
        if(np.any(denominator == 0)): raise ex.ForwardCalculationError("Denominator = 0 in some points")
        return inputs[0] / denominator
    
    def backward_calculation(self):
        inputs = [x.get_memory("forward") for x in self.get_childrens()]
        output = self.get_memory("backward")
        denominator2 = inputs[0]
        if(np.any(denominator2 == 0)): raise ex.BackwardCalculationError("Denominator = 0 in some points")
        
        return [output * inputs[1], output / denominator2]

    def get_string_structure(self, x):
        return f"{x[0]} / {x[1]}"
   
class sin(OperationNode):
    def __init__(self):
        super().__init__(1)
        
    def check_output_domain(self, output):
        return np.all(output >= -1 & output <= 1)
    
    def forward_calculation(self):
        return np.sin(self.get_childrens()[0].get_memory("forward"))
    
    def backward_calculation(self):
        real_output = self.get_memory("backward")
        if(np.any((real_output < -1) | (real_output > 1))): raise ex.BackwardCalculationError("Output not valid for backward sine")
        return [dm.get_ideal_input_sin(self.get_childrens()[0].get_memory("forward"), real_output)]

    def get_string_structure(self, x):
        return f"np.sin({x[0]})"

class cos(OperationNode):
    def __init__(self):
        super().__init__(1)
        
    def check_output_domain(self, output):
        return np.all(output >= -1 & output <= 1)
    
    def forward_calculation(self):
        return np.sin(self.get_childrens()[0].get_memory("forward"))
    
    def backward_calculation(self):
        real_output = self.get_memory("backward")
        if(np.any((real_output < -1) | (real_output > 1))): raise ex.BackwardCalculationError("Output not valid for backward cosine")
        return [dm.get_ideal_input_cos(self.get_childrens()[0].get_memory("forward"), real_output)]

    def get_string_structure(self, x):
        return f"np.cos({x[0]})"



#MANCA DA FARE OPERAZIONI

#exp richiede log
#log richiede log


#add       R        sposta
#sub       R        sposta
#mul       R        sposta
#div       c2 != 0  sposta
#exp       if 1/c2 % == 0 c1 >= 0 else R  taglia
#log       c1 > 0 (argomento) and c2 > 1 (argomento)  taglia
#sin       R                             
#cos       R
#tan       x != (pi / 2) + kn
#Valore assoluto R


#MANCA IL CALCOLO INVERSO GENERALE (QUINDI CREARE CLASSI APPOSITE DI INSIEMI CHE GESTISCONO PIù SOLUZIONI CONTEMPORANEAMENTE)
#PER FUNZIONI SURRIETTIVE
#MANCA LA GESTIONE DEL DOMINIO