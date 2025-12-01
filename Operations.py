import numpy as np

import matplotlib.pyplot as plt
from graphviz import Digraph
import matplotlib.pyplot as plt
from PIL import Image
import io

#Rivedere il view_graph un attimo

class OperationNode:
    
    def __init__(self, channels, name = None):
        
        if(type(channels) != int): raise ValueError(f"Channels number must be a int value, cannot be a {channels}")
        
        self.__channels = channels
        
        self.__father = None
        
        self.set_name(name)
        
        self.__childrens = [None for _ in range(channels)]

        self.__forward_calculation_memory = None
        self.__backward_calculation_memory = None
        self.__forward_gradient_memory = None
        self.__forward_hessian_memory = None
        
    def get_channels(self):
        return self.__channels
        
    def get_name(self):
        return self.__name
        
    def get_father(self):
        return self.__father
    
    def get_childrens(self): return list(self.__childrens)
    
    def get_forward_calculation_memory(self):
        return self.__forward_calculation_memory.copy()
    
    def get_backward_calculation_memory(self):
        return self.__backward_calculation_memory.copy()
    
    def get_forward_gradient_memory(self):
        return self.__forward_gradient_memory.copy()
    
    def get_forward_hessian_memory(self):
        return self.__forward_hessian_memory.copy()
    
    def check_set_name(self, name):
        return True
    
    def check_value_memory(self, value):
        if(not isinstance(value, np.array)): raise ValueError("Memory must contains only a numpy tensor")
        
    def set_name(self, name): 
        if(not self.check_set_name(name) and name != None): raise ValueError(f"For this node {self.__class__.__name__}, {name} like name is not acceptable")
        self.__name = name
        
    def free_connection(father, child):
        if(not isinstance(OperationNode, father)): raise ValueError(f"Father must be a Operation node type, not a {type(father)}")
        if(not isinstance(OperationNode, child)): raise ValueError(f"Child must be a Operation node type, not a {type(child)}")
        
        if(child.__father != father): raise ValueError(f"Child {child} has not like a father {father}, but has {child.__father}")
        if(child not in father.__childrens): raise ValueError(f"Father {father} has not {child} in own childrens")
        
        brothers = child.__father.__childrens
        brothers[brothers.index(child)] = None
        del brothers
        
        child.__father = None
        
    def new_connection(father, child, i):
        if(not isinstance(OperationNode, father)): raise ValueError(f"Father must be a Operation node type, not a {type(father)}")
        if(not isinstance(OperationNode, child)): raise ValueError(f"Child must be a Operation node type, not a {type(child)}")
        
        if(child.__father != None): raise ValueError(f"Child {child} has also a father {father}, so connection is useless")
        if(child in father.__childrens): raise ValueError(f"Father {father} has {child} in own childrens, so connection is useless")
        
        if(i >= father.__channels): raise ValueError(f"Cannot set a {i}-th channel in a node with {father.__channels} children for father")
        if(father.__childrens[i] != None): raise ValueError("Father has already a child in {i}-th position")
        
        father.__childrens[i] = child
        child.__father = father
    
    def set_forward_calculation_memory(self, value):
        self.check_value_memory(value)
        self.__forward_calculation_memory = value
    
    def set_backward_calculation_memory(self, value):
        self.check_value_memory(value)
        self.__backward_calculation_memory = value
    
    def set_forward_gradient_memory(self, value):
        self.check_value_memory(value)
        self.__forward_gradient_memory = value
    
    def set_forward_hessian_memory(self, value):
        self.check_value_memory(value)
        self.__forward_hessian_memory = value

    def forward_calculation(self):
        return NotImplemented
    
    def forward_gradient(self):
        return NotImplemented
    
    def forward_hessian(self):
        return NotImplemented
    
    def backward_calculation(self):
        return NotImplemented
    
    def copy_sub_tree(self):
        
        result = self.__class__(self.__channels, self.__name)
        
        result.__childrens = [None for _ in self.__childrens]
        for i, child in enumerate(self.__childrens):
            if(child == None): continue
            OperationNode.new_connection(result, child, i)

        result.set_forward_calculation_memory(self.__forward_calculation_memory)
        result.set_backward_calculation_memory(self.__backward_calculation_memory)
        result.set_forward_gradient_memory(self.__forward_gradient_memory)
        result.set_forward_hessian_memory(self.__forward_hessian_memory)
        
        return result
    
    
            
    def __str__(self):
        
        if(self.__class__ in [Constant, Variable]): return self.get_name()

        return f"{self.__class__.__name__}({', '.join([str(child) for child in self.get_childrens()])})"

    def view_sub_tree(self):
    
        class CounterGraph:
            def __init__(self):
                self.i = 0

        Printer = Digraph()

        count = CounterGraph()

        def recursion(node, level):

            if(type(node) in [Constant, Variable]):
                str_node = node.get_name()
                count.i += 1
                Printer.node(f"Child {count.i}", str_node, {'shape': 'ellipse', 'color': 'green' if type(node) == Variable else 'red', 'style': 'filled'})
                return f"Child {count.i}"
            
            Printer.node(str(id(node)), f"{node.__class__.__name__}", {'shape': 'ellipse', 'color': 'yellow', 'style': 'filled'})

            for child in [x for x in node.__childrens]: Printer.edge(str(id(node)), recursion(child, level+1))
                
            return str(id(node))
        
        recursion(self, 1)

        img_bytes = Printer.pipe(format='png')

        # Converte i byte in un'immagine
        image = Image.open(io.BytesIO(img_bytes))

        # Mostra l'immagine con matplotlib
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    
    
    
    
       
class Constant(OperationNode):
    def __init__(self):
        super().__init__(0)
        
    def check_set_name(self, name):
        return type(name) in [float, int]

class Variable(OperationNode):
    def __init__(self):
        super().__init__(0)
    
    def check_set_name(self, name):
        return type(name) in [str]