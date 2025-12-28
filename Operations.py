import numpy as np

import matplotlib.pyplot as plt
from graphviz import Digraph
import matplotlib.pyplot as plt
from PIL import Image
import io

class OperationNode:
    
    def __init__(self, channels, name = None):
        
        if(type(channels) != int): raise ValueError(f"Channels number must be a int value, cannot be a {channels}")
        
        self.__channels = channels
        
        self.__father = None
        
        self.set_name(name)
        
        self.__childrens = [None for _ in range(channels)]

        self.__forward_calculation_memory = None
        self.__backward_calculation_memory = None
        
        self.__weight_vector_memory = None
        
    def __str__(self):
        if(isinstance(self, Leaf)): return self.__name if type(self.__name) == "str" else str(self.__name)
        return f"{self.__class__.__name__}({', '.join([str(x) for x in self.__childrens])})"
        
    def get_channels(self):
        return self.__channels
        
    def get_name(self):
        return self.__name
        
    def get_father(self):
        return self.__father
    
    def get_childrens(self): return list(self.__childrens)
    
    def get_forward_calculation_memory(self):
        value = self.__forward_calculation_memory
        return value.copy() if isinstance(value, np.ndarray) else value
    
    def get_backward_calculation_memory(self):
        value = self.__backward_calculation_memory.copy()
        return value.copy() if isinstance(value, np.ndarray) else value
    
    def get_weight_vector_memory(self):
        value = self.__weight_vector_memory.copy()
        return value.copy() if isinstance(value, np.ndarray) else value
    
    def check_set_name(self, name):
        return True
    
    def __check_value_memory(self, value):
        if(not isinstance(value, np.ndarray) and type(value) not in [float, int] and not isinstance(value, np.generic) and value != None): raise ValueError("Memory must contains only a numpy tensor")

    def set_name(self, name):
        if(not self.check_set_name(name) and name != None): raise ValueError(f"For this node {self.__class__.__name__}, {name} like name is not acceptable")
        self.__name = name
        
    def free_connection(father, child):
        if(not issubclass(type(father), OperationNode)): raise ValueError(f"Father must be a Operation node type, not a {type(father)}")
        if(not issubclass(type(child), OperationNode)): raise ValueError(f"Child must be a Operation node type, not a {type(child)}")
        
        if(child.__father != father): raise ValueError(f"Child {child} has not like a father {father}, but has {child.__father}")
        if(child not in father.__childrens): raise ValueError(f"Father {father} has not {child} in own childrens")
        
        brothers = child.__father.__childrens
        brothers[brothers.index(child)] = None
        del brothers
        
        child.__father = None
        
    def new_connection(father, child, i):
        
        if(not issubclass(type(father), OperationNode)): raise ValueError(f"Father must be a Operation node type, not a {type(father)}")
        if(not issubclass(type(child), OperationNode)): raise ValueError(f"Child must be a Operation node type, not a {type(child)}")
        
        if(child.__father != None): raise ValueError(f"Child {child} has also a father {father}, so connection is useless")
        if(child in father.__childrens): raise ValueError(f"Father {father} has {child} in own childrens, so connection is useless")
        
        if(i >= father.__channels): raise ValueError(f"Cannot set a {i}-th channel in a node with {father.__channels} children for father")
        if(father.__childrens[i] != None): raise ValueError("Father has already a child in {i}-th position")
        
        father.__childrens[i] = child
        child.__father = father
    
    def set_forward_calculation_memory(self, value):
        self.__check_value_memory(value)
        self.__forward_calculation_memory = value.copy() if isinstance(value, np.ndarray) else value
    
    def set_backward_calculation_memory(self, value):
        self.__check_value_memory(value)
        self.__backward_calculation_memory = value.copy() if isinstance(value, np.ndarray) else value
        
    def set_weight_vector_memory(self, value):
        self.__check_value_memory(value)
        self.__weight_vector_memory = value.copy() if isinstance(value, np.ndarray) else value
        
    def __clear_memory_sub_tree(self, set_function):
        
        def recursion(node):
            if(not issubclass(type(node), Leaf)):
                for child in node.get_childrens(): recursion(child)
                set_function(child)
        
        recursion(self)
        
    def clear_all_memory_sub_tree(self):
        self.clear_forward_calculation_memory_sub_tree()
        self.clear_backward_calculation_memory_sub_tree()
        self.clear_weight_calculation_memory_sub_tree()
        
    def clear_forward_calculation_memory_sub_tree(self):
        self.__clear_memory_sub_tree(lambda x: x.set_forward_calculation_memory(None))
        
    def clear_backward_calculation_memory_sub_tree(self):
        self.__clear_memory_sub_tree(lambda x: x.set_backward_calculation_memory(None))
        
    def clear_weight_calculation_memory_sub_tree(self):
        self.__clear_memory_sub_tree(lambda x: x.set_weight_vector_memory(None))
        
    def get_string_structure(self, x):
        return NotImplemented

    #Calculate forward value
    def forward_calculation(self):
        return NotImplemented
    
    #Calculate backward value considering the forward values from childrens
    def backward_calculation(self):
        return NotImplemented
    
    #Calculate weights for indipendent backward, taking forward values from childrens
    def weight_vector_calculation(self):
        return NotImplemented
    
    #Calculate indipendent backwards using weights and output
    def backward_indipendent_calculation(self):
        return NotImplemented
    
    def check_input_domain(self):
        return NotImplemented
    
    def check_output_domain(self):
        return NotImplemented
     
    def copy_sub_tree(self):
        
        result = self.__class__(self.__name) if(self.__class__.__name__ == "Leaf") else self.__class__()
        
        result.__childrens = [None for _ in self.__childrens]
        
        #Non necessita della copia perchè i metodi set fanno una copia già di default
        result.set_forward_calculation_memory(self.__forward_calculation_memory)
        result.set_backward_calculation_memory(self.__backward_calculation_memory)
        result.set_weight_vector_memory(self.__weight_vector_memory)
        
        for i, child in enumerate(self.__childrens):
            if(child == None): continue
            OperationNode.new_connection(result, child.copy_sub_tree(), i)

        return result

    #brown -> magenta
    #blue -> cyan
    #purple -> yellow

    def view_sub_tree(self, f = lambda x: ""):
    
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
        plt.show(block = False)
     

class Leaf(OperationNode):
    def __init__(self, label):
        super().__init__(0, label)
    
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
        super().__init__(1)
        
    def forward_calculation(self):
        return -self.get_childrens()[0].get_forward_calculation_memory()
    
    def backward_calculation(self):
        return [-self.get_backward_calculation_memory()]
    
    def get_string_structure(self, x):
        return f"-{x[0]}"
    
    def weight_vector_calculation(self):
        return 0
    
    def backward_indipendent_calculation(self):
        return self.get_backward_calculation_memory()

class add(OperationNode):
    def __init__(self):
        super().__init__(2)
        
    def forward_calculation(self):
        childrens = self.get_childrens()
        return childrens[0].get_forward_calculation_memory() + childrens[1].get_forward_calculation_memory()
    
    def backward_calculation(self):
        childrens = self.get_childrens()
        output = self.get_backward_calculation_memory() 
        return [output - childrens[1].get_forward_calculation_memory(), output - childrens[0].get_forward_calculation_memory()]
    
    def get_string_structure(self, x):
        return f"{x[0]} + {x[1]}"
    
    def weight_vector_calculation(self):
        inputs = [self.get_childrens()[0].get_forward_calculation_memory(), self.get_childrens()[1].get_forward_calculation_memory()]
        output = self.get_backward_calculation_memory()
        return NotImplemented
    
    def backward_indipendent_calculation(self):
        output = self.get_backward_calculation_memory() 
        weights = self.get_weight_vector_memory()
        return [weights * output, (1 - weights) * output]
    
    
    
class sub(OperationNode):
    def __init__(self):
        super().__init__(2)
        
    def forward_calculation(self):
        childrens = self.get_childrens()
        return childrens[0].get_forward_calculation_memory() - childrens[1].get_forward_calculation_memory()
    
    def backward_calculation(self):
        childrens = self.get_childrens()
        output = self.get_backward_calculation_memory() 
        return [output + childrens[1].get_forward_calculation_memory(), -output + childrens[0].get_forward_calculation_memory()]
        
    def get_string_structure(self, x):
        return f"{x[0]} - {x[1]}"
    
    def weight_vector_calculation(self):
        inputs = [self.get_childrens()[0].get_forward_calculation_memory(), self.get_childrens()[1].get_forward_calculation_memory()]
        output = self.get_backward_calculation_memory()
        return NotImplemented
    
    def backward_indipendent_calculation(self):
        output = self.get_backward_calculation_memory() 
        weights = self.get_weight_vector_memory()
        return [weights * output, (1 - weights) * output]
    

class mul(OperationNode):
    def __init__(self):
        super().__init__(2)
        
    def forward_calculation(self):
        childrens = self.get_childrens()
        return childrens[0].get_forward_calculation_memory() * childrens[1].get_forward_calculation_memory()
    
    def backward_calculation(self):
        childrens = self.get_childrens()
        output = self.get_backward_calculation_memory() 
        return [output / childrens[1].get_forward_calculation_memory(), output / childrens[0].get_forward_calculation_memory()]

    def get_string_structure(self, x):
        return f"{x[0]} * {x[1]}"
    
    def weight_vector_calculation(self):
        inputs = [self.get_childrens()[0].get_forward_calculation_memory(), self.get_childrens()[1].get_forward_calculation_memory()]
        output = self.get_backward_calculation_memory()
        return NotImplemented
    
    def backward_indipendent_calculation(self):
        output = self.get_backward_calculation_memory() 
        weights = self.get_weight_vector_memory()
        return [output ** weights, output ** (1 - weights)]


class div(OperationNode):
    def __init__(self):
        super().__init__(2)
        
    def check_input_domain(self):
        return np.all(self.get_childrens[1].get_forward_calculation_memory() != 0)
    
    def forward_calculation(self):
        childrens = self.get_childrens()
        return childrens[0].get_forward_calculation_memory() / childrens[1].get_forward_calculation_memory()
    
    def backward_calculation(self):
        childrens = self.get_childrens()
        output = self.get_backward_calculation_memory() 
        return [output * childrens[1].get_forward_calculation_memory(), childrens[0].get_forward_calculation_memory() / output]

    def get_string_structure(self, x):
        return f"{x[0]} / {x[1]}"
    
    def weight_vector_calculation(self):
        inputs = [self.get_childrens()[0].get_forward_calculation_memory(), self.get_childrens()[1].get_forward_calculation_memory()]
        output = self.get_backward_calculation_memory()
        return NotImplemented
    
    def backward_indipendent_calculation(self):
        output = self.get_backward_calculation_memory() 
        weights = self.get_weight_vector_memory()
        return [output ** weights, output ** (weights - 1)]

#IDEA IN MODEL: LE FUNZIONI string_(calculation, gradient, hessian), servono per creare 1 o pià funzioni contemporaneamente 
# sotto forma di simboli diretti in python, poi il corpo verrà passato in un eval (e non è necessario prendere i nomi delle 
# foglie, perchè il modello passerà un dizionario ogni volta, che associa il nome della foglia ed il valore / feature assegnato)


#MANCA DA FARE OPERAZIONI


#exp richiede log
#log richiede log


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