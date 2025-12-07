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
        
        self.__string_calculation_memory = None
        self.__string_gradient_memory = None
        self.__string_hessian_memory = None
        
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
    
    def get_string_calculation_memory(self):
        return str(self.__string_calculation_memory)
    
    def get_string_gradient_memory(self):
        return self.__string_gradient_memory.copy()

    def get_string_hessian_memory(self):
        return self.__string_hessian_memory.copy()
    
    def check_set_name(self, name):
        return True
    
    def __check_value_memory(self, value):
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
        self.__check_value_memory(value)
        self.__forward_calculation_memory = value
    
    def set_backward_calculation_memory(self, value):
        self.__check_value_memory(value)
        self.__backward_calculation_memory = value
        
    def set_string_calculation_memory(self, value):
        if(type(value) != str and value != None): raise ValueError("String memory must contains only a string")
        self.__string_calculation_memory = value
    
    def set_string_gradient_memory(self, value):
        self.__check_value_memory(value)
        self.__string_gradient_memory = value
        
    def set_string_hessian_memory(self, value):
        self.__check_value_memory(value)
        self.__string_hessian_memory = value

    def forward_calculation(self):
        return NotImplemented
    
    def backward_calculation(self):
        return NotImplemented
    
    def string_calculation(self):
        return NotImplemented
    
    def string_gradient(self):
        return NotImplemented
    
    def string_hessian(self):
        return NotImplemented
    
    def check_input_domain(self):
        return NotImplemented
    
    def check_output_domain(self):
        return NotImplemented
     
    def copy_sub_tree(self):
        
        result = self.__class__(self.__channels, self.__name)
        
        result.__childrens = [None for _ in self.__childrens]
        for i, child in enumerate(self.__childrens):
            if(child == None): continue
            OperationNode.new_connection(result, child, i)

        result.set_forward_calculation_memory(self.__forward_calculation_memory.copy())
        result.set_backward_calculation_memory(self.__backward_calculation_memory.copy())
        result.set_string_calculation_memory(self.__string_calculation_memory.copy())
        result.set_string_gradient_memory(self.__string_gradient_memory.copy())
        result.set_string_hessian_memory(self.__string_hessian_memory.copy())
        
        return result


    def view_sub_tree(self, f):
    
        class CounterGraph:
            def __init__(self):
                self.i = 0

        Printer = Digraph()

        count = CounterGraph()

        def recursion(node, level):

            if(type(node) in [Leaf]):
                str_node = node.get_name()
                count.i += 1
                Printer.node(f"Child {count.i}", f"{str_node}\n{f(node)}", {'shape': 'ellipse', 'color': 'green' if type(node.get_name()) == str else 'red', 'style': 'filled'})
                return f"Child {count.i}"
            
            Printer.node(str(id(node)), f"{node.__class__.__name__}\n{f(node)}", {'shape': 'ellipse', 'color': 'yellow', 'style': 'filled'})

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
     

class Leaf(OperationNode):
    def __init__(self, label):
        super().__init__(0, label)
    
    def check_set_name(self, name):
        return type(name) in [str, int, float]  
    
    def forward_calculation(self):
        return self.get_forward_calculation_memory()
    
    def backward_calculation(self):
        return []
    
    def string_calculation(self):
        return self.get_name()
        
    def string_gradient(self):
        return np.array(["1" if type(self.get_name()) != str else "0"])
    
    def string_hessian(self):
        return np.array([["0"]])
    


class neg(OperationNode):
    def __init__(self):
        super().__init__(1)
        
    def forward_calculation(self):
        return -self.get_childrens()[0].get_forward_calculation_memory()
    
    def backward_calculation(self):
        return [-self.get_backward_calculation_memory()]
    
    def string_calculation(self):
        return f"-({self.get_childrens()[0].get_string_calculation_memory()})"
        
    def string_gradient(self):
        
        formula = self.get_childrens()[0].get_string_calculation_memory()
        gradients = self.get_childrens()[0].get_string_gradient_memory()
        
        return [f"-(({s}) * ({formula}))" for s in gradients]
    
    def string_hessian(self):
        return NotImplemented
    
    
    

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
    
    def string_calculation(self):
        childrens = self.get_childrens()
        return f"({childrens[0].get_string_calculation_memory()} + {childrens[1].get_string_calculation_memory()})"
        
    def string_gradient(self):
        childrens = self.get_childrens()
        return childrens[1].get_string_gradient_memory() + childrens[0].get_string_gradient_memory()
    
    def string_hessian(self):
        return NotImplemented
    
    
    
    
    
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
        
    def string_calculation(self):
        childrens = self.get_childrens()
        return f"({childrens[0].get_string_calculation_memory()} - {childrens[1].get_string_calculation_memory()})"
        
    def string_gradient(self):
        childrens = self.get_childrens()
        return childrens[0].get_string_gradient_memory() + [f"-({s})" for s in childrens[1].get_string_gradient_memory()]
    
    def string_hessian(self):
        return NotImplemented
    
    
    

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
        
    def string_calculation(self):
        childrens = self.get_childrens()
        return f"({childrens[0].get_string_gradient_memory()} * {childrens[1].get_string_gradient_memory()})"
       
    def string_gradient(self):
        
        childrens = self.get_childrens()
        
        str_calc1 = childrens[0].get_string_calculation_memory()
        str_calc2 = childrens[1].get_string_calculation_memory()
        
        str_grad1 = childrens[0].get_string_gradient_memory()
        str_grad2 = childrens[1].get_string_gradient_memory()
        
        return [f"({s} * {str_calc2})" for s in str_grad1] + [f"({str_calc1} * {s})" for s in str_grad2]
    
    def string_hessian(self):
        return NotImplemented
    



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
        
    def string_calculation(self):
        childrens = self.get_childrens()
        return f"({childrens[0].get_string_gradient_memory()} / {childrens[1].get_string_gradient_memory()})"
       
    def string_gradient(self):
        
        childrens = self.get_childrens()
        
        str_calc1 = childrens[0].get_string_calculation_memory()
        str_calc2 = childrens[1].get_string_calculation_memory()
        
        str_grad1 = childrens[0].get_string_gradient_memory()
        str_grad2 = childrens[1].get_string_gradient_memory()

        return [f"(1 / ({str_calc2}))" for _ in str_grad1] + [f"-(({str_calc1}) / (({str_calc2})**2))" for _ in str_grad2]
    
    def string_hessian(self):
        return NotImplemented





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