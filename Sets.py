import numpy as np

class Set:
    
    def __init__(self):
        pass
    
    def neighbour(self, values):
        return NotImplemented
    
class SimpleSet(Set):
    
    def __init__(self):
        super().__init__()
    
class ModuleSet(Set):
    
    def __init__(self):
        super().__init__()
        
class AbsSet(Set):
    
    def __init__(self):
        super().__init__()
        
        
    
class GoniometricSet(Set):
    
    def __init__(self, x):
        super().__init__()
        self.__x = x % (np.pi * 2)   
        
        
        

class SinSet(GoniometricSet):
    
    def __init__(self, x):
        super().__init__(x)
    
class CosSet(GoniometricSet):
    
    def __init__(self, x):
        super().__init__(x)
     
class TanSet(GoniometricSet):
    
    def __init__(self, x):
        super().__init__(x)
