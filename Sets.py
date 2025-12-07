import numpy as np

class Set:
    
    def __init__(self):
        pass
    
    def neighbour(self, x):
        return NotImplemented
    

class Pi:
    
    def __init__(self, x):
        self.__x = x % (np.pi * 2)
        
    def neighbour(self, x):
        
        n = x // (np.pi * 2)
        
        lower = (n * np.pi * 2) + self.__x
        upper = ((n+1) * np.pi * 2) + self.__x
        
        return lower if np.abs(lower - x) < np.abs(upper - x) else upper
        
