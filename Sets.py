import numpy as np

class Set:
    
    def __init__(self):
        pass
    
class SuriettiveInputDomainSet(Set):
    
    def neighbour(self, output):
        return NotImplemented
    

class DomainCheckSet(Set):
    
    def check(self):
        return NotImplemented