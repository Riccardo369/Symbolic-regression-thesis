#Method that flat all elements inside a type chain (like a chain of add or mul)
def flat_elements(node, binary_type):
        
    if(isinstance(node, binary_type)):
        result = []
        for c in node.get_childrens():  result += flat_elements(c, binary_type)
        return result
        
    else: return [node]