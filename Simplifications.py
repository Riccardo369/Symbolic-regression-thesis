import numpy as np

import Operations as op
import Exceptions as ex
import utils

def close_to(a, b):
    return np.abs(a - b) <= 1e-6

def prepare_chain(node, binary_type):

    elements = utils.flat_elements(node, binary_type)
    constants = [x.get_name() for x in elements if x.is_constant()]
    functions = [{"f": x} for x in elements if not x.is_constant()]
    del elements
        
    #Extract the coefficient from function multiplied by coefficient
    for i in range(len(functions)):
        
        #If is an multiplication (a coeff * f, f * coeff or f * g)
        if(isinstance(functions[i]["f"], op.mul)):
        
            childrens = functions[i]["f"].get_childrens()
            
            #Divide constant from function
            
            if(childrens[0].is_constant()):
                functions[i]["coeff"] = childrens[0].get_name()
                functions[i]["f"] = childrens[1]
                
            elif(childrens[1].is_constant()):
                functions[i]["coeff"] = childrens[1].get_name()
                functions[i]["f"] = childrens[0]
            
            #It is not possible to divide, so consider the multiplication like an unique piece
            else: functions[i]["coeff"] = 1
                
        #If is not a mul
        else: functions[i]["coeff"] = 1
        
        #Place the extracted coefficient
        if(isinstance(functions[i]["f"], op.neg)):
            
            functions[i]["f"] = functions[i]["f"].get_childrens()[0]
            functions[i]["coeff"] *= -1
    
    return {"functions": functions, "constants": constants}

def release_chain(nodes, binary_type):
    
    if(len(nodes) == 0): raise ex.SimplificationError
    
    elif(len(nodes) == 1): return nodes[0]
    
    #Connect first 2 elements
    nodes = list(nodes)
    
    result = binary_type()
    
    result.new_connection(nodes[0], 0)
    result.new_connection(nodes[1], 1)
    
    del nodes[0]
    del nodes[0]
    
    #Continue to connect until the list is empty
    while(len(nodes) > 0):
        
        new_result = binary_type()
        
        new_result.new_connection(nodes[0], 0)
        new_result.new_connection(result.copy_sub_tree(), 1)
        
        result = new_result
        
        del nodes[0]
        
    return result

#a oper1 (b oper2 c) --> (a oper1 b) oper2 (a oper1 c)
def distributivity_oper1_on_oper2(node, oper1, oper2):
    
    if(oper1().get_channels() != 2): raise ValueError(f"Operation 1 {oper1} must has 2 inputs")
    if(oper2().get_channels() != 2): raise ValueError(f"Operation 2 {oper2} must has 2 inputs")
    
    if(not isinstance(node, oper1)): raise ValueError(f"On node {node} it is impossible apply distributivity {oper1} on {oper2}")

    c = node.get_childrens()
        
    #Trace the element to split's position
    if(isinstance(c[1], oper2)): i = 0
    elif(isinstance(c[0], oper2)): i = 1
        
    else: raise ex.SimplificationError()
        
    element = c[i]
        
    c_observe = c[1-i].get_childrens()
    
    left  = oper1()
    right = oper1()
    
    left .new_connection(element.copy_sub_tree(), 0)
    right.new_connection(element.copy_sub_tree(), 0)
        
    left .new_connection(c_observe[0].copy_sub_tree(), 1)
    right.new_connection(c_observe[1].copy_sub_tree(), 1)
        
    result = oper2()
    
    result.new_connection(left, 0)
    result.new_connection(right, 1)
        
    return result

def scission_oper1_on_oper2(node, oper1, allow_operations):
    
    if(oper1().get_channels() != 1): raise ValueError(f"Operation 1 {oper1} must has 1 input")
    
    if(not isinstance(node, oper1)): raise ValueError(f"On node {node} it is impossible apply scission of {oper1} on own child")
    
    child = node.get_childrens()[0]
    
    if(child.__class__ not in allow_operations): raise ex.SimplificationError

    result = child.__class__()
    
    for i, g in enumerate(child.get_childrens()):
        
        new_child = oper1()
        new_child.new_connection(g.copy_sub_tree(), 0)
        result.new_connection(new_child, i)
    
    return result

def double_oper_null(node, oper):

    if(oper().get_channels() != 1): raise ValueError(f"Operation 1 {oper} must has 1 input")
    
    if(not isinstance(node, oper)): raise ValueError(f"On node {node} it is impossible apply scission of {oper} on own child")
    
    child = node.get_childrens()[0]
    
    try: gran_child = child.get_childrens()[0]
    except: raise ex.SimplificationError
    
    if(not isinstance(child, oper)): raise ex.SimplificationError
    
    return gran_child.copy_sub_tree()
    

class __Simplification:
    def __init__(self, top_down, bottom_up):
        self.__top_down = top_down
        self.__bottom_up = bottom_up
        
    def is_top_down(self):
        return self.__top_down
    
    def is_bottom_up(self):
        return self.__bottom_up
    
    def __call__(self, node):
        return NotImplemented

#BOTTOM UP SIMPLIFICATIONS

class approximation(__Simplification):
    def __init__(self):
        super().__init__(False, True)
        
    def __call__(self, node):
        if(not node.is_constant()): raise ex.SimplificationError
        
        #tolerance fissata in base alle capacità della macchina nella precisione numerica
        return op.Leaf(int(node.get_name())) if np.abs(node.get_name() - int(node.get_name())) <= 1e-6 else node

class constant(__Simplification):
    def __init__(self):
        super().__init__(False, True)
        
    def __call__(self, node):
        
        try: value = eval(node.get_string_calculation_function()['body'])
        except: raise ex.SimplificationError()
        
        if(not np.isfinite(value) and isinstance(node, op.div)): value = 0
        
        result = op.Leaf(value)
        result.set_memory("forward", value)
        return result

class add_chain(__Simplification):
    def __init__(self):
        super().__init__(False, True)
        
    def __call__(self, node):
        
        result_chain = prepare_chain(node, op.add)
        functions = result_chain["functions"]
        constants = result_chain["constants"]
        del result_chain
        
        #Take all distinct functions, considering that it is not possible to do with string (for commutative property of + and *, it is necessary __eq__)
        new_functions = []
        
        distinct_functions = []
        for x in functions:
            if(x["f"] not in distinct_functions): distinct_functions.append(x["f"])
          
        #Consider only the survival functions  
        for f in distinct_functions:
            
            coeff = sum([x["coeff"] for x in functions if x["f"] == f] + [0])
            if(coeff != 0): new_functions.append({"f": f, "coeff": coeff})
        
        del distinct_functions
        
        #Calculate the nodes to concatenate
        results = []
        
        for x in new_functions:
            
            if(x["coeff"] == 1): m = x["f"].copy_sub_tree()
            
            #elif(x["coeff"] == -1):
            #    m = op.neg()
            #    m.new_connection(x["f"].copy_sub_tree(), 0)
            
            else:
                m = op.mul()
                m.new_connection(op.Leaf(x["coeff"]), 0)
                m.new_connection(x["f"].copy_sub_tree(), 1)
            
            results.append(m)
        
        #Calculate the constant
        const = 0
        for c in constants: const += c
        del constants
        
        if(const != 0): results.append(op.Leaf(const))
        
        return release_chain(results, op.add)

class mul_chain(__Simplification):
    def __init__(self):
        super().__init__(False, True)
        
    def __call__(self, node):
        
        results = []
        
        result_chain = prepare_chain(node, op.mul)
        
        #Take the natural constants + all coefficients (all will be multiplied)
        constants = result_chain["constants"] + [x["coeff"] for x in result_chain["functions"]]
        functions = [x for x in result_chain["functions"]]
        del result_chain
        
        c = 1
        for x in constants: c *= x
        
        #If product of all constants is 0, everything is 0, and return immediatly 0
        if(c == 0): return op.Leaf(0)
        elif(c != 1): results.append(op.Leaf(c))
        
        #Consider which operation is numerator and which is denominator
        for x in functions:
            x["div"] = isinstance(x["f"], op.div)
            if(x["div"]): x["f"] = x["f"].get_childrens()[0]
            
        #Simplify and delete equal numerator and denominator
        new_functions = []
        
        #Take all distinct functions, considering that it is not possible to do with string (for commutative property of + and *, it is necessary __eq__)
        distinct_functions = []
        for x in functions:
            if(x["f"] not in distinct_functions): distinct_functions.append(x["f"])
          
        #Consider only the survival functions  
        for f in distinct_functions:
            coeff = sum([-1 if x["div"] else 1 for x in functions if x["f"] == f] + [0])
            if(coeff != 0): new_functions.append({"f": f, "mul_coeff": coeff})
        
        del distinct_functions
        
        for x in new_functions:
            
            standard = x["mul_coeff"] > 0
            
            for _ in range(abs(x["mul_coeff"])):
                
                if(standard): m = x["f"].copy_sub_tree()
                
                else:
                    m = op.div()
                    m.new_connection(x["f"].copy_sub_tree(), 0)
                    
                results.append(m)
        
        return release_chain(results, op.mul)

#TOP DOWN SIMPLIFICATIONS

class neg_null(__Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        return double_oper_null(node, op.neg)
    
class div_null(__Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        return double_oper_null(node, op.div)
    
class abs_null(__Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        return double_oper_null(node, op.abs)

class relax_neg(__Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        
        child = node.get_childrens()[0]
        
        result = op.mul()
        result.new_connection(op.Leaf(-1), 0)
        result.new_connection(child.copy_sub_tree(), 1)
        
        return result

class div_scission(__Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        return scission_oper1_on_oper2(node, op.div, [op.mul, op.neg])

class abs_scission(__Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        return scission_oper1_on_oper2(node, op.abs, [op.mul, op.div])

class abs_destroy_neg(__Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        
        child = node.get_childrens()[0]
        
        if(not isinstance(child, op.neg)): raise ex.SimplificationError
        
        result = op.abs()
        result.new_connection(child.get_childrens()[0].copy_sub_tree(), 0)
        
        return result

class abs_double_mul_null(__Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        
        child = node.get_childrens()[0]
        
        if(not isinstance(child, op.mul)): raise ex.SimplificationError
        
        childrens = child.get_childrens()
        
        if(childrens[0] == childrens[1]): return child
        
        raise ex.SimplificationError

class mul_distributivity_on_add(__Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        return distributivity_oper1_on_oper2(node, op.mul, op.add)