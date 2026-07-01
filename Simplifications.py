import numpy as np
import itertools

import Operations as op
import Exceptions as ex
import utils


def close_to(a, b):
    return np.abs(a - b) <= 1e-6

"""
Preconditions:

- node must be a chain node

Postconditions:

- all chain (of same type node) flatted (ex. add(add(a, b, c), d, e) -> add(a, b, c, d, e)

"""
def flat_chain(node):
    
    if(not isinstance(node, op.ChainOperationNode)): raise ValueError(f"Node {node} is not a chain")
    
    elements = node.get_childrens()
    
    oper = type(node)
    
    sub_chains = [None]
    
    #Flat all oper chain in one chain (like oper(a, b, oper(c, d)) = oper (a, b, c, d))
    while(len(sub_chains) > 0):
        
        sub_chains = [x for x in elements if isinstance(x, oper)]
        elements =   [x for x in elements if not isinstance(x, oper)]
        
        for s in sub_chains: elements += s.get_childrens()
        
    return elements
    
    

"""

Preconditions:

- node must be a mul or add chain
- oper must be add or mul (the same type of node)
- f must be + or * lambda binary function
- start must be the start value of constant

Post conditions:

- flatted chain of the same type
- 1 constant (all left)
- No sub functions that can be a constant will be remain

"""
def chain_simplification(node, f, start):
    
    elements = flat_chain(node)
        
    #Transform every element in a constant (if it is possible)
    for i in range(len(elements)):
        
        try: elements[i] = constant()(elements[i])
        except ex.SimplificationError: pass
        
    constants = [x.get_name() for x in elements if x.is_constant()]
    functions = [x for x in elements if not x.is_constant()]
    del elements
    
    c = start
    for i in constants: c = f(c, i)
    
    #result = oper()
    #result.new_connection(op.Leaf(c))
    
    #for i in functions: result.new_connection(i.copy_sub_tree())

    return c, functions




def accumulation_coefficients(functions):
    
    set_functions = []
    
    for i in functions:
        if(i["f"] not in set_functions): set_functions.append(i["f"])
    
    dict_accumulation = [{"f": x, "c": 0} for x in set_functions]
    for x in dict_accumulation:
        for y in [a for a in functions if a["f"] == x["f"]]: x["c"] += y["c"]
            
    return dict_accumulation
    


#a oper1 (b oper2 c) --> (a oper1 b) oper2 (a oper1 c)
def distributivity_oper1_on_oper2(node, oper1, oper2):
    
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
    
    left .new_connection(element.copy_sub_tree())
    right.new_connection(element.copy_sub_tree())
        
    left .new_connection(c_observe[0].copy_sub_tree())
    right.new_connection(c_observe[1].copy_sub_tree())
        
    result = oper2()
    
    result.new_connection(left)
    result.new_connection(right)
        
    return result

def scission_oper1_on_oper2(node, oper1, allow_operations):
    
    if(oper1().get_channels() != 1): raise ValueError(f"Operation 1 {oper1} must has 1 input")
    
    if(not isinstance(node, oper1)): raise ValueError(f"On node {node} it is impossible apply scission of {oper1} on own child")
    
    child = node.get_childrens()[0]
    
    if(child.__class__ not in allow_operations): raise ex.SimplificationError

    result = child.__class__()
    
    for i, g in enumerate(child.get_childrens()):
        
        new_child = oper1()
        new_child.new_connection(g.copy_sub_tree())
        result.new_connection(new_child)
    
    return result

def double_oper_null(node, oper):

    if(oper().get_channels() != 1): raise ValueError(f"Operation 1 {oper} must has 1 input")
    
    if(not isinstance(node, oper)): raise ValueError(f"On node {node} it is impossible apply scission of {oper} on own child")
    
    child = node.get_childrens()[0]
    
    try: gran_child = child.get_childrens()[0]
    except: raise ex.SimplificationError
    
    if(not isinstance(child, oper)): raise ex.SimplificationError
    
    return gran_child.copy_sub_tree()
    

class Simplification:
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

class approximation(Simplification):
    def __init__(self):
        super().__init__(False, True)
        
    def __call__(self, node):
        raise ex.SimplificationError
        if(not node.is_constant()): raise ex.SimplificationError
        
        #tolerance fissata in base alle capacità della macchina nella precisione numerica
        return op.Leaf(int(node.get_name())) if np.abs(node.get_name() - int(node.get_name())) <= 1e-6 else node

class constant(Simplification):
    def __init__(self):
        super().__init__(True, True)
        
    def __call__(self, node):
        
        try: value = eval(str(node.get_string_calculation_function()['body']))
        except: raise  ex.SimplificationError
            
        result = op.Leaf(value)
        result.set_memory("forward", value)
        return result


#Cleaning from:
#
# constant (1 in mul, 0 in add)
# destroy al mul chain if constant = 0
# cleaning add and mul chain if there is only 1 child

class cleaning(Simplification):
    
    def __init__(self):
        super().__init__(True, True)
        
    def __call__(self, node):
        
        result = node.copy_sub_tree()
        
        if(isinstance(result, op.ChainOperationNode)):
            
            c = result.get_childrens()[0]
            
            if(c.is_constant() and ((isinstance(result, op.add) and c.get_name() == 0) or
                                    (isinstance(result, op.mul) and c.get_name() == 1))): result.free_connection(c)
                        
        return result
    
            
        


class add_chain(Simplification):
    def __init__(self):
        super().__init__(False, True)
        
    def __call__(self, node):
        
        #Prepare
        constant, functions = chain_simplification(node, lambda a, b: a + b, 0)

        #Add the identical function, adding the coefficients
        for i in range(len(functions)):
            
            if(isinstance(functions[i], op.mul)):
                
                #Separate the constant from mul chain
                c = functions[i].get_childrens()[0]
                
                nucleus = op.mul()
                
                for r in functions[i].get_childrens()[1:]:
                        
                    functions[i].free_connection(r)
                    nucleus.new_connection(r)
                
                #Save the configuration from mul chain
                functions[i] = {"f": nucleus, "c": c.get_name()}
            
            #Save the configuration from a not mul chain
            else: functions[i] = {"f": functions[i], "c": 1}
        
        
        #Sum the configurations
        dict_accumulation = accumulation_coefficients(functions)

        #Create the result
        result = op.add()
        
        result.new_connection(op.Leaf(constant))

        for x in dict_accumulation: 
            
            element = x["f"].copy_sub_tree()
            
            part = op.mul()
            part.new_connection(op.Leaf(x["c"]))
            
            if(isinstance(element, op.mul)):
                for x in element.get_childrens(): part.new_connection(x)

            else: part.new_connection(element)
                            
            result.new_connection(part)

        return result
            
            

class mul_chain(Simplification):
    def __init__(self):
        super().__init__(False, True)
        
    def __call__(self, node):
        
        #Prepare
        constant, functions = chain_simplification(node, lambda a, b: a * b, 1)
        
        new_functions = []
        
        for i in range(len(functions)):
            
            if(isinstance(functions[i], op.div)):
                
                child_div = functions[i].get_childrens()[0]
                
                if(isinstance(child_div, op.mul)):
                    
                    c = child_div.get_childrens()[0].get_name()

                    #If c is a constant = 0, we are in a 1/0 situation, and the semplification is not possible (and launch an error)
                    if(c == 0): raise ZeroDivisionError("During the simplification, it was not possible to do the simplification (1/0 found)")
                    
                    constant *= 1/c
                    
                    #Save all configurations from mul chain under a div
                    for r in child_div.get_childrens()[1:]: new_functions.append({"f": r, "c": -1})
                    
                else:
                    
                    #Save configuration from a div
                    new_functions.append({"f": child_div, "c": -1})
                
            #Save a configuration
            else: new_functions.append({"f": functions[i], "c": 1})


        #Sum the configurations
        dict_accumulation = accumulation_coefficients(new_functions)
        
        if(0 == constant): return op.Leaf(0)

        numerator = [x for x in dict_accumulation if x["c"] > 0]
        denominator = [x for x in dict_accumulation if x["c"] < 0]
        
        total_num = []
        total_den = []
        
        for x in numerator:
            for _ in range(x["c"]): total_num.append(x["f"].copy_sub_tree())
            
        for x in denominator:
            for _ in range(-x["c"]): total_den.append(x["f"].copy_sub_tree())
            
        
        result_child = {"c": op.Leaf(constant), "num": None, "den": None}
    
    
    
        if(len(total_num) > 0):
 
            child = op.mul()
            for x in total_num: child.new_connection(x)
                
            result_child["num"] = child
            
            
            
        if(len(total_den) > 0):
        
            if(len(total_den) == 1): child = total_den[0]
            else: 
                child = op.mul()
                child.new_connection(op.Leaf(1))
                for x in total_den: child.new_connection(x)
                
            div_child = op.div()
            div_child.new_connection(child)
                
            result_child["den"] = div_child
    
    
        #If there is only constant, get only constant itself
        if(result_child["num"] == None and result_child["den"] == None): result = result_child["c"]        
        
        #If num exists, create a new mul node, inglobal c, all num childrens and den (if exists)
        elif(result_child["num"] != None):
            
            result = op.mul()
            
            #Inglobal c in result
            result.new_connection(result_child["c"])
            
            #Inglobal num in result
            for x in result_child["num"].get_childrens(): result.new_connection(x)
            
            #Inglobal den in num, if den exists
            if(result_child["den"] != None): result.new_connection(result_child["den"])



        #In this case, num not exists, and den exists   
        else:
            
            #Create a mul and concatenate c and den
            result = op.mul()
            result.new_connection(result_child["c"])
            result.new_connection(result_child["den"])
   
        return result


            

                
                
        
        

#TOP DOWN SIMPLIFICATIONS

class neg_null(Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        return double_oper_null(node, op.neg)
    
class div_null(Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        return double_oper_null(node, op.div)
    
class abs_null(Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        return double_oper_null(node, op.abs)
    
class exp_ln_null(Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        
        child = node.get_childrens()[0]
        
        if(isinstance(child, op.ln)): return child.get_childrens()[0]
        else: raise ex.SimplificationError
        
class ln_exp_null(Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        
        child = node.get_childrens()[0]
        
        if(isinstance(child, op.exp)): return child.get_childrens()[0]
        else: raise ex.SimplificationError

class relax_neg(Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        
        child = node.get_childrens()[0]
        
        result = op.mul()
        result.new_connection(op.Leaf(-1))
        result.new_connection(child.copy_sub_tree())
        
        return result

class div_scission(Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        return scission_oper1_on_oper2(node, op.div, [op.mul, op.neg])

class abs_scission(Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        return scission_oper1_on_oper2(node, op.abs, [op.mul, op.div])

class abs_destroy_neg(Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        
        child = node.get_childrens()[0]
        
        if(not isinstance(child, op.neg)): raise ex.SimplificationError
        
        result = op.abs()
        result.new_connection(child.get_childrens()[0].copy_sub_tree())
        
        return result

class abs_double_mul_null(Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        
        child = node.get_childrens()[0]
        
        if(not isinstance(child, op.mul)): raise ex.SimplificationError
        
        childrens = child.get_childrens()
        
        if(childrens[0] == childrens[1]): return child
        
        raise ex.SimplificationError

class mul_distributivity_on_add(Simplification):
    def __init__(self):
        super().__init__(True, False)
        
    def __call__(self, node):
        
        elements = [x.get_childrens() if isinstance(x, op.add) else [x] for x in node.get_childrens()]
        
        combinations = itertools.product(*elements)
        
        sub_mul_chains = []
        
        for i in combinations:
            
            sub_mul = op.mul()
            sub_mul.new_connection(op.Leaf(1))
            for x in i: sub_mul.new_connection(x.copy_sub_tree())
                
            sub_mul_chains.append(sub_mul)
            
        if(len(sub_mul_chains) == 1): return sub_mul_chains[0]
        
        else:
            
            result = op.add()
            result.new_connection(op.Leaf(0))
            for x in sub_mul_chains: result.new_connection(x)
            return result
