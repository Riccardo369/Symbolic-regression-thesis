import Operations as op
import Exceptions as ex
import numpy as np

def constant(node):
    
    c = node.get_childrens()
    
    if(not all([x.is_constant() for x in c])): raise ex.SimplificationError()
    
    try: value = eval(node.get_string_calculation_function()['body'])
    except: raise ex.SimplificationError()
    
    result = op.Leaf(value)
    result.set_memory("forward", value)
    return result
    
def addition_identity(node):
    
    c = node.get_childrens()
    
    if(not (c[0].is_constant() and c[0].get_name() == 0) or (c[1].is_constant() and c[1].get_name() == 0)): raise ex.SimplificationError()
    
    if(isinstance(c[0], op.Leaf) and c[0].get_name() == 0): return c[1].copy_sub_tree()
    if(isinstance(c[1], op.Leaf) and c[1].get_name() == 0): return c[0].copy_sub_tree()
    

def multiplication_identity(node):
    
    c = node.get_childrens()
    
    if(not (c[0].is_constant() and c[0].get_name() == 1) or (c[1].is_constant() and c[1].get_name() == 1)): raise ex.SimplificationError()
    
    if(c[0].is_constant() and c[0].get_name() == 1): return c[1].copy_sub_tree()
    if(c[1].is_constant() and c[1].get_name() == 1): return c[0].copy_sub_tree()

def division_identity(node):
    
    c = node.get_childrens()
    if(not (c[1].is_constant() and c[1].get_name() == 1)): raise ex.SimplificationError()
    return c[0].copy_sub_tree()


def multiplication_null(node):
    
    c = node.get_childrens()
    
    if(not (c[0].is_constant() and c[0].get_name() == 0) or (c[1].is_constant() and c[1].get_name() == 0)): raise ex.SimplificationError()
    
    result = op.Leaf(0)
    result.set_memory("forward", 0)
    return result

def division_null(node):
    
    c = node.get_childrens()
    if(not (str(c[0]) == str(c[1]))): raise ex.SimplificationError()
    
    result = op.Leaf(1)
    result.set_memory("forward", 1)
    return result

    

