import numpy as np
import itertools
import sympy as sp

import Operations as op
import Exceptions as ex


#Dict translation name -> Sympy node
sympy_convertion_dict = {
    "add": sp.Add,
    "mul": sp.Mul,
    "div": lambda x: 1/x,
    "sin": sp.sin,
    "cos": sp.cos,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "exp": sp.exp,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    }





#Convertion from tree to sympy tree
def to_sympy(node):
    
    if isinstance(node, op.Leaf):
        
        if isinstance(node.get_name(), (int, float)):
            return sp.Number(node.get_name())
        
        return sp.Symbol(str(node.get_name()), real = True)
    
    args = [to_sympy(c) for c in node.get_childrens()]
    
    for key in sympy_convertion_dict.keys():
        if(isinstance(node, eval(f"op.{key}"))): return sympy_convertion_dict[key](*args)
    
    raise ValueError(f"Operazione sconosciuta: {type(node)}")






#Convertion of a power sympy node, in equivalent standard form with in multiplication form
def power_in_mul(base, pow):
    
    float_pow = eval(str(pow))
    
    if(float_pow == 0): return op.Leaf(1)
    
    #Sign of element
    positive = float_pow > 0
    if(not positive): float_pow *= -1
    
    #Number of base ripetition
    ripetive_in_mul_chain = int(float_pow // 1)
   
    mul_chain = op.mul()
    for _ in range(ripetive_in_mul_chain): mul_chain.new_connection(base.copy_sub_tree())
    
    residual = float_pow % 1
    
    if(residual > 0):
        
        num = pow.p
        den = pow.q
        
        count_sqrt = np.log2(den)
        
        if(count_sqrt != int(count_sqrt)): raise ValueError(f"It is not possible to apply convertion with multiplication extension, residual poer is {num} / {den}")
        
        count_sqrt = int(count_sqrt)
        
        sqrt = op.sqrt()
        
        for _ in range(count_sqrt-1):

            new_sqrt = op.sqrt()
            sqrt.new_connection(new_sqrt)
            sqrt = new_sqrt
            
        sqrt.new_connection(base.copy_sub_tree())
        
        while(sqrt.get_father() != None): sqrt = sqrt.get_father()
        
        for _ in range(num): mul_chain.new_connection(sqrt.copy_sub_tree())
        
    #If mul chain is empty, return the base
    if(len(mul_chain.get_childrens()) == 0): mul_chain = base.copy_sub_tree()
    
    if(positive): node = mul_chain
    else: 
        node = op.div()
        node.new_connection(mul_chain)
        
    return node
        
        
        
    
    
    
    
    
    
    
#Convertion of a power sympy node, in equivalent standard form with exp form
def power_in_exp(base, pow, TR_inputs_dict):
    
    try:
                
        output_base = base.forward_calculation_memory_sub_tree(TR_inputs_dict).get_memory("forward")
        valid_base = np.all(np.isfinite(output_base)) and op.ln.check_input_domain(0, output_base)
                
    except: valid_base = False
            
    if(not valid_base): raise ex.ExponentialSimplificationError(f"It is not possible to translate the power {base} ^ {pow} in exp(ln({base}) * {pow})")
    
    node = op.exp()
    
    log = op.ln()
    log.new_connection(base)
            
    mul = op.mul()
    mul.new_connection(log)
    mul.new_connection(pow)
    
    node.new_connection(mul)
    
    return node
    


#Convertion from sympy tree to standard tree
def from_sympy(expr, TR_inputs_dict):
    
    """Riconverte un'espressione SymPy nel tuo albero di OperationNode."""
    if(isinstance(expr, sp.Symbol)): return op.Leaf(str(expr))
    if(isinstance(expr, (sp.Number, sp.Integer, sp.Float, sp.NumberSymbol))): return op.Leaf(float(expr.evalf()))
    
    if(expr.func == sp.Pow):
        
        base, pow = expr.args
        
        base = from_sympy(base, TR_inputs_dict)
        
        node = None
        
        try: node = power_in_exp(base, from_sympy(pow, TR_inputs_dict), TR_inputs_dict)
        except: 
            try: node = power_in_mul(base, pow)
            except: pass
        
        if(node == None): raise ValueError(f"It is not possible to convert {base}^{pow} in standard form")

    elif(expr.func in sympy_convertion_dict.values()):
    
        for key in sympy_convertion_dict.keys():
            if(expr.func == sympy_convertion_dict[key]): node = getattr(op, key)()
            
        for arg in expr.args:
            child_node = from_sympy(arg, TR_inputs_dict)
            node.new_connection(child_node)

    else: raise ValueError(f"Operatore SymPy non supportato: {expr.func}")
    
    if(isinstance(node, op.ChainOperationNode)):
        
        new_node = node.__class__()
        for x in flat_chain(node): new_node.new_connection(x)
        node = new_node
    
    return node




def prod(x):
    s = 1
    for i in x: s*= i
    return s

"""
Preconditions:

- node must be a chain node

Postconditions:

- all chain (of same type node) flatted (ex. add(add(a, b, c), d, e) -> add(a, b, c, d, e)

"""
def flat_chain(node):
    
    if(not isinstance(node, op.ChainOperationNode)): raise ValueError(f"Node {node} is not a chain")
    
    sub_chains = [None]
    elements = node.get_childrens()
    
    Oper = type(node)
    
    #Flat all Oper chain in one chain (like Oper(a, b, Oper(c, d)) = Oper (a, b, c, d))
    while(len(sub_chains) > 0):
        
        sub_chains = [x for x in elements if isinstance(x, Oper)]
        elements =   [x for x in elements if not isinstance(x, Oper)]
        
        for s in sub_chains: elements += s.get_childrens()
        
    constants = [x.get_name() for x in elements if x != None and x.is_constant()]
    elements = [x.copy_sub_tree() for x in elements if x != None and not x.is_constant()]
    
    if(len(constants) == 0): return elements
    
    new_constant = np.sum(constants) if Oper == op.add else np.prod(constants)
    
    if(not ((new_constant == 0.0 and Oper == op.add) or 
            (new_constant == 1.0 and Oper == op.mul))): elements.insert(0, op.Leaf(new_constant))
    
    return elements


