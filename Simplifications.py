import numpy as np
import itertools
import sympy as sp

import Operations as op
import Exceptions as ex



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






def to_sympy(node):
    """Converte un tuo OperationNode in un'espressione SymPy."""
    if isinstance(node, op.Leaf):
        # Gestisce variabili e costanti
        if isinstance(node.get_name(), (int, float)):
            return sp.Number(node.get_name())
        return sp.Symbol(str(node.get_name()), real = True)
    
    # Converte ricorsivamente i figli
    args = [to_sympy(c) for c in node.get_childrens()]
    
    for key in sympy_convertion_dict.keys():
        if(isinstance(node, eval(f"op.{key}"))): return sympy_convertion_dict[key](*args)
    
    raise ValueError(f"Operazione sconosciuta: {type(node)}")







def from_sympy(expr, TR_inputs_dict):
    
    """Riconverte un'espressione SymPy nel tuo albero di OperationNode."""
    if(isinstance(expr, sp.Symbol)): return op.Leaf(str(expr))
    if(isinstance(expr, (sp.Number, sp.Integer, sp.Float))): return op.Leaf(float(expr))
    
    node = None
    
    if(expr.func == sp.Pow):
        
        base, exp = expr.args
        
        base = from_sympy(base, TR_inputs_dict)
        
        try:
            
            output_base = base.forward_calculation_memory_sub_tree(TR_inputs_dict).get_memory("forward")
            valid_base = np.all(np.isfinite(output_base)) and op.ln.check_input_domain(0, output_base)
            
        except: valid_base = False
        
        if(not valid_base): raise ex.ExponentialSimplificationError(f"It is not possible to translate the power {base} ^ {exp} in exp(ln({base}) * {exp})")

        node = op.exp()
        
        log = op.ln()
        log.new_connection(base)
        
        mul = op.mul()
        mul.new_connection(log)
        mul.new_connection(from_sympy(exp, TR_inputs_dict))
        
        node.new_connection(mul)
        
            
    elif(expr.func in sympy_convertion_dict.values()):
    
        for key in sympy_convertion_dict.keys():
            if(expr.func == sympy_convertion_dict[key]): node = eval(f"op.{key}")()
            
        for arg in expr.args:
            child_node = from_sympy(arg, TR_inputs_dict)
            node.new_connection(child_node) # Imposta automaticamente il padre!
            
            
            
            
            
    elif(isinstance(expr, (sp.Number, sp.NumberSymbol))): return op.Leaf(float(expr.evalf()))
        
            
            
    else: raise ValueError(f"Operatore SymPy non supportato: {expr.func}")
        
    # Costruisci ricorsivamente garantendo l'integrità di _father
    
    if(isinstance(node, op.ChainOperationNode)):
        
        new_node = node.__class__()
        for x in flat_chain(node): new_node.new_connection(x)
        node = new_node
    
    return node






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



