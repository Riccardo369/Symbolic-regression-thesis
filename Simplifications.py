import numpy as np
import itertools
import sympy as sp

import Operations as op
import Exceptions as ex
import utils




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
    "sqrt": sp.sqrt
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



     
         
    


def from_sympy(expr):
    
    """Riconverte un'espressione SymPy nel tuo albero di OperationNode."""
    if(isinstance(expr, sp.Symbol)): return op.Leaf(str(expr))
    if(isinstance(expr, (sp.Number, sp.Integer, sp.Float))): return op.Leaf(float(expr))
    
    node = None
    
    if(expr.func == sp.Pow):
        
        base, exp = expr.args
        
        if(int(exp) != exp): raise ValueError(f"Not integer exp in Pow non supported: {exp}")
        
        else:
            
            bases = from_sympy(base)
            
            n = np.abs(exp)
            
            bases = [bases.copy_sub_tree() for _ in range(n)]

            if(n > 1):
                
                mul_chain = op.mul()
                for i in bases: mul_chain.new_connection(i)
                bases = mul_chain
                
            else: bases = bases[0]
            
            if(exp > 0): node = mul_chain
            
            else:
                
                node = op.div()
                node.new_connection(bases)
                
              
    elif(expr.func in sympy_convertion_dict.values()):
    
        for key in sympy_convertion_dict.keys():
            if(expr.func == sympy_convertion_dict[key]): node = eval(f"op.{key}")()
            
        for arg in expr.args:
            child_node = from_sympy(arg)
            node.new_connection(child_node) # Imposta automaticamente il padre!
            
            
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



