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
    "arcsin": sp.asin,
    "abs": sp.Abs,
    "exp": sp.exp,
    "ln": sp.log
    }



#def decorporate_functions(expr):
    
    #for 

#def accorporate_functions(expr):
    #pass


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



#-----------------------------------------------------------------------

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
def __chain_simplification(node, f, start):
    
    elements = flat_chain(node)
        
    #Transform every element in a constant (if it is possible)
    for i in range(len(elements)):
        
        try: elements[i] = constant(elements[i])
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

def __accumulation_coefficients(functions):
    
    set_functions = []
    
    for i in functions:
        if(i["f"] not in set_functions): set_functions.append(i["f"])
    
    dict_accumulation = [{"f": x, "c": 0} for x in set_functions]
    for x in dict_accumulation:
        for y in [a for a in functions if a["f"] == x["f"]]: x["c"] += y["c"]
            
    return dict_accumulation
    
    
    
    
    
def __double_vertical_deletion(node, type1, type2):
    
    if(not isinstance(node, type1)): raise ex.SimplificationError
    if(node.get_channels() != 1): raise ex.SimplificationError
    
    child = node.get_childrens()[0]
    
    if(not isinstance(child, type2)): raise ex.SimplificationError
    if(child.get_channels() != 1): raise ex.SimplificationError
    
    return child.get_childrens()[0].copy_sub_tree()
        
      
        
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------


    
    



def scission_oper1_on_oper2(node, oper1, allow_operations):
    
    if(node.get_channels() != 1): raise ValueError(f"Node {node} must has 1 input")
    
    if(not isinstance(node, oper1)): raise ValueError(f"On node {node} it is impossible apply scission of {oper1} on own child")
    
    child = node.get_childrens()[0]
    
    if(child.__class__ not in allow_operations): raise ex.SimplificationError

    result = child.__class__()
    
    for g in child.get_childrens():
        
        new_child = oper1()
        new_child.new_connection(g.copy_sub_tree())
        result.new_connection(new_child)
    
    return result






def approximation(node):

    raise ex.SimplificationError
    if(not node.is_constant()): raise ex.SimplificationError
        
    #tolerance fissata in base alle capacità della macchina nella precisione numerica
    return op.Leaf(int(node.get_name())) if np.abs(node.get_name() - int(node.get_name())) <= 1e-6 else node


def constant(node):
        
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

def cleaning(node):
        
    result = node.copy_sub_tree()
        
    if(isinstance(result, op.ChainOperationNode)):
            
        c = result.get_childrens()[0]
            
        if(c.is_constant() and ((isinstance(result, op.add) and c.get_name() == 0) or
                                (isinstance(result, op.mul) and c.get_name() == 1))): result.free_connection(c)
            
        result = result if result.get_channels() > 1 else result.get_childrens()[0]
        
    return result

def add_chain(node):
        
    #Prepare
    constant, functions = __chain_simplification(node, lambda a, b: a + b, 0)

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
    dict_accumulation = __accumulation_coefficients(functions)

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
 
def mul_chain(node):
        
    #Prepare
    constant, functions = __chain_simplification(node, lambda a, b: a * b, 1)
        
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
    dict_accumulation = __accumulation_coefficients(new_functions)
        
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




              



#DELETIONS
    
    
#div(div(x)) = x
def div_null(node):
    return __double_vertical_deletion(node, op.div, op.div)

#abs(abs(x)) = x
def abs_null(node):
    result = op.abs()
    result.new_connection(__double_vertical_deletion(node, op.abs, op.abs))
    return result

#exp(ln(x)) = x
def exp_ln_null(node):
    return __double_vertical_deletion(node, op.exp, op.ln)

#ln(exp(x)) = x
def ln_exp_null(node):
    return __double_vertical_deletion(node, op.ln, op.exp)


        
#SCISSIONS

def div_scission(node):
    return scission_oper1_on_oper2(node, op.div, [op.mul])

def abs_scission(node):
    return scission_oper1_on_oper2(node, op.abs, [op.mul, op.div])





   
def mul_distributivity_on_add(node):
        
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



"""
PRECONDITION: Qui, la costante deve essere già messa apposto.
Sia per i decorporation che per i accorporation.
"""

#ABS, EXP, LN

#Accorporation operations
def total_accorporation(node):
    
    childrens = node.get_childrens()

    #Extract constant
    residual_childrens = [childrens[0]]
    childrens = childrens[1:]
    
    if(isinstance(node, op.mul)):
        
        total_exp = []
        total_abs = []

        for c in childrens:
            
            if(isinstance(c, op.exp)): total_exp.append(c.get_childrens()[0])
            elif(isinstance(c, op.abs)): total_abs.append(c.get_childrens()[0])
            else: residual_childrens.append(c)
            
        if(len(total_exp) > 0): 
            child = op.exp()
            chain = op.add()
            for x in total_exp: chain.new_connection(x.copy_sub_tree())
            child.new_connection(chain)
            residual_childrens.append(child)
            
        if(len(total_abs) > 0): 
            child = op.abs()
            chain = op.mul()
            for x in total_abs: chain.new_connection(x.copy_sub_tree())
            child.new_connection(chain)
            residual_childrens.append(child)
    
    if(isinstance(node, op.add)):
        
        total_ln = []
        
        for c in childrens:
            
            if(isinstance(c, op.ln)): total_exp.append(c.get_childrens()[0])
            else: residual_childrens.append(c)
        
        if(len(total_ln) > 0): 
            child = op.ln()
            chain = op.mul()
            for x in total_abs: chain.new_connection(x.copy_sub_tree())
            child.new_connection(chain)
            residual_childrens.append(child)
            
        
            
    result = node.__class__()
    for x in residual_childrens: result.new_connection(x.copy_sub_tree())
    
    return result






#Decorporation operations
def decorporation_ln(node):
    
    if(not isinstance(node, op.ln)): raise ex.SimplificationError
    
    child = node.get_childrens()[0]
    
    if(not isinstance(child, op.mul)): raise ex.SimplificationError
    
    if(child.get_childrens()[0].get_name() <= 0): raise ex.SimplificationError
    
    result = op.add()
    
    for c in child.get_childrens():
        
        ln = op.ln()
        ln.new_connection(c.copy_sub_tree())
        result.new_connection(ln)
        
    return result



def decorporation_exp(node):
    
    if(not isinstance(node, op.exp)): raise ex.SimplificationError
    
    child = node.get_childrens()[0]
    
    if(not isinstance(child, op.add)): raise ex.SimplificationError

    result = op.mul()
    
    for c in child.get_childrens():
        
        exp = op.exp()
        exp.new_connection(c.copy_sub_tree())
        result.new_connection(exp)
        
    return result



def decorporation_abs(node):
    
    if(not isinstance(node, op.abs)): raise ex.SimplificationError
    
    child = node.get_childrens()[0]
    
    if(not isinstance(child, op.mul)): raise ex.SimplificationError

    result = op.mul()
    
    for c in child.get_childrens():
        
        abs = op.abs()
        abs.new_connection(c.copy_sub_tree())
        result.new_connection(abs)
        
    return result


