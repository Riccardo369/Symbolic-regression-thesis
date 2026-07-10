import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os

import LearningStrategy as ls
from Model import SymbolicRegression
import Operations as op
import Errors as er
import Indicators as ind
import InformationSOTA.artificial_formulas as af
import utils



tests = ["mul(X, 2, add(X, w), add(2, 3))",
         "mul(mul(X, 2), 3, Q, S)",
         "mul(mul(X, 2), 3, add(Q, S), A)",
         "mul(mul(X, 2), 3, mul(mul(A, B), B), B)",
         "add(mul(add(A, B), add(C , D)), mul(E, F))",
         "mul(add(A, B), add(C, D), add(E, F))",
         "mul(X, add(X, w), div(mul(w, q)))",
         "mul(X, add(X, w), div(w), div(q))",
         "div(2)",
         "mul(X, div(X), X, add(W, mul(-2, W)))",
         "div(div(div(x)))",
         "add(2.350108225108223, add(1.125899906842626e+16, add(1.8014398509482e+16, add(1.8014398509481996e+16, add(1.801439850948199e+16, mul(5.404319552844595e+16, X), mul(9007199254740992.0, X, add(6.0, mul(4.0, X)))), add(5.404319552844595e+16, mul(5.404319552844595e+16, X))), add(5.404319552844595e+16, add(5.404319552844595e+16, mul(5.404319552844595e+16, X)))), add(2.25179981368525e+16, add(2.2517998136852496e+16, add(2.2517998136852492e+16, mul(1.125899906842624e+16, add(6.0, mul(4.0, X)))))), add(5.404319552844595e+16, add(5.404319552844595e+16, add(5.404319552844595e+16, mul(5.404319552844595e+16, X))))), add(11.8, add(8.8, add(5.800000000000001, add(2.8000000000000003, mul(0.8000000000000002, X))))), add(3.500000000000001, mul(1.5000000000000007, X)), add(6.333333333333334, add(3.3333333333333335, mul(1.3333333333333335, X))), mul(1.246753246753246, X))",
         "add(0, mul(1, add(3.9999999999999964, mul(1, mul(0.0, add(0, abs(add(0, mul(1, X)))))))))",
         "ln(exp(x))",
         "exp(ln(x))",
         "arcsin(sin(x))",
         "sin(arcsin(x))",
         "mul(abs(x), abs(y))",
         "add(ln(x), ln(y), ln(z))",
         "add(ln(mul(x, y)), ln(w))",
         "mul(exp(x), exp(y), exp(z))",
         "mul(exp(x), exp(y), exp(z), div(exp(z)))",
         "exp(add(x, y, z))",
         "exp(add(x, y, z, mul(-1, z)))",
         "ln(mul(-2, x))",
         ]


for t in tests:
    
    print("BEFORE:", t)
    print("AFTER: ", SymbolicRegression.string_to_tree(t).simplify_model())
    print("")
    
