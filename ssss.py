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
         "mul(X, add(X, w), div(mul(w, q)))",
         "mul(X, add(X, w), div(w), div(q))",
         "div(2)",
         "mul(X, div(X), X, add(W, mul(-2, W)))"
         ]


for t in tests:
    
    
    SymbolicRegression.string_to_tree(t).simplify_model()
    print("")
    
