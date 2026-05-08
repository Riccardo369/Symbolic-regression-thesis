import numpy as np

import Indicators as ind

x = 3
y = np.array([1, 2, 3])

print(1 / y)

print(1 / (1 / y))

raise ValueError("STOP")

f = lambda x, y: ind.pearson_correlation(x, y)

print(f(x, x))
print(f(x, y))
print(f(y, x))
print(f(y, y))

f = lambda x, y: ind.spearman_correlation(x, y)

print(f(x, x))
print(f(x, y))
print(f(y, x))
print(f(y, y))

f = lambda x, y: ind.chatterjee_correlation(x, y)

print(f(x, x))
print(f(x, y))
print(f(y, x))
print(f(y, y))