import os
import sys

mingw_path = r'C:\mingw64\bin' 

if os.path.exists(mingw_path):
    os.add_dll_directory(mingw_path)

import Ball
import numpy as np

# Generiamo due vettori con una relazione non lineare (es. quadratica)
x = np.linspace(-1, 1, 100)
y = x**2 + np.sin(x)

# Calcolo della Ball Correlation
# Nota: Il modulo potrebbe chiamarsi 'Ball' o 'ball' a seconda di come lo importa Python
try:
    bcor = Ball.bcor(x, y)
    print(f"Successo! Ball Correlation calcolata: {bcor:.4f}")
except AttributeError:
    # Alcune versioni usano Ball.bcov.bcor
    from Ball import bcov
    bcor = bcov.bcor(x, y)
    print(f"Successo (via bcov)! Ball Correlation: {bcor:.4f}")