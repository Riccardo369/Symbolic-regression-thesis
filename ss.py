import numpy as np

CFU = np.array([   6,  9,  6,  6,  9,  9,  9,  9,  9,  9,  9,  6], dtype = np.float64)
Votes = np.array([26, 27, 30, 27, 26, 23, 25, 26, 29, 27, 24, 24], dtype = np.float64)



CFU /= np.sum(CFU)
mean = np.sum(Votes * CFU)


print(mean * (11/3))