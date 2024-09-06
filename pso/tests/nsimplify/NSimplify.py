import numpy as np
import matplotlib.pyplot as plt

def NCalc(t, w1, w2, w3):
    N = sum(np.cos(w1*t + w2*t**2 + w3*t**3)**2)
    return N

wvar = np.arange(0.5, 150.5, 0.5)
Ns = np.zeros(shape=(3, 4, 300))

for i in range(4):
    Ts = 10**(i-4)
    t = np.arange(0, 10, Ts)
    for j in range(300):
        Ns[0, i, j] = NCalc(t, wvar[j], 0, 0)
        Ns[1, i, j] = NCalc(t, 0, wvar[j], 0)
        Ns[2, i, j] = NCalc(t, 0, 0, wvar[j])

np.save('saved\\NSimplify.npy', Ns)