import numpy as np
from numpy import random
import freqGen as fg
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt

def freqToPhase(freq):
    # Assuming uniform spacings
    tgap = freq[0][1] - freq[0][0]
    phase = np.zeros(shape=(len(freq[0]), 2))
    phase[0, 1] = 0
    for i in range(len(freq[0])-1):
        phase[i+1, 0] = freq[0][i+1]
        phase[i+1, 1] = phase[i, 1] + tgap*freq[1][i+1]

    return phase

def genSignal(f0: float, band: float, A: float, M = 20):
    p = fg.genKnot(M, f0, band)
    freq = fg.genBSpline(p)
    phase = freqToPhase(freq)
    cleanSig = A*np.cos(phase[:, 1])
    return [phase[:, 0], cleanSig]

f0 = 60; band = 30; A = 2
signal = genSignal(f0, band, A)

fig = plt.figure(figsize=(15, 5))
plt.plot(signal[0], signal[1], '-')
plt.xlabel('t (s)'); plt.ylabel('q(t)')
plt.show()