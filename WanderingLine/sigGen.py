import numpy as np
import random as rand
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

def addVar(data, stnd: float):
    newdata = np.zeros(len(data))

    for i in range(len(data)):
        dy = rand.gauss(data[i], stnd)
        newdata[i] = data[i] + dy
    return newdata

def genSignal(f0: float, band: float, A: float, M = 20):
    p = fg.genKnot(M, f0, band)
    freq = fg.genBSpline(p)
    phase = freqToPhase(freq)
    cleanSig = A*np.cos(phase[:, 1])
    distSig = addVar(cleanSig, 0.3*A)
    return [phase[:, 0], distSig]

f0 = 60; band = 30; A = 2
signal = genSignal(f0, band, A)

fig = plt.figure(figsize=(15, 5))
plt.plot(signal[0], signal[1], 'o')
plt.xlabel('t (s)'); plt.ylabel('q(t)')
plt.show()