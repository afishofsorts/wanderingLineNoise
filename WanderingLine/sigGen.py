import numpy as np
import random as rand
import freqGen as fg
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt

# converts frequency to phase using rectangle approximation
def freqToPhase(freq):
    tgap = freq[0][1] - freq[0][0] # assumes uniform spacings of frequency
    phase = np.zeros(shape=(len(freq[0]), 2))
    phase[0, 1] = 0; phase[0, 0] = 0
    for i in range(len(freq[0])-1):
        phase[i+1, 0] = freq[0][i+1]
        phase[i+1, 1] = phase[i, 1] + tgap*freq[1][i+1]

    return phase

# adds iid noise to every value of a 1D list using a gaussian distribution
def addVar(data, stnd: float):
    newdata = np.zeros(len(data))

    for i in range(len(data)):
        dy = rand.gauss(0, stnd)
        newdata[i] = data[i] + dy
    return newdata

# generates a signal of amplitude A with iid noise and smooth frequency variation between M frequencies within the range f0 +- band
def genSignal(f0: float, band: float, A: float, M = 30):
    p = fg.genKnot(M, f0, band)
    freq = fg.genBSpline(p)
    phase = freqToPhase(freq)
    cleanSig = A*np.cos(phase[:, 1]*2*np.pi)
    distSig = addVar(cleanSig, 0.4*A)
    return [phase[:, 0], distSig, cleanSig, p, freq]

def plotScatter(x, y, w=15, h=5):
    fig = plt.figure(figsize=(w, h))
    plt.plot(x, y, 'o')
    plt.xlabel('t (s)'); plt.ylabel('q(t)')
    plt.show()

def plotLine(x, y, w=15, h=5):
    fig = plt.figure(figsize=(w, h))
    plt.plot(x, y, '-')
    plt.xlabel('t (s)'); plt.ylabel('q(t)')
    plt.show()
