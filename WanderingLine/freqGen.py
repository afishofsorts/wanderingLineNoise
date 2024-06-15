import numpy as np
import random as rand
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt

def genKnot(M: int, f0: float, band: float):
    points = np.zeros(shape=(M, 2))
    tgap = round(10/f0, 2)

    for i in range(M):
        epsi = rand.randrange(-100, 100)*band/100
        points[i, 0] = i*tgap
        points[i, 1] = f0 + epsi
    return points

def genBSpline(p):
    tck = splrep(p[:, 0], p[:, 1], s=0, k=4)
    xnew = np.linspace(0, p[-1, 0], 1000)
    smoothFit = BSpline(*tck)(xnew)
    return [xnew, smoothFit]

def plotSpline(p, spline):
    plt.plot(p[:, 0], p[:, 1], 'o')
    plt.plot(spline[0], spline[1], '-')
    plt.xlabel('t (s)'); plt.ylabel('freq (Hz)')
    plt.show()

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
    p = genKnot(M, f0, band)
    freq = genBSpline(p)
    phase = freqToPhase(freq)
    cleanSig = A*np.cos(phase[:, 1])
    return [phase[:, 0], cleanSig]

f0 = 60; band = 30; A = 2
signal = genSignal(f0, band, A)

fig = plt.figure(figsize=(15, 5))
plt.plot(signal[0], signal[1], '-')
plt.xlabel('t (s)'); plt.ylabel('q(t)')
plt.show()

