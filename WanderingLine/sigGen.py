import numpy as np
import random as rand
import freqGen as fg
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt

# converts frequency to phase using rectangle approximation
def freqToPhase(freq):
    Ts = freq[0][1] - freq[0][0] # assumes uniform spacings of frequency
    phase = np.zeros(shape=(len(freq[0])))
    phase[0] = 0
    for i in range(len(freq[0])-1):
        phase[i+1] = phase[i] + Ts*freq[1][i+1]

    return phase

# adds iid noise to every value of a 1D list using a gaussian distribution
def addVar(data, stnd: float):
    newdata = np.zeros(len(data))

    for i in range(len(data)):
        dy = rand.gauss(0, stnd)
        newdata[i] = data[i] + dy
    return newdata

# generates a signal of amplitude A with iid noise and smooth frequency variation between M frequencies within the range f0 +- band
def genSignal(f0: float, band: float, A: float, Ts, M = 30, drange = 0.4):
    freqKnots = fg.genKnot(M, f0, band) # knots for freq spline
    spline = fg.genBSpline(freqKnots, Ts) # freq spline
    times = spline[0]; freqs = spline[1]

    phase = freqToPhase(spline)

    cleanSig = A*np.cos(phase*2*np.pi) # creates sinusoidal signal
    distSig = addVar(cleanSig, drange*A) # add iid noise
    return [times, distSig, cleanSig, freqKnots, freqs]

def plotScatter(x, y, w=15, h=5, filename='ScatterPlot'):
    fig = plt.figure(figsize=(w, h))
    plt.plot(x, y, 'o')
    plt.xlabel('t (s)'); plt.ylabel('q(t)')
    plt.savefig(r'C:\Users\casey\Desktop\REU24\WanderingLine\SignalPlots\\' + filename + '.png')
    plt.show()

def plotLine(x, y, w=15, h=5, filename='LinePlot'):
    fig = plt.figure(figsize=(w, h))
    plt.plot(x, y, '-')
    plt.xlabel('t (s)'); plt.ylabel('q(t)')
    plt.savefig(r'C:\Users\casey\Desktop\REU24\WanderingLine\SignalPlots\\' + filename + '.png')
    plt.show()
