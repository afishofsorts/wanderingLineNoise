import numpy as np
import random as rand
from . import freqGen as fg
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt
import pandas as pd

# converts frequency to phase using rectangle approximation
def freqToPhase(t, freqs, Ts):
    # INPUTS:
    # t:         1D time array with Ts spacing
    # freqs:     1D frequency array
    # Ts:        Sampling rate
    # OUTPUTS:
    # phase:     1D array of approximate phase values 

    phase = np.zeros(shape=(len(t)))
    phase[0] = 0
    for i in range(len(t)-1):
        phase[i+1] = phase[i] + Ts*freqs[i+1]

    return phase

# adds iid noise to every value of a 1D list using a gaussian distribution
def addVar(data, stnd: float):
    # INPUTS:
    # data:      1D array of data values
    # stnd:      Standard deviation for added gaussian variation to data values
    # OUTPUTS:
    # newdata:   1D array of data with added gaussian variation for each value

    newdata = np.zeros(len(data))

    for i in range(len(data)):
        dy = rand.gauss(0, stnd)
        newdata[i] = data[i] + dy
    return newdata

# generates a cosine signal of amplitude A with iid noise and smooth frequency 
# variation between M frequencies within the range f0 +- band
def genSignal(t, freqs, A: float, Ts: float, stnd = 0.4):
    # INPUTS:
    # t:         1D time array with Ts spacing
    # freqs:     1D frequency array
    # A:         Signal amplitude
    # Ts:        Sampling rate
    # stnd:      Multiple of A to be used as standard deviation for added iid
    # OUTPUTS:
    # cleanSig   1D frequency array with smoothly varying values
    # distSig:   cleanSig with added iid noise
    # freqKnots: 2-tuple of times and frequencies for BSpline knots used in generating cleanSig

    phase = freqToPhase(t, freqs, Ts) # converts to phase

    cleanSig = A*np.cos(phase*2*np.pi) # creates sinusoidal signal
    distSig = addVar(cleanSig, stnd*A) # adds iid noise
    return cleanSig, distSig

def genWL(f0, band, A, Ts, M = 10):
    freqKnots = fg.genKnot(M, f0, band) # knots for freq spline
    t, freqs = fg.genBSpline(freqKnots, Ts) # generates time and freq spline arrays
    cleanSig, distSig = genSignal(t, freqs, A, Ts) # generates signal with smoothly varying frequency and iid noise
    return t, cleanSig, distSig, freqs, freqKnots
