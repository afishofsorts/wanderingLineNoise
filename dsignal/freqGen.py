import numpy as np
import random as rand
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt

# generates breakpoints evenly spaced in time and uniformly random in frequency within fixed range
def genKnot(M: int, f0: float, band: float, n = 10):
    # INPUTS:
    # M:         Number of breakpoints
    # f0:        Mid-range frequency
    # band:      Frequency range above and below f0
    # n:         Number of periods between breakpoints for f0
    # OUTPUTS:
    # tKnots:    1D array of times spaced n periods apart
    # freqKnots: 1D array of uniformly random frequency values

    tKnots, freqKnots = np.zeros(shape=(2, M))
    tgap = round(n/f0, 2) # time between breakpoints

    for i in range(M):
        epsi = rand.randrange(-100, 100)*band/100
        tKnots[i] = i*tgap
        freqKnots[i] = f0 + epsi
    return tKnots, freqKnots

# interpolates cubic BSpline fitting to an array of 2-tuple knots
def genBSpline(knots, Ts):
    # INPUTS:
    # knots:     2-tuple of times and frequencies
    # Ts:        Sampling rate
    # OUTPUTS:
    # t:         1D time array with Ts spacing
    # smoothFit: 1D array of BSpline y values

    tck = splrep(knots[0], knots[1], s=0, k=4) # returns 3-tuple of knots, BSpline coefficients, and polynomial order
    
    t = np.arange(0, knots[0][-1], Ts)
    smoothFit = BSpline(*tck)(t) # wraps tck into BSpline to generate fit
    return t, smoothFit

# plots a given BSpline and its knots
def plotSpline(t, spline, knots, dir: str):
    # INPUTS:
    # t:         1D time array with Ts spacing
    # spline:    1D array of BSpline y values
    # knots:     2-tuple of knots times and frequencies
    # filename:  string for saved image name

    plt.plot(knots[:, 0], knots[:, 1], 'o', color='pink')
    plt.plot(t, spline, '-', color='hotpink')
    plt.xlabel('t (s)'); plt.ylabel('freq (Hz)')
    plt.savefig(dir)
    plt.show()

# generates frequency step function with at least n periods per step
def genStepFreq(N: int, fmin: float, fmax: float, Ts: float, n=10):
    # INPUTS:
    # N:         Number of steps
    # fmin       Initial frequency
    # fmax       Final frequency
    # Ts         Sampling rate
    # n          Minimum number of periods per step
    # OUTPUTS:
    # t:         1D time array with Ts spacing
    # freqs:     1D array of step frequency values

    Tmin = n/fmin # minimum periods per step
    t = np.arange(0, N*Tmin, Ts) # time array
    fstep = (fmax-fmin)//N; tstep = len(t)//N # step frequency values and number of time values per step rounded down
    freqs = np.zeros(len(t))
    for i in range(N):
        freqs[tstep*i:tstep*(i+1)] = fmin + fstep*i # adds same frequency value to tstep length slice segment of freqs
    
    if len(freqs[tstep*(N-1):]) != 0:
         freqs[tstep*N:] = fmin + fstep*(N-1) # fills in final empty t values due to tstep rounding

    return t, freqs

# generates periodically changing frequency function
def genPeriodicFreq(f0, Ts, f1, Tnum, A):
    # INPUTS:
    # f0:        Middle of frequency range
    # Ts:        Sampling rate
    # f1:        Frequency of frequency variation
    # Tnum:      Number of frequency oscillations
    # OUTPUTS:
    # t:         1D time array with Ts spacing
    # freqs:     1D array of periodic frequency values

    t = np.arange(0, Tnum/f1, Ts)
    freqs = f0 + A*np.cos(2*np.pi*f1*t)

    return t, freqs

