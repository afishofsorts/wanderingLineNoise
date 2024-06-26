import numpy as np
import random as rand
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt

# generates M breakpoints spaced n periods apart each with a uniformly random frequency within f0 +- band
def genKnot(M: int, f0: float, band: float, n = 10):
    knots = np.zeros(shape=(M, 2))
    tgap = round(n/f0, 2) # converting to period

    for i in range(M):
        epsi = rand.randrange(-100, 100)*band/100
        knots[i, 0] = i*tgap
        knots[i, 1] = f0 + epsi
    return knots

# interpolates cubic BSpline fitting to an array of 2D knots
def genBSpline(knots, Ts):
    tck = splrep(knots[:, 0], knots[:, 1], s=0, k=4) # returns 3-tuple of knots, BSpline coefficients, and polynomial order
    
    xnew = np.arange(0, knots[-1, 0], Ts)
    smoothFit = BSpline(*tck)(xnew) # wraps tck into BSpline to generate fit
    return xnew, smoothFit

def plotSpline(times, spline, knots, filename = 'FreqSpline'):
    plt.plot(knots[:, 0], knots[:, 1], 'o', color='pink')
    plt.plot(times, spline, '-', color='hotpink')
    plt.xlabel('t (s)'); plt.ylabel('freq (Hz)')
    plt.savefig(r'C:\Users\casey\Desktop\REU24\WanderingLine\SignalPlots\\' + filename + '.png')
    plt.show()

def genStepFreq(M, f0, band, Ts, n=10):
    times = np.arange(0, M*n/f0, Ts)
    fstep = 0; tstep = len(times)//4
    freqs = np.zeros(len(times))
    for i in range(len(times)):
        if i%tstep==0:
            fstep = fstep + band
        freqs[i] = f0 - band + fstep

    return times, freqs

def genPeriodicFreq(M, f0, Ts, n=10):
    times = np.arange(0, M*n/f0, Ts)
    f1 = 0.1*f0

    freqs = f0 + f1*np.cos(2*np.pi*f1*times)

    return times, freqs

