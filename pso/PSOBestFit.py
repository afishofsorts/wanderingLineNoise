import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from sko.PSO import PSO
import time
import random

# cosine function with cubic phase
def polyCos(t, p0, w1, w2, w3):
    # INPUTS:
    # t:           1D time array with Ts spacing
    # p0           Initial phase
    # w1, w2, w3:  Coefficients for time dependent phase terms

    return np.cos(w1*t + w2*t**2 + w3*t**3 + p0)

# least squares for input data and model
def leastSquaresFit(t, data, model):
    # INPUTS:
    # t:         1D time array with Ts spacing
    # data:      1D array of input data
    # model:     1D array
    # OUTPUTS:
    # fit:       Least squares fit value

    ls = (model - data)**2
    fit = 0
    for i in range(len(t)):
        fit = fit + ls[i]
    return fit

def rootCheck(t, a, b, c):
    sqrtArg = (2*b)**2 - 4*3*a*c
    if sqrtArg < 0:
        return False
    root1 = (-2*b + np.sqrt(sqrtArg)) / (2 * 3*a)
    root2 = (-2*b - np.sqrt(sqrtArg)) / (2 * 3*a)
    if t[0] < root1 < t[-1] or t[0] < root1 < t[-1]:
        return True
    return False

# Simplifies polyCos fitting by substituting the to be maximized function with R
def RSub(omegas, t, x):
    # INPUTS:
    # omegas:    Coefficients for time dependent phase terms
    # t:         1D time array with Ts spacing
    # x:         1D array of input data
    # OUTPUTS:
    # R:         Omega dependent function whose argmax is the same as least squares minimization

    w1, w2, w3 = omegas

    isRoot = rootCheck(t, w1, w2, w3)

    if isRoot:
        return -999999
    A = sum(x*polyCos(t, 0, w1, w2, w3))
    B = -sum(x*polyCos(t, -np.pi/2, w1, w2, w3))
    R = np.sqrt(A**2+B**2)
    return R

# calculates upper and lower time indices for a given segment of time array
def bounds(t, n, Nseg):
    # INPUTS:
    # t:         1D time array with Ts spacing
    # n:         Number segment to find the bounds of
    # Nseg:      Total number of segments
    # OUTPUTS:
    # lt, ut     Integer number indices to bound time array

    tstep = len(t)//Nseg
    lt = tstep*n
    if n!=(Nseg-1):
        ut = lt + tstep + 1
    else:
        ut = len(t)+1
    return lt, ut

# Uses PSO to fit polyCos model to input data
def PSOPolyCosFit(t, x, lbounds, ubounds):
    # INPUTS:
    # t:                 1D time array with Ts spacing
    # x:                 1D array of input data
    # lbounds, ubounds:  Length 3 arrays of omega parameter  bounds
    # OUTPUTS:
    # w1, w2, w3, R:     Best fit omegas and minimal R

    RN = lambda omegas: -RSub(omegas, t, x) # generates function to be minimized over omegas

    random.seed()
    pso = PSO(func=RN, n_dim=3, pop=40, max_iter=50, lb=lbounds, ub=ubounds, w=0.7, c1=0.5, c2=0.5) # performs PSO fitting over omegas
    pso.run()

    w1, w2, w3 = pso.gbest_x; R = -pso.gbest_y # best fit omegas and minimal R
    return w1, w2, w3, R

# calculates best fit phase and amplitude given w1, w2, and w3
def parCalc(t, x, w1, w2, w3, R):
    # INPUTS:
    # t:                 1D time array with Ts spacing
    # x:                 1D array of input data
    # w1, w2, w3, R:     Best fit omegas and minimal R
    # OUTPUTS:
    # a:                 Amplitude
    # p0:                Phase

    N = sum(polyCos(t, 0, w1, w2, w3)**2) # variables for amplitude and phase calculation
    A = sum(x*polyCos(t, 0, w1, w2, w3))
    B = -sum(x*polyCos(t, -np.pi/2, w1, w2, w3))
    a = R/N; p0 = np.arctan(B/A)

    return a, p0

# checks if PSO fit hit the parameter bounds and needs to be expanded
def boundCheck(w1, w2, w3, lbounds, ubounds):
    # INPUTS:
    # w1, w2, w3:        Best fit omegas
    # lbounds, ubounds:  Length 3 arrays of omega parameter  bounds
    # OUTPUTS:
    # lbounds, ubounds:  Length 3 arrays of omega parameter  bounds

    if abs(w1) == abs(lbounds[0]):
        lbounds[0] = 1.5*lbounds[0]; ubounds[0] = 1.5*ubounds[0]
    if abs(w2) == abs(lbounds[1]):
        lbounds[1] = 1.5*lbounds[1]; ubounds[1] = 1.5*ubounds[1]
    if abs(w3) == abs(lbounds[2]):
        lbounds[2] = 1.5*lbounds[2]; ubounds[2] = 1.5*ubounds[2]
    return lbounds, ubounds

def PSOSegmenter(t, data, n, Nseg, lbounds, ubounds, dynBound = True):
    lt, ut = bounds(t, n, Nseg) # finds time index bounds for given fitting segment
    tseg = t[lt:ut]; xseg = data[lt:ut] # segmented t and data values

    w1, w2, w3, R = PSOPolyCosFit(tseg, xseg, lbounds, ubounds) # fits data to polyCos using PSO
    a, p0 = parCalc(tseg, xseg, w1, w2, w3, R) # calculates min amplitude and phase based on omegas

    if dynBound:
        lbounds, ubounds = boundCheck(w1, w2, w3, lbounds, ubounds)

    runSeg = a*polyCos(tseg, p0, w1, w2, w3) # generates model signal given previously calculated parameters
    segFit = leastSquaresFit(tseg, xseg, runSeg) # finds the fit of that model against original data
    return lt, ut, runSeg, segFit, lbounds, ubounds

# seperately fits to segments of input coordinate data
def PSOMultirun(t, data, Nseg: int, lbounds, ubounds, runs, dynBound = True):
    # INPUTS:
    # t:                 1D time array with Ts spacing
    # data:              1D array of input data
    # Nseg:              Number of segments to split the data into and fit seperately
    # lbounds, ubounds:  Bounds for omega parameters
    # OUTPUTS:
    # model:             1D array of fitted model

    model = np.zeros(len(data)) # preparing array for model's dependent values
    bestFits = np.zeros(Nseg)
    isBadFit =  [True for i in range(Nseg)]

    for i in range(runs):
        for n in range(Nseg):
            if i==0 or isBadFit[n]: # checks if
                lt, ut, runSeg, segFit, lbounds, ubounds = PSOSegmenter(t, data, n, Nseg, lbounds, ubounds, dynBound)

                if i==0 or segFit<bestFits[n]:
                    model[lt:ut] = runSeg # commits model to the best fit if runfit is better than any previous
                    bestFits[n] = segFit
                    if segFit/len(runSeg) < 0.1:
                        isBadFit[n] = False

    tfit = leastSquaresFit(t, data, model)
    return model, tfit, isBadFit

def PSOMR(tseg, xseg, runs, lbounds, ubounds, thresh = 0.1, dynBound = True):
    isBadFit = True; bestFit = 0
    for i in range(runs):
        if i==0 or isBadFit:
            w1, w2, w3, R = PSOPolyCosFit(tseg, xseg, lbounds, ubounds) # fits data to polyCos using PSO
            a, p0 = parCalc(tseg, xseg, w1, w2, w3, R) # calculates min amplitude and phase based on omegas

            if dynBound:
                lbounds, ubounds = boundCheck(w1, w2, w3, lbounds, ubounds)

            runSeg = a*polyCos(tseg, p0, w1, w2, w3) # generates model signal given previously calculated parameters
            segFit = leastSquaresFit(tseg, xseg, runSeg) # finds the fit of that model against original data

            if i==0 or segFit<bestFit:
                bestFit = segFit
                if segFit/len(runSeg) < thresh:
                    isBadFit = False
    return runSeg, isBadFit