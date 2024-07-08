import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from sko.PSO import PSO
import time

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

# Simplifies polyCos fitting by substituting the to be maximized function with R
def RSub(omegas, t, x):
    # INPUTS:
    # omegas:    Coefficients for time dependent phase terms
    # t:         1D time array with Ts spacing
    # x:         1D array of input data
    # OUTPUTS:
    # R:         Omega dependent function whose argmax is the same as least squares minimization

    w1, w2, w3 = omegas
    A = sum(x*polyCos(t, 0, w1, w2, w3))
    B = -sum(x*polyCos(t, -np.pi/2, w1, w2, w3))
    R = np.sqrt(A**2+B**2)
    return R

# Plots data set and related model
def plotPSOFit(t, data, model):
    # INPUTS:
    # t:           1D time array with Ts spacing
    # data, model: 1D arrays

    plt.figure(figsize=(15, 10))
    plt.plot(t, data)
    plt.plot(t, model)
    plt.show()

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
        ut = lt + tstep
    else:
        ut = len(t)+1

    return lt, ut

# Uses PSO to fit polyCos model to input data
def PSOPolyCosFit(t, x, lbounds, ubounds):
    # INPUTS:
    # t:                 1D time array with Ts spacing
    # x:                 1D array of input data
    # lbounds, ubounds:  Bounds for omega parameters
    # OUTPUTS:
    # w1, w2, w3, R:     Best fit omegas and minimal R

    RN = lambda omegas: -RSub(omegas, t, x) # generates function to be minimized over omegas

    pso = PSO(func=RN, n_dim=3, pop=40, max_iter=50, lb=lbounds, ub=ubounds, w=0.7, c1=0.5, c2=0.5) # performs PSO fitting over omegas
    pso.run()
    plt.plot(pso.gbest_y_hist)

    w1, w2, w3 = pso.gbest_x; R = -pso.gbest_y # best fit omegas and minimal R
    return w1, w2, w3, R

def parCalc(t, x, w1, w2, w3, R):
    N = sum(polyCos(t, 0, w1, w2, w3)**2) # variables for amplitude and phase calculation
    A = sum(x*polyCos(t, 0, w1, w2, w3))
    B = -sum(x*polyCos(t, -np.pi/2, w1, w2, w3))
    a = R/N; p0 = np.arctan(B/A)

    return a, p0

def PSOSegmenter(t, data, n, Nseg, lbounds, ubounds):
    lt, ut = bounds(t, n, Nseg) # finds time index bounds for given fitting segment
    tseg = t[lt:ut]; xseg = data[lt:ut] # segmented t and data values

    w1, w2, w3, R = PSOPolyCosFit(tseg, xseg, lbounds, ubounds) # fits data to polyCos using PSO
    a, p0 = parCalc(tseg, xseg, w1, w2, w3, R) # calculates min amplitude and phase based on omegas

    runSeg = a*polyCos(tseg, p0, w1, w2, w3) # generates model signal given previously calculated parameters
    runFit = leastSquaresFit(tseg, data[lt:ut], runSeg) # finds the fit of that model against original data
    return lt, ut, runSeg, runFit

# seperately fits to segments of input coordinate data
def PSOMultirun(t, data, Nseg: int, lbounds, ubounds, runs):
    # INPUTS:
    # t:                 1D time array with Ts spacing
    # data:              1D array of input data
    # Nseg:              Number of segments to split the data into and fit seperately
    # lbounds, ubounds:  Bounds for omega parameters
    # OUTPUTS:
    # model:             1D array of fitted model
    start = time.perf_counter()
    runstart = time.perf_counter()

    model = np.zeros(len(data)) # preparing array for model's dependent values
    bestFits = np.zeros(Nseg)
    isBadFit =  [True for i in range(Nseg)]

    for i in range(runs):
        print("Now running PSO fit " + str(i))
        for n in range(Nseg):
            if i==0 or isBadFit[n]: # checks if 
                print("Initiating segment " + str(n))
                lt, ut, runSeg, runFit = PSOSegmenter(t, data, n, Nseg, lbounds, ubounds)

                if i==0 or runFit<bestFits[n]:
                    model[lt:ut] = runSeg # commits model to the best fit if this run's fit is better than any previous
                    bestFits[n] = runFit
                    if runFit/len(runSeg) < 0.1:
                        bfend = time.perf_counter()
                        print("Best Fit for Segment " + str(n) + " found in " + str(bfend-start) + "s")
                        isBadFit[n] = False
        runend = time.perf_counter()
        print("Time elapsed for run " + str(i) + ": " + str(runend-runstart))
        runstart = runend

    tfit = leastSquaresFit(t, data, model)
    return model, tfit

