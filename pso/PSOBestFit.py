import matplotlib.pyplot as plt
import numpy as np
from sko.PSO import PSO
import sys
import random
import statistics
import math

# cosine function with cubic phase
def polyCos(t, p0, w0, w1, w2):
    # INPUTS:
    # t:           1D time array with Ts spacing
    # p0           Initial phase
    # w0, w1, w2:  Coefficients for time dependent phase terms

    return np.cos(w0*t + w1*t**2 + w2*t**3 + p0)

# least squares for input data and model
def leastSquaresFit(data, model):
    # INPUTS:
    # data:      1D array of input data
    # model:     1D array
    # OUTPUTS:
    # fit:       Least squares fit value

    fit = sum((model - data)**2)
    return fit

# checks if quadratic function has roots during t
def rootCheck(t, a, b, c):
    # INPUTS:
    # t:         1D time array with Ts spacing
    # a, b, c:   Quadratic coefficients
    # OUTPUTS:
    # Boolean:   True if function is negative during t

    sqrtArg = b**2 - 4*a*c # arguement under square root in quadratic formula
    if sqrtArg < 0:
        return False # roots would be imaginary, no real solution
    root1 = (-b + np.sqrt(sqrtArg)) / (2 * a) # quadratic formula
    root2 = (-b - np.sqrt(sqrtArg)) / (2 * a)
    if t[0] < root1 < t[-1] or t[0] < root2 < t[-1]:
        return True
    return False

# checks if omegas produce a frequency functions within freq range during t
def frangeCheck(t, w0, w1, w2, fmax, fmin):
    freqs = w0 + 2*w1*t + 3*w2*t**2
    low = np.where(fmin>freqs); high = np.where(freqs>fmax)
    if len(low[0])==0 and len(high[0])==0:
        return True
    return False

# simplifies polyCos fitting by substituting the to be maximized function with R
def RSub(omegas, t, x, fmax, fmin):
    # INPUTS:
    # omegas:    Coefficients for time dependent phase terms
    # t:         1D time array with Ts spacing
    # x:         1D array of input data
    # OUTPUTS:
    # R:         Omega dependent function whose argmax is the same as least squares minimization

    w0, w1, w2 = omegas

    isRoot = rootCheck(t, 3*w2, 2*w1, w0) # checks if omegas result in a function with negative frequency
    isRange = frangeCheck(t, w0, w1, w2, fmax, fmin)
    if isRoot and isRange:
        return -999999
    
    A = sum(x*polyCos(t, 0, w0, w1, w2))
    B = -sum(x*polyCos(t, -np.pi/2, w0, w1, w2))
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
def PSOPolyCosFit(t, x, lbounds, ubounds, fmax, fmin, mi = 50):
    # INPUTS:
    # t:                 1D time array with Ts spacing
    # x:                 1D array of input data
    # lbounds, ubounds:  Length 3 arrays of omega parameter  bounds
    # OUTPUTS:
    # w0, w1, w2, R:     Best fit omegas and minimal R

    RN = lambda omegas: -RSub(omegas, t, x, fmax, fmin) # generates function to be minimized over omegas

    seed = random.randrange(0, sys.maxsize); random.seed(seed)
    pso = PSO(func=RN, n_dim=3, pop=40, max_iter=mi, lb=lbounds, ub=ubounds, w=0.7, c1=0.5, c2=0.5) # performs PSO fitting over omegas
    pso.run()

    w0, w1, w2 = pso.gbest_x; R = -pso.gbest_y # best fit omegas and minimal R
    return w0, w1, w2, R

# calculates best fit phase and amplitude given w0, w1, and w2
def parCalc(t, x, w0, w1, w2, R):
    # INPUTS:
    # t:                 1D time array with Ts spacing
    # x:                 1D array of input data
    # w0, w1, w2, R:     Best fit omegas and minimal R
    # OUTPUTS:
    # a:                 Amplitude
    # p0:                Phase

    N = sum(polyCos(t, 0, w0, w1, w2)**2) # variables for amplitude and phase calculation
    A = sum(x*polyCos(t, 0, w0, w1, w2))
    B = -sum(x*polyCos(t, -np.pi/2, w0, w1, w2))
    a = R/N; p0 = np.arctan(B/A)

    return a, p0

# multiplies kth entry of upper and lower bounds by incr
def boundIncr(lbounds, ubounds, k, incr=1.5):
    # INPUTS:
    # lbounds, ubounds:  Length 3 arrays of omega parameter bounds
    # k:                 Index for bounds                
    # incr:              Multiplication term
    # OUTPUTS:
    # lbounds, ubounds:  Multiplied length 3 arrays

    lbounds[k] = incr*lbounds[k]; ubounds[k] = incr*ubounds[k]
    return lbounds, ubounds

# checks if PSO fit hit the parameter bounds and needs to be expanded
def boundCheck(omegas, lbounds, ubounds):
    # INPUTS:
    # omegas:            Best fit omegas
    # lbounds, ubounds:  Length 3 arrays of omega parameter bounds
    # OUTPUTS:
    # lbounds, ubounds:  Length 3 arrays of omega parameter bounds

    for i in range(1):
        if omegas[i+2] == lbounds[i+2] or omegas[i+2] == ubounds[i+2]:
            lbounds, ubounds = boundIncr(lbounds, ubounds, i+1)
    return lbounds, ubounds

# multiplies all search bounds by incr
def wideBI(lbounds, ubounds, incr):
    # INPUTS:
    # lbounds, ubounds:  Length 3 arrays of omega parameter bounds
    # incr:              Multiplication term
    # OUTPUTS:
    # lbounds, ubounds:  Multiplied length 3 arrays
    for i in range(1):
        lbounds[i+2] = incr*lbounds[i+2]; ubounds[i+2] = incr*ubounds[i+2]
    return lbounds, ubounds

# performs one PSO fit to signal segment
def PSOSingleRun(tseg, xseg, lbounds, ubounds, fmax, fmin, dynBound = True, mi = 70):
    # INPUTS:
    # tseg:              1D array segment of t
    # xseg:              1D array segment of data
    # lbounds, ubounds:  Length 3 arrays of omega parameter bounds
    # dynBound:          Boolean for search bound increase if best fit omegas are at bounds
    # mi:                Maximum iterations for PSO algorithm
    # OUTPUTS:
    # modSeg:            1D array of quadratic chirp model for this segment
    # segFit:            Least Squares Fit for model and xseg
    # lbounds, ubounds:  Length 3 arrays of omega parameter bounds with possible increases since input

    w0, w1, w2, R = PSOPolyCosFit(tseg, xseg, lbounds, ubounds, fmax, fmin, mi) # fits data to polyCos using PSO
    a, p0 = parCalc(tseg, xseg, w0, w1, w2, R) # calculates min amplitude and phase based on omegas
    if dynBound:
        lbounds, ubounds = boundCheck([w0, w1, w2], lbounds, ubounds)

    modSeg = a*polyCos(tseg, p0, w0, w1, w2) # generates model signal given previously calculated parameters
    segFit = leastSquaresFit(xseg, modSeg) # finds the fit of that model against original data
    segFitAlt = leastSquaresFit(xseg, -modSeg)

    if segFit>segFitAlt:
        modSeg = -modSeg
        segFit = segFitAlt
    return modSeg, segFit, lbounds, ubounds

# performs multiple independent PSO fits to a single segment and returns the best found
def PSOMultiRun(tseg, xseg, runs, lbounds, ubounds, fmax, fmin, thresh = 0, dynBound = True, mi = 70):
    # INPUTS:
    # tseg:              1D array segment of t
    # xseg:              1D array segment of data
    # runs:              Number of independent PSO fits
    # lbounds, ubounds:  Length 3 arrays of omega parameter bounds
    # thresh:            Least Squares Fit value below which a fit it considered good enough
    # dynBound:          Boolean for search bound increase if best fit omegas are at bounds
    # mi:                Maximum iterations for PSO algorithm
    # OUTPUTS:
    # bestSeg:           1D array of bets quadratic chirp model for this segment across all runs
    # isBadFit:          Boolean for if bestSeg's Least Squares Fit is less that thresh

    isBadFit = True; bestFit = 0; bestSeg = np.zeros(len(tseg))
    for i in range(runs):
        if isBadFit:
            if i % 10 == 0 and i!= 0 and dynBound: # increases search bounds automatically every 5 runs
                lbounds, ubounds = wideBI(lbounds, ubounds, 2.5)
            modSeg, segFit, lbounds, ubounds = PSOSingleRun(tseg, xseg, lbounds, ubounds, fmax, fmin, dynBound, mi) # performs one PSO run

            if i==0 or segFit<bestFit:
                bestFit = segFit # updates bestFit and bestSeg with better model from this run
                bestSeg = np.copy(modSeg)
                if segFit/len(modSeg) < thresh: # if Least Squares Fit is below thresh, ends runs there
                    isBadFit = False
    return bestFit, bestSeg

# breaks up input signal and performs multiple independent PSO fits for each segment returning best found for all
def PSOMultiSeg(t, data, segTime, runs, fmax, fmin, thresh = 1.0, dynBound = True, mi = 70, subdiv = True):
    # INPUTS:
    # t:                 1D array with Ts spacing
    # data:              1D array of input data
    # Nseg:              Number of segments to split t and data into
    # runs:              Number of independent PSO fits for each segment
    # fmax, fmin:        Range of input signal
    # thresh:            Least Squares Fit value below which a fit it considered good enough
    # dynBound:          Boolean for search bound increase if best fit omegas are at bounds
    # mi:                Maximum iterations for PSO algorithm
    # subdiv:            Boolean for further splitting segments that don't reach thresh in 1 multirun
    # OUTPUTS:
    # model:             1D array of bets quadratic chirp model for all segments across all runs
    # finalFit:          Least Squares Fit for final model and data
    # isBadFit:          Boolean for if bestSeg's Least Squares Fit is less that thresh

    Nseg = math.ceil(t[-1]/segTime)
    model = np.zeros(len(t)); isBadFit = [True for i in range(Nseg)]; segFits = np.zeros(Nseg)
    lb0 = [fmin, -(fmax-fmin)*segTime*np.pi, -(fmax-fmin)*np.pi/3]; ub0 = [fmax*2*np.pi, (fmax-fmin)*segTime*np.pi, (fmax-fmin)*np.pi/3]
    for n in range(Nseg):
        print('Now fitting seg ' + str(n) + ' of ' + str(Nseg-1))
        lbounds = np.copy(lb0); ubounds = np.copy(ub0)
        lt, ut = bounds(t, n, Nseg) # finds index bounds for nth segment
        tseg = t[lt:ut]-t[lt]; xseg = data[lt:ut]
        segFits[n], model[lt:ut] = PSOMultiRun(tseg, xseg, runs, lbounds, ubounds, fmax, fmin, thresh, dynBound, mi) # performs multiple PSO runs
        print('Fit with length averaged LSF of ' + str(segFits[n]/len(model[lt:ut])))
    if subdiv:
        print('Checking for poor outlier fits...')
        SFAvg = sum(segFits)/len(segFits); SFStnd = statistics.stdev(segFits)
        for n in range(Nseg):
            if segFits[n] > SFAvg + 2*SFStnd:
                print('Segment ' + str(n) + ' has an outlier fit, attempting to subdivide')
                lbounds = np.copy(lb0); ubounds = np.copy(ub0)
                lt, ut = bounds(t, n, Nseg)
                tseg = t[lt:ut]; xseg = data[lt:ut]
                halfInd = len(tseg)//2; modSeg = np.zeros(len(model[lt:ut]))
                SF1, modSeg[:halfInd] = PSOMultiRun(tseg[:halfInd], xseg[:halfInd], runs, lbounds, ubounds, fmax, fmin, thresh, dynBound, mi)
                lbounds = np.copy(lb0); ubounds = np.copy(ub0)
                SF2, modSeg[halfInd:] = PSOMultiRun(tseg[halfInd:], xseg[halfInd:], runs, lbounds, ubounds, fmax, fmin, thresh, dynBound, mi)
                SFtotal = leastSquaresFit(xseg, modSeg)
                if SFtotal < segFits[n]:
                    model[lt:ut] = modSeg
                    segFits[n] = SFtotal
                    print('Better fit found for segment ' + str(n) + ' with sub fits: ' + str(SF1/halfInd) + ', ' + str(SF2/halfInd))
                else:
                    print('No better fit was found, retaining original outlier')
    
    for n in range(Nseg):
        if segFits[n] < thresh:
            isBadFit[n] = False
        
    finalFit = leastSquaresFit(data, model); 
    print('Done! Model generated with fit ' + str(finalFit/len(t)))
    return model, finalFit, isBadFit