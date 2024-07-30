import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from dsignal import freqGen as fg, sigGen as sg, sigAnalysis as sa
import numpy as np
import psoBestFit as pbf
import matplotlib.pyplot as plt
import time
import random
from sko.PSO import PSO

def genWL(f0, band, M, Ts, A):
    freqKnots = fg.genKnot(M, f0, band) # knots for freq spline
    t, freqs = fg.genBSpline(freqKnots, Ts) # generates time and freq spline arrays
    cleanSig, distSig = sg.genSignal(t, freqs, A, Ts) # generates signal with smoothly varying frequency and iid noise

    return t, cleanSig, distSig, freqs, freqKnots

def PSOPolyCosFit(t, x, lbounds, ubounds, mi):
    # INPUTS:
    # t:                 1D time array with Ts spacing
    # x:                 1D array of input data
    # lbounds, ubounds:  Length 3 arrays of omega parameter  bounds
    # OUTPUTS:
    # w1, w2, w3, R:     Best fit omegas and minimal R

    RN = lambda omegas: -pbf.RSub(omegas, t, x) # generates function to be minimized over omegas

    seed = random.randrange(0, sys.maxsize); random.seed(seed)
    pso = PSO(func=RN, n_dim=3, pop=40, max_iter=mi, lb=lbounds, ub=ubounds, w=0.7, c1=0.5, c2=0.5) # performs PSO fitting over omegas
    pso.run()

    gbest = np.zeros(mi)
    for i in range(mi):
        gbest[i] = pso.gbest_y_hist[i][0]

    w1, w2, w3 = pso.gbest_x; R = -pso.gbest_y # best fit omegas and minimal R
    return w1, w2, w3, R, gbest

def PSOSingleRun(tseg, xseg, lbounds, ubounds, mi):
    w1, w2, w3, R, gbest = PSOPolyCosFit(tseg, xseg, lbounds, ubounds, mi) # fits data to polyCos using PSO
    a, p0 = pbf.parCalc(tseg, xseg, w1, w2, w3, R) # calculates min amplitude and phase based on omegas

    modSeg = a*pbf.polyCos(tseg, p0, w1, w2, w3) # generates model signal given previously calculated parameters
    segFit = pbf.leastSquaresFit(tseg, xseg, modSeg) # finds the fit of that model against original data
    return modSeg, segFit, lbounds, ubounds, gbest

def PSOMultiRun(tseg, xseg, runs, lbounds, ubounds, mi, thresh = 0.1):
    isBadFit = True; bestFit = 0; bestSeg = np.zeros(len(tseg))
    RGbest = np.zeros(shape=(runs, mi))
    MRDifs = np.zeros(runs)
    for i in range(runs):
        runStart = time.perf_counter()
        if isBadFit:
            modSeg, segFit, lbounds, ubounds, RGbest[i, :] = PSOSingleRun(tseg, xseg, lbounds, ubounds, mi)

            if i==0 or segFit<bestFit:
                bestFit = segFit
                bestSeg = np.copy(modSeg)
        runEnd = time.perf_counter()
        MRDifs[i] = runEnd - runStart
    return bestSeg, MRDifs, RGbest


f0 = 60; band = 30; A = 2; M = 300
fmax = f0 + band; Ts = 1/(10*fmax)

runs = 30; trials = 20; mi = 300

tsegs = np.arange(0, 0.2, 0.01)
newTS = np.zeros(shape=(trials, len(tsegs)-1))
for i in range(trials):
    newTS[i, :] = tsegs[1:]

lsf = np.zeros(shape=(trials, len(tsegs)-1))
segTimes = np.zeros(shape=(trials, len(tsegs)-1))
runDifs = np.zeros(shape=(trials, len(tsegs-1), runs))
NGB = np.zeros(shape=(trials, len(tsegs)-1, runs, mi))
t, cleanSig, distSig, freqs, freqKnots = genWL(f0, band, M, Ts, A)

for i in range(trials):
    print('Running trial ' + str(i))
    for j in range(len(tsegs)-1):
        print('Testing seg length ' + str(j))
        flim = sa.FFTPeaks(t, cleanSig, Ts)[-1]
        lbounds = [-flim, -flim, -flim]; ubounds = [flim, flim, flim]

        lt = np.where(t>tsegs[0])[0][0]; ut = np.where(t>tsegs[j+1])[0][0]
        tseg = t[lt:ut]; xseg = distSig[lt:ut]
        start = time.perf_counter()
        bestSeg, runDifs[i, j, :], NGB[i, j, :, :] = PSOMultiRun(tseg, xseg, runs, lbounds, ubounds, mi, thresh = 0)
        end = time.perf_counter()
        lsf[i, j] = pbf.leastSquaresFit(tseg, xseg, bestSeg)/len(tseg)
        segTimes[i, j] = end-start
    tsegs = tsegs + 1.2

filename = 'Short_SLT_data'
dir = 'pso\\tests\\seglen\\saved\\' + filename + '.npy'
np.save(dir, [newTS, lsf, segTimes])

filename = 'Short_SLT_runtimes'
dir = 'pso\\tests\\seglen\\saved\\' + filename + '.npy'
np.save(dir, runDifs)

filename = 'Short_SLT_MI'
dir = 'pso\\tests\\seglen\\saved\\' + filename + '.npy'
np.save(dir, NGB)
