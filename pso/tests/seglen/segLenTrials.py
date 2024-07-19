import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from dsignal import freqGen as fg, sigGen as sg, sigAnalysis as sa
import numpy as np
import psoBestFit as pbf
import matplotlib.pyplot as plt
import time

def genWL(f0, band, M, Ts, A):
    freqKnots = fg.genKnot(M, f0, band) # knots for freq spline
    t, freqs = fg.genBSpline(freqKnots, Ts) # generates time and freq spline arrays
    cleanSig, distSig = sg.genSignal(t, freqs, A, Ts) # generates signal with smoothly varying frequency and iid noise

    return t, cleanSig, distSig, freqs, freqKnots


def PSOSegment(t, x, lbounds, ubounds, dynBound = True):
    w1, w2, w3, R = pbf.PSOPolyCosFit(t, x, lbounds, ubounds) # fits data to polyCos using PSO
    a, p0 = pbf.parCalc(t, x, w1, w2, w3, R) # calculates min amplitude and phase based on omegas

    if dynBound:
        lbounds, ubounds = pbf.boundCheck(w1, w2, w3, lbounds, ubounds)

    runSeg = a*pbf.polyCos(t, p0, w1, w2, w3) # generates model signal given previously calculated parameters
    segFit = pbf.leastSquaresFit(t, x, runSeg) # finds the fit of that model against original data
    return runSeg, segFit, lbounds, ubounds

def PSOMultiRun(t, data, lbounds, ubounds, runs):
    bestFit = 0
    for j in range(runs):
        runSeg, segFit, lbounds, ubounds = PSOSegment(t, data, lbounds, ubounds)

        if j==0 or segFit<bestFit:
            bestFit = segFit
    return bestFit

def PSOMultLen(t, data, lbounds, ubounds, runs: int, tsegs):

    start = time.perf_counter()
    segTimes = np.zeros(len(tsegs)-1)
    lsf = np.zeros(len(tsegs)-1)

    for i in range(len(tsegs)-1):
        segstart = time.perf_counter()
        lt = np.where(t > tsegs[i])[0][0]; ut = np.where(t > tsegs[i+1])[0][0]
        tseg = t[lt:ut]; xseg = data[lt:ut]

        lsf[i] = PSOMultiRun(tseg, xseg, lbounds, ubounds, runs)

        segend = time.perf_counter()
        segTimes[i] = segend - segstart
        segstart = segend

    return lsf, segTimes

f0 = 60; band = 30; A = 2; M = 100
fmax = f0 + band; Ts = 1/(10*fmax)

runs = 20; trials = 15

tsegs = np.arange(0, 1.2, 0.1)
newTS = np.zeros(shape=(trials, len(tsegs)-1))
for i in range(trials):
    newTS[i, :] = tsegs[1:]

lsf = np.zeros(shape=(trials, len(tsegs)-1))
segTimes = np.zeros(shape=(trials, len(tsegs)-1))
t, cleanSig, distSig, freqs, freqKnots = genWL(f0, band, M, Ts, A)

filename = 'ML_alltime'
dir = 'pso\\tests\\seglen\\saved\\' + filename + '.png'
sa.plotAllTime(t, cleanSig, distSig, freqs, freqKnots, Ts, fmax, dir)

for i in range(trials):
    print('Running trial' + str(i))
    tsegs = tsegs + 0.9
    flim = sa.FFTPeaks(t, cleanSig, Ts)[-1]
    lbounds = [-flim, -flim, -flim]; ubounds = [flim, flim, flim]

    lsf[i, :], segTimes[i, :] = PSOMultLen(t, cleanSig, lbounds, ubounds, runs, tsegs)

filename = 'ML_data'
dir = 'pso\\tests\\seglen\\saved\\' + filename + '.npy'
np.save(dir, [newTS, lsf, segTimes])
