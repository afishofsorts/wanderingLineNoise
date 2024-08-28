import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from dsignal import freqGen as fg, sigGen as sg, sigAnalysis as sa
import numpy as np
import psoBestFit as pbf

def genWL(f0, band, M, Ts, A):
    freqKnots = fg.genKnot(M, f0, band) # knots for freq spline
    t, freqs = fg.genBSpline(freqKnots, Ts) # generates time and freq spline arrays
    cleanSig, distSig = sg.genSignal(t, freqs, A, Ts) # generates signal with smoothly varying frequency and iid noise

    return t, cleanSig, distSig, freqs, freqKnots

def PSOMultiRun(tseg, xseg, runs, lbounds, ubounds, mi, thresh = 0.1, dynBound = True):
    bestFit = np.zeros(runs)
    for i in range(runs):
        if i % 5 == 0 and i!= 0:
            lbounds, ubounds = pbf.wideBI(lbounds, ubounds, 2.5)
            mi = mi*2
        modSeg, segFit, lbounds, ubounds = pbf.PSOSingleRun(tseg, xseg, lbounds, ubounds, dynBound, mi)

        if i==0 or segFit<bestFit[i-1]:
            bestFit[i] = segFit
        else:
            bestFit[i] = bestFit[i-1]
                
    return bestFit

def PSOMultiSeg(t, data, Nseg, runs, lb0, ub0, mi, trials, thresh = 0.1, dynBound = True, subdiv = True):
    BFS = np.zeros(shape=(trials, runs))
    for n in range(trials):
        print('Beginning to fit seg ' + str(n))
        lbounds = np.copy(lb0); ubounds = np.copy(ub0)
        lt, ut = pbf.bounds(t, n, Nseg)
        tseg = t[lt:ut]; xseg = data[lt:ut]
        BFS[n, :] = PSOMultiRun(tseg, xseg, runs, lbounds, ubounds, mi, thresh, dynBound)
    
    return BFS

f0 = 60; band = 30; A = 2; M = 100
fmax = f0 + band; Ts = 1/(10*fmax)

trials = 1; runs = 30

lsf = np.zeros(shape=(trials, runs))
segTimes = np.zeros(shape=(trials, runs))
t, cleanSig, distSig, freqs, freqKnots = genWL(f0, band, M, Ts, A)

flim = sa.FFTPeaks(t, cleanSig)[-1]
lbounds = [-flim, -flim, -flim]; ubounds = [flim, flim, flim]

BFS = PSOMultiSeg(t, distSig, 100, 30, lbounds, ubounds, 100, trials)

filename = 'MR_BFS'
dir = 'pso\\tests\\multirun\\saved\\' + filename + '.npy'
np.save(dir, BFS)