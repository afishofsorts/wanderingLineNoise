import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from dsignal import freqGen as fg, sigGen as sg, sigAnalysis as sa
import numpy as np
import psoBestFit as pbf
import matplotlib.pyplot as plt
from sko.PSO import PSO
import random

# Uses PSO to fit polyCos model to input data
def PSOMIPCF(t, x, lbounds, ubounds, mi):
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

    return gbest

def PSOMIMR(tseg, xseg, runs, lbounds, ubounds, mi):
    gbest = np.zeros(shape=(runs, mi))
    for i in range(runs):
        if i % 10 == 0 and i!= 0:
            print('Bounds Widened')
            lbounds, ubounds = pbf.wideBI(lbounds, ubounds, 2)
        gbest[i, :] = PSOMIPCF(tseg, xseg, lbounds, ubounds, mi)
    return gbest

def PSOMIMS(t, data, runs, lbounds, ubounds, Nseg, mi):
    NGB = np.zeros(shape=(20, runs, mi))
    for n in range(20):
        print('Segment ' + str(n))
        lt, ut = pbf.bounds(t, n, Nseg)
        tseg = t[lt:ut]; xseg = data[lt:ut]
        NGB[n, :, :] = PSOMIMR(tseg, xseg, runs, lbounds, ubounds, mi)
    return NGB
    
    

f0 = 60; band = 30; fmax = f0 + band; Ts = 1/(10*fmax)
t, cleanSig, distSig, freqs, freqKnots = sg.genWL(f0, band, 2, Ts, M=100)

xf, yg = sa.FFT(t, distSig, Ts); flim = sa.FFTPeaks(xf, yg, Ts)[-1]
lb0 = [-flim, -flim, -flim]; ub0 = [flim, flim, flim]

NGB = PSOMIMS(t, distSig, 60, lb0, ub0, 100, 300)

filename = 'MI_NGB'
dir = 'pso\\tests\\maxIter\\saved\\' + filename + '.npy'
np.save(dir, NGB)


