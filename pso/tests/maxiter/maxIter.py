import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from dsignal import sigGen as sg, sigAnalysis as sa
import numpy as np
import psoBestFit as pbf
from sko.PSO import PSO
import random

##############################################################
# TEST TO SEE HOW FIT HISTORY CHANGES WITH SEARCH BOUND SIZE #
##############################################################

# uses PSO to fit polyCos model to input data, returns global best over iterations
def gbestHistory(t, x, lbounds, ubounds, mi):
    # INPUTS:
    # t:                 1D time array with Ts spacing
    # x:                 1D array of input data
    # lbounds, ubounds:  Length 3 arrays of omega parameter  bounds
    # mi:                Maximum PSO iterations
    # OUTPUTS:
    # gbest:             1D array of global best fits over PSO iterations

    RN = lambda omegas: -pbf.RSub(omegas, t, x) # generates function to be minimized over omegas

    seed = random.randrange(0, sys.maxsize); random.seed(seed)
    pso = PSO(func=RN, n_dim=3, pop=40, max_iter=mi, lb=lbounds, ub=ubounds, w=0.7, c1=0.5, c2=0.5) # performs PSO fitting over omegas
    pso.run()

    gbest = np.zeros(mi)
    for i in range(mi):
        gbest[i] = pso.gbest_y_hist[i][0]

    return gbest

# returns gbest over PSO iterations across multiple independent PSO runs
def gbestMultirun(tseg, xseg, runs, lbounds, ubounds, mi):
    # INPUTS:
    # tseg:              Segment of 1D time array with Ts spacing
    # xseg:              Segment of 1D array of input data
    # runs:              Number of independent PSO fits
    # lbounds, ubounds:  Length 3 arrays of omega parameter  bounds
    # mi:                Maximum PSO iterations
    # OUTPUTS:
    # GBRuns:            2D coordinate array of gbest history for different runs

    GBRuns = np.zeros(shape=(runs, mi))
    for i in range(runs):
        if i % 10 == 0 and i!= 0:
            print('Bounds Widened')
            lbounds, ubounds = pbf.wideBI(lbounds, ubounds, 2)
        GBRuns[i, :] = gbestHistory(tseg, xseg, lbounds, ubounds, mi)
    return GBRuns

# performs gbestMultirun over first 20 segments of input signal
def gbestMultiseg(t, data, runs, lbounds, ubounds, Nseg, mi):
    # INPUTS:
    # t:                 1D time array with Ts spacing
    # x:                 1D array of input data
    # runs:              Number of independent PSO fits
    # lbounds, ubounds:  Length 3 arrays of omega parameter  bounds
    # Nseg:              Number of segments to split t and data into
    # mi:                Maximum PSO iterations
    # OUTPUTS:
    # GBAll:             3D array of each segment's gbest history over different runs

    GBAll = np.zeros(shape=(20, runs, mi))
    for n in range(20):
        print('Segment ' + str(n))
        lt, ut = pbf.bounds(t, n, Nseg)
        tseg = t[lt:ut]; xseg = data[lt:ut]
        GBAll[n, :, :] = gbestMultirun(tseg, xseg, runs, lbounds, ubounds, mi)
    return GBAll
    
    

f0 = 60; band = 30; fmax = f0 + band; Ts = 1/(10*fmax)
t, cleanSig, distSig, freqs, freqKnots = sg.genWL(f0, band, 2, Ts, M=100)

xf, yg = sa.FFT(t, distSig, Ts); flim = sa.FFTPeaks(xf, yg)[-1]
lb0 = [-flim, -flim, -flim]; ub0 = [flim, flim, flim]

GBAll = gbestMultiseg(t, distSig, 60, lb0, ub0, 100, 300) # extra long signal so each segment becomes a trial

filename = 'MI_NGB'
dir = 'pso\\tests\\maxIter\\saved\\' + filename + '.npy'
np.save(dir, GBAll)


