import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from dsignal import freqGen as fg, sigGen as sg, sigAnalysis as sa
import numpy as np
from pso import psoBestFit as pbf, psoAnalysis as pa
import random
import matplotlib.pyplot as plt
import os

def PSOBetterSegs(tseg, xseg, runs, lbounds, ubounds, thresh = 0.1, dynBound = True, mi = 50):
    isBadFit = True; bestFit = 0
    betterSegs = [np.zeros(len(tseg))]
    for i in range(runs):
        if isBadFit:
            print('PSO run ' + str(i))
            if i % 5 == 0 and i!= 0:
                print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
                lbounds, ubounds = pbf.wideBI(lbounds, ubounds, 2.5)
                mi = mi*2
            modSeg, segFit, lbounds, ubounds = pbf.PSOSingleRun(tseg, xseg, lbounds, ubounds, dynBound, mi)

            if i==0 or segFit<bestFit:
                print('Better fit found')
                betterSegs = np.concatenate((betterSegs, [modSeg]), axis=0)
                bestFit = segFit
                if segFit/len(modSeg) < thresh:
                    print('Best fit found')
                    isBadFit = False
    return betterSegs[1:, :], isBadFit, modSeg

seed = 901230450639287252; random.seed(seed)
dir = 'pso\\tests\\pshift\\saved\\' + str(seed)
if not os.path.exists(dir):
    os.makedirs(dir)

f0 = 60; band = 30; fmax = f0 + band; Ts = 1/(10*fmax)
t, cleanSig, distSig, freqs, freqKnots = sg.genWL(f0, band, 2, Ts)
sa.plotAllTime(t, cleanSig, distSig, freqs, freqKnots, Ts, fmax, dir + '\\wl_time_plots_' + str(seed) + '.png')

# now that data has been generated, we can try and fit using pso
xf, yg = sa.FFT(t, cleanSig, Ts); flim = sa.FFTPeaks(xf, yg, Ts)[-1]
lb0 = [-flim, -flim, -flim]; ub0 = [flim, flim, flim]

oscT = t[-1]*sum(xf*yg)/sum(yg)
OPS = 10; Nseg = int(oscT/OPS); runs = 20
lbounds = np.copy(lb0); ubounds = np.copy(ub0)
lt, ut = pbf.bounds(t, 10, Nseg)
tseg = t[lt:ut]; xseg = cleanSig[lt:ut]
print(lt)
print(ut)
betterSegs, isBadFit, modSeg = PSOBetterSegs(tseg, xseg, runs, lb0, ub0, 0.01)
ibf = [isBadFit]

for i in range(len(betterSegs)):
    pa.plotPSOFit(tseg, xseg, betterSegs[i], ibf, dir + '\\pso_fit_' + str(i) + '.png')

pa.plotPSOFit(tseg, xseg, modSeg, ibf, dir + '\\pso_fit_modSeg' + '.png')