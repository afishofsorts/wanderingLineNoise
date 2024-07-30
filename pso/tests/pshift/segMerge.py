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

def PSOMerge(t, data, Nseg, runs, lb0, ub0, dir, thresh = 0.1, dynBound = True, subdiv = True):
    model = np.zeros(len(t)); isBadFit = [True for i in range(Nseg)]
    m = 0
    for n in range(Nseg):
        if n > 8:
            print('Beginning to fit seg ' + str(n))
            lbounds = np.copy(lb0); ubounds = np.copy(ub0)
            lt, ut = pbf.bounds(t, n, Nseg)
            print(lt)
            print(ut)
            tseg = t[lt:ut]; xseg = data[lt:ut]
            model[lt:ut], isBadFit[m] = pbf.PSOMultiRun(tseg, xseg, runs, lbounds, ubounds, thresh, dynBound)
            np.save(dir + '\\seg_data.npy', [model[lt:ut], tseg, xseg])
        m = m + 1
    
    tfit = pbf.leastSquaresFit(t, data, model)
    return model, tfit, isBadFit

seed = 901230450639287252; random.seed(seed)
dir = 'pso\\tests\\pshift\\saved\\merge_' + str(seed)
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
model, lsf, isBadFit = PSOMerge(t, cleanSig, Nseg, runs, lb0, ub0, dir, 0.01, subdiv = False)

pa.plotPSOFit(t, cleanSig, model, isBadFit, dir + '\\pso_fit_' + str(seed) + '.png')
modSeg, tseg, xseg = np.load(dir + '\\seg_data.npy')
print(pbf.leastSquaresFit(tseg, xseg, modSeg))
pa.plotPSOFit(tseg, xseg, modSeg, isBadFit, dir + '\\pso_fit_end_' + str(seed) + '.png')