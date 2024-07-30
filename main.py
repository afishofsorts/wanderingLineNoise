from dsignal import sigGen as sg, sigAnalysis as sa
import numpy as np
from pso import psoBestFit as pbf, psoAnalysis as pa
import random
import matplotlib.pyplot as plt
import os
import sys
import time

seed = random.randrange(0, sys.maxsize); random.seed(seed)
dir = 'saved\\' + str(seed) + '_stnd_1.0'
if not os.path.exists(dir):
    os.makedirs(dir)

f0 = 60; band = 30; fmax = f0 + band; Ts = 1/(10*fmax)
t, cleanSig, distSig, freqs, freqKnots = sg.genWL(f0, band, 2, Ts, stnd=1.0)
np.save(dir + '\\_data.npy', [t, cleanSig, distSig])
sa.plotAllTime(t, cleanSig, distSig, freqs, freqKnots, Ts, fmax, dir + '\\wl_time_plots_' + str(seed) + '.png')

# now that data has been generated, we can try and fit using pso
xf, yg = sa.FFT(t, distSig, Ts); flim = sa.FFTPeaks(xf, yg, Ts)[-1]
lb0 = [-flim, -flim, -flim]; ub0 = [flim, flim, flim]

oscT = t[-1]*sum(xf*yg)/sum(yg)
OPS = 10; Nseg = int(oscT/OPS); runs = 20
model, lsf, isBadFit = pbf.PSOMultiSeg(t, distSig, Nseg, runs, lb0, ub0, subdiv = False)
np.save(dir + '\\_model.npy', model)

pa.plotPSOFit(t, distSig, model, isBadFit, dir + '\\pso_fit_' + str(seed) + '.png')
pa.modelDif(t, distSig, model, dir + '\\pso_dif_' + str(seed) + '.png')
sa.plotSpectComp(t, model, freqs, freqKnots, Ts, fmax, 'PSO Fit Spectrogram and Input Frequency Spline', dir + '\\PSO_spectC_' + str(seed) + '.png', vmax=0.18)
sa.plotSpectComp(t, cleanSig, freqs, freqKnots, Ts, fmax, 'Clean WL Dummy Signal and Input Frequency Spline', dir + '\\orig_spectC_' + str(seed) + '.png', vmax=0.18)