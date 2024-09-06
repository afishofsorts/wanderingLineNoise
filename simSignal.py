from dsignal import sigGen as sg, sigAnalysis as sa
import numpy as np
from pso import psoBestFit as pbf, psoAnalysis as pa
import random
import os
import sys

# initiated rng seeding and folder for output
seed = random.randrange(0, sys.maxsize); random.seed(seed)
dir = 'saved\\simdata\\' + str(seed)
if not os.path.exists(dir):
    os.makedirs(dir)

# Parameters for simulated Wandering Line signal
f0 = 1300 # average frequency
band = 20 # abs distance from f0 for signal
fmax = f0 + band # maximum frequency
Ts = 1/(10*fmax) # sampling rate
a = 2
stnd = 1.0
n = 10

t, cleanSig, distSig, freqs, freqKnots = sg.genWL(f0, band, a, Ts, stnd=stnd, n = n) # creates signal
np.save(dir + '\\data_' + str(seed) + '.npy', [t, cleanSig, distSig, freqs])
np.save(dir + '\\params_' + str(seed) + '.npy', [f0, band, Ts, a, stnd])
sa.plotAllTime(t, cleanSig, distSig, freqs, freqKnots, Ts, fmax, dir + '\\wl_time_plots_' + str(seed) + '.png')

# now we choose how many segments to break the signal into before fitting
segTime = n/f0 # number of segments
runs = 20
fmax = 1350; fmin = 1250
model, lsf, isBadFit = pbf.PSOMultiSeg(t, distSig, segTime, runs, fmax, fmin, subdiv = True, dynBound=False, mi = 70)
np.save(dir + '\\model_' + str(seed) + '.npy', model)
np.save(dir + '\\freqKnots_' + str(seed) + '.npy', freqKnots)
np.save(dir + '\\isBadFit_' + str(seed) + '.npy', isBadFit)

# finally the result can be analyzed
dif = distSig-model
hop = 70
pa.plotPSOFit(t, distSig, model, isBadFit, dir + '\\pso_fit_' + str(seed) + '.png')
sa.plotSpectComp(t, model, freqs, freqKnots, Ts, fmax*1.1, hop, fmin=fmin*0.9, title='PSO Fit Spectrogram and Input Frequency Spline', dir=dir + '\\PSO_spectC_' + str(seed) + '.png', vmax=0.18)
sa.plotSpectComp(t, distSig, freqs, freqKnots, Ts, fmax*1.1, hop, fmin=fmin*0.9, title='iid Signal and Input Frequency Spline', dir=dir + '\\orig_spectC_' + str(seed) + '.png', vmax=0.18)
sa.plotPSD(dif, Ts, 'Power Spectral Density of WL Signal minus PSO Fitting', dir + '\\dif_PSD.png')
sa.plotPSD(distSig, Ts, 'Power Spectral Density of Dummy Signal', dir + '\\dist_PSD.png')
sa.plotSpectrogram(dif, Ts, fmax*1.1, hop, fmin=fmin*0.9, dir=dir + '\\final_spect.png', vmax=0.18)