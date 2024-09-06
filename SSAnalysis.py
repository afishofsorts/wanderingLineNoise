from dsignal import sigGen as sg, sigAnalysis as sa
import numpy as np
from pso import psoBestFit as pbf, psoAnalysis as pa
import random
import os
import sys

seed = 5065321641733889615
dir = 'saved\\simdata\\' + str(seed)

t, cleanSig, distSig, freqs = np.load(dir + '\\data_' + str(seed) + '.npy')
model = np.load(dir + '\\model_' + str(seed) + '.npy')
f0, band, Ts, a, stnd = np.load(dir + '\\params_' + str(seed) + '.npy')
freqKnots = np.load(dir + '\\freqKnots_' + str(seed) + '.npy')
isBadFit = np.load(dir + '\\isBadFit_' + str(seed) + '.npy')
fmax = f0 + band

hop = 70
dif = distSig-model
sa.plotAllTime(t, cleanSig, distSig, freqs, freqKnots, Ts, fmax, dir + '\\wl_time_plots_' + str(seed) + '.png')
pa.plotPSOFit(t, distSig, model, isBadFit, dir + '\\pso_fit_' + str(seed) + '.png')
sa.plotSpectComp(t, model, freqs, freqKnots, Ts, fmax*1.1, hop, fmin=1200, title='PSO Fit Spectrogram and Input Frequency Spline', dir=dir + '\\PSO_spectC_' + str(seed) + '.png', vmax=0.18)
sa.plotSpectComp(t, distSig, freqs, freqKnots, Ts, fmax*1.1, hop, fmin=1200, title='iid Signal and Input Frequency Spline', dir=dir + '\\orig_spectC_' + str(seed) + '.png', vmax=0.18)
sa.plotPSD(dif, Ts, 'Power Spectral Density of WL Signal minus PSO Fitting', dir + '\\dif_PSD.png')
sa.plotPSD(distSig, Ts, 'Power Spectral Density of Dummy Signal', dir + '\\dist_PSD.png')
sa.plotSpectrogram(dif, Ts, fmax*1.1, hop, fmin=1200, dir=dir + '\\final_spect.png', vmax=0.18)