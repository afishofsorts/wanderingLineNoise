from dsignal import sigGen as sg, sigAnalysis as sa
import numpy as np
from pso import psoBestFit as pbf, psoAnalysis as pa
import random
import os
import sys

# initiated rng seeding and folder for output
seed = random.randrange(0, sys.maxsize); random.seed(seed)
dir = 'saved\\' + str(seed) + '_stnd_1.0'
if not os.path.exists(dir):
    os.makedirs(dir)

# Parameters for simulated Wandering Line signal
f0 = 60 # average frequency
band = 30 # abs distance from f0 for signal
fmax = f0 + band # maximum frequency
Ts = 1/(10*fmax) # sampling rate
t, cleanSig, distSig, freqs, freqKnots = sg.genWL(f0, band, 2, Ts, stnd=1.0) # creates signal
np.save(dir + '\\_data.npy', [t, cleanSig, distSig])
sa.plotAllTime(t, cleanSig, distSig, freqs, freqKnots, Ts, fmax, dir + '\\wl_time_plots_' + str(seed) + '.png')

# now that data has been generated, we can try and fit using pso
xf, yg = sa.FFT(t, distSig, Ts) # performs fast fourier transform on signal
flim = sa.FFTPeaks(xf, yg)[-1] # returns highest frequency peak
lb0 = [-flim, -flim, -flim]; ub0 = [flim, flim, flim] # creates initial search bounds for pso

# now we choose how many segments to break the signal into before fitting
oscT = t[-1]*sum(xf*yg)/sum(yg) # avg total oscillations of signal
OPS = 10; Nseg = int(oscT/OPS) # oscillations per segment and number of segments
runs = 20
model, lsf, isBadFit = pbf.PSOMultiSeg(t, distSig, Nseg, runs, lb0, ub0, subdiv = True)
np.save(dir + '\\_model.npy', model)

pa.plotPSOFit(t, distSig, model, isBadFit, dir + '\\pso_fit_' + str(seed) + '.png')
pa.plotModelDif(t, distSig, model, dir + '\\pso_dif_' + str(seed) + '.png')
sa.plotSpectComp(t, model, freqs, freqKnots, Ts, fmax, 'PSO Fit Spectrogram and Input Frequency Spline', dir + '\\PSO_spectC_' + str(seed) + '.png', vmax=0.18)
sa.plotSpectComp(t, distSig-model, freqs, freqKnots, Ts, fmax, 'Filtered Signal and Input Frequency Spline', dir + '\\orig_spectC_' + str(seed) + '.png', vmax=0.18)