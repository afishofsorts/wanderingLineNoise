import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from dsignal import freqGen as fg, sigGen as sg, sigAnalysis as sa
from pso import psoBestFit as pbf, psoAnalysis as pa
import numpy as np
import os
import random
import matplotlib.pyplot as plt

seed = 3043026741456144152; random.seed(seed)
dir = 'pso\\tests\\stubseg\\saved\\dense\\' + str(seed)
if not os.path.exists(dir):
    os.makedirs(dir)

# Parameters for simulated Wandering Line signal
f0 = 60 # average frequency
band = 10 # abs distance from f0 for signal
fmax = f0 + band # maximum frequency
Ts = 1/(10*fmax) # sampling rate
a = 2
stnd = 1.0
n = 100

t, cleanSig, distSig, freqs, freqKnots = sg.genWL(f0, band, a, Ts, M=10, stnd=stnd, n = n) # creates signal
np.save(dir + '\\data_' + str(seed) + '.npy', [t, cleanSig, distSig, freqs])
np.save(dir + '\\params_' + str(seed) + '.npy', [f0, band, Ts, a, stnd])

Nseg = 10 # number of segments
pseg = 1
lt, ut = pbf.bounds(t, pseg, Nseg) # finds index bounds for nth segment
tseg = t[lt:ut]-t[lt]; xseg = distSig[lt:ut]

runs = 20
fmax = 110; fmin = 20
lb0 = [fmin, -(fmax-fmin)*Nseg/t[-1]*np.pi, -fmax*3*np.pi]; ub0 = [fmax*2*np.pi, (fmax-fmin)*Nseg/t[-1]*np.pi, fmax*np.pi/3]
lsf, model = pbf.PSOMultiRun(tseg, xseg, runs, lb0, ub0, fmax, fmin, mi=70)
plt.show()
np.save(dir + '\\model_' + str(seed) + '.npy', model)
np.save(dir + '\\freqKnots_' + str(seed) + '.npy', freqKnots)

# finally the result can be analyzed
dif = xseg-model
hop = 20

newTk, newFk = np.zeros(shape=(2, 2))
newTk[0] = freqKnots[0][3]-freqs[lt]; newTk[1] = freqKnots[0][4]-freqs[lt]
newFk[0] = freqKnots[0][3]; newFk[1] = freqKnots[0][4]

pa.plotPSOFit(tseg, xseg, model, [True], dir + '\\pso_fit_' + str(seed) + '.png')
sa.plotSpectComp(tseg, model, freqs[lt:ut], [newFk, newTk], Ts, fmax*1.5, hop, title='PSO Fit Spectrogram and Input Frequency Spline', dir=dir + '\\PSO_spectC_' + str(seed) + '.png', vmax=0.18)
sa.plotSpectComp(tseg, xseg, freqs[lt:ut], [newFk, newTk], Ts, fmax*1.5, hop, title='iid Signal and Input Frequency Spline', dir=dir + '\\orig_spectC_' + str(seed) + '.png', vmax=0.18)
sa.plotPSD(dif, Ts, 'Power Spectral Density of WL Signal minus PSO Fitting', dir + '\\dif_PSD.png')
sa.plotPSD(xseg, Ts, 'Power Spectral Density of Dummy Signal', dir + '\\dist_PSD.png')
sa.plotSpectrogram(dif, Ts, fmax*1.5, hop, dir=dir + '\\final_spect.png', vmax=0.18)