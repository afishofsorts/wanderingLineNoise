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

seed = 25; random.seed(seed)

f0 = 60; band = 30; fmax = f0 + band; Ts = 1/(10*fmax)
Tnum = 5; f1 = 1.5
t, freqs = fg.genPeriodicFreq(f0, Ts, f1, Tnum, 40) # generates time and frequency arrays
cleanSig, distSig = sg.genSignal(t, freqs, 2, Ts) # generates periodically varying frequency signal
plt.plot(t, freqs)
plt.title('Input Signal Spectrogram')
plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)')
plt.savefig('pso\\tests\\perbreak\\saved\\perb_input.png')
plt.close()

xf, yg = sa.FFT(t, cleanSig, Ts); flim = sa.FFTPeaks(xf, yg, Ts)[-1]
lb0 = [-2*flim, -2*flim, -2*flim]; ub0 = [2*flim, 2*flim, 2*flim]

Nseg = 2; runs = 2
model, lsf, isBadFit = pbf.PSOMultiSeg(t, cleanSig, Nseg, runs, lb0, ub0, subdiv = False)
# model, lsf, isBadFit = pbf.PSOMultirun(t, cleanSig, Nseg, lb0, ub0, runs)

pa.plotPSOFit(t, cleanSig, model, isBadFit, 'pso\\tests\\perbreak\\saved\\perb_pso_fit_24.png')
pa.modelDif(t, cleanSig, model, 'pso\\tests\\perbreak\\saved\\perb_dif_24.png')
sa.plotSpectComp(t, model, freqs, Ts, fmax, 'PSO Fit Spectrogram and Clean Input Frequency Spline', 'pso\\tests\\perbreak\\saved\\perb_spectC_24.png')