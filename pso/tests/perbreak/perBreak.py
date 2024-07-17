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
plt.show()

xf, yg = sa.FFT(t, cleanSig, Ts); flim = sa.FFTPeaks(xf, yg, Ts)[-1]
lbounds = [-flim, -flim, -flim]; ubounds = [flim, flim, flim]

Nseg = 24; runs = 20
model, lsf, isBadFit = pbf.PSOMultirun(t, cleanSig, Nseg, lbounds, ubounds, runs)

pa.plotPSOFit(t, cleanSig, model, isBadFit, 'pso\\tests\\perbreak\\saved\\perb_pso_fit_.png')
pa.modelDif(t, cleanSig, model, 'pso\\tests\\perbreak\\saved\\perb_dif.png')
sa.plotSpectComp(t, model, freqs, Ts, fmax, 'PSO Fit Spectrogram and Clean Input Frequency Spline', 'pso\\tests\\perbreak\\saved\\perb_spectC_.png')