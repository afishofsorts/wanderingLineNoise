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

seed = 23; random.seed(seed)

f0 = 60; band = 30; fmax = f0 + band; Ts = 1/(10*fmax)
t1, cleanSig1, distSig1, freqs1, freqKnots1 = sg.genWL(f0, band, 2, Ts)

freqKnots2 = fg.genKnot(10, f0, band) # knots for freq spline
freqKnots2[1][0] = freqKnots1[1][-1]

t2, freqs2 = fg.genBSpline(freqKnots2, Ts) # generates time and freq spline arrays
cleanSig2, distSig2 = sg.genSignal(t2, freqs2, 2, Ts) # generates signal with smoothly varying frequency and iid noise

t = np.append(t1, t2[1:]+t1[-1])
cleanSig = np.append(cleanSig1, cleanSig2[1:])
distSig = np.append(distSig1, distSig2[1:])
freqs = np.append(freqs1, freqs2[1:])
freqKnots = np.append(freqKnots1, freqKnots2[1][1:] + t1[-1])

sa.plotAllTime(t, cleanSig, distSig, freqs, freqKnots, Ts, fmax, 'pso\\tests\\stubseg\\saved\\extension_time_plots.png')

xf, yg = sa.FFT(t, cleanSig, Ts); flim = sa.FFTPeaks(xf, yg, Ts)[-1]
lbounds = [-flim, -flim, -flim]; ubounds = [flim, flim, flim]

Nseg = 20; runs = 20

model, lsf, isBadFit = pbf.PSOMultirun(t, cleanSig, Nseg, lbounds, ubounds, runs)

pa.plotPSOFit(t, cleanSig, model, isBadFit, 'pso\\tests\\stubseg\\saved\\extension_pso_fit_.png')
pa.modelDif(t, cleanSig, model, 'pso\\tests\\stubseg\\saved\\extension_dif.png')
sa.plotSpectComp(t, model, freqs, Ts, fmax, 'PSO Fit Spectrogram and Clean Input Frequency Spline', 'pso\\tests\\stubseg\\saved\\extension_spectC_.png')