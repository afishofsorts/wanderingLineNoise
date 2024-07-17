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
t, cleanSig, distSig, freqs, freqKnots = sg.genWL(f0, band, 2, Ts)
sa.plotAllTime(t, cleanSig, distSig, freqs, freqKnots, Ts, fmax, 'pso\\tests\\stubseg\\saved\\cutoff_time_plots.png')

xf, yg = sa.FFT(t, cleanSig, Ts); flim = sa.FFTPeaks(xf, yg, Ts)[-1]
lbounds = [-flim, -flim, -flim]; ubounds = [flim, flim, flim]

Nseg = 10; runs = 20

cutInd = int(9*len(t)/10)
cutSig = cleanSig[:cutInd]; cutTime = t[:cutInd]
model, lsf, isBadFit = pbf.PSOMultirun(cutTime, cutSig, Nseg, lbounds, ubounds, runs)

pa.plotPSOFit(cutTime, cutSig, model, isBadFit, 'pso\\tests\\stubseg\\saved\\cutoff_pso_fit_.png')
pa.modelDif(cutTime, cutSig, model, 'pso\\tests\\stubseg\\saved\\cutoff_dif.png')
sa.plotSpectComp(cutTime, model, freqs[:cutInd], Ts, fmax, 'PSO Fit Spectrogram and Clean Input Frequency Spline', 'pso\\tests\\stubseg\\saved\\cutoff_spectC_.png')