import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from dsignal import freqGen as fg, sigGen as sg, sigAnalysis as sa
import numpy as np
from pso import PSOBestFit as pbf
import random
import matplotlib.pyplot as plt

f0 = 60; band = 30; A = 2; M = 10
fmax = f0 + band; Ts = 1/(10*fmax)

random.seed(10) # fixed seed for testing
freqKnots = fg.genKnot(M, f0, band) # knots for freq spline
t, freqs = fg.genBSpline(freqKnots, Ts) # generates time and freq spline arrays
cleanSig, distSig = sg.genSignal(t, freqs, A, Ts) # generates signal with smoothly varying frequency and iid noise

sa.plotAllTime(t, cleanSig, distSig, freqs, freqKnots, Ts, fmax, 'pso\\tests\\stubseg\\saved\\wl_seg_time_plots.png')

# now that data has been generated, we can try and fit using pso
xf, yg = sa.FFT(t, cleanSig, Ts)
flim = 0; thresh = 0.0001 # peaks initialization and threshold for detecting peaks
for i in range(len(yg)-2):
    if thresh < yg[i]:
        flim = xf[i] # adds any FFT amplitudes above thresh

print(flim)

Nseg = 10; runs = 20
lt, ut = pbf.bounds(t, 0, Nseg) # finds time index bounds for given fitting segment
tseg = t[lt:ut]; xseg = cleanSig[lt:ut] # segmented t and data values

lbounds = [-flim, -flim, -flim]; ubounds = [flim, flim, flim] # w2 and w3 bounds are arbitrary right now

model, lsf = pbf.PSOMultirun(tseg, xseg, 1, lbounds, ubounds, runs)
print(lsf/len(model))

pbf.plotPSOFit(tseg, xseg, model, 'pso\\tests\\stubseg\\saved\\pso_seg_fit.png')
sa.plotSpectComp(tseg, model, freqs[lt:ut], Ts, fmax, 'PSO iid Fit Spectrogram and Clean Input Frequency Spline', 'pso\\tests\\stubseg\\saved\\wl_seg_spectC.png')