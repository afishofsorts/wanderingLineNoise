from dsignal import freqGen as fg, sigGen as sg, sigAnalysis as sa
import numpy as np
from pso import psoBestFit as pbf, psoAnalysis as pa
import random
import matplotlib.pyplot as plt

f0 = 60 # median frequency
band = 30 # range of variation above and below f0
fmax = f0 + band
A = 2 # amplitude of signal
M = 10 # number of breakpoints with 10 f0 oscillations between them
Ts = 1/(10*fmax) # sampling rate

random.seed(12)
freqKnots = fg.genKnot(M, f0, band) # knots for freq spline
t, freqs = fg.genBSpline(freqKnots, Ts) # generates time and freq spline arrays
cleanSig, distSig = sg.genSignal(t, freqs, A, Ts) # generates signal with smoothly varying frequency and iid noise

sa.plotAllTime(t, cleanSig, distSig, freqs, freqKnots, Ts, fmax, 'saved\\wl_time_plots.png')

# now that data has been generated, we can try and fit using pso
flim = sa.FFTPeaks(t, cleanSig, Ts)[-1]

lbounds = [-flim, -flim, -flim]; ubounds = [flim, flim, flim] # w2 and w3 bounds are arbitrary right now
Nseg = 10; runs = 20

model, lsf, isBadFit = pbf.PSOMultirun(t, cleanSig, Nseg, lbounds, ubounds, runs)
print(lsf/len(model))

plt.show()

pa.plotPSOFit(t, cleanSig, model, isBadFit, 'saved\\pso_fit.png')
pa.modelDif(t, cleanSig, model, 'saved\\pso_dif.png')
sa.plotSpectComp(t, model, freqs, Ts, fmax, 'PSO Fit Spectrogram and Clean Input Frequency Spline', 'saved\\wl_spectC.png')