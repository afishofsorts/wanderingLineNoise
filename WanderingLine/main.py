from DummySignal import freqGen as fg, sigGen as sg, sigAnalysis as sa
import numpy as np
import PSOBestFit as pbf

f0 = 60 # median frequency
band = 30 # range of variation above and below f0
fmax = f0 + band
A = 2 # amplitude of signal
M = 10 # number of breakpoints with 10 f0 oscillations between them
Ts = 1/(10*fmax) # sampling rate

freqKnots = fg.genKnot(M, f0, band) # knots for freq spline
t, freqs = fg.genBSpline(freqKnots, Ts) # generates time and freq spline arrays
cleanSig, distSig = sg.genSignal(t, freqs, A, Ts) # generates signal with smoothly varying frequency and iid noise

sa.plotAllTime(t, cleanSig, distSig, freqs, freqKnots, Ts, fmax)

flim = 1.2*fmax*2*np.pi
lbounds = [-flim, -200, -200]; ubounds = [flim, 200, 200] # w2 and w3 bounds are arbitrary right now

Nseg = M; runs = 20

model, lsf = pbf.PSOMultirun(t, cleanSig, Nseg, lbounds, ubounds, runs)
print(lsf/len(model))

pbf.plotPSOFit(t, cleanSig, model)

sa.plotSpectComp(t, model, freqs, Ts, fmax, 'PSO iid Fit Spectrogram and Clean Input Frequency Spline')