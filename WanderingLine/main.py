from DummySignal import freqGen as fg, sigGen as sg, sigAnalysis as sa
import numpy as np
import PSOBestFit as pbf

f0 = 6 # median frequency
band = 0 # range of variation above and below f0
fmax = f0 + band
A = 2 # amplitude of signal
M = 5 # number of breakpoints
Ts = 1/(10*fmax) # sampling rate

freqKnots = fg.genKnot(M, f0, band) # knots for freq spline
t, freqs = fg.genBSpline(freqKnots, Ts) # generates time and freq spline arrays

cleanSig, distSig = sg.genSignal(t, freqs, A, Ts) # generates signal with smoothly varying frequency and iid noise

dir = 'SavedFiles\\DataStore\\Dummy_WL.npy'
np.save(dir, [t, cleanSig])

# sa.plotAllTime(t, distSig, cleanSig, freqKnots, freqs, Ts, fmax)

osc = t[-1]*f0 # median number of oscillations

lbounds = [-40, -30, -30]; ubounds = [40, 30, 30] # w2 and w3 bounds are arbitrary right now
model= pbf.PSOSegmenter(dir, 1, lbounds, ubounds) 
sa.plotSpectComp(t, model, freqs, Ts, fmax, 'PSO iid Fit Spectrogram and Clean Input Frequency Spline')