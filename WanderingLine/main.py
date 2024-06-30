from DummySignal import sigGen as sg
from DummySignal import freqGen as fg
from DummySignal import sigAnalysis as sa
import numpy as np
import PSOBestFit as pbf

f0 = 5 # median frequency
band = 2 # range of variation above and below f0
fmax = f0 + band
A = 2 # amplitude of signal
M = 5 # number of breakpoints
Ts = 1/(10*fmax) # sampling rate

freqKnots = fg.genKnot(M, f0, band) # knots for freq spline
t, freqs = fg.genBSpline(freqKnots, Ts) # generates time and freq spline arrays

cleanSig, distSig = sg.genSignal(t, freqs, A, Ts) # generates signal with smoothly varying frequency and iid noise

filename = 'Dummy_WL'
dir = r'C:\Users\casey\Desktop\REU24\WanderingLine\SavedFiles\DataStore\\' + filename + '.npy'
np.save(dir, [t, distSig])

osc = t[-1]*f0 # median number of oscillations

lbounds = [-30, -30, -30]; ubounds = [30, 30, 30] # w2 and w3 bounds are arbitrary right now
model= pbf.PSOSegmenter(filename, int(osc/5), lbounds, ubounds) 
sa.plotSpectComp(t, model, freqs, Ts, fmax, 'PSO iid Fit Spectrogram and Clean Input Frequency Spline')