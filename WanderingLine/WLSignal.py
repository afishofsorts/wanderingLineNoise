import sigGen as sg
import freqGen as fg
import sigAnalysis as sa
import numpy as np
import PSOBestFit as pbf

f0 = 60 # median frequency
band = 30 # range of variation above and below f0
fmax = f0 + band; fmin = f0 - band
A = 2 # amplitude of signal
Ts = 1/(10*fmax) # sampling rate

times, distSig, cleanSig, freqKnots, freqs = sg.genSignal(f0, band, A, Ts) # generates signal with smoothly varying frequency and iid noise

dir = r'C:\Users\casey\Desktop\REU24\WanderingLine\DataStore\\distSig.npy'
np.save(dir, [times, distSig])

osc = times[-1]*f0 # median number of oscillations

lbounds = [0, 0, 1.1*fmin*2*np.pi, -1, -1]; ubounds = [5, 2*np.pi, 1.1*fmax*2*np.pi, 1, 1] # w2 and w3 bounds are arbitrary right now
model, avgfit = pbf.PSOSegmenter(int(osc/2), lbounds, ubounds)
sa.plotSpectComp(times, model, freqs, Ts, fmax, 'PSO iid Fit Spectrogram and Clean Input Frequency Spline')
print(avgfit)