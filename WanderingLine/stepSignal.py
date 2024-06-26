import sigGen as sg
import freqGen as fg
import sigAnalysis as sa
import numpy as np

f0 = 5 # constant pivot frequency
band = 2 # range of variation above and below f0
fmax = f0 + band
A = 2 # amplitude of signal
M = 5
Ts = 1/(10*fmax)

times, distSig, cleanSig, freqs = sg.genStepSig(f0, band, A, Ts, M) # generates step function 

dir = r'C:\Users\casey\Desktop\REU24\WanderingLine\DataStore\\distSig.npy'
np.save(dir, [times, cleanSig])

sa.plotFFT(times, cleanSig, Ts, 'FFT_Step')
sa.plotPSD(cleanSig, Ts, 'PSD_Step')
sa.plotSpectComp(times, distSig, freqs, Ts, fmax, 'SpectC_Step')