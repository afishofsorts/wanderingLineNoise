import sigGen as sg
import freqGen as fg
import sigAnalysis as sa
import numpy as np

fmin = 3; fmax = 7
N = 4 # number of steps
A = 2 # signal amplitude
Ts = 1/(10*fmax) # sampling rate

t, freqs = fg.genStepFreq(N, fmin, fmax, Ts) # generates time and frequency arrays

cleanSig, distSig = sg.genSignal(t, freqs, A, Ts) # generates frequency step function 

filename = 'Dummy_Step'
dir = r'C:\Users\casey\Desktop\REU24\WanderingLine\SavedFiles\DataStore\\' + filename + '.npy'
np.save(dir, [t, distSig])

sa.plotFFT(t, cleanSig, Ts, 'FFT_Step')
sa.plotPSD(cleanSig, Ts, 'PSD_Step')
sa.plotSpectComp(t, distSig, freqs, Ts, fmax, 'iid Spectrogram and Input Frequency Signal for Step Function', 'SpectC_Step')