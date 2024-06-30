import sigGen as sg
import sigAnalysis as sa
import freqGen as fg
import numpy as np

f0 = 60 # Mid-range frequency
f1 = 0.1*f0 # frequency of frequency variation
A = 2 # amplitude of signal
Ts = 1/(10*(f0 + f1)) # Sampling rate with f0 since freq variation is low
Tnum = 20 # number of frequency oscillations

t, freqs = fg.genPeriodicFreq(f0, Ts, f1, Tnum) # generates time and frequency arrays
cleanSig, distSig = sg.genSignal(t, freqs, A, Ts) # generates periodically varying frequency signal

xf, yg = sa.plotFFT(t, cleanSig, Ts, 'FFT_Periodic') # performs FFT

peaks = []; thresh = 0.001 # peaks initialization and threshold for detecting peaks
for i in range(len(yg)-2):
    if thresh < yg[i]:
        peaks = np.append(peaks, xf[i]) # adds any FFT amplitudes above thresh
        
print(peaks)

sa.plotPSD(cleanSig, Ts, 'PSD_Periodic')
sa.plotSpectComp(t, distSig, freqs, Ts, f0*1.1, 'SpectC_Periodic')