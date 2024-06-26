import sigGen as sg
import freqGen as fg
import sigAnalysis as sa
import numpy as np

f0 = 60 # constant pivot frequency
band = 30 # range of variation above and below f0
fmax = f0 + band
A = 2 # amplitude of signal
M = 30
Ts = 1/(10*fmax)

times, distSig, cleanSig, freqs = sg.genPeriodicSig(f0, band, A, Ts, M)

xf, yg = sa.plotFFT(times, cleanSig, Ts, 'FFT_Periodic')

print(xf)

peaks = np.zeros(100)
n = 0

for i in range(len(yg)-2):
    if 0.01 < yg[i]:
        peaks[n] = xf[i]
        n = n + 1

print(peaks)


# sa.plotPSD(cleanSig, Ts, 'PSD_Periodic')
# sa.plotSpectComp(times, distSig, freqs, Ts, fmax, 'SpectC_Periodic')