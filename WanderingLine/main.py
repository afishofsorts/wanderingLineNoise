import sigGen as sg
import freqGen as fg
import sigAnalysis as sa

f0 = 60 # constant pivot frequency
band = 30 # range of variation above and below f0
fmax = f0 + band
A = 2 # amplitude of signal
M = 30
Ts = 1/(10*fmax)

signal = sg.genSignal(f0, band, A, Ts, M) # generates signal with smoothly varying frequency and iid noise
times = signal[0]; distSig = signal[1]; cleanSig = signal[2]; freqKnots = signal[3]; freqs = signal[4]

sa.plotAllTime(signal, Ts, fmax)
sa.plotFFT(times, cleanSig, Ts)
sa.plotPSD(cleanSig, Ts)
#sa.plotSpectrogram(times, distSig, Ts, fmax)
#fg.plotSpline(times, freqs, freqKnots)
sa.plotSpectComp(times, distSig, freqs, Ts, fmax)