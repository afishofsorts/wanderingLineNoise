import sigGen as sg
import freqGen as fg
import sigAnalysis as sa

f0 = 60 # constant pivot frequency
band = 30 # range of variation above and below f0
A = 2 # amplitude of signal
M = 30
res = 1000 

signal = sg.genSignal(f0, band, A, res, M) # generates signal with smoothly varying frequency and iid noise
times = signal[0]; distSig = signal[1]; cleanSig = signal[2]; freqKnots = signal[3]; freqs = signal[4]

sa.plotAllTime(signal)
sa.plotFFT(times, cleanSig) # plots FFT of entire clean data set
sa.plotPSD(times, cleanSig)
sa.plotSpectrogram(times, distSig, res)
fg.plotSpline(times, freqs, freqKnots)