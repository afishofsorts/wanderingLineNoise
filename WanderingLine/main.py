import sigGen as sg
import freqGen as fg
import sigAnalysis as sa

f0 = 60; band = 30; A = 2; M = 30
signal = sg.genSignal(f0, band, A, M)

# sg.plotAll(signal)

fg.plotSpline(signal[3], signal[4])
# plots the frequency over time
sa.plotFFT(signal[0], signal[2])
# use signal[1] for signal with iid and signal[2] for clean signal