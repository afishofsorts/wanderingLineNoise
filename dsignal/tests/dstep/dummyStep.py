import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from dsignal import freqGen as fg, sigGen as sg, sigAnalysis as sa
import numpy as np

fmin = 3; fmax = 7
N = 4 # number of steps
A = 2 # signal amplitude
Ts = 1/(10*fmax) # sampling rate

t, freqs = fg.genStepFreq(N, fmin, fmax, Ts) # generates time and frequency arrays

cleanSig, distSig = sg.genSignal(t, freqs, A, Ts) # generates frequency step function 

dir = 'dsignal\\tests\\dstep\\saved\\Dummy_Step.npy'
np.save(dir, [t, distSig])

dir = 'dsignal\\tests\\dstep\\saved\\FFT_Step.png'
sa.plotFFT(t, cleanSig, Ts, dir)

dir = 'dsignal\\tests\\dstep\\saved\\PSD_Step.png'
sa.plotPSD(cleanSig, Ts, dir)

dir = 'dsignal\\tests\\dstep\\saved\\SpectC_Step.png'
sa.plotSpectComp(t, distSig, freqs, Ts, fmax, 'iid Spectrogram and Input Frequency Signal for Step Function', dir)