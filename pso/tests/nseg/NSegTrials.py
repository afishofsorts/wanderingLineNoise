import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from dsignal import freqGen as fg, sigGen as sg, sigAnalysis as sa
import numpy as np
import psoBestFit as pbf
import matplotlib.pyplot as plt
import time

def genWL(f0, band, M, Ts, A):
    freqKnots = fg.genKnot(M, f0, band) # knots for freq spline
    t, freqs = fg.genBSpline(freqKnots, Ts) # generates time and freq spline arrays
    cleanSig, distSig = sg.genSignal(t, freqs, A, Ts) # generates signal with smoothly varying frequency and iid noise

    return t, cleanSig

f0 = 60 # median frequency
band = 30 # range of variation above and below f0
fmax = f0 + band
A = 2 # amplitude of signal
M = 10 # number of breakpoints with 10 f0 oscillations between them
Ts = 1/(10*fmax) # sampling rate

oscT = M*10
OPSmin = 4; OPSmax = 14
OPS = np.arange(OPSmin, OPSmax, 1)
segTests = len(OPS)

Nrange = [0] * segTests
for i in range(segTests):
    Nrange[i] = int(oscT/OPS[i])
print(Nrange)

trials = 10; runs = 20
MRlsf = np.zeros(shape=(trials, segTests))
MRtimes = np.zeros(shape=(trials, segTests))

for i in range(trials):
    print("NSegTest Trial " + str(i))
    t, cleanSig = genWL(f0, band, M, Ts, A)

    flim = 1.2*fmax*2*np.pi

    for j in range(segTests):
        start = time.perf_counter()
        lbounds = [-flim, -flim, -flim]; ubounds = [flim, flim, flim]
        model, MRlsf[i, j], isBadFit = pbf.PSOMultirun(t, cleanSig, Nrange[j], lbounds, ubounds, runs)
        end = time.perf_counter()
        MRtimes[i, j] = end-start

extOPS = np.zeros(shape=(trials, segTests)); extNR = np.zeros(shape=(trials, segTests))
for i in range(trials):
    extNR[i, :] = Nrange
    for j in range(segTests):
        extOPS[i, j] = oscT/Nrange[j]

filename = 'Data_NSVar.npy'
dir = 'pso\\tests\\nseg\\saved\\' + filename + '.npy'
np.save(dir, [extOPS, extNR, MRlsf, MRtimes])
