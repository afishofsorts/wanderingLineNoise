import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from dsignal import freqGen as fg, sigGen as sg, sigAnalysis as sa
import numpy as np
import PSOBestFit as pbf
import matplotlib.pyplot as plt

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

osc0 = M*10
OPSmin = 5; OPSmax = 30

Nmax = int(osc0/OPSmin); Nmin = int(osc0/OPSmax)
Nrange = np.arange(Nmin, Nmax, 8)
print(Nrange)
segTests = len(Nrange)

trials = 10; runs = 20
lsf = np.zeros(shape=(segTests, trials))

for i in range(trials):
    print("OscTest Trial " + str(i))
    t, cleanSig = genWL(f0, band, M, Ts, A)

    flim = 1.2*fmax*2*np.pi
    lbounds = [-flim, -200, -200]; ubounds = [flim, 200, 200] # w2 and w3 bounds are arbitrary right now

    for j in range(segTests):
        model, lsf[j, i] = pbf.PSOMultirun(t, cleanSig, Nrange[j], lbounds, ubounds, runs)
    
print(lsf)
avglsf = np.zeros(segTests)
for k in range(segTests):
    avglsf[k] = sum(lsf[k, :])/trials

print(avglsf)
plt.plot(Nrange, avglsf)
plt.show()
