import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from dsignal import sigGen as sg, sigAnalysis as sa
import numpy as np
import psoBestFit as pbf
import random

ST = 30

lsf = np.zeros(ST)

for i in range(ST):
    stnd = i*0.1
    seed = random.randrange(0, sys.maxsize); random.seed(seed)

    f0 = 60; band = 30; fmax = f0 + band; Ts = 1/(10*fmax)
    t, cleanSig, distSig, freqs, freqKnots = sg.genWL(f0, band, 2, Ts, 10, stnd)

    # now that data has been generated, we can try and fit using pso
    xf, yg = sa.FFT(t, distSig, Ts); flim = sa.FFTPeaks(xf, yg)[-1]
    lb0 = [-flim, -flim, -flim]; ub0 = [flim, flim, flim]

    oscT = t[-1]*sum(xf*yg)/sum(yg)
    OPS = 10; Nseg = int(oscT/OPS); runs = 20
    model, lsf[i], isBadFit = pbf.PSOMultiSeg(t, distSig, Nseg, runs, lb0, ub0, subdiv = False, thresh = 0)

stnds = np.arange(0, 1, 0.1)

filename = 'DL_data'
dir = 'pso\\tests\\distlim\\saved\\' + filename + '.npy'
np.save(dir, [stnds, lsf])


