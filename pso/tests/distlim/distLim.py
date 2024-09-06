import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from dsignal import sigGen as sg, sigAnalysis as sa
import numpy as np
import psoBestFit as pbf
import random

#######################################################
# TEST TO SEE HOW FIT QUALITY DEGRADES WITH IID NOISE #
#######################################################

stnds = np.arange(0, 1, 0.1) # standard deviations for gaussian iid noise
lsf = np.zeros(len(stnds)) # initializing array

for i in range(len(stnds)):
    stnd = i*0.1
    seed = random.randrange(0, sys.maxsize); random.seed(seed)

    f0 = 60; band = 30; fmax = f0 + band; Ts = 1/(10*fmax) # f0 and band were chosed arbitrarily
    t, cleanSig, distSig, freqs, freqKnots = sg.genWL(f0, band, 2, Ts, 10, stnd) # creates simulated WL signal

    # now that data has been generated, we can try and fit using pso
    xf, yg = sa.FFT(t, distSig, Ts) # performs fast fourier transform on signal
    flim = sa.FFTPeaks(xf, yg)[-1] # returns highest frequency peak
    lb0 = [-flim, -flim, -flim]; ub0 = [flim, flim, flim] # creates initial search bounds for pso

    oscT = t[-1]*sum(xf*yg)/sum(yg)# avg total oscillations of signal
    OPS = 10; Nseg = int(oscT/OPS) # oscillations per segment and number of segments
    runs = 20 
    model, lsf[i], isBadFit = pbf.PSOMultiSeg(t, distSig, Nseg, runs, lb0, ub0, subdiv = False, thresh = 0) # adds lsf from this stnd to array

filename = 'DL_data'
dir = 'pso\\tests\\distlim\\saved\\' + filename + '.npy'
np.save(dir, [stnds, lsf])


