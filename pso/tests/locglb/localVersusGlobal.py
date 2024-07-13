import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from dsignal import freqGen as fg, sigGen as sg, sigAnalysis as sa
import numpy as np
from pso import psoBestFit as pbf, psoAnalysis as pa
import random
import matplotlib.pyplot as plt

def PSOSegW(t, data, n, Nseg, lbounds, ubounds, dynBound = True):
    lt, ut = pbf.bounds(t, n, Nseg) # finds time index bounds for given fitting segment
    tseg = t[lt:ut]; xseg = data[lt:ut] # segmented t and data values

    w1, w2, w3, R = pbf.PSOPolyCosFit(tseg, xseg, lbounds, ubounds) # fits data to polyCos using PSO
    a, p0 = pbf.parCalc(tseg, xseg, w1, w2, w3, R) # calculates min amplitude and phase based on omegas

    if dynBound:
        lbounds, ubounds = pbf.boundCheck(w1, w2, w3, lbounds, ubounds)

    runSeg = a*pbf.polyCos(tseg, p0, w1, w2, w3) # generates model signal given previously calculated parameters
    segFit = pbf.leastSquaresFit(tseg, data[lt:ut], runSeg) # finds the fit of that model against original data
    return lt, ut, runSeg, segFit, lbounds, ubounds, w1, w2, w3

# seperately fits to segments of input coordinate data
def PSOMultiW(t, data, Nseg: int, lbounds, ubounds, runs, dynBound = True):
    # INPUTS:
    # t:                 1D time array with Ts spacing
    # data:              1D array of input data
    # Nseg:              Number of segments to split the data into and fit seperately
    # lbounds, ubounds:  Bounds for omega parameters
    # OUTPUTS:
    # model:             1D array of fitted model

    model = np.zeros(len(data)) # preparing array for model's dependent values
    bestFits = np.zeros(Nseg)
    w1, w2, w3 = [0, 0, 0]
    isBadFit =  [True for i in range(Nseg)]

    for i in range(runs):
        print("Now running PSO fit " + str(i))
        for n in range(Nseg):
            if i==0 or isBadFit[n]: # checks if 
                print("Initiating segment " + str(n))
                lt, ut, runSeg, segFit, lbounds, ubounds, w1, w2, w3 = PSOSegW(t, data, n, Nseg, lbounds, ubounds, dynBound)

                if i==0 or segFit<bestFits[n]:
                    model[lt:ut] = runSeg # commits model to the best fit if runfit is better than any previous
                    bestFits[n] = segFit
                    if segFit/len(runSeg) < 0.1:
                        isBadFit[n] = False

    tfit = pbf.leastSquaresFit(t, data, model)
    return model, tfit, isBadFit, w1, w2, w3

def pSampler(w1, w2, w3):
    var1 = 0.01*w1; var2 = 0.01*w2; var3 = 0.01*w3
    smpl1 = np.arange(w1 - var1, w1 + var1, 0.1*var1)
    smpl2 = np.arange(w2 - var2, w2 + var2, 0.1*var2)
    smpl3 = np.arange(w3 - var3, w3 + var3, 0.1*var3)
    return smpl1, smpl2, smpl3

def fitSampler(t, x, w1, w2, w3):
    smpl1, smpl2, smpl3 = pSampler(w1, w2, w3)
    print(smpl1)
    fitProximity = np.zeros(len(smpl1))
    for i in range(len(smpl1)):
        fitProximity[i] = pbf.RSub([w1, w2, w3], t, x)
    return fitProximity

f0 = 60; band = 30; A = 2; M = 10
fmax = f0 + band; Ts = 1/(10*fmax)

random.seed(12) # fixed seed for testing
freqKnots = fg.genKnot(M, f0, band) # knots for freq spline
t, freqs = fg.genBSpline(freqKnots, Ts) # generates time and freq spline arrays
cleanSig, distSig = sg.genSignal(t, freqs, A, Ts) # generates signal with smoothly varying frequency and iid noise

# now that data has been generated, we can try and fit using pso
Nseg = 10; runs = 20
lt, ut = pbf.bounds(t, 1, Nseg) # finds time index bounds for given fitting segment
tseg = t[lt:ut]; xseg = cleanSig[lt:ut] # segmented t and data values

flim = sa.FFTPeaks(t, cleanSig, Ts)[-1]*5
lbounds = [-flim, -flim, -flim]; ubounds = [flim, flim, flim]

model, lsf, isBadFit, w1, w2, w3 = PSOMultiW(tseg, xseg, 1, lbounds, ubounds, runs, False)

pa.plotPSOFit(tseg, xseg, model, isBadFit, 'pso\\tests\\locglb\\saved\\pso_seg_fit.png')

fitProximity = fitSampler(tseg, xseg, w1, w2, w3)
print(fitProximity)
