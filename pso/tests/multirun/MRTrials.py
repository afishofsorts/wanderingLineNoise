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

# Modifies PSOBestFit's PSOMultiRun to measure the time taken for each run
def PSORunTimes(t, data, Nseg: int, lbounds, ubounds, runs: int):
    # INPUTS:
    # t:                 1D time array with Ts spacing
    # data:              1D array of input data
    # Nseg:              Number of segments to split the data into and fit seperately
    # lbounds, ubounds:  Bounds for omega parameters
    # runs               Maximum number of repeated PSOSegmenter attempts
    # OUTPUTS:
    # runTimes:          1D array of times elapsed for each sequential run
    # fitHist:           1D array of lsf values for each sequential run

    start = time.perf_counter()
    runTimes = np.zeros(runs); fitHist = np.zeros(runs)

    model = np.zeros(len(data)) # preparing array for model's dependent values
    bestFits = np.zeros(Nseg)
    isBadFit =  [True for i in range(Nseg)]

    for i in range(runs):
        runstart = time.perf_counter()
        for n in range(Nseg):
            if i==0 or isBadFit[n]: # checks if 
                lt, ut, runSeg, segFit, lbounds, ubounds = pbf.PSOSegmenter(t, data, n, Nseg, lbounds, ubounds)

                if i==0 or segFit<bestFits[n]:
                    model[lt:ut] = runSeg # commits model to the best fit if runfit is better than any previous
                    bestFits[n] = segFit
                    if segFit/len(runSeg) < 0.1:
                        isBadFit[n] = False

        runend = time.perf_counter()
        runTimes[i] = runend - runstart
        runstart = runend

        fitHist[i] = pbf.leastSquaresFit(t, data, model) # only e-6 seconds added

    return runTimes, fitHist

f0 = 60; band = 30; A = 2; M = 10
fmax = f0 + band; Ts = 1/(10*fmax)

Nseg = 10
trials = 20; runs = 20

runRange = np.arange(1, runs+1, 1)
fitHist = np.zeros(shape=(trials, runs))
runTimes = np.zeros(shape=(trials, runs))

for i in range(trials):
    print("OscTest Trial " + str(i))
    t, cleanSig = genWL(f0, band, M, Ts, A)

    flim = sa.FFTPeaks(t, cleanSig, Ts)[-1]
    lbounds = [-flim, -flim, -flim]; ubounds = [flim, flim, flim]

    runTimes[i, :], fitHist[i, :] = PSORunTimes(t, cleanSig, Nseg, lbounds, ubounds, runs)

newRR = np.zeros(shape=(trials, runs))
for i in range(trials):
    newRR[i, :] = runRange

filename = 'Data_Nseg10_DB'
dir = 'pso\\tests\\multirun\\saved\\' + filename + '.npy'
np.save(dir, [newRR, runTimes, fitHist])