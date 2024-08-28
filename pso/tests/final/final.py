import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from dsignal import sigGen as sg, sigAnalysis as sa
from pso import psoAnalysis as pa
import numpy as np
import psoBestFit as pbf
import random
import os
import statistics

stnd = np.zeros(30); avg = np.zeros(30)
trueStnd = np.zeros(30); trueAvg = np.zeros(30)

for i in range(15):
    seed = random.randrange(0, sys.maxsize); random.seed(seed)
    dir = 'pso\\tests\\final\\saved\\' + str(seed) + '_SD' + str(i)
    if not os.path.exists(dir):
        os.makedirs(dir)

    f0 = 60; band = 30; fmax = f0 + band; Ts = 1/(10*fmax)
    t, cleanSig, distSig, freqs, freqKnots = sg.genWL(f0, band, 2, Ts, stnd=0.1*i)
    np.save(dir + '\\_data.npy', [t, cleanSig, distSig])
    sa.plotAllTime(t, cleanSig, distSig, freqs, freqKnots, Ts, fmax, dir + '\\wl_time_plots_' + str(seed) + '.png')

    # now that data has been generated, we can try and fit using pso
    xf, yg = sa.FFT(t, distSig, Ts); flim = sa.FFTPeaks(xf, yg)[-1]
    lb0 = [-flim, -flim, -flim]; ub0 = [flim, flim, flim]

    oscT = t[-1]*sum(xf*yg)/sum(yg)
    OPS = 10; Nseg = int(oscT/OPS); runs = 20
    model, lsf, isBadFit = pbf.PSOMultiSeg(t, distSig, Nseg, runs, lb0, ub0, subdiv = False)
    np.save(dir + '\\_model.npy', model)

    pa.plotPSOFit(t, distSig, model, isBadFit, dir + '\\pso_fit_' + str(seed) + '.png')
    pa.modelDif(t, distSig, model, dir + '\\pso_dif_' + str(seed) + '.png')
    sa.plotSpectComp(t, model, freqs, freqKnots, Ts, fmax, 'PSO Fit Spectrogram and Distorted Input Frequency Spline', 
                     dir + '\\wl_spectC_' + str(seed) + '.png', vmax=0.18)
    dif = distSig-model
    sa.plotPSD(dif, 1/900, 'Power Spectral Density of WL Signal minus PSO Fitting', dir + '\\dif_PSD.png')
    sa.plotPSD(distSig, 1/900, 'Power Spectral Density of WL Signal', dir + '\\dist_PSD.png')
    pa.plotDifHist(cleanSig, distSig, model, dir + '\\dif_hist.png')
    sa.plotSpectrogram(t, dif, 1/900, 90, dir + '\\final_spect.png', vmax=0.18)

    trueDif = distSig - cleanSig
    avg[i] = sum(dif)/len(dif)
    stnd[i] = statistics.stdev(dif)
    trueStnd[i] = statistics.stdev(trueDif)
    trueAvg[i] = sum(trueDif)/len(trueDif)

dir = 'pso\\tests\\final\\saved\\final_data.npy'
np.save(dir, [stnd, avg, trueStnd, trueAvg])
