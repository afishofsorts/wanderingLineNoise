import numpy as np
from dsignal import sigAnalysis as sa
from pso import psoAnalysis as pa

dir = 'saved\\3898574613585262069'
t, cleanSig, distSig = np.load(dir + '\\_data.npy')
model = np.load(dir + '\\_model.npy')

dif = distSig-model

sa.plotPSD(dif, 1/900, 'Power Spectral Density of WL Signal minus PSO Fitting', dir + '\\dif_PSD.png')
sa.plotPSD(distSig, 1/900, 'Power Spectral Density of Dummy Signal', dir + '\\dist_PSD.png')
pa.plotDifHist(cleanSig, distSig, model, dir + '\\dif_hist.png')
sa.plotSpectrogram(t, dif, 1/900, 90, dir + '\\final_spect.png', vmax=0.18)

