import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from dsignal import sigGen as sg, sigAnalysis as sa
from pso import psoBestFit as pbf, psoAnalysis as pa
import random
import os

seed = random.randrange(sys.maxsize); random.seed(seed)
dir = 'pso\\tests\\lowfreq\\saved\\' + str(seed)
if not os.path.exists(dir):
    os.makedirs(dir)

f0 = 10; band = 7; fmax = f0 + band; Ts = 1/(10*fmax)
t, cleanSig, distSig, freqs, freqKnots = sg.genWL(f0, band, 2, Ts)
sa.plotAllTime(t, cleanSig, distSig, freqs, freqKnots, Ts, fmax, dir + '\\wl_time_plots_' + str(seed) + '.png')

# now that data has been generated, we can try and fit using pso
xf, yg = sa.FFT(t, cleanSig, Ts); flim = sa.FFTPeaks(xf, yg)[-1]
lbounds = [-flim, -flim, -flim]; ubounds = [flim, flim, flim]

oscT = t[-1]*sum(xf*yg)/sum(yg)
OPS = 10; Nseg = int(oscT/OPS); runs = 20
model, lsf, isBadFit = pbf.PSOMultirun(t, cleanSig, Nseg, lbounds, ubounds, runs)

pa.plotPSOFit(t, cleanSig, model, isBadFit, dir + '\\pso_fit_' + str(seed) + '.png')
pa.modelDif(t, cleanSig, model, dir + '\\pso_dif_' + str(seed) + '.png')
sa.plotSpectComp(t, model, freqs, Ts, fmax, 'PSO Fit Spectrogram and Clean Input Frequency Spline', dir + '\\wl_spectC_' + str(seed) + '.png')