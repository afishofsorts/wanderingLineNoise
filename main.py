import scipy.io
import numpy as np
from dsignal import sigAnalysis as sa
from pso import psoBestFit as pbf

dir = 'saved\\ligodata\\'
filename = 'H-H1_LOSC_2KHZ_V1-1126309888-4096_condat_10to210sec.mat'
mat = scipy.io.loadmat(dir + filename)

t = mat['dataX'][0]; y = mat['dataY'][:, 0]

ind = np.where(t>40)[0][0]

tseg = t[ind:] - t[ind]; yseg = y[ind:]
Ts = tseg[1]-tseg[0]

sa.plotPSD(yseg, Ts, 'PSD of LIGO Data', dir+'ligoPSD.png')
# sa.plotSpectrogram(yseg, Ts, 1290, 1024, fmin=1300)

# now we can try and fit using pso
xf, yg = sa.FFT(tseg, yseg, Ts) # performs fast fourier transform on signal
bandMin = np.where(xf>1200)[0][0]; bandMax = np.where(xf>1400)[0][0]
xf = xf[bandMin:bandMax]; yg = yg[bandMin:bandMax]
lb0 = [-1, -1, -1]; ub0 = [1, 1, 1] # creates initial search bounds for pso

# now we choose how many segments to break the signal into before fitting
f0 = sum(xf*yg)/sum(yg)
oscT = tseg[-1]*f0 # avg total oscillations of signal
# downsampling
newN = int(len(tseg)/2000)
tnew = np.zeros(newN); ynew = np.zeros(newN)
for i in range(newN):
    tnew[i] = tseg[i*2000]
    ynew[i] = yseg[i*2000]
sa.plotPSD(ynew, Ts, 'PSD of Downsampled LIGO Data', dir+'downsampledPSD.png')

Nseg = 16
runs = 20
model, lsf, isBadFit = pbf.PSOMultiSeg(tseg, yseg, Nseg, runs, lb0, ub0, subdiv = True)
np.save(dir + '\\model.npy', model)

sa.plotPSD(yseg-model, Ts, 'PSD of Filtered LIGO Data', dir+'finalPSD.png')