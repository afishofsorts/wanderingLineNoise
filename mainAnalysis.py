from dsignal import sigAnalysis as sa
import numpy as np
import scipy.io

dir = 'saved\\ligodata\\'
filename = 'H-H1_LOSC_2KHZ_V1-1126309888-4096_condat_10to210sec.mat'
mat = scipy.io.loadmat(dir + filename)

t = mat['dataX'][0]; y = mat['dataY'][:, 0]

ind = np.where(t>40)[0][0]

tseg = t[ind:] - t[ind]; yseg = y[ind:]
Ts = tseg[1]-tseg[0]

model = np.load(dir + '\\model.npy')

print(model)
print(yseg)
dif = yseg-model
print(dif)

sa.plotPSD(model, Ts, 'PSD of Filtered LIGO Data', dir+'finalPSD.png')
sa.plotSpectrogram(yseg, Ts, 1500, 1024, fmin=0)
sa.plotSpectrogram(model, Ts, 1500, 1024, fmin=0)