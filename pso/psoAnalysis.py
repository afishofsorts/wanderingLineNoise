import numpy as np
import matplotlib.pyplot as plt
from . import psoBestFit as pbf
import statistics

# plots data set and related model
def plotPSOFit(t, data, model, isBadFit, dir='none'):
    # INPUTS:
    # t:           1D time array with Ts spacing
    # data, model: 1D arrays
    # isBadFit:    Boolean for if fit is above LSF threshold
    # dir:       Directory to save plot PNG to
    # OUTPUTS:
    # Image:     PNG of FFT plot saved to dir

    f, ax = plt.subplots()

    plt.figure(figsize=(15, 10))
    plt.plot(t, data)
    plt.plot(t, model, 'r')
    for i in range(len(isBadFit)):
        if not isBadFit[i]:
            lt, ut = pbf.bounds(t, i, len(isBadFit))
            plt.plot(t[lt:ut], model[lt:ut], 'orange')
    
    lsf = pbf.leastSquaresFit(data, model)
    plt.text(2.1, 2.29, 'LSF: ' + str(round(lsf/len(t), 3)),
     horizontalalignment='left',
     verticalalignment='top', transform=ax.transAxes)
    plt.title('Clean WL Signal and Resulting PSO Fit')
    plt.xlabel('Time (s)'); plt.ylabel('Strain')
    if dir=='none':
        plt.show()
    else:
        plt.savefig(dir)
        plt.close()

# plots difference between data set and model
def plotModelDif(t, data, model, dir='none'):
    # INPUTS:
    # t:           1D time array with Ts spacing
    # data, model: 1D arrays
    # dir:         Directory to save plot PNG to
    # OUTPUTS:
    # Image:       PNG of FFT plot saved to dir

    dif = data - model
    plt.plot(t, dif)
    plt.title('Difference Between PSO Fit and Clean Signal')
    plt.xlabel('Time (s)'); plt.ylabel('Strain')
    if dir=='none':
        plt.show()
    else:
        plt.savefig(dir)
        plt.close()

# plots histogram of iid noise by subtracting out clean signal
def plotDifHist(cleanSig, distSig, model, dir='none'):
    # INPUTS:
    # cleanSig:  1D array of clean signal
    # distSig:   1D array of cleanSig with added iid
    # model:     1D array
    # dir:       Directory to save plot PNG to
    # OUTPUTS:
    # Image:     PNG of FFT plot saved to dir
    
    dif = distSig - model
    trueDif = distSig - cleanSig

    avg = sum(dif)/len(dif)
    stnd = statistics.stdev(dif)

    trueStnd = statistics.stdev(trueDif)
    trueAvg = sum(trueDif)/len(trueDif)

    f, ax = plt.subplots()
    bins = np.arange(-2, 2.5, 0.25)
    plt.title('PSO Fit Subtraction Histogram')
    plt.xlabel('Strain'); plt.ylabel('Frequency')
    plt.hist([dif, trueDif], bins, rwidth = 0.9)
    plt.text(0.81, 0.84, r'$\sigma_{pso}$ = ' + str(round(stnd, 3)),
        horizontalalignment='left',
        verticalalignment='top', transform=ax.transAxes)
    plt.text(0.81, 0.80, r'$\mu_{pso}$ = ' + str(round(avg, 3)),
        horizontalalignment='left',
        verticalalignment='top', transform=ax.transAxes)
    plt.text(0.81, 0.74, r'$\sigma_{orig}$ = ' + str(round(trueStnd, 3)),
        horizontalalignment='left',
        verticalalignment='top', transform=ax.transAxes)
    plt.text(0.81, 0.70, r'$\mu_{orig}$ = ' + str(round(trueAvg, 3)),
        horizontalalignment='left',
        verticalalignment='top', transform=ax.transAxes)
    plt.legend(['PSO Subtraction', 'Original Noise'])
    f.tight_layout()
    if dir=='none':
        plt.show()
    else:
        plt.savefig(dir)
        plt.close()