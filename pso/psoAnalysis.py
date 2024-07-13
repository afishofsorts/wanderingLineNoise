import numpy as np
import matplotlib.pyplot as plt
from . import psoBestFit as pbf

# Plots data set and related model
def plotPSOFit(t, data, model, isBadFit, dir: str):
    # INPUTS:
    # t:           1D time array with Ts spacing
    # data, model: 1D arrays

    plt.figure(figsize=(15, 10))
    plt.plot(t, data)
    plt.plot(t, model)
    for i in range(len(isBadFit)):
        if isBadFit[i]:
            lt, ut = pbf.bounds(t, i, len(isBadFit))
            plt.plot(t[lt:ut], model[lt:ut], 'r')
    plt.title('Clean WL Signal and Resulting PSO Fit')
    plt.xlabel('Time (s)'); plt.ylabel('Strain')
    plt.savefig(dir)
    plt.show()

def modelDif(t, data, model, dir: str):
    dif = data - model
    plt.plot(t, dif)
    plt.title('Difference Between PSO Fit and Clean Signal')
    plt.xlabel('Time (s)'); plt.ylabel('Strain')
    plt.savefig(dir)
    plt.show()