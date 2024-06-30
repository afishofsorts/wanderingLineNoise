import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from sko.PSO import PSO

def polyCos(t, A, p0, w1, w2, w3):
    return A*np.cos(w1*t + w2*t**2 + w3*t**3)

def leastSquaresFit(params, tf, distSigf):
    w1, w2, w3 = params; A = 2; p0 = 0
    ls = (polyCos(tf, A, p0, w1, w2, w3) - distSigf)**2
    fit = 0
    for i in range(len(distSigf)):
        fit = fit + ls[i]
    return fit

def plotPSOFit(t, data, model):
    plt.figure(figsize=(15, 10))
    plt.plot(t, data)

    plt.plot(t, model)
    plt.show()

# seperately fits to segments of input coordinate data
def PSOSegmenter(filename: str, N: int, lbounds, ubounds):
    dir = r'C:\Users\casey\Desktop\REU24\WanderingLine\SavedFiles\DataStore\\' + filename + '.npy'
    t, distSig = np.load(dir)
    model = np.zeros(len(distSig))
    tseg = len(t)//N
    fitT = 0

    A = 2; p0 = 0
    for i in range(N+1):
        lt = tseg*i; 
        if i!=N:
            ut = lt + tseg
        else:
            ut = len(t)+1
        tf = t[lt:ut]
        distSigf = distSig[lt:ut]
        myFunc = lambda params: leastSquaresFit(params, tf, distSigf)

        pso = PSO(func=myFunc, n_dim=3, pop=40, max_iter=150, lb=lbounds, ub=ubounds, w=0.8, c1=0.5, c2=0.5)
        pso.run()

        w1, w2, w3 = pso.gbest_x
        model[lt:ut] = polyCos(tf, A, p0, w1, w2, w3)
        plt.plot(pso.gbest_y_hist)
        print(pso.gbest_x)
    
    plotPSOFit(t, distSig, model)
    return model

def bfPhase(t, data, w1: float, w2: float, w3: float):
    A = 0; B = 0
    temp = w1*t + w2*t**2 + w3*t**3
    for i in range(len(t)):
        A = A + data[i]*np.cos(temp[i])
        B = B + data[i]*np.sin(temp[i])


