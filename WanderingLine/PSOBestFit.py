import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from sko.PSO import PSO
import sigAnalysis as sa

def leastSquaresFit(p, tf, distSigf):
    A, p0, w1, w2, w3 = p
    ls = (A*np.cos(p0 + w1*tf + w2*tf**2 + w3*tf**3) - distSigf)**2
    fit = 0
    for i in range(len(distSigf)):
        fit = fit + ls[i]
    return fit

def leastSquaresFitS(n, tf, distSigf):
    return lambda p: leastSquaresFit(p, tf, distSigf)

def plotPSOFit(t, data, model):
    plt.figure(figsize=(15, 10))
    plt.plot(t, data)

    plt.plot(t, model)
    plt.show()

def PSOSegmenter(n, lbounds, ubounds):
    dir = r'C:\Users\casey\Desktop\REU24\WanderingLine\DataStore\\distSig.npy'
    t, distSig = np.load(dir)
    model = np.zeros(len(distSig))
    tseg = len(t)//n
    fitT = 0

    for i in range(n+1):
        lt = tseg*i; 
        if i!=n:
            ut = lt + tseg
        else:
            ut = len(t)+1
        tf = t[lt:ut]
        distSigf = distSig[lt:ut]
        myFunc = leastSquaresFitS(n, tf, distSigf)

        pso = PSO(func=myFunc, n_dim=5, pop=40, max_iter=70, lb=lbounds, ub=ubounds, w=0.8, c1=0.5, c2=0.5)
        pso.run()
        # print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

        A, p0, w1, w2, w3 = pso.gbest_x
        model[lt:ut] = A*np.cos(p0 + w1*tf + w2*tf**2 + w3*tf**3)
        fitT = fitT + pso.gbest_y
        plt.plot(pso.gbest_y_hist)
    
    # plotPSOFit(t, distSig, model)
    avgfit = fitT/(n+1)
    return model, avgfit


