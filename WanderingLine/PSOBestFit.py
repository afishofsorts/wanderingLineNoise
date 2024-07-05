import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from sko.PSO import PSO

def polyCos(t, p0, w1, w2, w3):
    return np.cos(w1*t + w2*t**2 + w3*t**3 + p0)

def leastSquaresFit(params, t, x):
    w1, w2, w3 = params; a = 2; p0 = 0
    ls = (a*polyCos(t, p0, w1, w2, w3) - x**2)
    fit = 0
    for i in range(len(x)):
        fit = fit + ls[i]
    return fit

def RSub(params, t, x):
    w1, w2, w3 = params
    A = sum(x*polyCos(t, 0, w1, w2, w3))
    B = -sum(x*polyCos(t, -np.pi/2, w1, w2, w3))
    R = np.sqrt(A**2+B**2)
    return R

def plotPSOFit(t, data, model):
    plt.figure(figsize=(15, 10))
    plt.plot(t, data)

    plt.plot(t, model)
    plt.show()

# seperately fits to segments of input coordinate data
def PSOSegmenter(t, signal, dir: str, Nseg: int, lbounds, ubounds):
    model = np.zeros(len(signal))
    tstep = len(t)//Nseg
    fitT = 0

    for i in range(Nseg):
        lt = tstep*i; 
        if i!=Nseg:
            ut = lt + tstep + 1
        else:
            ut = len(t)+1
        tseg = t[lt:ut]
        xseg = signal[lt:ut]

        R = lambda params: -RSub(params, tseg, xseg)

        pso = PSO(func=R, n_dim=3, pop=40, max_iter=250, lb=lbounds, ub=ubounds, w=0.7, c1=0.5, c2=0.5)
        pso.run()

        w1, w2, w3 = pso.gbest_x; R = -pso.gbest_y
        N = sum(polyCos(tseg, 0, w1, w2, w3)**2)
        A = sum(xseg*polyCos(tseg, 0, w1, w2, w3))
        B = -sum(xseg*polyCos(tseg, -np.pi/2, w1, w2, w3))

        a = R/N; p0 = np.arctan(B/A)
        model[lt:ut] = a*polyCos(tseg, p0, w1, w2, w3)
        plt.plot(pso.gbest_y_hist)
        print('Omegas: ' + str(pso.gbest_x) + '  p0: ' + str(p0) + '  A: ' + str(a))
    
    plotPSOFit(t, signal, model)
    return model

