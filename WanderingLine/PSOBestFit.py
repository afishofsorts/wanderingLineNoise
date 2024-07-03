import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from sko.PSO import PSO

def polyCos(t, p0, w1, w2, w3):
    return np.cos(w1*t + w2*t**2 + w3*t**3 + p0)

def leastSquaresFit(params, t, x):
    w1, w2, w3 = params; A = 2; p0 = 0
    ls = (A*polyCos(t, p0, w1, w2, w3) - x**2)
    fit = 0
    for i in range(len(x)):
        fit = fit + ls[i]
    return fit

def RSub(params, t, x):
    w1, w2, w3 = params
    A = sum(x*polyCos(t, 0, w1, w2, w3))
    B = sum(x*polyCos(t, np.pi/2, w1, w2, w3))
    R = np.sqrt(A**2+B**2)
    return R

def plotPSOFit(t, data, model):
    plt.figure(figsize=(15, 10))
    plt.plot(t, data)

    plt.plot(t, model)
    plt.show()

# seperately fits to segments of input coordinate data
def PSOSegmenter(dir: str, Nseg: int, lbounds, ubounds):
    t, distSig = np.load(dir)
    model = np.zeros(len(distSig))
    tstep = len(t)//Nseg
    fitT = 0

    for i in range(Nseg):
        lt = tstep*i; 
        if i!=Nseg:
            ut = lt + tstep
        else:
            ut = len(t)+1
        tseg = t[lt:ut]
        xseg = distSig[lt:ut]

        R = lambda params: -RSub(params, tseg, xseg)

        pso = PSO(func=R, n_dim=3, pop=40, max_iter=250, lb=lbounds, ub=ubounds, w=0.8, c1=0.5, c2=0.5)
        pso.run()

        w1, w2, w3 = pso.gbest_x; R = -pso.gbest_y
        N = sum(polyCos(tseg, 0, w1, w2, w3)**2)
        B = sum(xseg*polyCos(tseg, 0, w1, w2, w3))
        C = -sum(xseg*polyCos(tseg, np.pi/2, w1, w2, w3))

        A = R/N; p0 = np.arctan(C/B)
        model[lt:ut] = A*polyCos(tseg, p0, w1, w2, w3)
        plt.plot(pso.gbest_y_hist)
        print('Omegas: ' + str(pso.gbest_x) + '  p0: ' + str(p0) + '  A: ' + str(A))
    
    plotPSOFit(t, distSig, model)
    return model

def bfPhase(t, data, w1: float, w2: float, w3: float):
    A = 0; B = 0
    temp = w1*t + w2*t**2 + w3*t**3
    for i in range(len(t)):
        A = A + data[i]*np.cos(temp[i])
        B = B + data[i]*np.sin(temp[i])

dir = 'SavedFiles\\DataStore\\Dummy_WL.npy'
t, x = np.load(dir)
plt.plot(t[:100], x[:100], 'o')
plt.show()
for i in range(10):
        
    tnew = t[:100]; xnew = x[i:100+i]
    # t = np.arange(0, 10, 0.1)
    # x = np.cos(10*np.pi*t)
    Rmax = RSub([12*np.pi, 0, 0], tnew, xnew)
    N = sum(polyCos(tnew, 0, 12*np.pi, 0, 0)**2)
    B = sum(xnew*polyCos(tnew, 0, 12*np.pi, 0, 0))
    C = -sum(xnew*polyCos(tnew, -np.pi/2, 12*np.pi, 0, 0))
    A = Rmax/N; p0 = np.arctan(C/B)

    print(p0)