import numpy as np
import random as rand
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt

def genKnot(M: int, f0: float, band: float):
    points = np.zeros(shape=(M, 2))
    tgap = round(10/f0, 2)

    for i in range(M):
        epsi = rand.randrange(-100, 100)*band/100
        points[i, 0] = i*tgap
        points[i, 1] = f0 + epsi
    return points

def genBSpline(p):
    tck = splrep(p[:, 0], p[:, 1], s=0, k=4)
    xnew = np.linspace(0, p[-1, 0], 1000)
    smoothFit = BSpline(*tck)(xnew)
    return [xnew, smoothFit]

def plotSpline(p, spline):
    plt.plot(p[:, 0], p[:, 1], 'o')
    plt.plot(spline[0], spline[1], '-')
    plt.xlabel('t (s)'); plt.ylabel('freq (Hz)')
    plt.show()

