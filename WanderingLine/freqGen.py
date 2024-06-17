import numpy as np
import random as rand
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt

# generates M breakpoints spaced n periods apart each with a
# uniformly random frequency within f0 +- band
def genKnot(M: int, f0: float, band: float, n = 10):
    points = np.zeros(shape=(M, 2))
    tgap = round(n/f0, 2)
    # converting to period

    for i in range(M):
        epsi = rand.randrange(-100, 100)*band/100
        points[i, 0] = i*tgap
        points[i, 1] = f0 + epsi
    return points

# interpolates cubic BSpline fitting to an array of 2D points
def genBSpline(points, res = 2000):
    tck = splrep(points[:, 0], points[:, 1], s=0, k=4)
    # returns 3-tuple of knots, BSpline coefficients, and polynomial order
    xnew = np.linspace(0, points[-1, 0], res)
    smoothFit = BSpline(*tck)(xnew)
    # wraps tck into BSpline to generate fit
    return [xnew, smoothFit]

def plotSpline(p, spline):
    plt.plot(p[:, 0], p[:, 1], 'o', color='pink')
    plt.plot(spline[0], spline[1], '-', color='hotpink')
    plt.xlabel('t (s)'); plt.ylabel('freq (Hz)')
    plt.show()

