import numpy as np
from pso import psoAnalysis as pa, psoBestFit as pbf
from dsignal import freqGen as fg, sigGen as sg
import random
import matplotlib.pyplot as plt

freqKnots = fg.genKnot(10, 100, 20)
t, freqs = fg.genBSpline(freqKnots, 1/1000)
ind0 = np.where(t>=freqKnots[0][1])[0][0]; indF = np.where(t>freqKnots[0][-2])[0][0]
t = t[ind0:indF]
print(t)