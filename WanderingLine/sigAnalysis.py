from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np

def plotFFT(x, y):
    N = len(x)
    T = x[1]-x[0]
    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]), 'o')
    plt.grid()
    plt.show()