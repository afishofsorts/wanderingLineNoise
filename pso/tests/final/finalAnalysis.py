import numpy as np
import matplotlib.pyplot as plt

dir = 'pso\\tests\\final\\saved\\final_data.npy'
stnd, avg, trueStnd, trueAvg = np.load(dir)

print(stnd)

SD_pdif = np.zeros(15); A_dif = np.zeros(15)
TSperA = np.zeros(15)

for i in range(15):
    SD_pdif[i] = np.abs(stnd[i]-trueStnd[i])/trueStnd[i]
    A_dif[i] = np.abs(avg[i]-trueAvg[i])
    TSperA[i] = trueStnd[i]/2

plt.plot(TSperA, SD_pdif)
plt.show()