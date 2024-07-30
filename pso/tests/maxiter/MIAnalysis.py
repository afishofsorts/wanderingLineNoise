import numpy as np
import matplotlib.pyplot as plt
import statistics

filename = 'MI_NGB'
dir = 'pso\\tests\\maxIter\\saved\\' + filename + '.npy'
NGB = np.load(dir)

bestInd = np.zeros(shape=(20, 60))

for i in range(20):
    for j in range(60):
        bestInd[i, j] = np.where(NGB[i, j, :]<(NGB[i, j, -1]+10))[0][0]


BChange = np.zeros(shape=(6, 200))
BStnds = np.zeros(6); BAvgs = np.zeros(6)

for i in range(6):
    trialComp = np.zeros(10*20)
    for j in range(20):
        trialComp[(j*10):((j+1)*10)] = bestInd[j, :][i*10:(i+1)*10]
    BChange[i, :] = trialComp
    BStnds[i] = statistics.stdev(BChange[i, :])
    BAvgs[i] = sum(trialComp)/200

t = np.arange(1, 7, 1)
x = t**2

plt.title('Required Max Iterations for Relative Bound Size')
plt.xlabel('Relative Bound Size'); plt.ylabel('Avg. Last PSO Iteration within 10 of Final R')
plt.errorbar(x, BAvgs, yerr = BStnds, capsize=3, fmt="r--o", ecolor = "black")
plt.savefig(dir + 'MaxIterBS.png')











