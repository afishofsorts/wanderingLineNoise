import numpy as np
import matplotlib.pyplot as plt
import statistics

dir = 'pso\\tests\\maxIter\\saved\\'
NGB = np.load(dir + 'MI_NGB.npy')

bestInd = np.zeros(shape=(20, 60)) # 20 unique segments, 60 pso runs on each

for i in range(20):
    for j in range(60):
        bestInd[i, j] = np.where(NGB[i, j, :]<(NGB[i, j, -1]+10))[0][0] # finds last iteration to increase gbest by 10 


searchRuns = np.zeros(shape=(6, 200)) # 6 search bound sizes, 200 runs across 20 unique segments for each
trialAvgs = np.zeros(shape=(6, 20)) # searchRuns with every 10 runs that share search bounds and same segment averaged
SRAvgs = np.zeros(6) # searchRuns with all runs across all segments averaged for each search bound size
IQends = np.zeros(shape=(2, 6)); IQmids = np.zeros(shape=(2, 6)) # initializing error bars

for i in range(6):
    for j in range(20):
        singleSR = bestInd[j, :][i*10:(i+1)*10] # takes the jth 10 runs for same search bounds and adds them to 1D array
        searchRuns[i, (j*10):((j+1)*10)] = singleSR
        trialAvgs[i, j] = sum(singleSR)/10
    SRAvgs[i] = sum(searchRuns[i, :])/200
    searchRuns[i, :].sort()
    IQends[0, i] = SRAvgs[i] - searchRuns[i, :][0] # finds max and min difference from avg
    IQends[1, i] = searchRuns[i, :][-1] - SRAvgs[i]
    IQmids[0, i] = SRAvgs[i] - searchRuns[i, :][50] # finds IQ2 and 3 difference from avg
    IQmids[1, i] = searchRuns[i, :][149] - SRAvgs[i]



t = np.arange(1, 7, 1)
x = 2**t # search bounds double in maxIter.py

plt.title('Avg. Max Iterations With Substantial Decrease vs. Relative Bound Size')
plt.xlabel('Relative Bound Size'); plt.ylabel('Avg. Last PSO Iteration within 10 of Final R')
plt.errorbar(x, SRAvgs, yerr = IQends, capsize=3, fmt="r--o", ecolor = "black")
plt.errorbar(x, SRAvgs, yerr = IQmids, capsize=3, fmt="r--o", ecolor = "black")
plt.savefig(dir + 'MaxIterBS.png')
plt.close()

bins = np.arange(0, 200, 25)
plt.title('Max Iterations With Substantial Decrease Histogram for Initial Search Bounds')
plt.xlabel('Avg. Last PSO Iteration within 10 of Final R'); plt.ylabel('Frequency')
plt.hist(searchRuns[0, :], bins, rwidth = 0.9)
plt.savefig(dir + 'SB0_Hist.png')












