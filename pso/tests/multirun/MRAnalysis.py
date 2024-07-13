import numpy as np
import matplotlib.pyplot as plt

filename = 'Data_Nseg10_DB'
dir = 'pso\\tests\\multirun\\saved\\' + filename + '.npy'

runRange, runTimes, fitHist = np.load(dir)

trials = len(runRange[:, 0])
print(trials)

filename = 'RTPlot_Nseg10_DB'
dir = 'pso\\tests\\multirun\\saved\\' + filename + '.png'

for i in range(trials):
    plt.plot(runRange[0, :], runTimes[i, :])
plt.title('Runtume versus Runs for 10 Trials')
plt.xlabel('Run'); plt.ylabel('Time For Single Run')
plt.savefig(dir)
plt.close()

filename = 'LSFPlot_Nseg10_DB'
dir = 'pso\\tests\\multirun\\saved\\' + filename + '.png'

for i in range(trials):
    plt.plot(runRange[0, :], fitHist[i, :])
plt.title('Least Squares Fit versus Runs for 10 Trials')
plt.xlabel('Run'); plt.ylabel('Least Squares Fit')
plt.savefig(dir)
plt.close()