import numpy as np
import matplotlib.pyplot as plt
import statistics

filename = 'ML_DMI_data'
dir = 'pso\\tests\\seglen\\saved\\' + filename + '.npy'
newTS, lsf, segTimes = np.load(dir)
indices = [0, 1, 9]
newTS = np.delete(newTS[2:], 9, 0)
lsf = np.delete(lsf[2:], 9, 0)
segTimes = np.delete(segTimes[2:], 9, 0)


filename = 'ML_DMI_runtimes'
dir = 'pso\\tests\\seglen\\saved\\' + filename + '.npy'
runDifs = np.load(dir)

filename = 'Short_SLT_MI'
dir = 'pso\\tests\\seglen\\saved\\' + filename + '.npy'
NGB = np.load(dir)

trials = len(newTS[:, 0]); tsegs = newTS[0, :]

OPS = np.zeros(trials)
for i in range(trials):
    OPS = newTS[i, :]*60

for i in range(trials):
    plt.plot(OPS, lsf[i, :])
filename = 'ML_DMI_lsf'
dir = 'pso\\tests\\seglen\\saved\\' + filename + '.png'
plt.title('Length Weighted LSF versus OPS for Distorted Signal')
plt.xlabel('Average oscillations per Segment'); plt.ylabel('Weighted LSF')
plt.savefig(dir)
plt.close()

for i in range(trials):
    plt.plot(newTS[0, :], segTimes[i, :])
filename = 'Short_SLT_times'
dir = 'pso\\tests\\seglen\\saved\\' + filename + '.png'
plt.title('PSO Multirun Time with Segment Length Variation (runs=20)')
plt.xlabel('Segment Length (s)'); plt.ylabel('Multirun Time (s)')
plt.savefig(dir)
plt.close()

avglsf = np.zeros(len(tsegs))
errs = np.zeros(len(tsegs))
for i in range(len(tsegs)):
    avglsf[i] = sum(lsf[:, i])/trials
    errs[i] = statistics.stdev(lsf[:, i])
plt.title('Length weighted LSF versus OPS for Distorted Signal')
plt.xlabel('Average Oscillations per Segment'); plt.ylabel('Average Weighted LSF')
plt.errorbar(OPS, avglsf, yerr = errs, capsize=3, fmt="r--o", ecolor = "black")
filename = 'ML_DMI_avglsf'
dir = 'pso\\tests\\seglen\\saved\\' + filename + '.png'
plt.savefig(dir)
plt.close()

runs = len(runDifs[0, 0, :])

AvgSRT = np.zeros(shape=(len(tsegs), runs))
for i in range(len(tsegs)-5):
    for j in range(runs):
        AvgSRT[i, j] = sum(runDifs[:, i, j]/trials)
    plt.plot(AvgSRT[i, :])

plt.title('Single Run Times versus Runs For Various Segment Lengths')
plt.legend(tsegs)
plt.xlabel('Run'); plt.ylabel('Time (s)')
filename = 'SML_DMI_SRT'
dir = 'pso\\tests\\seglen\\saved\\' + filename + '.png'
plt.savefig(dir)
plt.close()

bestInd = np.zeros(shape=(len(tsegs), 30))
for i in range(len(tsegs)):
    for j in range(30):
        bestInd[i, j] = np.where(NGB[0, i, j, :]<(NGB[0, i, j, -1]+10))[0][0]

AvgBI = np.zeros(len(tsegs)); BIStnds = np.zeros(len(tsegs))

for i in range(len(tsegs)):
    AvgBI[i] = sum(bestInd[i, :])/30
    BIStnds[i] = statistics.stdev(bestInd[i, :])

# print(BIStnds)
# print(AvgBI)