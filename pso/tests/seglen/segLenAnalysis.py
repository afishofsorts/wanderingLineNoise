import numpy as np
import matplotlib.pyplot as plt

filename = 'ML_data'
dir = 'pso\\tests\\seglen\\saved\\' + filename + '.npy'
newTS, lsf, segTimes = np.load(dir)

trials = len(newTS[:, 0])

OPS = np.zeros(trials)
for i in range(trials):
    OPS = newTS[i, :]*60

for i in range(trials):
    plt.plot(OPS, lsf[i, :])
plt.title('Segment Length LSF')
filename = 'ML_lsf'
dir = 'pso\\tests\\seglen\\saved\\' + filename + '.png'
plt.savefig(dir)
plt.close()

for i in range(trials):
    plt.plot(OPS, segTimes[i, :])
filename = 'ML_times'
dir = 'pso\\tests\\seglen\\saved\\' + filename + '.png'
plt.savefig(dir)
plt.close()