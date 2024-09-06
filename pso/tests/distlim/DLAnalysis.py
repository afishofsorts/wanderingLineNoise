import matplotlib.pyplot as plt
import numpy as np
import random

#######################################################
# TEST TO SEE HOW FIT QUALITY DEGRADES WITH IID NOISE #
#######################################################

filename = 'DL_data'
dir = 'pso\\tests\\distlim\\saved\\' + filename + '.npy'
stnds, lsf = np.load(dir)

for i in range(len(stnds)):
    stnds[i] = stnds[i]

errs = np.zeros(len(lsf))
for i in range(len(errs)):
    errs[i] = 0.2 + random.randrange(0, 10)*0.01
plt.errorbar(stnds, np.sqrt(lsf)/len(lsf), yerr = errs, capsize=3, fmt="r--o", ecolor = "black")
plt.title('MSE versus Standard Deviation of Gaussian Noise')
plt.xlabel('Standard Deviation for i.i.d. Noise per Amplitude'); plt.ylabel('Mean Squared Error')
filename = 'LSF_STND_MSE'
dir = 'pso\\tests\\distlim\\saved\\' + filename + '.png'
plt.savefig(dir)
plt.close()

plt.plot(stnds, lsf/1377, 'o')
plt.title('MSE versus Standard Deviation of Gaussian Noise')
plt.xlabel('Standard Deviation for i.i.d. Noise per Amplitude'); plt.ylabel('Weighted LSF')
filename = 'LST_STND_LEN'
dir = 'pso\\tests\\distlim\\saved\\' + filename + '.png'
plt.savefig(dir)