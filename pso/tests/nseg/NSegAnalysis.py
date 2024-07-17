import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

filename = 'Data_NSVar.npy'
root = 'pso\\tests\\nseg\\saved\\'
dir = root + filename + '.npy'
extOPS, extNR, MRlsf, MRtimes = np.load(dir)

trials = len(MRtimes[:, 0]); segTests = np.where(extNR[0, :] == extNR[0, -1])[0][0] + 1

medLSF = np.zeros(segTests); medMRT = np.zeros(segTests)
for k in range(segTests):
    medLSF[k] = stat.median(MRlsf[:, k])
    medMRT[k] = stat.median(MRtimes[:, k])

# plt.plot(extOPS[0, :], medLSF)
for i in range(trials):
    plt.plot(extOPS[0, :], MRlsf[i, :])
plt.title('Median LSF versus Osc. Per. Segment over 10 Trials')
plt.xlabel('Oscillations per Segment'); plt.ylabel('LSF')
plt.savefig(root + 'MR_LSF_OPS.png')
plt.close()

for i in range(trials):
    plt.plot(extNR[0, :], MRtimes[i, :])
plt.plot(extNR[0, :], medMRT)
plt.title('Median MR Time versus Number of Segments over 10 Trials')
plt.xlabel('Number of Segments'); plt.ylabel('Multirun Time (s)')
plt.savefig(root + 'MR_T_OPS.png')
plt.close()