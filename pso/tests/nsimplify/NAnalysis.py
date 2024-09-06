import numpy as np
import matplotlib.pyplot as plt

Ns = np.load('saved\\NSimplify.npy')

wvar = np.arange(0.5, 150.5, 0.5)

for i in range(4):
    Ts = 10**(i-4)
    plt.plot(wvar, Ns[0, i, :])
    plt.plot(wvar, Ns[1, i, :])
    plt.plot(wvar, Ns[2, i, :])

    plt.title('Cos Squared Sum vs Varied Omega Parameters (fs=' + str(round(1/Ts)) + rf'Hz, $\Delta t=10s$)')
    plt.legend(['w1', 'w2', 'w3'])
    plt.xlabel('Omega'); plt.ylabel('Cosine Squared')
    filename = 'NvsOmega'
    plt.savefig('pso\\tests\\nsimplify\\saved\\' + filename + str(Ts) + '.png')
    plt.close()

    colors = ['blue', 'orange', 'green']
    plt.hist([Ns[0, i, :], Ns[1, i, :], Ns[2, i, :]], 10, density=True, histtype='bar', color=colors)
    plt.legend(['w1', 'w2', 'w3'])
    plt.title('Cos Squared Histogram w/ Omega Variation (fs=' + str(round(1/Ts)) + rf'Hz, $\Delta t=10s$)')
    filename = 'NHist'
    plt.savefig('pso\\tests\\nsimplify\\saved\\' + filename + str(Ts) + '.png')
    plt.close()