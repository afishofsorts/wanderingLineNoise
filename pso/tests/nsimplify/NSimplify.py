import numpy as np
import matplotlib.pyplot as plt

def NCalc(t, w1, w2, w3):
    N = sum(np.cos(w1*t + w2*t**2 + w3*t**3)**2)
    return N

for j in range(3):
    tstep = 10**(j-3)
    # w1 variation
    w2 = 0; w3 = 0
    t = np.arange(0, 10, tstep)

    N1 = np.zeros(300); w1 = np.zeros(300)
    for i in range(300):
        w1[i] = i/2 + 1/2
        N1[i] = NCalc(t, w1[i], w2, w3)

    plt.plot(w1, N1)

    # w2 variation
    N2 = np.zeros(300); w2 = np.zeros(300); w1 = 0
    for i in range(300):
        w2[i] = i/2 + 1/2
        N2[i] = NCalc(t, w1, w2[i], w3)

    plt.plot(w2, N2)

    # w3 variation
    N3 = np.zeros(300); w3 = np.zeros(300); w2 = 0
    for i in range(300):
        w3[i] = i/2 + 1/2
        N3[i] = NCalc(t, w1, w2, w3[i])

    plt.plot(w3, N3)

    plt.title('Cosine Squared Sum versus Varied Omega Parameters (tstep=' + str(tstep) + ')')
    plt.legend(['w1', 'w2', 'w3'])
    plt.xlabel('Omega'); plt.ylabel('Cosine Squared')
    filename = 'NvsOmega'
    plt.savefig('pso\\tests\\nsimplify\\saved\\' + filename + str(tstep) + '.png')
    plt.close()

    colors = ['blue', 'orange', 'green']
    plt.hist([N1, N2, N3], 10, density=True, histtype='bar', color=colors)
    plt.legend(['w1', 'w2', 'w3'])
    plt.title('Cosine Squared Histogram w/ Omega Variation (tstep=' + str(tstep) + ')')
    filename = 'NHist'
    plt.savefig('pso\\tests\\nsimplify\\saved\\' + filename + str(tstep) + '.png')
    plt.close()



