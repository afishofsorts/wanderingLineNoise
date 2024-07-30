import numpy as np
import matplotlib.pyplot as plt

filename = 'MR_BFS'
dir = 'pso\\tests\\multirun\\saved\\' + filename + '.npy'
BFS = np.load(dir)

plt.plot(BFS[0, :])
plt.show()