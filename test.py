import numpy as np
from sko.PSO import PSO
import matplotlib.pyplot as plt

def demo_func(x):
    x1, x2 = x
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.e


constraint_ueq = (
    lambda x: (x[0] - 1) ** 2 + (x[1] - 0) ** 2 - 0.5 ** 2
    ,
)

max_iter = 40
pso = PSO(func=demo_func, n_dim=2, pop=40, max_iter=max_iter, lb=[-2, -2], ub=[2, 2]
          , constraint_ueq=constraint_ueq)
pso.record_mode = True
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

best = np.zeros(40)
for i in range(40):
    best[i] = pso.gbest_y_hist[i]

plt.plot(best)
plt.title('Global Best PSO Fit Value versus Iterations')
plt.xlabel('PSO Iterations'); plt.ylabel('Global Best Fit')
plt.show()
# %% Now Plot the animation
