
import os
import numpy as np
from tqdm import tqdm
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

folder = 'synthetic4'

n = 10000
minimum = -4
maximum = 4

def compute_f(x_1, x_2):
    f1 = 4*np.sin(x_1) + 2*np.sin(2*x_1)
    f2 = 3*np.cos(3*x_2) + 4*np.sin(5*x_2)
    f12 = np.exp(-(x_1 + x_2)**2)
    f = f1 + f2 + f12
    return f / 4

observed = []
real = []
x = make_blobs(n_samples=n, centers=3, cluster_std=0.4, random_state=0)[0]
x_1, x_2 = np.split(x, 2, 1)
f = compute_f(
    x_1, x_2
    )
y = f + np.random.normal(loc=0, scale=2e-1, size=f.shape)

observed = np.concatenate([y.reshape(-1,1), x], -1)
x1 = np.linspace(-4, 4, 100)
x2 = np.linspace(-4, 4, 100)
for x_1 in tqdm(x1, total=len(x1)):
    for x_2 in tqdm(x2, total=len(x1)):
        f = compute_f(
            x_1, x_2
            )
        real.append(
            (
                f,
                x_1,
                x_2,
                )
            )

os.makedirs('./data/{}'.format(folder), exist_ok=True)
real = np.asarray(real)
plt.plot(real[:,1],real[:,0], c='black')
plt.scatter(x=observed[:,1], y=observed[:,0])
np.save('./data/{}/observed'.format(folder),observed)
np.save('./data/{}/real'.format(folder),real)
