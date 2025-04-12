
import os
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('ggplot')
# sns.set_palette("Dark2")
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({
    'font.size': 8,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['axes.edgecolor'] = 'gray'

folder = 'synthetic2'

num_var = 2
n = 10000
minimum = -4
maximum = 4

def compute_f(x_1):
    f = np.sin(x_1)**2 + np.cos(x_1)**2 + np.sin(3*x_1) + np.cos(
        5*x_1) + np.sqrt(
        abs(x_1)) / 2
    return f

observed = []
real = []
for i in range(n):
    x_1 = np.random.uniform(-4, 4)
    f = compute_f(
        x_1
        )
    y = f + np.sin(2*np.pi*f) * np.random.normal(
        loc=0, scale=3e-1, size=f.shape
        )
    observed.append(
        (
            y,
            x_1,
            )
        )

observed = np.asarray(observed)
x1 = np.linspace(-4, 4, 100)
for x_1 in tqdm(x1, total=len(x1)):
    f = compute_f(
        x_1
        )
    real.append(
        (
            f,
            x_1,
            )
        )

os.makedirs('./data/{}'.format(folder), exist_ok=True)
real = np.asarray(real)
plt.plot(real[:,1],real[:,0], c='black')
plt.scatter(
    x=observed[:,1],
    y=observed[:,0],
    alpha=0.4,
    c='r',
    edgecolors='black',
    marker='+'
    )
np.save('./data/{}/observed'.format(folder),observed)
np.save('./data/{}/real'.format(folder),real)