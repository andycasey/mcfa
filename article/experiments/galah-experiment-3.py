
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

from matplotlib.ticker import MaxNLocator

sys.path.insert(0, "../../")


with open("galah-experiment-1.pkl", "rb") as fp:
    X_H, label_names, mask = pickle.load(fp)


X_Fe = np.copy(X_H)

feh_index = label_names.index("fe_h")
for i, label_name in enumerate(label_names):
    if i == feh_index: continue
    X_Fe[:, i] = X_H[:, i] - X_H[:, feh_index]




from mcfa import mcfa, mpl_utils, utils

matplotlib.style.use(mpl_utils.mpl_style)

J = 5
K = 1

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


whitened_X_H = utils.whiten(X_H)


model_h = mcfa.MCFA(init_method="svd+kmeans++", n_latent_factors=J, n_components=K, tol=1e-8)
model_h.fit(X_H)


"""
model_fe = mcfa.MCFA(n_latent_factors=J, n_components=K, tol=1e-8)
model_fe.fit(X_Fe)

pi, A_h, xi, omega, psi = model_h.theta_
A_fe = model_fe.theta_[1]


R = utils.rotation_matrix(A_fe, A_h)

A_X_FeR = A_fe @ R


D, J = A_fe.shape

# OK, let's look at the factor loads.
xi = np.arange(D)
fig, ax = plt.subplots()

for j in range(J):
    ax.plot(xi, A_h.T[j], "-", c=colors[j])
    ax.plot(xi, A_X_FeR.T[j], "-", c=colors[j], alpha=0.5)

raise a
"""

A = model_h.theta_[1]

N, D = X_H.shape
xi = np.arange(D)
fig, ax = plt.subplots()

for j in range(J):
    ax.plot(xi, A.T[j], "-", c=colors[j])



elements = [ea.split("_")[0].title() for ea in label_names]
latex_elements = [r"$\textrm{{{0}}}$".format(el) for el in elements]

ax.set_xticks(xi)
ax.set_xticklabels(latex_elements)

ax.axhline(0, -0.5, D + 0.5, linestyle=":", linewidth=0.5, c="#000000", zorder=-1)

ylim = np.max(np.abs(ax.get_ylim()))
ax.set_ylim(-ylim, +ylim)

ax.yaxis.set_major_locator(MaxNLocator(3))
ax.set_ylabel(r"$\mathbf{A}$")

fig.tight_layout()