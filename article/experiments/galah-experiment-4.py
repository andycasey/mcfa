


import sys
import os
import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, "../../")
from astropy.table import Table



galah = Table.read("../../catalogs/GALAH_DR2.1_catalog.fits")


all_element_label_names = [ln for ln in galah.dtype.names \
    if ln.endswith(("_fe", "fe_h")) and not ln.startswith(("flag_", "e_", "alpha_"))]



def savefig(fig, suffix):
    prefix = os.path.basename(__file__)[:-3]
    filename = f"{prefix}-{suffix}.png"
    fig.savefig(filename, dpi=150)


elements = [
    "fe",
    "eu",
    "ba", "y",
    "mg", "ca",
    "na",
    "ni", "cr",
]


# Select a subset of stars and abundances
common_mask = (galah["flag_cannon"] == 0)
for element in elements:
    if element != "fe":
        common_mask *= (galah[f"flag_{element}_fe"] == 0)


# order by atomic number.

periodic_table = """H                                                  He
                    Li Be                               B  C  N  O  F  Ne
                    Na Mg                               Al Si P  S  Cl Ar
                    K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
                    Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe
                    Cs Ba Lu Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
                    Fr Ra Lr Rf"""

lanthanoids    =   "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb"
actinoids      =   "Ac Th Pa U  Np Pu Am Cm Bk Cf Es Fm Md No"

periodic_table = periodic_table.replace(" Ba ", " Ba " + lanthanoids + " ") \
                               .replace(" Ra ", " Ra " + actinoids + " ") \
                               .split()



idxs = [periodic_table.index(k.split("_")[0].title()) for k in elements]
sorted_elements = [elements[idx] for idx in np.argsort(idxs)]

label_names = []
for sorted_element in sorted_elements:
    if sorted_element == "fe":
        label_names.append("fe_h")
    else:
        label_names.append(f"{sorted_element}_fe")


X = np.array([galah[ln][common_mask] for ln  in label_names]).T

# make all relative to  x/h
X_H = X.copy()

feh_index = label_names.index("fe_h")
for i, label_name in enumerate(label_names):
    if i == feh_index: continue

    X_H[:, i] = X[:, i] + X[:, feh_index]

label_names_wrt_h = ["{0}_h".format(ln.split("_")[0]) for ln in label_names]

assert len(label_names_wrt_h) == X_H.shape[1]

output_path = __file__.replace(".py", "")
with open(f"{output_path}.pkl", "wb") as fp:
    pickle.dump((X_H, label_names_wrt_h, common_mask), fp)


# Run MCFA

def init_astrophysical_factor_loads(X, J):
    A = mcfa._initial_factor_loads_by_noise(X, J)
    A[-1, 0] = 1 # r-process
    A[-3:-1, 1] = 1 # s-process
    A[3:6, 2] = 1 # fe-peak
    A[:2, 3] = 1 # alpha-process

    return A

from mcfa import mcfa, utils


np.random.seed(0)

output_path = "galah-experiment-4-logL-data.pkl"

if os.path.exists(output_path):
    with open(output_path, "rb") as fp:
        log_L, init_component_descriptions, init_factor_descriptions, mcfa_kwds = pickle.load(fp)
        C, F, N_monte_carlo_draws = log_L.shape

else:

    N_monte_carlo_draws = 30
    init_component_descriptions = ("random", "kmeans++")
    init_component_functions = ("random", "kmeans++")

    init_factor_descriptions = ("random", "noise", "svd", "astrophysical")
    init_factor_functions = ("random", "noise", "svd", init_astrophysical_factor_loads)

    C = len(init_component_functions)
    F = len(init_factor_functions)

    mcfa_kwds = dict(n_components=4, n_latent_factors=4, tol=1e-8)

    log_L = np.zeros((C, F, N_monte_carlo_draws))

    for c, init_component in enumerate(init_component_functions):

        for f, init_factor in enumerate(init_factor_functions):

            kwds = mcfa_kwds.copy()
            kwds.update(init_component=init_component, init_factor=init_factor)

            for i in range(N_monte_carlo_draws):

                model = mcfa.MCFA(**kwds)
                model.fit(utils.whiten(X))

                log_L[c, f, i] = model.log_likelihood_

                print(c, f, i, log_L[c, f, i])


    serialised = (log_L, init_component_descriptions, init_factor_descriptions, mcfa_kwds)

    with open(output_path, "wb") as fp:
        pickle.dump(serialised, fp)


latex_labels = {
    "random": r"$\textrm{random}$",
    "noise": r"$\textrm{noise}$",
    "kmeans++": r"$\textrm{k-means++}$",
    "svd": r"$\textrm{SVD}$",
    "astrophysical": r"$\textrm{astrophysical}$"
}


N_bins = 10
bins = np.linspace(log_L.min(), log_L.max(), 1 + N_bins)

fig, axes = plt.subplots(F, C)
axes = np.atleast_2d(axes).T

for c, init_component_description in enumerate(init_component_descriptions):
    for f, init_factor_description in enumerate(init_factor_descriptions):

        ax = axes[c, f]
        ax.hist(log_L[c, f], bins=bins)

        ax.xaxis.set_major_locator(MaxNLocator(3))

        ax.set_yticks([])

        if ax.is_first_row():
            ax.set_title(latex_labels.get(init_component_description, init_component_description))

        if ax.is_first_col():
            ax.set_ylabel(latex_labels.get(init_factor_description, init_factor_description))

        if not ax.is_last_row():
            ax.set_xticks([])


fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)

savefig(fig, "log-L-hist")



raise a





N, D = X.shape
J, K = (4, 4)

mcfa_kwds = dict(init_method="random",
                 n_components=K, n_latent_factors=J, tol=1e-8)

model = mcfa.MCFA(**mcfa_kwds)

A = mcfa._initial_factor_loads_by_noise(X, J)
A[-1, 0] = 1 # r-process
A[-3:-1, 1] = 1 # s-process
A[3:6, 2] = 1 # fe-peak
A[:2, 3] = 1 # alpha-process

pi, xi, omega = mcfa._initial_components_by_kmeans_pp(utils.whiten(X), A, J)
psi = np.ones(D)
theta = (pi, A, xi, omega, psi)


model.fit(utils.whiten(X), init_params=theta)

model_b = mcfa.MCFA(**mcfa_kwds)
model_b.fit(utils.whiten(X))


A_fit = model.theta_[1]
D, J = A_fit.shape

fig, ax = plt.subplots()

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for j in range(J):
    ax.plot(A_fit.T[j], c=colors[j])
    ax.plot(A.T[j], alpha=0.5, c=colors[j])

ax.set_xticks(np.arange(D))
ax.set_xticklabels(label_names_wrt_h)

