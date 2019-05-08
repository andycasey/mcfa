
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import yaml
from scipy import linalg

from matplotlib.ticker import MaxNLocator

sys.path.insert(0, "../")

from mcfa import (mcfa, grid_search, mpl_utils, utils)

matplotlib.style.use(mpl_utils.mpl_style)

here = os.path.dirname(os.path.realpath(__file__))  


with open("config.yml") as fp:
    config = yaml.load(fp)


np.random.seed(config.get("random_seed", 0))


def savefig(fig, suffix):
    prefix = os.path.basename(__file__)[:-3]
    filename = os.path.join(here, f"{prefix}-{suffix}")
    fig.savefig(f"{filename}.pdf", dpi=300)
    fig.savefig(f"{filename}.png", dpi=150)
    print(f"Created figures {filename}.png and {filename}.pdf")
    

from astropy.table import Table

mcfa_kwds = dict(init_factors="random", init_components="random", tol=1e-5,
                 max_iter=10000)



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

t = Table.read(os.path.join(here, "../catalogs/barklem_t3.fits"))

elements = np.unique(t["El"])
stars = np.unique(t["Name"])

S, E = (len(stars), len(elements))
X = np.nan * np.ones((S, E))

for i, row in enumerate(t):
    star_index = np.where(stars == row["Name"])[0][0]
    element_index = np.where(elements == row["El"])[0][0]

    X[star_index, element_index] = row["logEpsX"]

N_ok = np.zeros(E)
for i, element in enumerate(elements):
    N_ok[i] = np.sum(np.isfinite(X[:, i]))

element_indices = np.argsort(N_ok)[::-1]
for i in element_indices:
    print(elements[i], N_ok[i])

# Take top 13
use_element_indices = element_indices[:15]
use_elements = elements[use_element_indices]

use_star_indices = np.all(np.isfinite(X[:, use_element_indices]), axis=1)

label_names = list(map(str.strip, use_elements))
ei = np.argsort(np.array([periodic_table.index(el) for el in label_names]))
label_names = [label_names[ei_] for ei_ in ei]

X = X[use_star_indices, :][:, use_element_indices[ei]]

print(f"Elements: {', '.join(label_names)}")
print(f"Stars: {X.shape[0]}")



def convert_xh_to_xy(X_H, label_names, y_label):

    index = label_names.index(y_label)
    y_h = X_H[:, index]

    offsets = np.zeros_like(X_H)
    for i, label_name in enumerate(label_names):
        if label_name == y_label: continue
        offsets[:, i] = y_h

    return X_H - offsets


if config["wrt_x_fe"]:
    X = convert_xh_to_xy(X, label_names, "Fe")
if config["subtract_mean"]:
    X = (X - np.mean(X, axis=0))


# Run a grid search.
max_n_latent_factors = 7
max_n_components = 3

Js = 1 + np.arange(max_n_latent_factors)
Ks = 1 + np.arange(max_n_components)

Jg, Kg, converged, metrics = grid_search.grid_search(Js, Ks, X, N_inits=5,
                                                     mcfa_kwds=mcfa_kwds)

ll = metrics["ll"]
bic = metrics["bic"]
pseudo_bic = metrics["pseudo_bic"]
message_length = metrics["message_length"]

J_best_ll, K_best_ll = grid_search.best(Js, Ks, -ll)
J_best_bic, K_best_bic = grid_search.best(Js, Ks, bic)
J_best_mml, K_best_mml = grid_search.best(Js, Ks, message_length)

print(f"Best log likelihood  at J = {J_best_ll} and K = {K_best_ll}")
print(f"Best BIC value found at J = {J_best_bic} and K = {K_best_bic}")
print(f"Best MML value found at J = {J_best_mml} and K = {K_best_mml}")

# Plot some contours.
plot_filled_contours_kwds = dict(converged=converged,
                                 marker_function=np.nanargmin,
                                 cmap="Spectral_r")
fig_ll = mpl_utils.plot_filled_contours(Jg, Kg, -ll,
                                        colorbar_label=r"$-\log\mathcal{L}$",
                                        **plot_filled_contours_kwds)
savefig(fig_ll, "gridsearch-ll")


fig_ll = mpl_utils.plot_filled_contours(Jg, Kg, message_length,
                                        colorbar_label=r"$\textrm{MML}$",
                                        **plot_filled_contours_kwds)
savefig(fig_ll, "gridsearch-mml")


fig_bic = mpl_utils.plot_filled_contours(Jg, Kg, bic,
                                         colorbar_label=r"$\textrm{BIC}$",
                                         **plot_filled_contours_kwds)
savefig(fig_bic, "gridsearch-bic")


# Re-run model with best J, K.
if config["adopted_metric"].lower() == "bic":
    J_best, K_best = (J_best_bic, K_best_bic)

elif config["adopted_metric"].lower() == "mml":
    J_best, K_best = (J_best_mml, K_best_mml)

elif config["adopted_metric"].lower() == "ll":
    J_best, K_best = (J_best_ll, K_best_ll)

else:
    raise ValueError(f"unknown adopted metric")



model = mcfa.MCFA(n_components=K_best, n_latent_factors=J_best, **mcfa_kwds)
model.fit(X)

A_est = model.theta_[model.parameter_names.index("A")]


grouped_elements = config["grouped_elements"]

A_astrophysical = np.zeros_like(A_est)
for i, tes in enumerate(grouped_elements):
    for j, te in enumerate(tes):
        #idx = label_names.index("{0}_h".format(te.lower()))
        if te.title() not in label_names:
            print(f"ignoring {te}")
            continue

        idx = label_names.index(te.title())
        A_astrophysical[idx, i] = 1.0

A_astrophysical /= np.sqrt(np.sum(A_astrophysical, axis=0))

R, p_opt, cov, *_ = utils.find_rotation_matrix(A_astrophysical, A_est, 
                                               full_output=True)

R_opt = utils.exact_rotation_matrix(A_astrophysical, A_est, 
                                    p0=np.random.uniform(-np.pi, np.pi, model.n_latent_factors**2))

chi1 = np.sum(np.abs(A_est @ R - A_astrophysical))
chi2 = np.sum(np.abs(A_est @ R_opt - A_astrophysical))

R = R_opt if chi2 < chi1 else R

# Now make it a valid rotation matrix.
model.rotate(R, X=X, ensure_valid_rotation=True)
J = model.n_latent_factors
L = model.theta_[model.parameter_names.index("A")]
cmap = mpl_utils.discrete_cmap(2 + J, base_cmap="Spectral_r")
colors = [cmap(1 + j) for j in range(J)]

fig = mpl_utils.visualize_factor_loads(L, label_names, colors=colors)
savefig(fig, "latent-factors-visualize")
fig = mpl_utils.visualize_factor_loads(L, label_names, colors=colors, absolute_only=True)
savefig(fig, "latent-factors-visualize-abs")



# Plot the specific variances.
xticklabels = [r"$\textrm{{{0}}}$".format(ln.split("_")[0].title()) for ln in label_names]
fig_scatter = mpl_utils.plot_specific_scatter(model, 
                                              steps=True,
                                              xlabel="", 
                                              xticklabels=xticklabels,
                                              ylabel=r"$\textrm{specific scatter / dex}$")
#fig_scatter.axes[0].set_yticks(np.linspace(0, 0.25, 6))
savefig(fig_scatter, "specific-scatter")



# Plot a visualisation of latent factors.
#A_est
# Get fractions of absolute values per element.
f_A = np.abs(A_est)/np.atleast_2d(np.sum(np.abs(A_est), axis=1)).T
indices = np.argsort(1.0/f_A, axis=1)

#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cmap = mpl_utils.discrete_cmap(2 + J, base_cmap="Spectral_r")
#cmap = matplotlib.cm.Set2
colors = [cmap(1 + j) for j in range(J)]


fig, ax = plt.subplots(figsize=(3.19, 6))

E, J = f_A.shape

for e, (f_A_, indices_) in enumerate(zip(f_A, indices)):

    left = 0
    for i in indices_:
        ax.barh(E - e, f_A_[i], left=left, facecolor=colors[i])
        left += f_A_[i]

    ax.barh(E - e, 1, left=0, edgecolor='#000000', zorder=-1, linewidth=2)

vals = np.linspace(0, 1, 2 * J + 1)
for j in range(J):

    left = vals[2*j + 0] + 0.25/J
    #w = vals[2*j + 1]

    ax.barh(E + 1.5, 0.5/J, left=left, facecolor=colors[j], edgecolor="#000000")
    ax.text(left + 0.5 * 0.5/J, E + 1.5, r"$\mathbf{{L}}_{{{0}}}$".format(j),
            zorder=100, verticalalignment="center",
            horizontalalignment="center")

ax.set_yticks(np.arange(1, 1 + E))
ax.set_yticklabels([r"$\textrm{{{0}}}$".format(ea) for ea in label_names[::-1]])
ax.set_ylim(0.5, E + 2)
ax.set_xlim(-0.01, 1.01)
ax.set_xticks([])
ax.yaxis.set_tick_params(width=0)

ax.set_frame_on(False)

ax.set_xlabel(r"${|\mathbf{L}_\textrm{d}|} / {\sum_{j}|\mathbf{L}_\textrm{j,d}|}$")

fig.tight_layout()

savefig(fig, "latent-contributions")


