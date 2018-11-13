
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

from matplotlib.ticker import MaxNLocator

sys.path.insert(0, "../../")


with open("galah-experiment-1.pkl", "rb") as fp:
    X, label_names, mask = pickle.load(fp)


print(f"Labels {len(label_names)}: {label_names}")
from mcfa import mcfa, mpl_utils, utils

matplotlib.style.use(mpl_utils.mpl_style)


elements = [ea.split("_")[0].title() for ea in label_names]
latex_elements = [r"$\textrm{{{0}}}$".format(el) for el in elements]


fig, axes = plt.subplots(7, 2)
axes = np.array(axes).flatten()
idx = 0
feh_index = label_names.index("fe_h")
for i, label_name in enumerate(label_names):
    if label_name == "fe_h": continue

    ax = axes[idx]
    ax.scatter(X.T[feh_index], X.T[i], s=1)
    ax.set_ylabel(latex_elements[i])

    idx += 1

for ax in axes[i:]:
    ax.set_visible(False)

fig.tight_layout()



mcfa_kwds = dict(tol=1e-5, init_method="random")

get_output_path = lambda J, K: f"2/galah-experiment-1.{J:.0f}.{K:.0f}.pkl"

def savefig(fig, suffix):
    prefix = os.path.basename(__file__)[:-3]
    filename = f"{prefix}-{suffix}.png"
    fig.savefig(filename, dpi=150)


X_whitened = utils.whiten(X)

# Run grid search.
OVERWRITE = False
MAX_NUM_LATENT_FACTORS = 12
MAX_NUM_COMPONENTS = 40
PLOT_ONLY = True

if not PLOT_ONLY:

    for n_components in range(1, 1 + MAX_NUM_COMPONENTS):
        for n_latent_factors in range(1, 1 + MAX_NUM_LATENT_FACTORS):

            print(f"K = {n_components} (of {MAX_NUM_COMPONENTS}); J = {n_latent_factors} (of {MAX_NUM_LATENT_FACTORS})")

            output_path = get_output_path(n_latent_factors, n_components)

            kwds = mcfa_kwds.copy()

            if os.path.exists(output_path):
                with open(output_path, "rb") as fp:
                    model = pickle.load(fp)

                if (model.n_iter_ + 1) < model.max_iter and not OVERWRITE:
                    print(f"Skipping because {output_path} exists")
                    continue

                else:
                    kwds.update(max_iter=model.max_iter * 10)

            else:
                model = mcfa.MCFA(n_components, n_latent_factors, **mcfa_kwds)


            try:
                model.fit(X_whitened)

            except:
                print("Failed")
                raise
                continue

            else:

                print(f"log-likelihood: {model.log_likelihood_}")

                with open(output_path, "wb") as fp:
                    pickle.dump(model, fp)


# Make a contour plot of the log-likelihood.
Js = np.arange(1, 1 + MAX_NUM_LATENT_FACTORS)
Ks = np.arange(1, 1 + MAX_NUM_COMPONENTS)

Jm, Km = np.meshgrid(Js, Ks)

LL = np.nan * np.ones_like(Jm)
BIC = np.nan * np.ones_like(Jm)
pseudo_BIC = np.nan * np.ones_like(Jm)
pseudo_BIC_kwds = dict(omega=1, gamma=0.1)

converged = np.zeros(Jm.shape, dtype=bool)

n_iters = []

for j, J in enumerate(Js):
    for k, K in enumerate(Ks):

        output_path = get_output_path(J, K)

        if not os.path.exists(output_path):
            continue

        with open(output_path, "rb") as fp:
            model = pickle.load(fp)

            n_iters.append(model.n_iter_)

        LL[k, j] = model.log_likelihood_
        BIC[k, j] = model.bic(X_whitened)
        pseudo_BIC[k, j] = model.pseudo_bic(X_whitened, **pseudo_BIC_kwds)

        # Earlier versions of the code had a -1 index bug for n_iter_
        converged[k, j] = (model.max_iter > (1 + model.n_iter_))

print("GS-2: {0:.0f} {1:.0f} {2:.0f}".format(
    np.median(n_iters), np.std(n_iters), np.sum(n_iters)))

# Make a contour plot of the BIC.
N_contours = 100

fig, axes = plt.subplots(2, 1, figsize=(4.5, 9))

ax = axes[0]

cf = ax.contourf(Jm, Km, LL, N_contours)
ax.set_xlabel(r"$\textrm{Number of latent factors } J$")
ax.set_ylabel(r"$\textrm{Number of clusters } K$")

cbar = plt.colorbar(cf, ax=(ax, ))
cbar.set_label(r"$\log\mathcal{L}$")

ax = axes[1]
cf = ax.contourf(Jm, Km, BIC, N_contours)
ax.set_xlabel(r"$\textrm{Number of latent factors } J$")
ax.set_ylabel(r"$\textrm{Number of clusters } K$")

cbar = plt.colorbar(cf, ax=(ax, ))
cbar.set_label(r"$\textrm{BIC}$")

for ax in axes:
    ax.scatter(Jm[~converged], Km[~converged], marker="x", c="#000000",
             s=10, linewidth=1,
               alpha=0.3, zorder=10)

idx = np.nanargmin(BIC)
j_min, k_min = Js[idx % BIC.shape[1]], Ks[int(idx / BIC.shape[1])]

for ax in axes:
    ax.scatter([j_min], [k_min],
               facecolor="#ffffff", edgecolor="#000000", linewidth=1.5, s=50,
               zorder=15)

    ax.set_xticks(Js.astype(int))
    ax.set_xlim(Js[0], Js[-1])
    ax.set_ylim(Ks[0], Ks[-1])

print(f"BIC is lowest at J = {j_min} and K = {k_min}")



def plot_contours(J, K, Z, N=100, colorbar_label=None, 
                  converged=None, converged_kwds=None, 
                  marker_function=None, marker_kwds=None, 
                  ax=None, **kwargs):
    
    power = np.min(np.log10(Z).astype(int))

    Z = Z.copy() / (10**power)

    if ax is None:
        w = 0.2 + 4 + 0.1
        h = 0.5 + 4 + 0.1
        if colorbar_label is not None:
            w += 1
        fig, ax = plt.subplots(figsize=(w, h))
    else:
        fig = ax.figure

    cf = ax.contourf(J, K, Z, N, **kwargs)

    ax.set_xlabel(r"$\textrm{Number of latent factors } J$")
    ax.set_ylabel(r"$\textrm{Number of clusters } K$")

    if converged is not None:
        kwds = dict(marker="x", c="#000000", s=10, linewidth=1, alpha=0.3)
        if converged_kwds is not None:
            kwds.update(converged_kwds)

        ax.scatter(J[~converged], K[~converged], **kwds)

    if marker_function is not None:
        idx = marker_function(Z)
        j_m, k_m = (J[0][idx % Z.shape[1]], K.T[0][int(idx / Z.shape[1])])
        kwds = dict(facecolor="#ffffff", edgecolor="#000000", linewidth=1.5,
                    s=50, zorder=15)
        if marker_kwds is not None:
            kwds.update(marker_kwds)

        ax.scatter(j_m, k_m, **kwds)

    if colorbar_label is not None:
        cbar = plt.colorbar(cf)
        cbar.set_label(colorbar_label + " $/\,\,10^{0}$".format(power))
        cbar.ax.yaxis.set_major_locator(MaxNLocator(5))

    edge_percent = 0.025
    x_range = np.ptp(J)
    y_range = np.ptp(K)
    ax.set_xlim(J.min() - x_range * edge_percent,
                J.max() + x_range * edge_percent)

    ax.set_ylim(K.min() - y_range * edge_percent,
                K.max() + y_range * edge_percent)
    
    #ax.set_xlim(J.min() - 0.25, J.max() + 0.25)
    #ax.set_ylim(K.min() - 0.25, K.max() + 0.25)
    ax.xaxis.set_major_locator(MaxNLocator(9))
    ax.yaxis.set_major_locator(MaxNLocator(9))

    ax.set_xticks(Js.astype(int))
    ax.yaxis.set_tick_params(width=0)
    ax.xaxis.set_tick_params(width=0)

    
    fig.tight_layout()

    return fig

kwds = dict(converged=converged)
fig_ll = plot_contours(Jm, Km, -LL,
                       marker_function=lambda *_: np.nanargmin(-LL),
                       colorbar_label=r"$-\log\mathcal{L}$", 
                       **kwds)

savefig(fig_ll, "ll-contours")

fig_bic = plot_contours(Jm, Km, BIC,
                        marker_function=np.nanargmin,
                        colorbar_label=r"$\textrm{BIC}$",
                        **kwds)

savefig(fig_bic, "bic-contours")

fig_pseudo_BIC = plot_contours(Jm, Km, pseudo_BIC,
                               marker_function=np.nanargmin,
                               colorbar_label=r"$\textrm{pseudo-BIC}$",
                               **kwds)

savefig(fig_pseudo_BIC, "pseudobic-contours")


# plot other contours
N, D = X.shape
Q = np.log(N) * ((Km - 1) + D + Jm * (D + Km) + (Km * Jm * (Jm + 1))/2 - Jm**2)

fig_scalar = plot_contours(Jm, Km, Q, colorbar_label="scalar")
savefig(fig, "scalar-contours")


# Plot the inferred factor loads as they change throughout the grid.
max_J, max_K = (5, 10)
fls_J = np.arange(1, 1 + max_J).astype(int)
fls_K = np.arange(1, 1 + max_K).astype(int)

fig, axes = plt.subplots(fls_K.size, fls_J.size)#, figsize=(fls_J.size, fls_K.size))

axes = np.atleast_2d(axes)
xi = np.arange(D)
prev_fs = []

for j, J in enumerate(fls_J):

    signs = np.ones((J, D))
    prev_A = np.ones_like(signs)

    for k, K in enumerate(fls_K):

        ax = axes[k, j]
        output_path = get_output_path(J, K)
        with open(output_path, "rb") as fp:
            model = pickle.load(fp)

        pi, A, *_ = model.theta_

        if k == 0:
            signs = np.sign(A)
            prev_A = A.copy()
            prev_fs.append(model.factor_scores(X)[1])


        #fs = model.factor_scores(X)[1]
        R = utils.rotation_matrix(A, prev_A)

        #R = utils.rotation_matrix(fs, prev_fs[0])

        A_rotated = A @ R

        for i in range(J):

            ax.plot(xi, A_rotated.T[i], "-", lw=2)
        
        """
        if i >= 2 and k > 0:

            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']



            R = utils.rotation_matrix(prev_A, A)

            new_A = prev_A @ R

            fig, ax_after = plt.subplots()
            fig, ax_before = plt.subplots()
            for ii in range(1 + i):
                ax_before.plot(A.T[ii], c=colors[ii])
                ax_before.plot(prev_A.T[ii], c=colors[ii], alpha=0.5)

                ax_after.plot(A.T[ii], c=colors[ii])
                ax_after.plot(new_A.T[ii], c=colors[ii], alpha=0.5)

            ax_before.set_title("before")
            ax_after.set_title("after")





            raise a
        """

        ax.set_xticks([])
        ax.set_yticks([])

        ylim = np.max(np.abs(ax.get_ylim()))
        ax.axhline(0, -1, D + 1, linestyle=":", linewidth=1, c="#666666")
        ax.set_ylim(-ylim, ylim)

        ax.set_xlim(-0.5, D + 0.5)

        if ax.is_first_col() and ax.is_first_row():
            ax2 = ax.twiny()
            ax2.set_xticks([])
            ax2.set_xlabel(r"$\textrm{Increasing number of latent factors}$ $J$ $\rightarrow$",
                         horizontalalignment="left", x=0.0)
            ax.set_ylabel(r"$\leftarrow$ $\textrm{Increasing number of clusters}$ $K$",
                          horizontalalignment="right", y=1.0)



fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0, right=0.99, bottom=0.01)

savefig(fig, "factors-wrt-J-and-K")





# Now plot the inferred loads for fixed J with many K.
fixed_J = 5

possible_K = 1 + np.arange(50).astype(int)

fig, ax = plt.subplots()

xi = np.arange(D)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

prev_A = np.ones((fixed_J, D))

for k, K in enumerate(possible_K):

    output_path = get_output_path(fixed_J, K)
    if not os.path.exists(output_path):
        break

    with open(output_path, "rb") as fp:
        model = pickle.load(fp)

    pi, A, *_ = model.theta_
    if k == 0:
        prev_A = A.copy()

    R = utils.rotation_matrix(A, prev_A)
    A_approx = A @ R

    for j in range(fixed_J):

        color = colors[j]

        
        ax.plot(xi, A_approx.T[j], "-", c=color, alpha=0.3)

ax.axhline(0, -1, D + 1, linestyle=":", linewidth=1, zorder=-1)
ylim = np.max(np.abs(ax.get_ylim()))
ax.xaxis.set_tick_params(width=0)
ax.set_xlim(-0.5, D - 0.5)
ax.set_xticks(xi)
ax.set_xticklabels(latex_elements)

ax.set_ylim(-ylim, +ylim)
ax.set_yticks([])
ax.set_ylabel(r"$\textrm{Factor load strength (arb. units)}$")
fig.tight_layout()



# Now show all up until some J value.
up_to_J = 5
possible_K = 1 + np.arange(50).astype(int)


xi = np.arange(D)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


fig, ax = plt.subplots()

for j, J in enumerate(range(1, 1 + up_to_J)):

    prev_A = np.ones((J, D))

    for k, K in enumerate(possible_K):


        output_path = get_output_path(J, K)
        if not os.path.exists(output_path):
            break

        with open(output_path, "rb") as fp:
            model = pickle.load(fp)

        pi, A, *_ = model.theta_
        if k == 0:
            prev_A = A.copy()

        R = utils.rotation_matrix(A, prev_A)
        A_approx = A @ R

        for j in range(J):

            color = colors[j]

            ax.plot(xi, A_approx.T[j], "-", c=color, alpha=0.3)



ax.axhline(0, -1, D + 1, linestyle=":", linewidth=1, zorder=-1)
ylim = np.max(np.abs(ax.get_ylim()))
ax.xaxis.set_tick_params(width=0)
ax.set_xlim(-0.5, D - 0.5)
ax.set_xticks(xi)
ax.set_xticklabels(latex_elements)

ax.set_ylim(-ylim, +ylim)
ax.set_yticks([])
ax.set_ylabel(r"$\textrm{Factor load strength (arb. units)}$")
fig.tight_layout()
