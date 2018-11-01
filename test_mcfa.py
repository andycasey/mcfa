
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis

from mcfa import mcfa, utils


n_features = 15
n_components = 2
n_latent_factors = 3
n_samples = 10000

omega_scale = 1
noise_scale = 1
random_seed = 42

X, truth = utils.generate_data(n_samples=n_samples,
                               n_features=n_features,
                               n_latent_factors=n_latent_factors,
                               n_components=n_components,
                               omega_scale=omega_scale,
                               noise_scale=noise_scale,
                               random_seed=random_seed)

model = mcfa.MCFA(n_components=n_components, n_latent_factors=n_latent_factors)
model.fit(X)

# Just check that the ML solution is at least as good as the generated value
# (or at least within the convergence tolerance)
diff = model.expectation(X, *utils.parameter_vector(**truth))[0] \
     - model.expectation(X, *model.theta_)[0]

assert diff <= model.tol

pi, A, xi, omega, psi = model.theta_
scores = model.factor_scores(X)[2]


def common_limits(ax):
    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    limits = (np.min(limits), np.max(limits))

    ax.set_xlim(limits)
    ax.set_ylim(limits)


fig, axes = plt.subplots(n_latent_factors)

for i, ax in enumerate(axes):
    ax.plot(A.T[i], c="tab:blue")
    ax.plot(truth["A"].T[i], c="tab:red")


# Generate data.
X2 = scores @ A.T

def corner_scatter(X, label_names=None, show_ticks=False, fig=None, figsize=None,
                   **kwargs):
    """
    Make a corner plot where the data are shown as scatter points in each axes.

    :param X:
        The data, :math:`X`, which is expected to be an array of shape
        [n_samples, n_features].

    :param label_names: [optional]
        The label names to use for each feature.

    :param show_ticks: [optional]
        Show ticks on the axes.

    :param fig: [optional]
        Supply a figure (with [n_features, n_features] axes) to plot the data.

    :param figsize: [optional]
        Specify a size for the figure. This parameter is ignored if a `fig` is
        supplied.

    :returns:
        A figure with a corner plot showing the data.
    """

    N, D = X.shape
    assert N > D, "I don't believe that you have more dimensions than data"
    K = D - 1

    if fig is None:
        if figsize is None:
            figsize = (2 * K, 2 * K)
        fig, axes = plt.subplots(K, K, figsize=figsize)

    axes = np.array(fig.axes).reshape((K, K)).T

    kwds = dict(s=1, c="tab:blue", alpha=0.5)
    kwds.update(kwargs)
    
    for i, x in enumerate(X.T):
        for j, y in enumerate(X.T):
            if j == 0: continue

            try:
                ax = axes[i, j - 1]

            except:
                continue

            if i >= j:
                ax.set_visible(False)
                continue

            ax.scatter(x, y, **kwds)

            if not show_ticks:
                ax.set_xticks([])
                ax.set_yticks([])

            if ax.is_last_row() and label_names is not None:
                ax.set_xlabel(label_names[i])
                
            if ax.is_first_col() and label_names is not None:
                ax.set_ylabel(label_names[j])

    fig.tight_layout()
    
    return fig


# Draw samples.
fig = corner_scatter(X, c="tab:blue", s=1, alpha=0.5, figsize=(8, 8))
corner_scatter(X2, c="tab:red", s=1, alpha=0.25, zorder=10, fig=fig)

#corner_scatter(X2, c="tab:red", s=1, alpha=0.5, figsize=(8, 8))

fig, ax = plt.subplots()
ax.plot(truth["psi"], c="tab:blue")
ax.plot(psi, c="tab:red")
ax.set_ylabel(r"$\psi$")
ax.set_xlabel(r"$\textrm{dimension}$")
ax.set_xticks([])

