
import numpy as np


def whiten(X, axis=0):
    return (X - np.mean(X, axis=axis))/np.std(X, axis=axis)



def generate_data_from_leuton_drton():

    A = np.array([
        [ 0.97,  0.00,  0.00],
        [ 0.04,  0.90,  0.00],
        [ 1.00, -1.12,  0.57],
        [ 2.03,  0.42,  0.57],
        [ 0.31,  0.47,  0.09],
        [ 0.43, -0.21, -0.35],
        [ 0.75,  0.31,  0.68],
        [ 0.45, -0.48, -1.50],
        [-2.21,  1.45,  0.38],
        [ 1.98, -0.30,  0.96],
        [-2.63,  0.41,  1.09],
        [-0.72,  1.39,  0.97],
        [-0.88,  2.01, -0.39],
        [-0.53,  0.04,  0.59],
        [-0.95,  1.39,  0.37]
    ])

    raise NotImplementedError

def generate_data(n_samples=20, n_features=5, n_latent_factors=3, n_components=2,
                  omega_scale=1, noise_scale=1, latent_scale=1, random_seed=0):

    rng = np.random.RandomState(random_seed)

    #A = rng.randn(n_features, n_latent_factors)

    sigma_L = np.abs(rng.normal(0, latent_scale))
    choose = lambda x, y: int(np.math.factorial(x) \
                        / (np.math.factorial(y) * np.math.factorial(x - y)))

    M = n_latent_factors * (n_features - n_latent_factors) \
      + choose(n_latent_factors, 2)

    beta_lower_triangular = rng.normal(0, sigma_L, size=M)
    beta_diag = np.abs(rng.normal(0, latent_scale, size=n_latent_factors))

    A = np.zeros((n_features, n_latent_factors), dtype=float)
    A[np.tril_indices(n_features, -1, n_latent_factors)] = beta_lower_triangular
    A[np.diag_indices(n_latent_factors)] = np.sort(beta_diag)[::-1]


    # latent variables
    pvals = np.ones(n_components) / n_components
    R = np.argmax(rng.multinomial(1, pvals, size=n_samples), axis=1)
    pi = np.array([np.sum(R == i) for i in range(n_components)])/n_samples

    xi = rng.randn(n_latent_factors, n_components)
    omega = np.zeros((n_latent_factors, n_latent_factors, n_components))
    for i in range(n_components):
        omega[(*np.diag_indices(n_latent_factors), i)] = \
            rng.gamma(1, scale=omega_scale, size=n_latent_factors)**2

    scores = np.empty((n_samples, n_latent_factors))
    for i in range(n_components):
        match = (R == i)
        scores[match] = rng.multivariate_normal(xi.T[i], omega.T[i], 
                                                size=sum(match))

    #psi = rng.gamma(1, scale=noise_scale, size=n_features)
    psi = np.abs(rng.normal(0, noise_scale, n_features))
    
    noise = np.sqrt(psi) * rng.randn(n_samples, n_features)

    X = scores @ A.T + noise

    truth = dict(A=A, pi=pi, xi=xi, omega=omega, psi=psi,
                 noise=noise, R=R, scores=scores)

    return (X, truth)


def parameter_vector(pi, A, xi, omega, psi, **kwargs):
    return [np.array(_) for _ in (pi, A, xi, omega, psi)]



def simulate_example_data(N, D, J, K=1, seed=None, full_output=False, **kwargs):
    """
    Simulate data that have common latent factors and clustering in the latent
    space.

    :param N:
        The number of samples (data points) to simulate.

    :param D:
        The dimensionality (or number of features) of the data.

    :param J:
        The number of common latent factors.

    :param K:
        The number of clusters to simulate.

    :param seed: [optional]
        A random seed.
    """

    assert D == 10 and J == 3, \
        "Sorry these are fixed until we generate latent factors randomly"
    if seed is not None:
        np.random.seed(seed)

    mu_theta = np.zeros(J)
    mu_epsilon = np.zeros(D)

    phi = np.eye(J)
    psi_scale = kwargs.get("__psi_scale", 1)
    psi = np.diag(np.abs(np.random.normal(0, 1, size=D))) * psi_scale

    # TODO: generate latent factors randomly... but keep near orthogonality
    L = np.array([
        [0.99, 0.00, 0.25, 0.00, 0.80, 0.00, 0.50, 0.00,  0.00,  0.00],
        [0.00, 0.90, 0.25, 0.40, 0.00, 0.50, 0.00, 0.00, -0.30, -0.30],
        [0.00, 0.00, 0.85, 0.80, 0.00, 0.75, 0.75, 0.00,  0.80,  0.80]
    ])

    truths = dict(N=N, D=D, J=J, K=K, seed=seed)

    if K == 1:
        theta = np.random.multivariate_normal(mu_theta, phi, size=N)
        responsibility = np.ones(N)

    else:
        # Calculate number of members per cluster.
        p = np.abs(np.random.normal(0, 1, K))
        responsibility = np.random.choice(np.arange(K), N, p=p/p.sum())

        scale = kwargs.get("__cluster_mu_theta_scale", 1)
        cluster_mu_theta = np.random.multivariate_normal(
            np.zeros(J), scale * np.eye(J), size=K)

        #S = 1 if kwargs.get("__cluster_common_scale", True) else J
        #cluster_mu_sigma = np.abs(np.random.multivariate_normal(
        #    np.zeros(J), 
        #    cluster_scale * np.random.normal(0, 1, size=S) * np.eye(J), 
        #    size=K))

        scale = kwargs.get("__cluster_sigma_theta_scale", 1)
        cluster_sigma_theta = scale * np.abs(np.random.normal(0, 1, size=K))

        truths.update(dict(cluster_mu_theta=cluster_mu_theta,
                           cluster_sigma_theta=cluster_sigma_theta))

        theta = np.zeros((N, J), dtype=float)
        for k, (mu, cov) in enumerate(zip(cluster_mu_theta, cluster_sigma_theta)):
            members = (responsibility == k)
            theta[members] = np.random.multivariate_normal(mu, np.eye(J) * cov, 
                                                           size=sum(members))
        

    epsilon = np.random.multivariate_normal(mu_epsilon, psi, size=N)

    X = np.dot(theta, L) + epsilon
    truths.update(dict(L=L, psi=psi, epsilon=epsilon, theta=theta, 
                       responsibility=responsibility))

    return (X, truths) if full_output else X




def rotation_matrix(A_prime, A):
    r"""
    Return a rotation amtrix :math:`\mathbf{R}` that will orient and flip the 
    latent factors `A_prime` to be as close as possible to `A` such that

    .. math:

        A \approx A_{prime} R

    :param A_prime:
        The latent factors to perform the rotation on.

    :param A:
        The latent factors that we seek to approximate.

    :returns:
        A rotation matrix :math:`\mathbf{R}` that can be used as `A_prime @ R`
        to approximate `A`.
    """

    D, J = A.shape
    D_prime, J_prime = A_prime.shape

    if A.shape != A_prime.shape:
        raise ValueError("A_prime and A must have the same shape")

    # We need to identify each factor (e.g. compare to closest) and allow for it
    # to be flipped, and off-centered.
    I = np.eye(D)

    chi2 = np.inf * np.ones((J, J_prime))
    all_params = np.empty((J, J_prime, 2))

    for j, A_j in enumerate(A.T):
        for j_prime, A_jstar in enumerate(A_prime.T):

            DM = np.vstack((np.ones(D), A_jstar)).T
            C = np.linalg.inv(DM.T @ np.linalg.solve(I, DM))
            P = np.atleast_2d(C @ (DM.T @ np.linalg.solve(I, A_j)))

            all_params[j, j_prime] = P
            chi2[j, j_prime] = np.sum(((P @ DM.T) - A_j)**2)

    # Rank order the matrix.
    R = np.zeros((J, J), dtype=int)

    # Here we perform the assignments based on the best chi-sq achievable for
    # each factor, so that if one factor is very similar to two others, we do
    # not end up with one factor being assigned to two others, and one factor
    # being zero-d out  entirely.
    order_assignments = np.argsort(np.min(chi2, axis=1))

    order = np.zeros(J, dtype=int)
    for i, order_assignment in enumerate(order_assignments):

        # Check the index for the best chi-sq for this factor, and only adopt it
        # if that index has not been assigned yet.
        for idx in np.argsort(chi2[order_assignment]):
            if idx not in order[order_assignments[:i]]:
                break

        order[order_assignment] = idx

    # If we didn't have to worry about multiple assignments to different factors
    # then we could just do:
    #order = np.argmin(chi2, axis=1)

    R[order, np.arange(J)] = np.sign(np.diag(all_params[:, :, 1][:, order]))

    # Check that we have only made one re-assignment per factor!
    assert np.alltrue(np.sum(R != 0, axis=1) == np.ones(J, dtype=int))

    return R



if __name__ == "__main__":


    import numpy as np

    np.random.seed(42)

    J = 10
    D = 15

    A = np.random.uniform(-1, 1, size=(D, J))

    # Randomly flip them.
    signs = np.sign(np.random.uniform(-1, 1, size=J))

    # Add a little noise to the signs
    signs = np.random.normal(signs, 0.05 * np.ones(J))


    # Add a little bias.
    bias = np.random.normal(0, 0.05 * np.ones(J))

    A_star = signs * A + bias

    # Re-order them.
    indices = np.random.choice(J, J, replace=False)
    A_star = A_star[:, indices]

    R = rotation_matrix(A_star, A)

    A_prime = A_star @ R

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax_before, ax_after = axes[0]
    ax_before_diff, ax_after_diff = axes[1]

    for j in range(J):

        color = colors[j]

        ax_before.plot(A.T[j], c=color)
        ax_before.plot(A_star.T[j], c=color, alpha=0.5)

        ax_after.plot(A.T[j], c=color)
        ax_after.plot(A_prime.T[j], c=color, alpha=0.5)

        before_diff = A.T[j] - A_star.T[j]
        after_diff = A.T[j] - A_prime.T[j]

        ax_before_diff.plot(before_diff - np.mean(before_diff), c=color)
        ax_after_diff.plot(after_diff - np.mean(after_diff), c=color)


    ax_before.set_title("before")
    ax_after.set_title("after")

    for ax in (ax_before, ax_after):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhline(0, -1, D + 1, c="k", linestyle=":", linewidth=0.5, zorder=-1)

        ax.set_xlim(0 - 0.5, D - 0.5)

        abs_ylim = np.max(np.abs(ax.get_ylim()))
        ax.set_ylim(-abs_ylim, +abs_ylim)

    abs_ylim = np.max([np.max(np.abs(ax.get_ylim())) for ax in (ax_before_diff, ax_after_diff)])

    for ax in (ax_before_diff, ax_after_diff):

        ax.set_xlim

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhline(0, -1, D + 1, c="k", linestyle=":", linewidth=0.5, zorder=-1)

        ax.set_xlim(0 - 0.5, D - 0.5)

        ax.set_ylim(-abs_ylim, +abs_ylim)

    ax_before.set_ylabel(r"$A$")
    ax_before_diff.set_ylabel(r"$\Delta{A} - \langle\Delta{A}\rangle$")

    fig.tight_layout()

    fig.savefig("../rotate_factors.png", dpi=150)


