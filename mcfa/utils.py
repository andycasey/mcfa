
import numpy as np



def generate_data(n_samples=20, n_features=5, n_latent_factors=3, n_components=2,
                  omega_scale=1, noise_scale=1, random_seed=0):

    rng = np.random.RandomState(random_seed)

    A = rng.randn(n_features, n_latent_factors)

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

    psi = rng.gamma(1, scale=noise_scale, size=n_features)

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




def latent_factor_rotation_matrix(A_star, A):
    r"""
    Return a rotation amtrix :math:`\mathbf{R}` that will orient and flip the 
    latent factors `A_star` to be as close as possible to `A` such that

    .. math:

        A \approx A_star @ R


    :param A_star:
        The latent factors to perform the rotation on.

    :param A:
        The latent factors that we seek to approximate.

    :returns:
        A rotation matrix :math:`\mathbf{R}`.
    """
    D, J = A.shape
    D_star, J_star = A_star.shape

    # We need to identify each factor (e.g. compare to closest) and allow for it
    # to be flipped, and off-centered.

    # This requires us to determine best fitting coefficients for each pair-wise
    # comparison, and then decide on a rank ordering.
    I = np.eye(D)

    chi2 = np.inf * np.ones((J, J_star))
    all_params = np.empty((J, J_star, 2))

    for j, A_j in enumerate(A.T):
        for j_star, A_jstar in enumerate(A_star.T):

            DM = np.vstack((np.ones(D), A_jstar)).T
            C = np.linalg.inv(DM.T @ np.linalg.solve(I, DM))
            P = np.atleast_2d(C @ (DM.T @ np.linalg.solve(I, A_j)))

            all_params[j, j_star] = P
            chi2[j, j_star] = np.sum(((P @ DM.T) - A_j)**2)

    # Rank order the matrix.
    order = np.argmin(chi2, axis=1)

    R = np.zeros((J, J))
    R[order, np.arange(J)] = np.sign(np.diag(all_params[:, :, 1][:, order]))

    return R.astype(int)



if __name__ == "__main__":


    import numpy as np

    np.random.seed(42)

    J = 6
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
    print(f"True indices: {indices}")
    print(f"True signs: {signs}")

    R = latent_factor_rotation_matrix(A_star, A)

    A_prime = A_star @ R

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax_before, ax_after = axes

    for j in range(J):

        color = colors[j]

        ax_before.plot(A.T[j], c=color)
        ax_before.plot(A_star.T[j], c=color, alpha=0.5)

        ax_after.plot(A.T[j], c=color)
        ax_after.plot(A_prime.T[j], c=color, alpha=0.5)

    ax_before.set_title("before")
    ax_after.set_title("after")


    raise a


