
import numpy as np
from sklearn.datasets import make_blobs


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

    noise = psi * rng.randn(n_samples, n_features)

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

