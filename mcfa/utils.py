
import numpy as np
from scipy import (linalg, optimize as op, stats)

def whiten(X, axis=0):
    return (X - np.mean(X, axis=axis))/np.std(X, axis=axis)


def generate_data(n_samples, n_features, n_latent_factors, n_components,
                  omega_scale=1, noise_scale=1, random_seed=0, force=None):

    rng = np.random.RandomState(random_seed)

    A = stats.special_ortho_group.rvs(n_features, random_state=rng)
    A = A[:, :n_latent_factors]
    AL = linalg.cholesky(A.T @ A)
    A = A @ linalg.solve(AL, np.eye(n_latent_factors))

    pvals = np.ones(n_components) / n_components
    R = np.argmax(rng.multinomial(1, pvals, size=n_samples), axis=1)
    pi = np.array([np.sum(R == i) for i in range(n_components)])/n_samples

    xi = rng.randn(n_latent_factors, n_components)
    omega = np.zeros((n_latent_factors, n_latent_factors, n_components))

    if force is not None:
        if "xi" in force:
            xi = force["xi"]
            print("using forced xi")
        if "A" in force:
            A = force["A"]
            print("using forced A")

    for i in range(n_components):
        omega[(*np.diag_indices(n_latent_factors), i)] = \
            rng.gamma(1, scale=omega_scale, size=n_latent_factors)**2

    if force is not None:
        if "omega" in force:
            omega = force["omega"]
            print("using forced omega")


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


def old_generate_data(n_samples=20, n_features=5, n_latent_factors=3, n_components=2,
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


def euclidean_2d_rotation_matrix(theta):

    tr = theta * np.pi/180.0

    R = np.array([
        [np.cos(tr), -np.sin(tr)],
        [np.sin(tr), +np.cos(tr)]
    ])

    return R

def generalized_rotation_matrix(psi, theta, phi):

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(psi), -np.sin(psi)],
        [0, np.sin(psi), np.cos(psi)]
    ])

    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])

    return Rz @ (Ry @ Rx)



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



def givens_rotation_matrix(*angles):

    angles = np.atleast_1d(angles)
    D = len(angles)
    R = np.ones((D, D, D))

    for i, theta in enumerate(angles):

        s = np.sin(theta)

        R[i] = np.eye(D)
        R[i, -i, -i] = R[i, -i + 1, -i + 1] = np.cos(theta)
        R[i, -i, -i + 1] = +s
        R[i, -i + 1, -i] = -s

    R = np.linalg.multi_dot(R)
    assert np.allclose(R @ R.T, np.eye(R.shape[0]))
    return R



cost = lambda B, *p: (B @ givens_rotation_matrix(*p)).flatten()


def find_rotation_matrix(A, B, init=None, n_inits=25, full_output=True, **kwargs):
    """
    Find the Euler angles that produce the rotation matrix such that

    .. math:

        A \approx B @ R
    """

    kwds = dict(maxfev=10000)
    kwds.update(kwargs)

    A, B = (np.atleast_2d(A), np.atleast_2d(B))

    D, J = A.shape

    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")

    diff = lambda R: np.abs(A - B @ R)
    L = lambda R: np.sum(diff(R))

    def objective_function(angles):
        return L(givens_rotation_matrix(*angles))

    if init is None:
        inits = np.random.uniform(0, 2 * np.pi, size=(n_inits, J))
        inits[0] = 0

    else:
        inits = np.atleast_2d(inits).reshape((-1, J))

    best_R = None
    best_cost = None
    best_opt = None

    for i, init in enumerate(inits):

        p_opt = op.minimize(objective_function, init, method="BFGS")
        #p_opt = op.basinhopping(objective_function, init, niter=10)
        #p_opt = op.minimize(objective_function, init, method="Nelder-Mead")

        R = givens_rotation_matrix(*p_opt.x)
        cost = L(R)

        # Try flipping axes
        flips = np.ones(J)
        for j in range(J):

            R_flip = R.copy()
            R_flip[:, j] *= -1

            if L(R_flip) < cost:
                flips[j] = -1

        R *= flips
        cost = L(R)

        print(p_opt.x, cost)

        if best_cost is None or cost < best_cost:
            best_R = R
            best_cost = cost
            best_opt = p_opt

    print("Average cost per entry: {}".format(best_cost / A.size))
    
    if full_output:

        def objective(angles):
            cost = diff(givens_rotation_matrix(*angles)).flatten()
            if not np.all(angles > 0) \
            or not np.all((2 * np.pi) >= angles):
                return np.nan * cost
            return cost

        try:
            p_opt, cov, infodict, mesg, ier = op.leastsq(objective, 
                                                         best_opt.x % (2 * np.pi),
                                                         full_output=True)

        except:
            p_opt, cov, infodict, mesg, ier = best_opt.x, None, dict(), 0, -1
    
        return (best_R, p_opt, cov, infodict, mesg, ier)

    return best_R


def exact_rotation_matrix(A_true, A_est, p0=None, full_output=False, **kwargs):

    N, J = A_true.shape

    def cost(R):
        return np.sum(np.abs((A_true - A_est @ R.reshape((J, J))).flatten()))

    if p0 is None:
        p0 = np.zeros(J**2)

    kwds = dict(method="Powell")
    kwds.update(kwargs)
    p_opt = op.minimize(cost, p0, **kwds)
    
    return p_opt.x.reshape((J, J)) if not full_output else p_opt




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


