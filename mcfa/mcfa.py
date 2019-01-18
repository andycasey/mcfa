
""" Mixture of common factor analyzers. """

import json
import logging
import numpy as np
import warnings
from copy import deepcopy
from inspect import signature as inspect_signature
from scipy import linalg, spatial, stats
from scipy.special import logsumexp, gammaln, gamma
from sklearn import cluster
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms
from time import time
from tqdm import tqdm

logger = logging.getLogger(__name__)




class MCFA(object):

    r""" A mixture of common factor analyzers model. """

    def __init__(self, n_components, n_latent_factors, 
                 covariance_regularization=0, max_iter=10000, tol=1e-5,
                 init_components="kmeans++", init_factors="svd", verbose=1, 
                 random_seed=None, **kwargs):
        r"""
        A mixture of common factor analyzers model.

        :param n_components:
            The number of components (clusters) in the mixture.

        :param n_latent_factors:
            The number of common latent factors.

        :param covariance_regularization: [optional]
            An additive term to apply to the covariance matrices of factor
            scores in latent space, which aids in numerical stability.

        :param max_iter: [optional]
            The maximum number of expectation-maximization iterations.
    
        :param tol: [optional]
            Relative tolerance before declaring convergence.

        :param init_components: [optional]
            The iniitialisation method to use when assigning data points to
            components. Available options include: 'kmeans++', 'random', or a
            callable function that takes three arguments: the data, an initial
            estimate of the latent factors, and the number of components. This
            function should return three quantities: an array of the relative
            weights, the means in factor scores, the covariance matrices in
            factor scores.

        :param init_factors: [optional]
            The initialisation method to use for the latent factors. Available
            options include: 'svd', 'random', or a callable functiomn that takes
            the input data as a single argument and returns a matrix of latent
            factors.

        :param verbose: [optional]
            Show warning messages.

        :param random_seed: [optional]
            A seed for the random number generator.
        """

        self.n_components = int(n_components)
        self.n_latent_factors = int(n_latent_factors)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_seed = random_seed
        self.init_components = init_components
        self.init_factors = init_factors
        self.verbose = int(verbose)
        self.covariance_regularization = float(covariance_regularization)

        if self.covariance_regularization < 0:
            raise ValueError("covariance_regularization must be non-negative")

        if self.n_latent_factors < 1:
            raise ValueError("n_latent_factors must be a positive integer")

        if self.n_components < 1:
            raise ValueError("n_components must be a positive integer")

        if self.max_iter < 1:
            raise ValueError("number of iterations must be greater than one")

        if self.tol <= 0:
            raise ValueError("tol must be greater than zero")

        available_init_components = ("random", "kmeans++")
        if self.init_components not in available_init_components \
        and not hasattr(self.init_components, "__call__"):
            raise ValueError(f"init_components must be one of {available_init_components} "
                             f"or be a callable function")

        available_init_factors = ("random", "noise", "svd")
        if self.init_factors not in available_init_factors \
        and not hasattr(self.init_factors, "__call__"):
            raise ValueError(f"init_factors must be one of {available_init_factors} "
                             f"or be a callable function")

        self.log_likelihoods_ = []

        return None


    def serialize(self):
        r"""
        Serialize the object so that it can be saved to disk.
        """
        result = dict()
        for result_attr in ("tau_", "theta_", "log_likelihood_", "n_iter_"):
            value = getattr(self, result_attr, None)
            if value is not None:
                result[result_attr] = value

        return json.dumps({
            "class": self.__class__.__name__,
            "args": (self.n_components, self.n_latent_factors),
            "kwargs": dict(max_iter=self.max_iter,
                           n_init=self.n_init,
                           tol=self.tol,
                           verbose=self.verbose,
                           random_seed=self.random_seed),
            "result": result
        })


    @classmethod
    def deserialize(cls, data):
        r"""
        De-serialize the data and return an object.

        :param data:
            Serialized data describing the object.

        :returns:
            A :class:`mcfa.MCFA` object.
        """

        params = json.loads(data)

        klass = cls(*params["args"], **params["kwargs"])
        for k, v in params["result"].items():
            setattr(klass, k, v)

        return klass


    def _check_data(self, X, warn_about_whitening=False):
        r""" 
        Verify that the latent space has lower dimensionality than the data
        space.

        :param X:
            The data, which is expected to be an array with shape [n_samples, 
            n_features].

        :raises ValueError:
            If the data array has non-finite entries, or if there are more
            latent factors than there are dimensions.

        :returns:
            The data array, ensuring it is a 2D array.
        """

        X = np.atleast_2d(X)
        if not np.all(np.isfinite(X)):
            raise ValueError("data has non-finite entries")

        N, D = X.shape
        if D > N:
            logger.warning(f"There are more dimensions than data ({D} > {N})!")

        if D <= self.n_latent_factors:
            raise ValueError(f"there are more factors than dimensions "\
                             f"({self.n_latent_factors} >= {D})")

        # Check to see if the data are whitened.
        mu, sigma = (np.mean(X, axis=0), np.std(X, axis=0))
        if warn_about_whitening and \
        not (np.allclose(mu, np.zeros(D)) or np.allclose(sigma, np.ones(D))):
            logger.warn("Supplied data do not appear to be whitened. "\
                        "Use mcfa.utils.whiten(X) to whiten the data.")

        # Pre-calculate X2, because we will use this at every EM step.
        self._check_precomputed_X2(X)

        return X


    def _check_precomputed_X2(self, X, **kwargs):
        r"""
        Compute and store X^2 if it is not already calculated. 
        If it is pre-computed, check a random entry of the matrix, and raise a
        `ValueError` exception if it does not meet expected tolerance.

        :param X:
            The data, which is expected to be an array with shape [n_samples,
            n_features].
        """

        try:
            self._X2

        except AttributeError:
            self._X2 = np.square(X)

        else:
            # Check a single entry.
            i, j = (np.random.choice(X.shape[0]), np.random.choice(X.shape[1]))

            expected, actual = (np.square(X[i, j]), self._X2[i, j])
            if not np.allclose(expected, actual, **kwargs):
                raise ValueError(
                    f"pre-computed X^2 does not match actual X^2 at {i}, {j} "\
                    f"({expected} != {actual})")

        return self._X2


    def _initial_parameters(self, X, random_state=None):
        r"""
        Estimate the initial parameters of the model.

        :param X:
            The data, :math:`X`, which is expected to be an array of shape
            [n_samples, n_features].

        :raises ValueError:
            If no valid initialisation point could be found.

        :returns:
            A five-length tuple containing the updated parameter estimates for
            the mixing weights :math:`\pi`, the common factor loads :math:`A`,
            the means of the components in latent space :math:`\xi`, the
            covariance matrices of components in latent space :math:`\omega`,
            and the variance in each dimension :math:`\psi`.
        """

        previous_theta = getattr(self, "theta_", None)
        if previous_theta is not None:
            logger.warn("Running from previous estimates of theta")
            return previous_theta

        initial_factor_func = {
            "random": _initial_factor_loads_by_random,
            "noise": _initial_factor_loads_by_noise,
            "svd": _initial_factor_loads_by_svd
        }.get(self.init_factors, self.init_factors)
        
        A = initial_factor_func(X, self.n_latent_factors, random_state)

        initial_components_func = {
            "random": _initial_components_by_random,
            "kmeans++": _initial_components_by_kmeans_pp,
        }.get(self.init_components, self.init_components)

        # TODO: use random state.
        pi, xi, omega = initial_components_func(X, A, self.n_components)

        N, D = X.shape
        psi = np.ones(D)

        return (pi, A, xi, omega, psi)


    @property
    def parameter_names(self):
        r""" Return the names of the parameters in this model. """
        return tuple(inspect_signature(self.expectation).parameters.keys())[1:]


    def number_of_parameters(self, D):
        r"""
        Return the number of model parameters :math:`Q` required to describe 
        data of :math:`D` dimensions.

        .. math::

            Q = (K - 1) + D + J(D + K) + \frac{1}{2}KJ(J + 1) - J^2

        Where :math:`K` is the number of components, :math:`D` is the number of
        dimensions in the data, and :math:`J` is the number of latent factors.

        :param D:
            The dimensionality of the data (the number of features).

        :returns:
            The number of model parameters, :math:`Q`.
        """

        J, K = self.n_latent_factors, self.n_components
        return int((K - 1) + D + J*(D + K) + (K*J*(J+1))/2 - J**2)


    def expectation(self, X, pi, A, xi, omega, psi):
        r"""
        Compute the conditional expectation of the complete-data log-likelihood
        given the observed data :math:`X` and the given model parameters.

        :param X:
            The data, which is expected to be an array with shape [n_samples, 
            n_features].

        :param pi:
            The relative weights for the components in the mixture. This should
            have size `n_components` and the entries should sum to one.

        :param A:
            The common factor loads between mixture components. This should have
            shape [n_features, n_latent_factors].

        :param xi:
            The mean factors for the components in the mixture. This should have 
            shape [n_latent_factors, n_components].

        :param omega:
            The covariance matrix of the mixture components in latent space.
            This array should have shape [n_latent_factors, n_latent_factors, 
            n_components].

        :param psi:
            The variance in each dimension. This should have size [n_features].

        :raises scipy.linalg.LinAlgError:
            If the covariance matrix of any mixture component in latent space
            is ill-conditioned or singular.

        :returns:
            A two-length tuple containing the sum of the log-likelihood for the
            data given the model, and the responsibility matrix :math:`\tau`
            giving the partial associations between each data point and each
            component in the mixture.
        """

        ll, tau = _expectation(X, pi, A, xi, omega, psi, verbose=self.verbose)
        self.log_likelihoods_.append(ll)

        return (ll, tau)


    def maximization(self, X, tau, pi, A, xi, omega, psi):
        r"""
        Compute the updated estimates of the model parameters given the data,
        the responsibility matrix :math:`\tau`, and the current estimates of the
        model parameters.

        :param X:
            The data, which is expected to be an array with shape [n_samples, 
            n_features].

        :param tau:
            The responsibility matrix, which is expected to have shape
            [n_samples, n_components]. The sum of each row is expected to equal
            one, and the value in the i-th row (sample) of the j-th column
            (component) indicates the partial responsibility (between zero and
            one) that the j-th component has for the i-th sample.

        :param pi:
            The relative weights for the components in the mixture. This should
            have size `n_components` and the entries should sum to one.

        :param A:
            The common factor loads between mixture components. This should have
            shape [n_features, n_latent_factors].

        :param xi:
            The mean factors for the components in the mixture. This should have 
            shape [n_latent_factors, n_components].

        :param omega:
            The covariance matrix of the mixture components in latent space.
            This array should have shape [n_latent_factors, n_latent_factors, 
            n_components].

        :param psi:
            The variance in each dimension. This should have size [n_features].

        :returns:
            A five-length tuple containing the updated parameter estimates for
            the mixing weights :math:`\pi`, the common factor loads :math:`A`,
            the means of the components in latent space :math:`\xi`, the
            covariance matrices of components in latent space :math:`\omega`,
            and the variance in each dimension :math:`\psi`.
        """

        return _maximization(X, tau, pi, A, xi, omega, psi,
                             X2=self._check_precomputed_X2(X),
                             covariance_regularization=self.covariance_regularization)


    def _check_convergence(self, previous, current):
        r"""
        Check whether the expectation-maximization algorithm has converged based
        on the previous cost (e.g., log-likelihood or message length), and the
        current cost.

        :param previous:
            The cost of the previous estimate of the model parameters.

        :param current:
            The cost of the current estimate of the model parameters.

        :returns:
            A three-length tuple containing: a boolean flag indicating whether
            convergence has been achieved, the current cost, and the ratio of
            the current cost relative to the previous cost.
        """

        assert np.isfinite(current)
        if previous > current:
            logger.warn(f"Log-likelihood *decreased* by {previous-current:.2f}")

        #assert current > previous # depends on objective function

        ratio = abs((current - previous)/current)
        converged = self.tol >= ratio

        return (converged, current, ratio)


    def fit(self, X, init_params=None, **kwargs):
        r"""
        Fit the model to the data, :math:`Y`.

        :param X:
            The data, :math:`X`, which is expected to be an array of shape
            [n_samples, n_features].

        :param init_params: [optional]
            A dictionary of initial values to run expectation-maximization from.

        :returns:
            The fitted model.
        """

        np.random.seed(self.random_seed)

        X = self._check_data(X)

        theta = self._initial_parameters(X, self.random_seed) \
                if init_params is None else deepcopy(init_params) 

        prev_ll, tau = self.expectation(X, *theta)

        # TODO: start n_iter counter from previous value if we are starting from
        #       a previously optimised value.

        # Run E-M.
        tqdm_kwds = dict(desc="E-M", total=self.max_iter)
        for n_iter in tqdm(range(1, 1 + self.max_iter), **tqdm_kwds):

            theta = self.maximization(X, tau, *theta)            
            ll, tau = self.expectation(X, *theta)

            converged, prev_ll, ratio = self._check_convergence(prev_ll, ll)

            if converged: break

        else:
            if self.verbose >= 0:
                logger.warning(f"Convergence not achieved after at least "\
                               f"{self.max_iter} iterations ({ratio} > {self.tol})."\
                               f" Consider increasing the maximum number of iterations.")

        pi, A, xi, omega, psi = theta

        # Make A.T @ A = I
        AL = linalg.cholesky(A.T @ A)
        A = A @ linalg.solve(AL, np.eye(self.n_latent_factors))

        A = _post_check_factor_loads(A)
        xi = AL @ xi

        for i in range(self.n_components):
            omega[:, :, i] = AL @ omega[:, :, i] @ AL.T

        self.tau_ = tau
        self.theta_ = [pi, A, xi, omega, psi]
        self.log_likelihood_ = ll
        self.n_iter_ = n_iter

        return self


    def factor_scores(self, X):
        r"""
        Estimate the posterior factor scores given the model parameters.

        :param X:
            The data, :math:`X`, which is expected to be an array of shape
            [n_samples, n_features].
        # TODO: What should we return, exactly?
        """
        return _factor_scores(X, self.tau_, *self.theta_)


    def bic(self, X, theta=None):
        r"""
        Estimate the Bayesian Information Criterion given the model and the
        data.

        :param X:
            The data, :math:`X`, which is expected to be an array of shape
            [n_samples, n_features].

        :param theta: [optional]
            The model parameters :math:`\theta`. If None is given then the
            model parameters from `self.theta_` will be used.

        :returns:
            The Bayesian Information Criterion (BIC) for the model and the data.
            A smaller BIC value is often used as a statistic to select a single
            model from a class of models.
        """

        if theta is None:
            theta = self.theta_

        N, D = X.shape
        log_likelihood, tau = self.expectation(X, *theta)
        return np.log(N) * self.number_of_parameters(D) - 2 * log_likelihood


    def pseudo_bic(self, X, gamma=0.1, omega=1, theta=None):
        r"""
        Estimate the pseudo Bayesian Information Criterion given the model and
        the data as per Gao and Carroll (2017):

        .. math:

            \textrm{pseudo - BIC} = 6(1 + \gamma)\omega\log{N}Q - 2\log{mathcal{L}}

        :param X:
            The data, :math:`X`, which is expected to be an array of shape
            [n_samples, n_features].

        :param theta: [optional]
            The model parameters :math:`\theta`. If None is given then the
            model parameters from `self.theta_` will be used.

        :returns:
            The Bayesian Information Criterion (BIC) for the model and the data.
            A smaller BIC value is often used as a statistic to select a single
            model from a class of models.
        """

        if not gamma > 0:
            raise ValueError("gamma should be greater than zero")

        if not omega >= 1:
            raise ValueError("omega should be at least unity or higher")

        if theta is None:
            theta = self.theta_

        N, D = np.atleast_2d(X).shape
        log_likelihood, tau = self.expectation(X, *theta)
        Q = self.number_of_parameters(D)

        return 6 * (1 + gamma) * omega * np.log(N) * Q - 2 * log_likelihood


    def message_length(self, X, theta=None):
        r"""
        Estimate the explanation length given the model and the data.

        :param X:
            The data, :math:`X`, which is expected to be an array of shape
            [n_samples, n_features].

        :param theta: [optional]
            The model parameters :math:`\theta`. If None is given then the
            model parameters from `self.theta_` will be used.
        """

        if theta is None:
            theta = self.theta_

        N, D = np.atleast_2d(X).shape
        log_likelihood, tau = self.expectation(X, *theta)

        # For just MLF:
        pi, A, xi, omega, psi = theta
        J, K = self.n_latent_factors, self.n_components

        # In Wallace nomenclature, J = number of factors (here also J)
        #                          K = dimensionality of data (here is D)
        #                          N = number of data (here also N)

        # Recall:
        # X! = gamma(X + 1)
        # S_m = (m * np.pi**(m/2))/((m/2)!)
        log_S = lambda m: np.log(m) + (m/2) * np.log(np.pi) - gammaln(m/2 + 1)

        # Approximate the normalization constants (p308 of Wallace 2005)
        #  J = [1     2     3     4     5      6]
        A_Js = [0, 0.24, 1.35, 3.68, 7.45, 12.64]
        B_Js = [0, 0.58, 0.64, 0.73, 0.86,  1.21]

        #bj_sq = A**2
        #v = self.factor_scores(X)[1]
        #vj_sq = np.diag((v.T @ v)) # very unsure about this.
        

        if J > len(A_Js):

            #raise NotImplementedError("too many latent factors; "
            #                          "cannot approximate normalization constants")
            logger.warn("Too many latent factors; "
                        "cannot approximate normalization constants when J > 6")

            A_J, B_J = (A_Js[-1], B_Js[-1])

        else:
            A_J, B_J = (A_Js[int(J) - 1], B_Js[int(J) - 1])

        log_GNK = -(A_J + 0.5 * J * (J - 1) * np.log(D - B_J))


        # Add weights for mixture (A.4 of GMMMML paper)
        # kappa varies between 1/12 and 1/(2*pi*e), so assume kappa = 1/12
        kappa = 1/12.
        I_1 = 0.5 * (K - 1) * np.log(N) - 0.5 * np.sum(np.log(pi)) \
            - gammaln(K) + 0.5 * K * np.log(kappa)
            
        # Encode covariance matrices. (A.11 of GMMMML paper)
        # Wishart prior on covariance matrix.
        # Note here we use J instead of D because the mixing is done in latent
        # space.
        slogdet_omega = np.linalg.slogdet(omega.T)[1]

        I_2 = 0.25 * J * (J + 3) * np.sum(np.log(N * pi)) \
            - 0.5 * (2 * J + 3) * np.sum(slogdet_omega) \
            - 0.5 * (K * J) * np.log(2)

        return 0#I_0 + I_1 + I_2


    def sample(self, n_samples=1, theta=None):
        r"""
        Generate random samples from the fitted model.

        :param n_samples: [optional]
            Number of samples to generate. Defaults to 1.

        # TODO: return docs
        """

        if theta is None:
            theta = self.theta_

        pi, A, xi, omega, psi = theta

        # Draw which component it is from.
        taus = np.random.choice(self.n_components, size=n_samples, p=pi)

        # Draw from the latent space and project to real space.
        N, D = (n_samples, psi.size)

        X = np.empty((N, D))
        for k in range(self.n_components):

            match = (taus == k)
            scores = np.random.multivariate_normal(xi.T[k], omega.T[k],
                                                   size=sum(taus == k))

            # Project to real space.
            X[match] = (A @ scores.T).T

        # Add noise.
        X += np.random.multivariate_normal(np.zeros(D), np.diag(psi), N)

        return X


    def rotate(self, R, X=None, ensure_valid_rotation=True, atol=1e-3, rtol=1e-5):
        r"""
        Rotate the factor loads and factor scores by a valid rotation matrix.

        :param R:
            A `J` times `J` rotation matrix, where `J` is the number of latent
            factors.
        
        :param X: [optional]
            The data, which is expected to be an array with shape [n_samples,
            n_features]. If given, the log-likelihood will be evaluated before
            and after rotation. A warning will be raised if the log-likelihood
            changes by more than the convergence tolerance.

        :param ensure_valid_rotation: [optional]
            If the rotation matrix does not follow R @ R.T = I, then the nearest
            rotation matrix with this property will be used.

        :param atol: [optional]
            The absolute tolerance acceptable for individual entries in the
            matrix I - R @ R.T. Default is 1e-3.

        :param rtol:
            The relative tolerance acceptable for individual entries in the
            matrix I - R @ R.T. Default is 1e-5.
        """

        # Check that it is a rotation matrix.
        R = np.atleast_2d(R)
        J, J_ = R.shape
        if J != J_:
            raise ValueError("rotation matrix is not square")

        if not np.allclose(R @ R.T, np.eye(J), atol=atol, rtol=rtol):

            # Do our own rotation.
            L = linalg.cholesky(R.T @ R)
            R = R @ linalg.solve(L, np.eye(self.n_latent_factors))

            if not np.allclose(R @ R.T, np.eye(J), atol=atol, rtol=rtol):
                raise ValueError("R is not a valid rotation matrix")

        if X is not None:
            ll, tau = self.expectation(X, *self.theta_)

        pi, A, xi, omega, psi = self.theta_

        # A covariance matrix can be decomposed as:
        # \Sigma = Q @ \Lambda @ Q'
        # Where Q is a valid rotation matrix and \Lambda acts like a scaling
        # matrix.

        # Thus a rotated covariance matrix is:
        # \Sigma_rot = R.T @ omega_ @ R
        self.theta_ = (pi, A @ R, (xi.T @ R).T, (R.T @ omega.T @ R).T, psi)

        if X is not None:
            ll_r, tau_r = self.expectation(X, *self.theta_)

            if np.abs(ll - ll_r) > self.tol:
                logger.warning(f"Log-likelihood changed after rotation: "\
                               f"{ll - ll_r} (>{self.tol})")

        return True


    




def _initial_factor_loads_by_random(X, n_latent_factors, random_state=None):
    N, D = X.shape
    A = stats.ortho_group.rvs(D, random_state=random_state)[:, :n_latent_factors]
    AL = linalg.cholesky(A.T @ A)
    A = A @ linalg.solve(AL, np.eye(n_latent_factors))
    return A


def _initial_factor_loads_by_noise(X, n_latent_factors, random_state=None,
                                   scale=1e-2):

    # TODO: use random state.
    N, D = X.shape
    return np.random.normal(0, scale, size=(D, n_latent_factors))

def _initial_factor_loads_by_svd(X, n_latent_factors, random_state=None,
                                 n_svd_max=1000):

    # TODO: use random state
    N, D = X.shape
    n_svd_max = N if n_svd_max < 0 or n_svd_max > N else int(n_svd_max)
    idx = np.random.choice(N, n_svd_max, replace=False)
    U, S, V = linalg.svd(X[idx].T/np.sqrt(N - 1))
    return U[:, :n_latent_factors]


def _initial_assignments_by_random(X, A, n_components):
    N, D = X.shape
    return np.random.choice(n_components, N)


def _initial_assignments_by_kmeans_pp(X, A, n_components, random_state=None):
    r"""
    Estimate the initial assignments of each sample to each component in the
    mixture.

    :param X:
        The data, which is expected to be an array with shape [n_samples,
        n_features].

    :param n_components:
        The number of components (clusters) in the mixture.

    :returns:
        An array of shape [1, n_samples] with initial assignments, 
        where each integer entry in the matrix indicates which component the 
        sample is to be initialised to.
    """

    random_state = check_random_state(random_state)

    Y = X @ A # run k-means++ in the latent space

    squared_norms = row_norms(Y, squared=True)

    means = cluster.k_means_._k_init(Y, n_components,
                                     random_state=random_state,
                                     x_squared_norms=squared_norms)
    return np.argmin(spatial.distance.cdist(means, Y), axis=0)


def _initial_components(X, A, n_components, assignments):

    N, D = X.shape
    D, J = A.shape

    n_components = len(set(assignments))

    xi = np.empty((J, n_components))
    omega = np.empty((J, J, n_components))
    pi = np.empty(n_components)

    for i in range(n_components):
        match = (assignments == i)
        pi[i] = float(sum(match)) / N
        xs = X[match] @ A
        xi[:, i] = np.mean(xs, axis=0)
        omega[:, :, i] = np.cov(xs.T)

    return (pi, xi, omega)


def _initial_components_by_random(X, A, n_components):
    assignments = _initial_assignments_by_random(X, A, n_components)
    return _initial_components(X, A, n_components, assignments)

def _initial_components_by_kmeans_pp(X, A, n_components):
    assignments = _initial_assignments_by_kmeans_pp(X, A, n_components)
    return _initial_components(X, A, n_components, assignments)



def _expectation(X, pi, A, xi, omega, psi, verbose=1, full_output=False):
    r"""
    Compute the conditional expectation of the complete-data log-likelihood
    given the observed data :math:`X` and the given model parameters.

    :param X:
        The data, which is expected to be an array with shape [n_samples, 
        n_features].

    :param pi:
        The relative weights for the components in the mixture. This should
        have size `n_components` and the entries should sum to one.

    :param A:
        The common factor loads between mixture components. This should have
        shape [n_features, n_latent_factors].

    :param xi:
        The mean factors for the components in the mixture. This should have 
        shape [n_latent_factors, n_components].

    :param omega:
        The covariance matrix of the mixture components in latent space.
        This array should have shape [n_latent_factors, n_latent_factors, 
        n_components].

    :param psi:
        The variance in each dimension. This should have size [n_features].

    :param verbose: [optional]
        The verbosity.

    :raises scipy.linalg.LinAlgError:
        If the covariance matrix of any mixture component in latent space
        is ill-conditioned or singular.

    :returns:
        A two-length tuple containing the sum of the log-likelihood for the
        data given the model, and the responsibility matrix :math:`\tau`
        giving the partial associations between each data point and each
        component in the mixture.
    """

    N, D = X.shape
    K = pi.size

    I_D = np.eye(D)
    I_psi = I_D * psi
    U = (np.diag(1.0/psi) @ A).T

    log_prob = np.zeros((N, K))
    log_prob_constants = -0.5 * (np.sum(np.log(psi)) + D * np.log(2*np.pi))
    
    for i, (xi_, omega_) in enumerate(zip(xi.T, omega.T)):

        mu = A @ xi_
        A_omega_ = A @ omega_
        cov = A_omega_ @ A.T + I_psi

        precision = _compute_precision_cholesky_full(cov)
        dist = np.sum(
            np.square(np.dot(X, precision) - np.dot(mu, precision)), axis=1)

        # Use matrix determinant lemma:
        # det(A + U @V.T) = det(I + V.T @ A^-1 @ U) * det(A)
        # here A = omega^{-1}; U = (\psi^-1 @ A).T; V = A.T
        
        # and making use of det(A) = 1.0/det(A^-1)

        with warnings.catch_warnings():
            if 1 > verbose: warnings.simplefilter("ignore")

            # log(det(omega)) + log(det(I + A @ omega @ U)/det(omega)) \
            # = log(det(I + A @ omega @ U))
            _, logdetD = np.linalg.slogdet(I_D + A @ omega_ @ U)

        log_prob[:, i] = -0.5 * dist - 0.5 * logdetD + log_prob_constants

    weighted_log_prob = log_prob + np.log(pi)
    log_prob = logsumexp(weighted_log_prob, axis=1)
    with np.errstate(under="ignore"):
        log_tau = weighted_log_prob - log_prob[:, np.newaxis]

    if not full_output:
        return (np.sum(log_prob), np.exp(log_tau))

    else:
        return (np.sum(log_prob), np.exp(log_tau))


def _maximization(X, tau, pi, A, xi, omega, psi,
                  X2=None, covariance_regularization=0):
    r"""
    Compute the updated estimates of the model parameters given the data,
    the responsibility matrix :math:`\tau`, and the current estimates of the
    model parameters.

    :param X:
        The data, which is expected to be an array with shape [n_samples, 
        n_features].

    :param tau:
        The responsibility matrix, which is expected to have shape
        [n_samples, n_components]. The sum of each row is expected to equal
        one, and the value in the i-th row (sample) of the j-th column
        (component) indicates the partial responsibility (between zero and
        one) that the j-th component has for the i-th sample.

    :param pi:
        The relative weights for the components in the mixture. This should
        have size `n_components` and the entries should sum to one.

    :param A:
        The common factor loads between mixture components. This should have
        shape [n_features, n_latent_factors].

    :param xi:
        The mean factors for the components in the mixture. This should have 
        shape [n_latent_factors, n_components].

    :param omega:
        The covariance matrix of the mixture components in latent space.
        This array should have shape [n_latent_factors, n_latent_factors, 
        n_components].

    :param psi:
        The variance in each dimension. This should have size [n_features].

    :param X2: [optional]
        The square of X, :math:`X^2`. If `None` is given then this will be
        computed at each maximization step.

    :param covariance_regularization: [optional]
        An additive term to apply to the covariance matrices of factor
        scores in latent space, which aids in numerical stability.

    :returns:
        A five-length tuple containing the updated parameter estimates for
        the mixing weights :math:`\pi`, the common factor loads :math:`A`,
        the means of the components in latent space :math:`\xi`, the
        covariance matrices of components in latent space :math:`\omega`,
        and the variance in each dimension :math:`\psi`.
    """

    if X2 is None:
        X2 = np.square(X)

    N, D = X.shape
    D, J = A.shape

    inv_D = np.diag(1.0/psi)


    A1 = np.zeros((D, J))
    A2 = np.zeros((J, J))
    Di = np.zeros(D)

    inv_D_A = inv_D @ A

    ti = np.sum(tau, axis=0).astype(float)
    
    I_J = np.eye(J)

    for i, tau_ in enumerate(tau[np.newaxis].T):

        W = linalg.solve(omega[:, :, i], I_J)
        C = linalg.solve(W + A.T @ inv_D_A, I_J)
        gamma = (inv_D - inv_D_A @ C @ inv_D_A.T) @ A @ omega[:, :, i]

        xi_ = np.copy(xi[:, [i]])

        Y_Axi_i = X.T - A @ xi_
        tY_Axi_i = Y_Axi_i * tau_.T

        xi[:, i] += gamma.T @ (np.sum(tY_Axi_i, axis=1) / ti[i])

        diff = (xi_ - xi[:, [i]])

        omega[:, :, i] = (I_J - gamma.T @ A) @ omega[:, :, i] \
                       + gamma.T @ Y_Axi_i @ tY_Axi_i.T @ gamma / ti[i] \
                       - diff @ diff.T \
                       + covariance_regularization * I_J

        A1 += np.atleast_2d(np.sum(X * tau_, axis=0)).T @ xi_.T \
            + X.T @ tY_Axi_i.T @ gamma
        A2 += (omega[:, :, i] + xi[:, [i]] @ xi[:, [i]].T) * ti[i]
    
    A = A1 @ linalg.solve(A2, I_J)

    A = _post_check_factor_loads(A)  

    Di = np.sum(X2.T @ tau, axis=1)

    psi = (Di - np.sum((A @ A2) * A, axis=1)) / N
    
    pi = ti / N

    return (pi, A, xi, omega, psi)



def _factor_scores(X, tau, pi, A, xi, omega, psi):
    r"""
    Estimate the factor scores for each data point, given the model.

    :param X:
        The data, which is expected to be an array with shape [n_samples, 
        n_features].

    :param tau:
        The responsibility matrix, which is expected to have shape
        [n_samples, n_components]. The sum of each row is expected to equal
        one, and the value in the i-th row (sample) of the j-th column
        (component) indicates the partial responsibility (between zero and
        one) that the j-th component has for the i-th sample.

    :param pi:
        The relative weights for the components in the mixture. This should
        have size `n_components` and the entries should sum to one.

    :param A:
        The common factor loads between mixture components. This should have
        shape [n_features, n_latent_factors].

    :param xi:
        The mean factors for the components in the mixture. This should have 
        shape [n_latent_factors, n_components].

    :param omega:
        The covariance matrix of the mixture components in latent space.
        This array should have shape [n_latent_factors, n_latent_factors, 
        n_components].

    :param psi:
        The variance in each dimension. This should have size [n_features].

    # TODO: returns
    """

    N, D = X.shape
    J, K = xi.shape

    U = np.zeros((N, J, K))
    gamma = np.zeros((D, J, K))

    inv_D = np.diag(1.0/psi)

    I = np.eye(J)

    for k in range(K):
        C = linalg.solve(linalg.solve(omega[:, :, k], I) \
                        + A.T @ inv_D @ A, I)
        gamma[:, :, k] = (inv_D - inv_D @ A @ C @ A.T @ inv_D) \
                       @ A @ omega[:, :, k]
        U[:, :, k] = np.repeat(xi[:, [k]], N).reshape((J, N)).T \
                   + (X - (A @ xi[:, [k]]).T) @ gamma[:, :, k]

    cluster = np.argmax(tau, axis=1)

    UC = np.zeros((N, J))
    Umean = np.zeros((N, J))

    for i in range(N):
        UC[i] = U[i, :, cluster[i]]
        Umean[i] = tau[i] @ U[i].T

    return (U, UC, Umean)


def _compute_precision_cholesky_full(cov):
    r"""
    Compute the Cholesky decomposition of the precision of the given covariance
    matrix, which is expected to be a square matrix with non-zero off-diagonal
    terms.

    :param cov:
        The given covariance matrix.

    :raises scipy.linalg.LinAlgError:
        If the Cholesky decomposition failed.

    :returns:
        The Cholesky decomposition of the precision of the covariance matrix.
    """

    try:
        cholesky_cov = linalg.cholesky(cov, lower=True)

    except linalg.LinAlgError:
        raise linalg.LinAlgError("failed to do Cholesky decomposition")

    # Return the precision matrix.
    D, _ = cov.shape
    return linalg.solve_triangular(cholesky_cov, np.eye(D), lower=True).T



def _post_check_factor_loads(A):
    D, J = A.shape
    #A[np.triu_indices(J, 1)] = 0 
    #A[np.diag_indices(J)] = np.abs(A[np.diag_indices(J)])

    return A 


if __name__ == "__main__":

    
    from sklearn.datasets import load_iris
    
    X = load_iris().data

    model = MCFA(n_components=3, n_latent_factors=2)
    model.fit(X)


    try:
        from .mpl_utils import plot_latent_space
    except ModuleNotFoundError:
        from mpl_utils import plot_latent_space # TODO don't do this


    fig = plot_latent_space(model, X)


"""
        S = lambda m: (m*np.pi**(0.5 * m))/np.exp(gammaln(0.5*m + 1))

        pi, A, xi, omega, psi = theta

        # Approximate the normalization constants (p308 of Wallace 2005)
        #  J = [1     2     3     4     5      6]
        A_Js = [0, 0.24, 1.35, 3.68, 7.45, 12.64]
        B_Js = [0, 0.58, 0.64, 0.73, 0.86,  1.21]

        bj_sq = A**2
        v = self.factor_scores(X)[1]
        vj_sq = np.diag((v.T @ v)) # very unsure about this.
        

        if J > len(A_Js):

            #raise NotImplementedError("too many latent factors; "
            #                          "cannot approximate normalization constants")
            logger.warn("Too many latent factors; "
                        "cannot approximate normalization constants when J > 6")

            A_J, B_J = (A_Js[-1], B_Js[-1])

        else:
            A_J, B_J = (A_Js[int(J) - 1], B_Js[int(J) - 1])

        log_GNK = -(A_J + 0.5 * J * (J - 1) * np.log(D - B_J))

        # I_0 is term for MLF model (no mixture)
        # TODO: missing term related to S(N-1) ?
        # TODO: commented-out missing terms
        I_0 = (D/2 - 2 * J) * np.log(2) + D * np.log(N) \
            + 0.5 * J * N * np.log(2 * np.pi) \
            - gammaln(J + 1) - J * np.log(S(N)) \
            - J * np.log(S(D)/S(D + 1)) \
            + (N - 1) * np.sum(0.5 * np.log(psi)) \
            + np.sum(log_likelihood) \
            - 0.5 * D * np.log(2 * np.pi) + 0.5 * np.log(D * np.pi) \
            - log_GNK #\
            #+ 0.5 * np.sum(vj_sq)
            #+ 0.5 * (D - J + 1) * np.sum(np.log(vj_sq))
            #+ 0.5 * (N + D - 1) * np.sum(np.log(1 + bj_sq)) 
            #+ 0.5 * np.sum(np.sum(vj_sq, axis=0) * np.sum(bj_sq, axis=0))
            #- np.sum(b_j.T @ X @ b_j / Q_j) \
        
        if J > 3 and K > 2:
            raise a
"""

