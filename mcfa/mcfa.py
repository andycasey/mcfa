
""" Mixture of common factor analyzers. """

import logging
import numpy as np
import warnings
from copy import deepcopy
from inspect import signature as inspect_signature
from scipy import linalg
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from time import time

logger = logging.getLogger(__name__)


class MCFA(object):

    r""" A mixture of common factor analyzers model. """

    def __init__(self, n_components, n_latent_factors, max_iter=500, n_init=5, 
                 tol=1e-5, verbose=0, random_seed=None, **kwargs):
        r"""
        A mixture of common factor analyzers model.

        :param n_components:
            The number of components (clusters) in the mixture.

        :param n_latent_factors:
            The number of common latent factors.

        :param max_iter: [optional]
            The maximum number of expectation-maximization iterations.

        :param n_init: [optional]
            The number of initialisations to run using the k-means algorithm.

        :param tol: [optional]
            Relative tolerance before declaring convergence.

        :param verbose: [optional]
            Show warning messages.

        :param random_seed: [optional]
            A seed for the random number generator.
        """

        self.n_components = int(n_components)
        self.n_latent_factors = int(n_latent_factors)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.tol = float(tol)
        self.random_seed = random_seed

        if self.n_latent_factors < 1:
            raise ValueError("n_latent_factors must be a positive integer")

        if self.n_components < 1:
            raise ValueError("n_components must be a positive integer")

        if self.max_iter < 1:
            raise ValueError("number of iterations must be greater than one")

        if self.n_init < 1:
            raise ValueError("n_init must be a positive integer")

        if self.tol <= 0:
            raise ValueError("tol must be greater than zero")

        self.verbose = int(verbose)
        return None


    def _check_data(self, X):
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
        if D <= self.n_latent_factors:
            raise ValueError(f"there are more factors than dimensions "\
                             f"({self.n_latent_factors} >= {D})")


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

        return True


    def _initial_parameters(self, X):
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

        # Do initial partitions (either randomly or by k-means).
        assignments = _initial_assignments(X, self.n_components, self.n_init)
        best_theta, best_log_likelihood = (None, -np.inf)

        for i, assignment in enumerate(assignments):
            try:
                params = _initial_parameters(X, 
                                             self.n_components, 
                                             self.n_latent_factors,
                                             assignment)

            except ValueError:
                if self.verbose > 0:
                    logger.exception("Exception in initializing:")

            else:

                # Run one E-M step from this initial guess.
                try:                
                    log_likelihood, tau = self.expectation(X, *params)
                    theta = self.maximization(X, tau, *params)
                    
                    if log_likelihood > best_log_likelihood:
                        best_theta, best_log_likelihood = (theta, log_likelihood)

                except:
                    if self.verbose > 0:
                        logger.exception("Exception in running E-M step from "\
                                         "initialisation point:")

        if best_theta is None:
            raise ValueError("no initialisation point found")

        return best_theta


    @property
    def parameter_names(self):
        r""" Return the names of the parameters in this model. """
        return tuple(inspect_signature(self.expectation).parameters.keys())[1:]


    def number_of_parameters(self, D):
        r"""
        Return the number of model parameters :math:`Q` required to describe 
        data of :math:`D` dimensions.

        .. math:
            Q = (K - 1) + D + J(D + K) + \frac{1}{2}KJ(J + 1) - J^2


        Where :math:`K` is the number of components,
              :math:`D` is the number of dimensions in the data, and
              :math:`J` is the number of latent factors.

        :param D:
            The dimensionality of the data (the number of features).

        :returns:
            The number of model parameters, :math:`Q`.
        """

        J, K = self.n_components, self.n_latent_factors
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

        N, D = X.shape

        I_D = np.eye(D)
        I_psi = I_D * psi
        U = (np.diag(1.0/psi) @ A).T

        log_prob = np.zeros((N, self.n_components))
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
                if 1 > self.verbose: warnings.simplefilter("ignore")

                # log(det(omega)) + log(det(I + A @ omega @ U)/det(omega)) \
                # = log(det(I + A @ omega @ U))
                _, logdetD = np.linalg.slogdet(I_D + A @ omega_ @ U)

            precision = _compute_precision_cholesky_full(cov)
            diff = np.dot(X, precision) - np.dot(mu, precision)
            dist = np.sum(np.square(diff), axis=1)

            log_prob[:, i] = -0.5 * dist - 0.5 * logdetD + log_prob_constants

        weighted_log_prob = log_prob + np.log(pi)
        log_likelihood = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            log_tau = weighted_log_prob - log_likelihood[:, np.newaxis]

        tau = np.exp(log_tau)

        return (sum(log_likelihood), tau)


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

        self._check_precomputed_X2(X)

        N, D = X.shape

        inv_D = np.diag(1.0/psi)

        A1 = np.zeros((D, self.n_latent_factors))
        A2 = np.zeros((self.n_latent_factors, self.n_latent_factors))
        Di = np.zeros(D)

        inv_D_A = inv_D @ A

        ti = np.sum(tau, axis=0).astype(float)
        pi = ti / N
        
        I_J = np.eye(self.n_latent_factors)

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
                           - diff @ diff.T

            A1 += np.atleast_2d(np.sum(X * tau_, axis=0)).T @ xi_.T \
                + X.T @ tY_Axi_i.T @ gamma
            A2 += (omega[:, :, i] + xi[:, [i]] @ xi[:, [i]].T) * ti[i]
            Di += np.sum(self._X2 * tau_, axis=0)

        A = A1 @ linalg.solve(A2, I_J)
        psi = (Di - np.sum((A @ A2) * A, axis=1)) / N
        
        return (pi, A, xi, omega, psi)


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
        assert current > previous # depends on objective function

        ratio = abs((current - previous)/current)
        converged = self.tol >= ratio

        return (converged, current, ratio)


    def fit(self, X, init_params=None):
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
        theta = self._initial_parameters(X) if init_params is None \
                                            else deepcopy(init_params) 

        # Calculate initial log-likelihood.
        prev_ll, tau = self.expectation(X, *theta)

        # Run E-M.
        for n_iter in range(self.max_iter):

            theta = self.maximization(X, tau, *theta)            
            ll, tau = self.expectation(X, *theta)

            converged, prev_ll, ratio = self._check_convergence(prev_ll, ll)

            if converged:
                break

        else:
            if self.verbose >= 0:
                logger.warning(f"Convergence not achieved after at least "\
                               f"{self.max_iter} iterations ({ratio} > {self.tol})."\
                               f" Consider increasing the maximum number of iterations.")

        # Make A.T @ A = I
        pi, A, xi, omega, psi = theta

        AL = linalg.cholesky(A.T @ A)
        A = A @ linalg.solve(AL, np.eye(self.n_latent_factors))
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

        N, D = np.atleast_2d(X).shape
        log_likelihood, tau = self.expectation(X, *theta)
        return np.log(N) * self.number_of_parameters(D) - 2 * log_likelihood


    def sample(self, n_samples=1):
        r"""
        Generate random samples from the fitted model.

        :param n_samples: [optional]
            Number of samples to generate. Defaults to 1.

        # TODO: return docs
        """

        pi, A, xi, omega, psi = self.theta_

        # Draw which component it is from.
        taus = np.random.choice(self.n_components, size=n_samples, p=pi)

        # Draw from the latent space and project to real space.
        N, D = (n_samples, psi.size)

        X = np.empty((N, D))
        for k in range(self.n_components):

            match = (tau == k)
            S = sum(match)
            scores = np.random.multivariate_normal(xi.T[k], omega.T[k], size=S)

            # Project to real space.
            X[match] = A @ scores

        # Add noise.
        X += np.random.multivariate_normal(np.zeros((1, D)), np.diag(psi), N)

        # TODO
        raise NotImplementedError("nope")


def _initial_assignments(X, n_components, n_kmeans_init):
    r"""
    Estimate the initial assignments of each sample to each component in the
    mixture.

    :param X:
        The data, which is expected to be an array with shape [n_samples,
        n_features].

    :param n_components:
        The number of components (clusters) in the mixture.

    :param n_kmeans_init:
        The number of initializations to run using k-means algorithm.

    :returns:
        An array of shape [n_kmeans_init, n_samples] with initial assignments, 
        where each integer entry in the matrix indicates which component the 
        sample is to be initialised to.
    """

    assignments = np.empty((n_kmeans_init, X.shape[0]))
    for i in range(n_kmeans_init):
        assignments[i] = KMeans(n_components).fit(X).labels_

    # Check for duplicate initial assignments. Don't allow them.
    return np.atleast_2d(np.vstack({tuple(a) for a in assignments}).astype(int))


def _initial_parameters(X, n_components, n_latent_factors, assignments,
                        n_svd_max=1000):
    r"""
    Estimate the initial parameters for a model with a mixture of common factor
    analyzers.

    :param X:
        The data, which is expected to be an array with shape [n_samples, 
        n_features].

    :param n_components:
        The number of components (clusters) in the mixture.

    :param n_latent_factors:
        The number of common latent factors.
    
    :param assignments:
        An array of size [n_samples, ] that indicates the initial assignments
        of each sample to each component.

    :param n_svd_max: [optional]
        The number of samples to use when initialising the latent factors.

    :returns:
        A five-length tuple containing: (0) the initial weights; (1) the common
        factor loadings, (2) the initial means in latent space, (3) the initial
        covariance matrices in latent space, and (4) the variance in each data
        dimension.
    """

    N, D = X.shape
    xi = np.empty((n_latent_factors, n_components))
    omega = np.empty((n_latent_factors, n_latent_factors, n_components))
    pi = np.empty(n_components)

    n_svd_max = N if n_svd_max < 0 or n_svd_max > N else int(n_svd_max)
    idx = np.random.choice(N, n_svd_max, replace=False)
    U, S, V = linalg.svd(X[idx].T/np.sqrt(N - 1))
    A = U[:, :n_latent_factors]

    for i in range(n_components):
        match = (assignments == i)
        pi[i] = float(sum(match)) / N

        xs = X[match] @ A
        xi[:, i] = np.mean(xs, axis=0)
        omega[:, :, i] = np.cov(xs.T)

    small_w = S[D - n_latent_factors:]
    psi = np.ones(D) * np.nanmean(small_w[small_w > 0]**2)

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
