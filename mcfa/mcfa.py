
""" Mixture of common factor analyzers. """

import inspect
import logging as logger
import numpy as np
import warnings
from copy import deepcopy
from scipy import linalg
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from time import time

try:
    from .mpl_utils import plot_latent_space
except ModuleNotFoundError:
    from mpl_utils import plot_latent_space # TODO don't do this


class MCFA(object):

    def __init__(self, n_components, n_latent_factors, max_iter=500, n_init=5, 
        tol=1e-5, verbose=0, **kwargs):
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
        """

        self.n_components = int(n_components)
        self.n_latent_factors = int(n_latent_factors)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.tol = float(tol)

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
            raise ValueError("there are more factors than dimensions ({} >= {})"\
                             .format(self.n_latent_factors, D))
        return X


    def _initial_parameters(self, X):
        """
        Estimate the initial parameters of the model.

        :param X:
            The data, :math:`X`, which is expected to be an array of shape
            [n_samples, n_features].
        """

        # Do initial partitions (either randomly or by k-means).
        assignments = _initial_assignments(X, self.n_components, self.n_init)

        best_model, best_log_likelihood = (None, -np.inf)

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

                # Run E-M from this initial guess.
                model = self.__class__(self.n_components, 
                                       self.n_latent_factors, 
                                       max_iter=self.max_iter,
                                       tol=self.tol,
                                       verbose=-1)

                try:
                    model.fit(X, init_params=params)

                except ValueError:
                    raise
                    if self.verbose > 0:
                        logger.exception("Exception in fitting:")

                else:
                    if model.log_likelihood_ > best_log_likelihood:
                        best_model = model
                        best_log_likelihood = model.log_likelihood_

        return best_model.theta_


    @property
    def parameter_names(self):
        r""" Return the names of the parameters in this model. """
        return tuple(inspect.signature(self.expectation).parameters.keys())[1:]


    def _expectation(self, X, pi, A, xi, omega, psi):
        r"""
        Evaluate the conditional expectation of the complete-data log-likelihood
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

        # calculate constant terms.
        log_prob_constants = -0.5 * (np.sum(np.log(psi)) + D * np.log(2*np.pi))

        psi_eye = np.diag(psi)
        inv_D = np.diag(1.0/psi)
        inv_D_A = inv_D @ A

        I = np.eye(self.n_latent_factors)
        log_prob = np.zeros((N, self.n_components))
        

        for i in range(self.n_components):

            try:
                W = linalg.solve(omega[:, :, i], I) + inv_D_A.T @ A
                inv_O = linalg.solve(W, I)
                inv_S = inv_D - inv_D_A @ inv_O @ inv_D_A.T
        
            except linalg.LinAlgError:
                raise linalg.LinAlgError(
                    "ill-conditioned or singular Sigma[:, :, {}]".format(i))


            diff = X - (A @ xi[:, i])
            dist = (diff @ inv_S @ diff.T)
            dist = np.diag(dist)
           
            with warnings.catch_warnings():
                if 1 > self.verbose:
                    warnings.simplefilter("ignore")

                logdetD = np.log(linalg.det(omega[:, :, i])) \
                        - np.log(linalg.det(inv_O))


            log_prob[:, i] = -0.5 * dist - 0.5 * logdetD + log_prob_constants




        weighted_log_prob = log_prob + np.log(pi)
        log_likelihood = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            log_tau = weighted_log_prob - log_likelihood[:, np.newaxis]

        tau = np.exp(log_tau)

        #_, __ = self._expectation(X, pi, A, xi, omega, psi)
        
        #print("expectation_slow: {:.2f}s {}".format(np.ptp(times), np.diff(times)))
        return (sum(log_likelihood), tau)



    def expectation(self, X, pi, A, xi, omega, psi):
        r"""
        Evaluate the conditional expectation of the complete-data log-likelihood
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

            # Use matrix determinant lemma:
            # det(A + U @V.T) = det(I + V.T @ A^-1 @ U) * det(A)
            # here A = omega^{-1}; U = (\psi^-1 @ A).T; V = A.T
            
            # and making use of det(A) = 1.0/det(A^-1)

            with warnings.catch_warnings():
                if 1 > self.verbose: warnings.simplefilter("ignore")

                # log(det(omega)) + log(det(I + A @ omega @ U)/det(omega)) \
                # = log(det(I + A @ omega @ U))
                _, logdetD = np.linalg.slogdet(I_D + A @ omega_ @ U)

            mu = A @ xi_
            cov = A @ omega_ @ A.T + I_psi

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
        Evaluate the updated estimates of the model parameters given the data,
        the responsibility matrix :math:`\tau` and the current estimates of the
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

        N, D = X.shape

        inv_D = np.diag(1.0/psi)

        A1 = np.zeros((D, self.n_latent_factors))
        A2 = np.zeros((self.n_latent_factors, self.n_latent_factors))
        Di = np.zeros(D)

        inv_D_A = inv_D @ A

        I = np.eye(self.n_latent_factors)

        for i in range(self.n_components):

            W = linalg.solve(omega[:, :, i], I)
            C = linalg.solve(W + A.T @ inv_D_A, I)
            gamma = (inv_D - inv_D_A @ C @ inv_D_A.T) @ A @ omega[:, :, i]

            ti = np.sum(tau[:, i])
            xi_ = np.copy(xi[:, [i]])

            tY = X * tau[:, [i]]
            Y_Axi_i = X.T - A @ xi_
            tY_Axi_i = Y_Axi_i * tau[:, [i]].T

            xi[:, i] += gamma.T @ (np.sum(tY_Axi_i, axis=1) / ti)

            diff = (xi_ - xi[:, [i]])

            omega[:, :, i] = (I - gamma.T @ A) @ omega[:, :, i] \
                           + gamma.T @ Y_Axi_i @ tY_Axi_i.T @ gamma / ti \
                           - diff @ diff.T

            A1 += np.atleast_2d(np.sum(tY, axis=0)).T @ xi_.T \
                + X.T @ tY_Axi_i.T @ gamma
            A2 += (omega[:, :, i] + xi[:, [i]] @ xi[:, [i]].T) * ti
            Di += np.sum(X * tY, axis=0)

            pi[i] = ti / N

        A = A1 @ linalg.solve(A2, I)
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


    def fit(self, X, slow=False, init_params=None):
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

        X = self._check_data(X)
        theta = self._initial_parameters(X) if init_params is None \
                                            else deepcopy(init_params) 

        # Calculate initial log-likelihood.
        prev_ll, tau = self.expectation(X, *theta)

        # Do E-M iterations.

        for i in range(self.max_iter):
            ta = time()

            theta = self.maximization(X, tau, *theta)

            tb = time()

            if slow:
                ll, tau = self._expectation(X, *theta)
            else:
                ll, tau = self.expectation(X, *theta)

            tc = time()
            print("i/{}: expectation {:.1e} maximisation {:.1e} ll {:.1e}".format(
                i, tb - ta, tc - tb, ll))

            converged, prev_ll, ratio = self._check_convergence(prev_ll, ll)

            if converged:
                break

        else:
            if self.verbose >= 0:
                logger.warning("Convergence not achieved after more than {} "
                               "iterations ({:.1e} > {:.1e}). Consider "
                               "increasing the maximum number of iterations."\
                               .format(self.max_iter, ratio, self.tol))

        # Make A.T @ A = I
        pi, A, xi, omega, psi = theta

        CH = linalg.cholesky(A.T @ A)
        A = A @ linalg.solve(CH, np.eye(self.n_latent_factors))
        xi = CH @ xi

        for i in range(self.n_components):
            omega[:, :, i] = CH @ omega[:, :, i] @ CH.T

        self.tau_ = tau
        self.theta_ = [pi, A, xi, omega, psi]
        self.log_likelihood_ = ll

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


    def plot_latent_space(self, X, **kwargs):
        r"""
        Plot the latent space.
        """
        return plot_latent_space(self, X, **kwargs)






def _initial_assignments(X, n_components, n_init):
 
    assignments = np.empty((n_init, X.shape[0]))
    for i in range(n_init):
        assignments[i] = KMeans(n_components).fit(X).labels_

    # Check for duplicate initial assignments. Don't allow them.
    return np.unique(assignments, axis=0).astype(int)


def _initial_parameters(X, n_components, n_latent_factors, partition):

    N, D = X.shape
    xi = np.empty((n_latent_factors, n_components))
    omega = np.empty((n_latent_factors, n_latent_factors, n_components))
    pi = np.empty(n_components)

    U, S, V = linalg.svd(X.T/np.sqrt(N - 1))
    A = U[:, :n_latent_factors]

    for i in range(n_components):
        match = (partition == i)
        pi[i] = float(sum(match)) / N

        uiT = X[match] @ A
        xi[:, i] = np.mean(uiT, axis=0)
        omega[:, :, i] = np.cov(uiT.T)

    small_w = S[D - n_latent_factors:]
    psi = np.ones(D) * np.nanmean(small_w[small_w > 0]**2)

    return (pi, A, xi, omega, psi)


def _factor_scores(X, tau, pi, A, xi, omega, psi):

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

    :returns:
        The Cholesky decomposition of the precision of the covariance matrix.
    """

    D, _ = cov.shape

    try:
        cholesky_cov = linalg.cholesky(cov, lower=True)

    except linalg.LinAlgError:
        raise linalg.LinAlgError("failed to do Cholesky decomposition")

    # Return the precision matrix.
    return linalg.solve_triangular(cholesky_cov, np.eye(D), lower=True).T




if __name__ == "__main__":

    
    from sklearn.datasets import load_iris
    
    X = load_iris().data

    model = MCFA(n_components=3, n_latent_factors=2)
    model.fit(X)


    fig = model.plot_latent_space(X)
