
""" Mixture of common factor analyzers. """

import inspect
import logging as logger
import numpy as np
import warnings
from copy import deepcopy
from scipy.special import logsumexp
from sklearn.cluster import KMeans


class MCFA(object):

    def __init__(self, n_components, n_latent_factors, max_iter=500, n_init=5, 
        n_random_init=1, init_method=None, tol=1e-5, verbose=0, **kwargs):
        """
        A mixture of common factor analyzers model.

        :param n_components:
            The number of components (clusters) in the mixture.

        :param n_latent_factors:
            The number of common latent factors.

        :param max_iter: [optional]
            The maximum number of expectation-maximization iterations.

        :param n_init: [optional]
            The number of initialisations to run using the k-means algorithm.

        :param n_random_init: [optional]
            The number of random initialisations to run.

        :param init_method: [optional]
            The initialisation method(s) to use. Available options are 'eigen-A'
            or 'rand-A'.

        :param tol: [optional]
            Relative tolerance before declaring convergence.

        :param verbose: [optional]
            Show warning messages.
        """

        self.n_components = int(n_components)
        self.n_latent_factors = int(n_latent_factors)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.n_random_init = int(n_random_init)
        self.tol = float(tol)

        if self.n_latent_factors < 1:
            raise ValueError("n_latent_factors must be a positive integer")

        if self.n_components < 1:
            raise ValueError("n_components must be a positive integer")

        if self.max_iter < 1:
            raise ValueError("number of iterations must be greater than one")

        if self.n_init < 1:
            raise ValueError("n_init must be a positive integer")

        if self.n_random_init < 1:
            raise ValueError("n_random_init must be a positive integer")

        if self.tol <= 0:
            raise ValueError("tol must be greater than zero")

        methods = ("eigen-A", "rand-A", )
        if init_method is None:
            init_method = methods

        else:
            for method in methods:
                if method.lower().startswith(init_method.lower()):
                    init_method = (method, )
                    break
            else:
                raise ValueError("init_method must be in {}".format(methods))

        self.init_method = init_method
        self.verbose = int(verbose)
        return None


    def _check_data(self, X):
        """ 
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


    @property
    def parameter_names(self):
        """ Return the names of the parameters in this model. """
        args = inspect.getargspec(self.expectation).args
        return tuple([arg for arg in args if arg not in ("self", "X")])


    def expectation(self, X, pi, A, xi, omega, psi):
        """
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

        :raises np.linalg.LinAlgError:
            If the covariance matrix of any mixture component in latent space
            is ill-conditioned or singular.

        :returns:
            A two-length tuple containing the sum of the log-likelihood for the
            data given the model, and the responsibility matrix :math:`\tau`
            giving the partial associations between each data point and each
            component in the mixture.
        """

        N, D = X.shape

        with warnings.catch_warnings():
            if 1 > self.verbose:
                warnings.simplefilter("ignore")
            _, slogdet_psi = np.linalg.slogdet(psi)

        inv_D = np.diag(1.0/np.diag(psi))
        inv_D_A = inv_D @ A

        I = np.eye(self.n_latent_factors)
        log_prob = np.zeros((N, self.n_components))
        

        for i in range(self.n_components):

            try:
                C = np.linalg.solve(omega[:, :, i], I)
                W = C + inv_D_A.T @ A
                inv_O = np.linalg.solve(W, I)

                inv_S = inv_D - inv_D_A @ inv_O @ inv_D_A.T
        
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError(
                    "ill-conditioned or singular Sigma[:, :, {}]".format(i))

            with warnings.catch_warnings():
                if 1 > self.verbose:
                    warnings.simplefilter("ignore")

                logdetD = np.log(np.linalg.det(omega[:, :, i])) \
                        + slogdet_psi - np.log(np.linalg.det(inv_O))

            diff = X - (A @ xi[:, i])
            dist = np.diag(diff @ inv_S @ diff.T)

            log_prob[:, i] = - 0.5 * dist - 0.5 * D * np.log(2 * np.pi) \
                             - 0.5 * logdetD

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

        inv_D = np.diag(1.0/np.diag(psi))

        A1 = np.zeros((D, self.n_latent_factors))
        A2 = np.zeros((self.n_latent_factors, self.n_latent_factors))
        Di = np.zeros(D)

        inv_D_A = inv_D @ A

        I = np.eye(self.n_latent_factors)

        for i in range(self.n_components):

            W = np.linalg.solve(omega[:, :, i], I)
            C = np.linalg.solve(W + A.T @ inv_D_A, I)
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

        A = A1 @ np.linalg.solve(A2, I)
        psi = np.diag(Di - np.sum((A @ A2) * A, axis=1)) / N

        return (pi, A, xi, omega, psi)


    def fit(self, X, init_params=None):
        """
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

            theta = self.maximization(X, tau, *theta)

            ll, tau = self.expectation(X, *theta)

            converged, prev_ll, ratio = self._check_convergence(prev_ll, ll)

            if converged:
                break

        # Make A.T @ A = I
        pi, A, xi, omega, psi = theta

        CH = np.linalg.cholesky(A.T @ A)
        A = A @ np.linalg.solve(CH, np.eye(self.n_latent_factors))
        xi = CH @ xi

        for i in range(self.n_components):
            omega[:, :, i] = CH @ omega[:, :, i] @ CH.T

        self.tau_ = tau
        self.theta_ = [pi, A, xi, omega, psi]
        self.log_likelihood_ = ll

        return self


    def factor_scores(self, X):
        """
        Estimate the posterior factor scores given the model parameters.

        :param X:
            The data, :math:`X`, which is expected to be an array of shape
            [n_samples, n_features].

        # TODO: Consider taking model parameters.
        # TODO: What should we return, exactly?
        """

        try:
            pi, A, xi, omega, psi = self.theta_

        except AttributeError:
            raise AttributeError("you must run fit() first")

        N, D = X.shape
        F, C = (self.n_latent_factors, self.n_components)

        U = np.zeros((N, F, C))
        gamma = np.zeros((D, F, C))

        inv_D = np.diag(1.0/np.diag(psi))

        I = np.eye(self.n_latent_factors)

        for i in range(self.n_components):

            C = np.linalg.solve(np.linalg.solve(omega[:, :, i], I) \
                                + A.T @ inv_D @ A, I)
            gamma[:, :, i] = (inv_D - inv_D @ A @ C @ A.T @ inv_D) \
                           @ A @ omega[:, :, i]
            U[:, :, i] = np.repeat(xi[:, [i]], N).reshape((F, N)).T \
                       + (X - (A @ xi[:, [i]]).T) @ gamma[:, :, i]

        cluster = np.argmax(self.tau_, axis=1)

        UC = np.zeros((N, F))
        Umean = np.zeros((N, F))

        for i in range(N):
            UC[i] = U[i, :, cluster[i]]
            Umean[i] = self.tau_[i] @ U[i].T

        return (U, UC, Umean)
        

    def _check_convergence(self, previous, current):
        """
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



    def _initial_parameters(self, X):
        """
        Estimate the initial parameters of the model.

        :param X:
            The data, :math:`X`, which is expected to be an array of shape
            [n_samples, n_features].
        """

        # Do initial partitions (either randomly or by k-means).
        initial_partitions = _initial_partitions(X, 
                                                 self.n_components, 
                                                 self.n_init, 
                                                 self.n_random_init)

        best_model, best_log_likelihood = (None, -np.inf)

        for i, partition in enumerate(initial_partitions):
    
            for j, init_method in enumerate(self.init_method):

                try:
                    params = _initial_parameters(X, init_method,
                                                 self.n_components, 
                                                 self.n_latent_factors,
                                                 partition)

                except (ValueError, AssertionError):
                    if self.verbose > 0:
                        logger.exception("Exception in initializing {} with {}:"\
                                         .format(i, init_method))

                else:

                    # Run E-M from this initial guess.
                    model = self.__class__(self.n_components, 
                                           self.n_latent_factors, 
                                           tol=self.tol, max_iter=self.max_iter)

                    try:
                        model.fit(X, init_params=params)

                    except (ValueError, AssertionError):
                        if self.verbose > 0:                        
                            logger.exception("Exception in fitting {} using {}"\
                                             .format(i, init_method))

                    else:
                        if model.log_likelihood_ > best_log_likelihood:
                            best_model = model
                            best_log_likelihood = model.log_likelihood_

        return best_model.theta_


def _initial_partitions(X, n_components, n_init, n_random_init, n_checks=10):

    N, D = X.shape
    n_partitions = n_random_init + n_init

    partitions = np.empty((n_partitions, N))

    for i in range(n_init):
        partitions[i] = KMeans(n_components).fit(X).labels_

    partitions[n_init:] = np.random.choice(np.arange(n_components),
                                           size=(n_random_init, N))

    # TODO: Check for duplicate initial partitions. Don't allow them.
    for i in range(n_checks):
        unique_partitions = np.unique(partitions, axis=0)

        u_partitions, _ = unique_partitions.shape
        if u_partitions < n_partitions:
            # Fill up the remainder with random partitions.
            k = n_partitions - u_partitions
            partitions = np.vstack([
                unique_partitions,
                np.random.choice(np.arange(n_components), size=(k, N))
            ])
            
        else:
            break

    partitions = np.unique(partitions, axis=0).astype(int)
    if partitions.shape[0] < n_partitions:
        logger.warn("Duplicate initial partitions exist after {} trials"\
                    .format(n_checks))

    return partitions



def _initial_parameters(X, method, *args):

    func = dict([
        ("rand-A", _initial_parameters_by_rand_a),
        ("eigen-A", _initial_parameters_by_eigen_a),
        #("gmf", _initial_parameters_by_gmf),
    ]).get(method, None)

    if func is None:
        raise ValueError("unknown initialisation method '{}'".format(method))

    return func(X, *args)


def _initial_parameters_by_rand_a(X, n_components, n_latent_factors, partition):

    N, D = X.shape
    xi = np.empty((n_latent_factors, n_components))
    omega = np.empty((n_latent_factors, n_latent_factors, n_components))
    pi = np.empty(n_components)

    A = np.random.normal(0, 1, size=(D, n_latent_factors))
    C = np.linalg.cholesky(A.T @ A).T
    A = A @ np.linalg.solve(C, np.eye(n_latent_factors))

    psi = np.zeros(D, dtype=float)
    for i in range(n_components):
        match = (partition == i)
        psi += (sum(match) - 1) * np.var(X[match], axis=0) / (N - n_components)
        
    _ = np.sqrt(psi)

    psi = np.diag(psi)
    sqrt_psi = np.diag(_)
    inv_sqrt_psi = np.diag(1.0/_)

    for i in range(n_components):
        match = (partition == i)

        pi[i] = float(sum(match)) / N
        xi[:, i] = np.mean(X[match] @ A, axis=0)
        
        # w, v = lambda, H
        w, v = np.linalg.eigh(inv_sqrt_psi @ np.cov(X[match].T) @ inv_sqrt_psi)

        g = D - n_latent_factors
        if n_latent_factors == D:
            var = 0

        else:
            small_w = w[:g]
            # Only take finite and non-zero eigenvalues
            var = np.nanmean(small_w[small_w > 0])

        omega[:, :, i] = A.T @ sqrt_psi @ v[:, :g] @ np.diag(w[:g] - var) \
                       @ v[:, :g].T @ sqrt_psi @ A

    return (pi, A, xi, omega, psi)


def _initial_parameters_by_eigen_a(X, n_components, n_latent_factors, partition):

    N, D = X.shape
    xi = np.empty((n_latent_factors, n_components))
    omega = np.empty((n_latent_factors, n_latent_factors, n_components))
    pi = np.empty(n_components)

    U, S, V = np.linalg.svd(X.T/np.sqrt(N - 1))
    A = U[:, :n_latent_factors]

    for i in range(n_components):
        match = (partition == i)
        pi[i] = float(sum(match)) / N

        uiT = X[match] @ A
        xi[:, i] = np.mean(uiT, axis=0)
        omega[:, :, i] = np.cov(uiT.T)

    small_w = S[D - n_latent_factors:]
    psi = np.diag(np.ones(D) * np.nanmean(small_w[small_w > 0]**2))

    return (pi, A, xi, omega, psi)