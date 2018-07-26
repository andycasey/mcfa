
import logging as logger # todo: don't be lazy
import numpy as np
from scipy.special import logsumexp

from sklearn.cluster import KMeans

        
class MCFA(object):

    def __init__(self, g, q, itmax=10000, nkmeans=5, nrandom=20, tol=1e-5,
        init_method=None, conv_measure="diff", warn_messages=True, **kwargs):

        self.g, self.q = (int(g), int(q))
        self.itmax = int(itmax)
        self.nkmeans = int(nkmeans)
        self.nrandom = int(nrandom)
        self.tol = float(tol)

        if self.q < 1:
            raise ValueError("q must be a positive integer")

        if self.g < 1:
            raise ValueError("p must be a positive integer")

        if self.itmax < 1:
            raise ValueError("max number of iterations must be greater than one")

        if self.nkmeans < 1:
            raise ValueError("nkmeans must be a positive integer")

        if self.nrandom < 1:
            raise ValueError("nrandom must be a positive integer")

        if self.tol < 0:
            raise ValueError("tol must be greater than zero")

        methods = ("eigen-A", "rand-A", )#"gmf")
        init_method = _validate_str_input("init_method", init_method, methods,
                                          allow_none=True)

        self.init_method = methods if init_method is None else (init_method, )

        self.conv_measure = _validate_str_input("conv_measure", conv_measure,
            ("diff", "ratio"))

        self.warn_messages = bool(warn_messages)

        return None


    def fit(self, Y, init_params=None, init_clust=None):

        Y = _validate_data_array(Y, self.q)

        if init_params is None:

            # Get initial parameters.
            init_params = self._initial_parameters(Y)

            raise a


        theta = _unpack(**init_params)

        # Calculate initial log-likelihood.
        prev_ll, tau = _mcfa_expectation(Y, *theta)

        print("initial {}".format(prev_ll))

        # Do E-M iterations.
        for i in range(self.itmax):

            # Do M-step.
            theta = _mcfa_maximization(Y, tau, *theta)

            # Do E-step.
            ll, tau = _mcfa_expectation(Y, *theta)

            prev_ll, converged = self._check_convergence(prev_ll, ll)

            if converged:
                break

            

        raise a


    def _check_convergence(self, previous, current):
        if self.conv_measure == "diff":
            value = current - previous
        elif self.conv_measure:
            value = (current - previous)/current

        print(value)
        return (current, value < self.tol)



    def _initial_parameters(self, Y):

        # Do initial partitions.
        initial_partitions = _initial_partitions(Y, self.g,
                                                 self.nkmeans, self.nrandom)

        
        for i, partition in enumerate(initial_partitions):

            for j, init_method in enumerate(self.init_method):

                try:
                    params = _initial_parameters(Y, self.g, self.q,
                                                 partition, init_method)

                except:
                    logger.exception("Exception in initial trial {} using {}:"\
                                     .format(i, init_method))
                    raise

                else:

                    # Run E-M.
                    model = self.__class__(self.g, self.q, 
                                           tol=self.tol, 
                                           itmax=self.itmax,
                                           conv_measure=self.conv_measure)
                    model.fit(Y, init_params=params)



        raise a



        p, n = Y.shape


_pack_order = ("g", "q", "pi", "A", "xi", "omega", "D")
def _unpack(**param_dict):
    return [param_dict[k] for k in _pack_order]


def _pack(*param_args):
    return dict(zip(_pack_order, param_args))




def _initial_partitions(Y, g, nkmeans, nrandom, n_duplicate_checks=10):

    n, p = Y.shape
    n_partitions = nrandom + nkmeans

    partitions = np.empty((n_partitions, n))

    for i in range(nkmeans):
        partitions[i] = KMeans(g).fit(Y).labels_

    partitions[nkmeans:] = np.random.choice(np.arange(g), size=(nrandom, n))

    # TODO: Check for duplicate initial partitions. Don't allow them.
    for i in range(n_duplicate_checks):
        unique_partitions = np.unique(partitions, axis=0)

        u_partitions, _ = unique_partitions.shape

        if u_partitions < n_partitions:

            # Fill up the remainder with random partitions.
            k = n_partitions - u_partitions
            partitions = np.vstack([
                unique_partitions,
                np.random.choice(np.arange(g), size=(k, n))
            ])
            
        else:
            break

    partitions = np.unique(partitions, axis=0).astype(int)
    if partitions.shape[0] < n_partitions:
        logger.warn("Duplicate initial partitions exist after {} trials"\
                    .format(n_duplicate_checks))

    return partitions



def _initial_parameters(Y, g, q, partitions, method):

    func = dict([
        ("rand-A", _initial_parameters_by_rand_a),
        ("eigen-A", _initial_parameters_by_eigen_a),
        #("gmf", _initial_parameters_by_gmf),
    ]).get(method, None)

    if func is None:
        raise ValueError("unknown initialisation method '{}'".format(method))

    return func(Y, g, q, partitions)


def _initial_parameters_by_rand_a(Y, g, q, partition):

    n, p = Y.shape
    xi = np.empty((q, g))
    omega = np.empty((q, q, g))
    pi = np.empty(g)

    A = np.random.normal(0, 1, size=(p, q))
    C = np.linalg.cholesky(A.T @ A).T
    A = A @ np.linalg.solve(C, np.eye(q))

    D = np.zeros(p, dtype=float)
    for i in range(g):
        match = (partition == i)
        D += (sum(match) - 1) * np.var(Y[match], axis=0) / (n - g)
        
    _ = np.sqrt(D)

    D = np.diag(D)
    sqrt_D = np.diag(_)
    inv_sqrt_D = np.diag(1.0/_)

    for i in range(g):
        match = (partition == i)

        pi[i] = float(sum(match)) / n
        xi[:, i] = np.mean(Y[match] @ A, axis=0)
        
        # w, v = lambda, H
        w, v = np.linalg.eigh(inv_sqrt_D @ np.coy(Y[match].T) @ inv_sqrt_D)

        if q == p:
            var = 0

        else:
            small_w = w[:p - q]
            # Only take finite and non-zero eigenvalues
            var = np.nanmean(small_w[small_w > 0])

        # FUTURE ANDY: If there is a bug here it might be due to the ordering of
        #              the eigenvalues. R orders them in the opposite order, and
        #              we are using p-q to get the last number of elements.
        #              This was tested for a set of p, q but unit test++

        omega[:, :, i] = A.T @ sqrt_D @ v[:, :p - q] @ np.diag(w[:p - q] - var) \
                       @ v[:, :p - q].T @ sqrt_D @ A

    return dict(g=g, q=q, pi=pi, A=A, xi=xi, omega=omega, D=D)


def _initial_parameters_by_eigen_a(Y, g, q, partition):

    # Not tested.

    n, p = Y.shape
    xi = np.empty((q, g))
    omega = np.empty((q, q, g))
    pi = np.empty(g)

    U, S, V = np.linalg.svd(Y.T/np.sqrt(n - 1))
    A = U[:, :q]

    for i in range(g):
        match = (partition == i)
        pi[i] = float(sum(match)) / n

        uiT = Y[match] @ A
        xi[:, i] = np.mean(uiT, axis=0)
        omega[:, :, i] = np.cov(uiT.T)


    small_w = S[p - q:]
    D = np.diag(np.ones(p) * np.nanmean(small_w[small_w > 0]**2))

    return dict(g=g, q=q, pi=pi, A=A, xi=xi, omega=omega, D=D)






def _mcfa_expectation(Y, g, q, pi, A, xi, omega, D):

    n, p = Y.shape

    log_prob = np.zeros((n, g))

    inv_D = np.diag(1.0/np.diag(D))

    inv_D_A = inv_D @ A

    I = np.eye(q)
        
    for i in range(g):

        try:
            C = np.linalg.solve(omega[:, :, i], I)
            W = C + inv_D_A.T @ A
            inv_O = np.linalg.solve(W, I)

            inv_S = inv_D - inv_D_A @ inv_O @ inv_D_A.T
    
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(
                "ill-conditioned or singular Sigma[:, :, {}]".format(i))

    
        logdetD = np.log(np.linalg.det(omega[:, :, i])) \
                + np.linalg.slogdet(D)[1] \
                - np.log(np.linalg.det(inv_O))

        diff = Y - (A @ xi[:, i])
        dist = np.diag(diff @ inv_S @ diff.T)

        log_prob[:, i] = -0.5 * dist - 0.5 * p * np.log(2 * np.pi) - 0.5 * logdetD


    """
    # This is the other way, but it is less (numerically) stable (I think)
    Fji = Fji + np.log(pi)
    Fjmax = np.max(Fji, axis=1)
    Fji_new = Fji - Fjmax[:, np.newaxis]

    ln_likelihood = np.sum(Fjmax) + logsumexp(Fji_new)
    """


    weighted_log_prob = log_prob + np.log(pi)
    log_likelihood = logsumexp(weighted_log_prob, axis=1)
    with np.errstate(under="ignore"):
        log_tau = weighted_log_prob - log_likelihood[:, np.newaxis]

    tau = np.exp(log_tau)

    return (sum(log_likelihood), tau)



def _mcfa_maximization(Y, tau, g, q, pi, A, xi, omega, D):

    n, p = Y.shape

    inv_D = np.diag(1.0/np.diag(D))
    assert len(inv_D.shape) == 2

    A1 = np.zeros((p, q))
    A2 = np.zeros((q, q))
    Di = np.zeros(p)

    inv_D_A = inv_D @ A

    I = np.eye(q)

    for i in range(g):

        C = np.linalg.solve(np.linalg.solve(omega[:, :, i], I) + A.T @ inv_D_A, I)
        gamma = (inv_D - inv_D_A @ C @ inv_D_A.T) @ A @ omega[:, :, i]

        ti = np.sum(tau[:, i])

        xi_ = np.copy(xi[:, [i]])

        tY = Y * tau[:, [i]]
        Y_Axi_i = Y.T - A @ xi_
        tY_Axi_i = Y_Axi_i * tau[:, [i]].T

        xi[:, i] += gamma.T @ (np.sum(tY_Axi_i, axis=1) / ti)

        diff = (xi_ - xi[:, [i]])

        omega[:, :, i] = (I - gamma.T @ A) @ omega[:, :, i] \
                       + gamma.T @ Y_Axi_i @ tY_Axi_i.T @ gamma / ti \
                       - diff @ diff.T

        #A1 += np.sum(tY, axis=0) @ xi_.T + Y.T @ tY_Axi_i @ gamma
        A1 += np.atleast_2d(np.sum(tY, axis=0)).T @ xi_.T + Y.T @ tY_Axi_i.T @ gamma
        A2 += (omega[:, :, i] + xi[:, [i]] @ xi[:, [i]].T) * ti
        Di += np.sum(Y * tY, axis=0)

        pi[i] = ti / n


    A = A1 @ np.linalg.solve(A2, I)

    D = np.diag(Di - np.sum((A @ A2) * A, axis=1)) / n

    return _unpack(g=g, q=q, pi=pi, A=A, xi=xi, omega=omega, D=D)




    # This is in mstep_mcfa.R but I think it is wrong because it is already
    # done just above
    # inv_D = np.diag(1 / np.diag(D))

    raise a
    #inv_D_A = 



def _tau(Y, g, q, pi, A, xi, omega, D):

    n, p = Y.shape

    Fji = np.zeros((n, ))





def _validate_data_array(Y, q):

    Y = np.atleast_2d(Y)
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y has non-finite entries")

    n, p = Y.shape

    if p <= q:
        raise ValueError("the number of factors (q) "\
                         "is less than the number of dimensions (p)")
    return Y


def _validate_str_input(descriptor, input_value, acceptable_inputs, 
                        allow_none=False):

    for acceptable_input in acceptable_inputs:
        if input_value is None and allow_none:
            break

        elif input_value is not None \
        and acceptable_input.lower().startswith(input_value.lower()):
            input_value = acceptable_input
            break

    else:
        raise ValueError("{} must be in: {} (not {})".format(
            descriptor, acceptable_inputs, input_value))

    return input_value




if __name__ == "__main__":
    from sklearn.datasets import load_iris

    iris = load_iris()


    model = MCFA(g=3, q=2)
    model.fit(iris.data)

    raise a