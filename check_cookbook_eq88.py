

# Check eq 88 of matrix cookbook
import numpy as np
from scipy.optimize import check_grad
from sklearn.utils import check_random_state

from mcfa.mcfa import _initial_factor_loads_by_random, _initial_components_by_random, _factor_scores, _expectation

N = 100
D = 15
J = 3
K = 2

random_seed = 0
np.random.seed(random_seed)
random_state = check_random_state(random_seed)




X = np.random.uniform(-1, 1, size=(D, N))
A = np.random.uniform(-1, 1, size=(D, J))
s = np.random.uniform(-1, 1, size=(J, N))

W = np.diag(np.random.uniform(0, 1, size=D))

def func(X, A, s, W, idx):
    diff = X[:, [idx]] - A @ s[:, [idx]]
    return -0.5 * diff.T @ W @ diff

def grad(X, A, s, W, idx):
    diff = X[:, [idx]] - A @ s[:, [idx]]
    return W @ diff @ s[:, [idx]].T



def f(A, idx=0):
    return func(X, A.reshape((D, J)), s, W, idx=idx)

def g(A, idx=0):
    return grad(X, A.reshape((D, J)), s, W, idx=idx).flatten()

foo = check_grad(f, g, A.flatten())


#bar = check_grad(f2, g2, A.flatten())

A = _initial_factor_loads_by_random(X.T, J, random_state)
pi, xi, omega = _initial_components_by_random(X.T, A, K, random_state)
psi = np.random.uniform(0.5, 1.5, size=D)

s = _factor_scores(X.T, pi, A, xi, omega, psi)[1].T

def f_p1(A, n=0, k=0):
    #S = s[:, [n]]
    S = xi[:, [k]]
    diff = X[:, [n]] - A @ S
    O = omega[:, :, k]
    return -0.5 * diff.T @ np.linalg.inv(A @ O @ A.T + np.diag(psi)) @ diff


def g_p1(A, n=0, k=0):

    D_ = np.diag(psi)
    #S = s[:, [n]]
    S = xi[:, [k]]

    T0 = X[:, [n]] - A @ S
    O = omega[:, :, k]
    T1 = np.linalg.inv(D_ + A @ O @ A.T)
    T2 = np.linalg.inv(D_ + A @ O.T @ A.T)
    T3 = T2 @ T0
    T4 = T1 @ T0

    # T3 === T4
    # T1 === T2

    return 0.5 * (T3 @ S.T + T4 @ T0.T @ T1 @ A @ O.T + T3 @ T0.T @ T2 @ A @ O + T4 @ S.T)



bar = check_grad(lambda _: f_p1(_.reshape((D, J))).flatten(),
                 lambda _: g_p1(_.reshape((D, J))).flatten(),
                 A.flatten())

def f_p2(A, n=0, k=0):
    O = omega[:, :, k]
    T0 = np.diag(1.0/psi)
    return -0.5 * np.linalg.slogdet(np.eye(D) + A @ O @ (T0 @ A).T)[1]

def g_p2(A, n=0, k=0):
    O = omega[:, :, k]
    T0 = np.diag(1.0/psi)
    #return -A.T @ np.linalg.inv(np.eye(D) + T0 @ A @ O.T @ A.T) @ T0 @ A    

    T1 = np.eye(D) + T0.T @ A @ O.T @ A.T
    T2 = np.eye(D) + A @ O @ A.T @ T0

    # T1 === T2.T
    return -0.5 * (np.linalg.inv(T1) @ T0.T @ A @ O.T + T0 @ np.linalg.inv(T2) @ A @ O)


bar2 = check_grad(lambda _: f_p2(_.reshape((D, J))).flatten(),
                  lambda _: g_p2(_.reshape((D, J))).flatten(),
                  A.flatten())


def f_all(A, **kwargs):
    constants = - 0.5 * np.sum(np.log(psi)) - 0.5 * D * np.log(2 * np.pi)
    return f_p1(A, **kwargs) + f_p2(A, **kwargs) + constants

def g_all(A, **kwargs):
    return g_p1(A, **kwargs) + g_p2(A, **kwargs)

bar3 = check_grad(lambda _: f_all(_.reshape((D, J))).flatten(),
                  lambda _: g_all(_.reshape((D, J))).flatten(),
                  A.flatten())

def f_actual(A, n=0, k=0):
    ll, tau, lp = _expectation(X.T, pi, A, xi, omega, psi, full_output=True)
    return lp[n, k]

raise a



def my_func(X, A, xi, omega, psi, n=0, k=0):

    D = psi.size
    Sigma = A @ omega[:, :, k] @ A.T + np.eye(D) * psi
    #Sigma = np.eye(D) * psi
    Sigma_inv = np.linalg.inv(Sigma)

    log_prob_constants = -0.5 * np.sum(np.log(psi)) - 0.5 * D * np.log(2*np.pi)


    U = (np.diag(1.0/psi) @ A).T
    logdet_D = np.linalg.slogdet(np.eye(D) + A @ omega[:, :, k] @ U)[1]

    logdet_D = 0
    log_prob_constants = 0

    diff = (X[:, [n]] - A @ xi[:, [k]])
    return - 0.5 * diff.T @ Sigma_inv @ diff \
           - 0.5 * logdet_D + log_prob_constants



def my_grad(X, A, xi, omega, psi, n=0, k=0):

    D = psi.size
    Sigma = A @ omega[:, :, k] @ A.T + np.eye(D) * psi
    #Sigma = np.eye(D) * psi
    Sigma_inv = np.linalg.inv(Sigma)

    diff = (X[:, [n]] - A @ xi[:, [k]])

    # f = (X-As).T @ C^-1 @ (X-As)

    # Let C^-1 = (A @ omega @ A.T + Psi)^-1
    #     C = A @ omega @ A.T + Psi

    # df/dA = (df/dC) * (dC/dA)

    # dC/dA = 2 * omega @ A.T

    # Eq 396:
    # df/dC = -0.5 * (-C^-1 + C^-1 * (X-A @ s) @ (X - A@s).T @ C^-1)
    # (might be some scalar multiple here because f is missing -0.5 at front)


    return Sigma_inv @ diff @ xi[:, [k]].T


# Eq 52:
def C(A, k=0):
    return np.linalg.det(A @ omega[:, :, k] @ A.T)

def dC(A, k=0):
    foo = A @ omega[:, :, k] @ A.T
    bar = np.linalg.inv(foo)
    return 2 * np.linalg.det(A @ omega[:, :, k] @ A.T) * omega[:, :, k] @ A.T @ bar



C_grad = check_grad(lambda A: C(A.reshape((D, J))),
                    lambda A: dC(A.reshape((D, J))).flatten(),
                    A.flatten())

raise a



def my_f(A):
    return my_func(X, A.reshape((D, J)), xi, omega, psi)

def my_g(A):
    return my_grad(X, A.reshape((D, J)), xi, omega, psi).flatten()

bar = check_grad(my_f, my_g, A.flatten())




"""
        mu = A @ xi_
        A_omega_ = A @ omega_
        cov = A_omega_ @ A.T + I_psi

        if X_var is not None:
            raise NotImplementedError("this needs some linear algebra to be fast")

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
"""