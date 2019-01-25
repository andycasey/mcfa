
import sys
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy import (linalg, optimize as op)
from matplotlib.ticker import MaxNLocator
from time import time

sys.path.insert(0, "../../")

from mcfa import (mcfa, mpl_utils, utils)

matplotlib.style.use(mpl_utils.mpl_style)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

np.random.seed(0)

n_features = 10
n_components = 8
n_latent_factors = 3
n_samples = 100

omega_scale = 1
noise_scale = 1
random_seed = 1

data_kwds = dict(n_features=n_features,
                 n_components=n_components,
                 n_latent_factors=n_latent_factors,
                 n_samples=n_samples,
                 omega_scale=omega_scale,
                 noise_scale=noise_scale,
                 random_seed=random_seed)


Y, truth = utils.generate_data(**data_kwds)



model = mcfa.MCFA(n_components=n_components, n_latent_factors=n_latent_factors)

pi, A, xi, omega, psi = model._initial_parameters(Y)


N, D = Y.shape
J, K = (model.n_latent_factors, model.n_components)


def f(A_flat, Y, pi, xi, omega, psi):

    N, D = Y.shape
    ll, tau = model.expectation(Y, pi, A_flat.reshape((D, -1)), xi, omega, psi)
    return np.sum(ll)




def g(A_flat, Y, pi, xi, omega, psi):

    N, D = Y.shape
    J, _, K = omega.shape

    A = A_flat.reshape((D, J))

    I_psi = np.eye(D) * psi

    v = np.ones((N, K, D, J))

    ll, tau = model.expectation(Y, pi, A, xi, omega, psi)
    
    for i in range(N):

        for k in range(K):
            O = omega[:, :, k]
            M = np.atleast_2d(xi.T[k]).T

            T1 = A @ M

            T0_inv = (I_psi + A @ O @ A.T)
            T0 = np.linalg.solve(T0_inv, np.eye(D))

            T2 = T0 @ (Y[[i]].T - T1)
            v[i, k, :] = T2 @ M.T \
                        + T2 @ (Y[[i]] + (-T1).T) @ T0 @ A @ O \
                        - T0 @ A @ O

    return np.sum(np.sum(v.T * tau.T, axis=-1), axis=-1).T.flatten()


def h(A_flat, Y, pi, xi, omega, psi):

    N, D = Y.shape
    J, _, K = omega.shape

    A = A_flat.reshape((D, J))

    I_psi = np.eye(D) * psi

    assert K == 1
    O = omega[:, :, 0]
    M = np.atleast_2d(xi.T[0]).T

    v = np.ones((N, D, J))

    # (inv(D+A*O*A') * (y-A*m))*m + (inv(D+A*O*A') * (y-A*m))*(y-A*m) * inv(D+A*O*A') * A * O - inv(D+A*O*A') * A * O
    # A, m, y = scalar

    for i in range(N):

        y = Y[[i, 0]]

        am = np.sum(A[0] * M)
        raise a

        raise a




    """
    T0_inv = (I_psi + A @ O @ A.T)
    T0 = np.linalg.solve(T0_inv, np.eye(D))
    T1 = T0 @ O
    T2 = A @ M
    T3 = T0 @ M
    T4 = Y + (-T2).T
    T5 = Y.T - T2
    T6 = T0 @ T5
    T7 = T0 @ (Y.T - O @ A.T @ T0 @ A @ M)
    T8 = T0 @ A
    T9 = T8 @ O @ T0
    T10 = T9 @ T5 @ M.T
    T11 = T8 @ M @ T4 @ T0 @ O


    grad_by_parts = np.array([
        -N * T1,
            + N * T1 @ A.T @ T0 @ A @ O,
            + N * T9 @ A @ O,
        - N * T3 @ T4 @ T0 @ A @ O, 
        - N * T6 @ M.T @ T0 @ A @ O,
        - N * T3 @ M.T,
        - N * T7 @ T4 @ T0 @ A @ O,
        - N * T6 @ (Y - T2.T @ T0 @ A @ O) @ T0 @ A @ O,
        -N *T7 @ M.T,
        -N * T10,
            + N * T11 @ A.T @ T0 @ A @ O,
            + N * T10 @ A.T @ T0 @ A @ O,
        -N * T11

    ])
    """

    # -(inv(D+A*O*A')*(Y'-A*M)*M'+inv(D+A*O*A')*(Y'-A*M)*(Y+(-(A*M)')*inv(D+A*O*A')*A*O)))

    raise a



#f = -0.5 * tr(((Y - (A*M)')*inv(A*O*A' + D)*(Y - (A*M)')')) - 0.5 * N * log(det(A*O*A' + D)) = -(N*inv(D+A*O*A')*A*O-(inv(D+A*O*A')*(Y'-A*M)*M'+inv(D+A*O*A')*(Y'-A*M)*(Y+(-A*M)')*inv(D+A*O*A')*A*O))
#g = 

"""

from scipy.misc import derivative
def f_for_pi(pi, Y, A, xi, omega, psi):

    ll, tau = model.expectation(Y, pi, A, xi, omega, psi)
    return np.sum(ll)


args = (Y, truth["A"], truth["xi"], truth["omega"], truth["psi"])
F_w = -derivative(f_for_pi, truth["pi"], dx=1e-3, n=2, args=args)

raise a


"""


#-N*tr(inv(D+A*O*A')*A*O-(inv(D+A*O*A')*(Y'-A*M)*M'+inv(D+A*O*A')*(Y'-A*M)*(Y+(-(A*M)')*inv(D+A*O*A')*A*O)))
# A: matrix
# D: symmetric matrix
# N: scalar
# O: symmetric matrix
# Y: matrix

args = (Y, pi, xi, omega, psi)

At = truth["A"].flatten()

#f(At, *args)
#g(At, *args)
#h(At, *args)




A_ = truth["A"].flatten()
approx = op.approx_fprime(A_, f, 1e-10, *args)
print(approx)

actual = g(A_, *args)


diff = op.check_grad(f, g, A_, *args)
print(f"Difference {diff}")


h_approx = lambda A_, *args: g(A_, *args)[0]
op.approx_fprime(A_, h_approx, 1e-10, *args)


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(approx.flatten(), np.abs((actual - approx).flatten()))
ax.semilogy()



