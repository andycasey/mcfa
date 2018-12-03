

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import stan_utils as stan
from scipy import stats
from mcfa import (mcfa, mpl_utils, utils)

matplotlib.style.use(mpl_utils.mpl_style)

N = 100
D = 1
J = 5
eta = 0.1


y = np.random.normal(0, 1, size=(N, D))

data_dict = dict(N=N, D=D, J=J, y=y, eta=eta)



def LKJCorrelationMatrix(n, eta, size=None):

    beta = eta - 1. + n/2.
    r12 = 2. * stats.beta.rvs(a=beta, b=beta, size=size) - 1.
    P = np.eye(n)[:, :, np.newaxis] * np.ones(size)
    P[0, 1] = r12
    P[1, 1] = np.sqrt(1. - r12**2)
    for mp1 in range(2, n):
        beta -= 0.5
        y = stats.beta.rvs(a=mp1 / 2., b=beta, size=size)
        z = stats.norm.rvs(loc=0, scale=1, size=(mp1, ) + size)
        z = z / np.sqrt(np.einsum('ij,ij->j', z, z))
        P[0:mp1, mp1] = np.sqrt(y) * z
        P[mp1, mp1] = np.sqrt(1. - y)
    C = np.einsum('ji...,jk...->...ik', P, P)
    triu_idx = np.triu_indices(n, k=1)
    return C[..., triu_idx[0], triu_idx[1]]



model = stan.load_model("lkj.stan")

#p_opt = model.optimizing(data=data_dict)

p_samples = model.sampling(**stan.sampling_kwds(data=data_dict, chains=2))

k = -1
L = p_samples["L"]
L_lower = np.array([L_[np.tril_indices(J, k)] for L_ in L])
Omega = np.array([L_ @ L_.T for L_ in L])
Omega_lower = np.array([O[np.tril_indices(J, k)] for O in Omega])


#fig, ax = plt.subplots()
#ax.hist(L_lower.flatten())

S = Omega_lower.flatten()


M = LKJCorrelationMatrix(J, eta, size=(L.shape[0], )).flatten()


fig, ax = plt.subplots()
ax.hist([S, M])