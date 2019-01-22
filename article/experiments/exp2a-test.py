
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from scipy import linalg

from matplotlib.ticker import MaxNLocator

sys.path.insert(0, "../../")
from mcfa import (mcfa, grid_search, mpl_utils, utils)

matplotlib.style.use(mpl_utils.mpl_style)

here = os.path.dirname(os.path.realpath(__file__))  

random_seed = 10

np.random.seed(random_seed)

def savefig(fig, suffix):
    prefix = os.path.basename(__file__)[:-3]
    filename = os.path.join(here, f"{prefix}-{suffix}")
    fig.savefig(f"{filename}.pdf", dpi=300)
    fig.savefig(f"{filename}.png", dpi=150)
    print(f"Created figures {filename}.png and {filename}.pdf")
    

from astropy.table import Table


periodic_table = """H                                                  He
                    Li Be                               B  C  N  O  F  Ne
                    Na Mg                               Al Si P  S  Cl Ar
                    K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
                    Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe
                    Cs Ba Lu Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
                    Fr Ra Lr Rf"""

lanthanoids    =   "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb"
actinoids      =   "Ac Th Pa U  Np Pu Am Cm Bk Cf Es Fm Md No"

periodic_table = periodic_table.replace(" Ba ", " Ba " + lanthanoids + " ") \
                               .replace(" Ra ", " Ra " + actinoids + " ") \
                               .split()

t = Table.read(os.path.join(here, "../../catalogs/barklem_t3.fits"))

elements = np.unique(t["El"])
stars = np.unique(t["Name"])

S, E = (len(stars), len(elements))
X = np.nan * np.ones((S, E))

for i, row in enumerate(t):
    star_index = np.where(stars == row["Name"])[0][0]
    element_index = np.where(elements == row["El"])[0][0]

    X[star_index, element_index] = row["logEpsX"]

N_ok = np.zeros(E)
for i, element in enumerate(elements):
    N_ok[i] = np.sum(np.isfinite(X[:, i]))

element_indices = np.argsort(N_ok)[::-1]
for i in element_indices:
    print(elements[i], N_ok[i])

# Take top 13
use_element_indices = element_indices[:15]
use_elements = elements[use_element_indices]

use_star_indices = np.all(np.isfinite(X[:, use_element_indices]), axis=1)

label_names = list(map(str.strip, use_elements))
ei = np.argsort(np.array([periodic_table.index(el) for el in label_names]))
label_names = [label_names[ei_] for ei_ in ei]

X = X[use_star_indices, :][:, use_element_indices[ei]]

print(f"Elements: {', '.join(label_names)}")
print(f"Stars: {X.shape[0]}")

"""
raise a


with open(os.path.join(here, "../../catalogs/barklem.pkl"), "rb") as fp:
    X_H, label_names, mask = pickle.load(fp)

raise a
# Do not include C
ignore_elements = ["c"]
element_mask = np.array([ln.split("_")[0] not in ignore_elements for ln in label_names])

X_H = X_H[:, element_mask]
label_names = list(np.array(label_names)[element_mask])
"""


X = utils.whiten(X)
#X = X_H


grouped_elements = [
    ["sr", "y", "ba", "eu"],
    ["ni", "co", "fe", "mn", "cr", "ti", "sc"],
    ["al", "ca", "mg"]
]


grouped_elements = [
    ["eu", "sr", "y", "ba"],
    ["al", "ca", "mg", "ni", "co", "fe", "mn", "cr", "ti", "sc"],
]

mcfa_kwds = dict(init_factors="random", init_components="random", tol=1e-5,
                 max_iter=10000, random_seed=random_seed)


model = mcfa.MCFA(n_components=1, n_latent_factors=len(grouped_elements),
                  **mcfa_kwds)

model.fit(X)

# Rotate the latent factors to be something close to astrophysical.

A_est = model.theta_[model.parameter_names.index("A")]

A_astrophysical = np.zeros_like(A_est)
for i, tes in enumerate(grouped_elements):
    for j, te in enumerate(tes):
        #idx = label_names.index("{0}_h".format(te.lower()))
        if te.title() not in label_names:
            print(f"ignoring {te}")
            continue
        idx = label_names.index(te.title())
        A_astrophysical[idx, i] = 1.0

A_astrophysical /= np.sqrt(np.sum(A_astrophysical, axis=0))

R, p_opt, cov, *_ = utils.find_rotation_matrix(A_astrophysical, A_est, 
                                               full_output=True)

R_opt = utils.exact_rotation_matrix(A_astrophysical, A_est, 
                                    p0=np.random.uniform(-np.pi, np.pi, model.n_latent_factors**2))

chi1 = np.sum(np.abs(A_est @ R - A_astrophysical))
chi2 = np.sum(np.abs(A_est @ R_opt - A_astrophysical))

R = R_opt if chi2 < chi1 else R


# Now make it a valid rotation matrix.
model.rotate(R, X=X, ensure_valid_rotation=True)

J = model.n_latent_factors
L = model.theta_[model.parameter_names.index("A")]
cmap = mpl_utils.discrete_cmap(2 + J, base_cmap="Spectral_r")
colors = [cmap(1 + j) for j in range(J)]


fig = mpl_utils.visualize_factor_loads(L, label_names, colors=colors)

raise a
