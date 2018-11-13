


import sys
import numpy as np
import matplotlib

import pickle
import operator

sys.path.insert(0, "../../")
from astropy.table import Table


REQUIRE_ELEMENTS = ["fe", "eu", "ba"]
EXCLUDE_LABEL_NAMES = ["o_fe"]

galah = Table.read("../../catalogs/GALAH_DR2.1_catalog.fits")


all_element_label_names = [ln for ln in galah.dtype.names \
    if ln.endswith(("_fe", "fe_h")) and not ln.startswith(("flag_", "e_", "alpha_"))]


# Select a subset of stars and abundances
common_mask = (galah["flag_cannon"] == 0)
for element in REQUIRE_ELEMENTS:
    if element != "fe":
        common_mask *= (galah[f"flag_{element}_fe"] == 0)


is_ok = dict()
for label_name in set(all_element_label_names).difference(EXCLUDE_LABEL_NAMES):
    flag_name = "flag_cannon" if label_name == "fe_h" else f"flag_{label_name}"
    is_ok[label_name] = sum((galah[flag_name] == 0) * common_mask)

sorted_items = sorted(is_ok.items(), key=operator.itemgetter(1))[::-1]


# Cumulative

x = np.arange(len(all_element_label_names))
y = np.zeros_like(x)
y_cumulative = np.zeros_like(x)

sorted_label_names = []

for i, (k, v) in enumerate(sorted_items):
    y[i] = v
    sorted_label_names.append(k)

    # Do cumulative count.
    mask = (galah["flag_cannon"] == 0)
    for k2, _ in sorted_items[:i + 1]:
        if k2 != "fe_h":
            mask *= (galah[f"flag_{k2}"] == 0)

    y_cumulative[i] = np.sum(mask)


D = 14

# Generate data array with all things relative to [X/H]
mask = (galah["flag_cannon"] == 0)
for ln, v in sorted_items[:D]:

    flag_label = f"flag_{ln}"
    if flag_label in galah.dtype.names:
        mask *= (galah[flag_label] == 0)

N = sum(mask)
print(f"There are {N} data points (D = {D})")

# order by atomic number.

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

idxs = [periodic_table.index(k.split("_")[0].title()) for k, v in sorted_items[:D]]
label_names = [sorted_items[:D][idx][0] for idx in np.argsort(idxs)]

X = np.array([galah[ln][mask] for ln  in label_names]).T

# make all relative to  x/h
X_H = X.copy()


feh_index = label_names.index("fe_h")
for i, label_name in enumerate(label_names):
    if i == feh_index: continue

    X_H[:, i] = X[:, i] + X[:, feh_index]

label_names_wrt_h = ["{0}_h".format(ln.split("_")[0]) for ln in label_names]

assert len(label_names_wrt_h) == X_H.shape[1]

output_path = __file__.replace(".py", "")
with open(f"{output_path}.pkl", "wb") as fp:
    pickle.dump((X_H, label_names_wrt_h, mask), fp)


