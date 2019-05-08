

import numpy as np
from astropy.table import Table
from collections import OrderedDict
from operator import itemgetter
import os
#data = Table.read("GALAH_iDR3_OpenClusters.fits")

here = os.path.dirname(os.path.realpath(__file__))

data = Table.read(os.path.join(here, "../catalogs/GALAH_DR2.1_catalog.fits"))

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

available_elements = list(set([ea.split("_")[0].title() \
                        for ea in data.dtype.names if (ea.endswith("_fe") or ea == "fe_h") and ea.split("_")[0].lower() not in ("e", "flag", "alpha")]))

indices = np.argsort([periodic_table.index(el) for el in available_elements])
available_elements = [available_elements[i] for i in indices]

def _get_elements(data):
    return [ln.split("_")[1] for ln in data.dtype.names \
            if ln.startswith("A_") and ln.split("_")[1] in periodic_table]

def _abundance_label(element):
    if element.lower() == "fe":
        return "fe_h"
    else:
        return f"{element.lower()}_fe"

def _abundance_flag_label(element):
    if element.lower() == "fe":
        return "flag_cannon"
    else:
        return f"flag_{element.lower()}_fe"




def get_abundance_mask(elements, use_galah_flags=False,
                       full_output=False):

    if elements is None:
        elements = []

    use = np.ones(len(data))

    counts = dict()
    for element in elements:
        mask = _abundance_mask(element, use_galah_flags)
        use *= mask
        counts[element] = sum(mask)

    use = use.astype(bool)
    return (use, counts) if full_output else use


def _abundance_mask(element, use_galah_flags):

    mask = np.isfinite(data[_abundance_label(element)])
    if use_galah_flags:
        mask *= (data[_abundance_flag_label(element)] == 0)

    return mask


def get_abundances_breakdown(elements, use_galah_flags=False):

    use, counts = get_abundance_mask(elements,
                                     use_galah_flags=use_galah_flags, full_output=True)
 
    return OrderedDict(sorted(counts.items(), key=itemgetter(1), reverse=True))


def suggest_abundances_to_include(mask, elements, use_galah_flags=False, 
                                  percentage_threshold=10):

    N = sum(mask)

    all_elements = _get_elements(data)
    consider_elements = set(all_elements).difference(elements)

    # Consider each in turn.
    updated = dict()
    for element in consider_elements:
        new_N = sum(mask * _abundance_mask(element, use_galah_flags))
        if new_N >= (N * (100 - percentage_threshold)/100.):
            updated[element] = new_N

    return OrderedDict(sorted(updated.items(), key=itemgetter(1), reverse=True))

def get_abundances_wrt_h(elements, mask=None, cluster_names=None, use_galah_flags=False):

    # Prepare data array.
    asplund_2009 = {
        "Pr": 0.72, "Ni": 6.22, "Gd": 1.07, "Pd": 1.57, "Pt": 1.62, "Ru": 1.75, 
        "S": 7.12, "Na": 6.24, "Nb": 1.46, "Nd": 1.42, "Mg": 7.6, "Li": 1.05,
        "Pb": 1.75, "Re": 0.26, "Tl": 0.9, "Tm": 0.1, "Rb": 2.52, "Ti": 4.95, 
        "As": 2.3, "Te": 2.18, "Rh": 0.91, "Ta": -0.12, "Be": 1.38, "Xe": 2.24, 
        "Ba": 2.18, "Tb": 0.3, "H": 12.0, "Yb": 0.84, "Bi": 0.65, "W": 0.85, 
        "Ar": 6.4, "Fe": 7.5, "Br": 2.54, "Dy": 1.1, "Hf": 0.85, "Mo": 1.88, 
        "He": 10.93, "Cl": 5.5, "C": 8.43, "B": 2.7, "F": 4.56, "I": 1.55, 
        "Sr": 2.87, "K": 5.03, "Mn": 5.43, "O": 8.69, "Ne": 7.93, "P": 5.41, 
        "Si": 7.51, "Th": 0.02, "U": -0.54, "Sn": 2.04, "Sm": 0.96, "V": 3.93, 
        "Y": 2.21, "Sb": 1.01, "N": 7.83, "Os": 1.4, "Se": 3.34, "Sc": 3.15, 
        "Hg": 1.17, "Zn": 4.56, "La": 1.1, "Ag": 0.94, "Kr": 3.25, "Co": 4.99, 
        "Ca": 6.34, "Ir": 1.38, "Eu": 0.52, "Al": 6.45, "Ce": 1.58, "Cd": 1.71, 
        "Ho": 0.48, "Ge": 3.65, "Lu": 0.1, "Au": 0.92, "Zr": 2.58, "Ga": 3.04, 
        "In": 0.8, "Cs": 1.08, "Cr": 5.64, "Cu": 4.19, "Er": 0.92
    }

    # Sort elements by atomic number.
    indices = np.argsort([periodic_table.index(el) for el in elements])
    sorted_elements = [elements[idx] for idx in indices]

    if mask is None:
        mask = get_abundance_mask(sorted_elements, cluster_names, 
                                 use_galah_flags=use_galah_flags)

    X_H = np.array([data[_abundance_label(el)][mask] - asplund_2009[el] \
                    for el in sorted_elements]).T

    label_names_wrt_h = ["{0}_h".format(el).lower() for el in sorted_elements]

    return (X_H, label_names_wrt_h)


