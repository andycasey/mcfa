


import sys
import numpy as np
import matplotlib.pyplot as plt
from time import time
import operator

sys.path.insert(0, "../../")
from astropy.table import Table


from mcfa import mpl_utils

matplotlib.style.use(mpl_utils.mpl_style)


galah = Table.read("../../catalogs/GALAH_DR2.1_catalog.fits")


element_label_names = [ln for ln in galah.dtype.names \
    if ln.endswith(("_fe", "fe_h")) and not ln.startswith(("flag_", "e_", "alpha_"))]

# Select a subset of stars and abundances
is_ok = dict()
for label_name in element_label_names:
    flag_name = "flag_cannon" if label_name == "fe_h" else f"flag_{label_name}"
    is_ok[label_name] = sum((galah[flag_name] == 0) * (galah["flag_cannon"] == 0))

x = np.arange(len(element_label_names))
y = np.zeros_like(x)
y_cumulative = np.zeros_like(x)

sorted_label_names = []

sorted_items = sorted(is_ok.items(), key=operator.itemgetter(1))[::-1]

for i, (k, v) in enumerate(sorted_items):
    y[i] = v
    sorted_label_names.append(k)

    # Do cumulative count.
    mask = (galah["flag_cannon"] == 0)
    for k2, _ in sorted_items[:i + 1]:
        if k2 != "fe_h":
            mask *= (galah[f"flag_{k2}"] == 0)

    y_cumulative[i] = np.sum(mask)


fig, ax = plt.subplots()
ax.plot(x, y, "-", drawstyle="steps-mid", c="#000000", lw=2,
        label=r"$\textrm{Independent}$")
ax.plot(x, y_cumulative, "-", c="tab:blue", drawstyle="steps-mid", lw=2, zorder=10,
        label=r"$\textrm{Cumulative}$")
ax.set_xticks(x)
ax.set_xticklabels([ln.split("_")[0].title() for ln in sorted_label_names])
ax.set_xlim(-0.5, x.max() + 0.5)

ax.semilogy()
ax.set_ylim(10**3, ax.get_ylim()[1])
ax.set_ylabel(r"$\textrm{GALAH stars with reliable abundance measurements}$")


plt.legend(loc="lower left")

ax2 = ax.twiny()
ax2.set_xticks(x)
ax2.set_xticklabels([f"{n}" for n in 1 + x])
ax2.set_xlim(ax.get_xlim())
ax2.set_xlabel(r"$\textrm{Number of chemical abundances for cumulative curve}$")


fig.tight_layout()


figure_name = __file__.lower().replace(".py", "")
fig.savefig(f"{figure_name}.png", dpi=150)
fig.savefig(f"{figure_name}.pdf", dpi=300)

