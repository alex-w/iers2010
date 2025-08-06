#! /usr/bin/python

import os
import sys
import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import argparse

parser = argparse.ArgumentParser(
    description="Plot tidal & non-tidal displacements on reference point(s).",
    epilog=(
        """National Technical University of Athens,
    Dionysos Satellite Observatory\n
    Send bug reports to:
    Xanthos Papanikolaou, xanthos@mail.ntua.gr
    Apr, 2024"""
    ),
)

parser.add_argument(
    "--only",
    metavar="CONSIDER_ONLY",
    dest="only",
    default=None,
    required=False,
    choices=["solid_earth", "ocean", "pole"],
    help="Consider only the given phenomenon.",
)

parser.add_argument(
    "--remove",
    metavar="REMOVE",
    dest="remove",
    default=[],
    required=False,
    choices=["solid_earth", "ocean", "pole"],
    nargs="+",
    help="Do not consider the given phenomenae.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    t = []
    dr_se = []
    dr_ot = []
    dr_ep = []

    for line in sys.stdin:
        if not line.startswith("#"):
            t.append(datetime.datetime.strptime(line[0:20], "%Y/%m/%d %H:%M:%S "))
            l = [float(x) for x in line.split()[2:]]
            dr_se.append(np.array(l[0:3]))
            dr_ot.append(np.array(l[3:6]))
            dr_ep.append(np.array(l[6:9]))
        else:
            if line.startswith("# Ref. Frame "):
                ref_frame = line.split()[3]

    if ref_frame == "xyz":
        components = ["X", "Y", "Z"]
    elif ref_frame == "enu":
        components = ["East", "North", "Up"]
    else:
        raise RuntimeError(f"Error! Unknow ref frame in file: {ref_frame}")

    dps = 0.3
    fig, ax = plt.subplots(3, 1, sharex=True)
    # Solid Earth Tides
    if (
        args.only is None and "solid_earth" not in args.remove
    ) or args.only == "solid_earth":
        ax[0].scatter(
            t,
            [x[0] for x in dr_se],
            s=dps,
            label="_nolegend_",
        )
        ax[1].scatter(
            t,
            [x[1] for x in dr_se],
            s=dps,
            label="_nolegend_",
        )
        ax[2].scatter(
            t,
            [x[2] for x in dr_se],
            s=dps,
            label="Solid Earth",
        )
    # Ocean Tides
    if (args.only is None and "ocean" not in args.remove) or args.only == "ocean":
        ax[0].scatter(
            t,
            [x[0] for x in dr_ot],
            s=dps,
            label="_nolegend_",
        )
        ax[1].scatter(
            t,
            [x[1] for x in dr_ot],
            s=dps,
            label="_nolegend_",
        )
        ax[2].scatter(
            t,
            [x[2] for x in dr_ot],
            s=dps,
            label="Ocean",
        )
    # Earth Pole Tide
    if (args.only is None and "pole" not in args.remove) or args.only == "pole":
        ax[0].scatter(
            t,
            [x[0] for x in dr_ep],
            s=dps,
            label="_nolegend_",
        )
        ax[1].scatter(
            t,
            [x[1] for x in dr_ep],
            s=dps,
            label="_nolegend_",
        )
        ax[2].scatter(
            t,
            [x[2] for x in dr_ep],
            s=dps,
            label="Pole",
        )

    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)

    ax[0].set_ylabel(f"{components[0]} [m]")
    ax[1].set_ylabel(f"{components[1]} [m]")
    ax[2].set_ylabel(f"{components[2]} [m]")
    ax[2].set_xlabel("Epoch")
    ax[0].set_title(r"Tide Displacement")

    fig.subplots_adjust(hspace=0)
    plt.tight_layout()

    ax[2].legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))

    plt.show()
