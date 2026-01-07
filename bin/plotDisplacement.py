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

parser.add_argument(
    "--compare-to",
    metavar="COMPARISSON_FILE",
    default=None,
    dest="compare_to",
    required=False,
    help="Compare displacement to the one given in another (this) file.",
)


def parseDisFile(fn=None):
    t = []
    dr_se = []  # Solid Earth
    dr_ot = []  # Ocean
    dr_ep = []  # Earth Pole

    istr = sys.stdin if fn is None else open(fn, "r")

    for line in istr:
        if not line.startswith("#"):
            t.append(datetime.datetime.strptime(line[0:20], "%Y/%m/%d %H:%M:%S "))
            l = [float(x) for x in line.split()[2:]]
            dr_se.append(np.array(l[0:3]))
            dr_ot.append(np.array(l[3:6]))
            dr_ep.append(np.array(l[6:9]))
        else:
            if line.startswith("# Ref. Frame "):
                ref_frame = line.split()[3]
    return ref_frame, t, dr_se, dr_ot, dr_ep


if __name__ == "__main__":
    args = parser.parse_args()

    ref_frame, t, dr_se, dr_ot, dr_ep = parseDisFile()

    if args.compare_to:
        ref_frame2, t2, dr_se2, dr_ot2, dr_ep2 = parseDisFile(args.compare_to)
        if ref_frame2 != ref_frame:
            raise RuntimeError(
                f"Error! The two data specified are aligned to different frames!"
            )

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
        if args.compare_to:
            ax[0].scatter(
                t2,
                [x[0] for x in dr_se2],
                s=dps,
                label="_nolegend_",
            )
            ax[1].scatter(
                t2,
                [x[1] for x in dr_se2],
                s=dps,
                label="_nolegend_",
            )
            ax[2].scatter(
                t2,
                [x[2] for x in dr_se2],
                s=dps,
                label="Solid Earth (2nd)",
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
        if args.compare_to:
            ax[0].scatter(
                t2,
                [x[0] for x in dr_ot2],
                s=dps,
                label="_nolegend_",
            )
            ax[1].scatter(
                t2,
                [x[1] for x in dr_ot2],
                s=dps,
                label="_nolegend_",
            )
            ax[2].scatter(
                t2,
                [x[2] for x in dr_ot2],
                s=dps,
                label="Ocean (2nd)",
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
        if args.compare_to:
            ax[0].scatter(
                t2,
                [x[0] for x in dr_ep2],
                s=dps,
                label="_nolegend_",
            )
            ax[1].scatter(
                t2,
                [x[1] for x in dr_ep2],
                s=dps,
                label="_nolegend_",
            )
            ax[2].scatter(
                t2,
                [x[2] for x in dr_ep2],
                s=dps,
                label="Pole (2nd)",
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
