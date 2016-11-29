"""Plot statistics on calibration counts per "position"

HIRS is supposed to dwell for both the space and IWCT views (NOAA KLM
User's Guide, Table 3.2.1.1-1 and Table 3.2.2.1-1), but only uses 48
elements (NOAA KLM User's Guide, page 3-30, PDF page 125, or Table J-2).
That means that statistically, each of those 48 positions should on
average measure the same.  Let's verify this.
"""

from .. import common
import argparse

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = common.add_to_argparse(parser,
        include_period=True,
        include_sat=True,
        include_channels=True,
        include_temperatures=False)

    p = parser.parse_args()
    return p

parsed_cmdline = parse_cmdline()

import logging
logging.basicConfig(
    format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
            "%(lineno)s: %(message)s"),
    filename=parsed_cmdline.log,
    level=logging.DEBUG if parsed_cmdline.verbose else logging.INFO)

import matplotlib
matplotlib.use("Agg")
import pathlib
pathlib.Path("/dev/shm/gerrit/cache").mkdir(parents=True, exist_ok=True)

import datetime
import itertools
import scipy.stats

import numpy
import matplotlib.pyplot
import matplotlib.ticker

import typhon.plots
matplotlib.pyplot.style.use(typhon.plots.styles("typhon"))
import pyatmlab.graphics

from .. import fcdr

def plot_calibcount_stats(h, Mall, channels,
        title="", filename=""):
    N = len(channels)

    (nrow, ncol) = typhon.plots.common.get_subplot_arrangement(N)

    x = numpy.arange(0, 49, 2)
    (f, ax_all) = matplotlib.pyplot.subplots(nrow, ncol,
        sharex=True,
        figsize=(4+3*nrow, 4+2*ncol))
    view_space = Mall[h.scantype_fieldname] == h.typ_space
    view_iwct = Mall[h.scantype_fieldname] == h.typ_iwt
    Msp = Mall[view_space]
    Miwt = Mall[view_iwct]
    for (a1, c) in zip(ax_all.ravel(), channels):
        a2 = a1.twinx()
        for (Mlocal, a, color, lab) in (
                (Msp, a1, "blue", "space"),
                (Miwt, a2, "green", "IWCT")):
            typhon.plots.plot_distribution_as_percentiles(a,
                numpy.arange(1, 49), Mlocal["counts"][:, 8:, c-1].T, bins=x, color=color,
                ptiles=[5, 25, 50, 75, 95],
                linestyles=[":", "--", "-", "--", ":"],
                label=lab,
                linewidth=1.0)
            a.grid(True, which="major")
            for ax in (a.xaxis, a.yaxis):
                ax.set_major_locator(
                    matplotlib.ticker.MaxNLocator(nbins=4, prune=None))
            a.yaxis.set_minor_locator(
                matplotlib.ticker.AutoMinorLocator(5))
            a.set_xlim(0, 48)
        a1.xaxis.set_minor_locator(
            matplotlib.ticker.MultipleLocator(1))
        if c == channels[-1]:
            # http://stackoverflow.com/a/10129461/974555
            lines, labels = a1.get_legend_handles_labels()
            lines2, labels2 = a2.get_legend_handles_labels()
            a1.legend(lines + lines2, labels + labels2,
                loc="upper left", bbox_to_anchor=(1.5, 1.15))
        for tl in a1.get_yticklabels():
            tl.set_color("blue")
        for tl in a2.get_yticklabels():
            tl.set_color("green")
        a1.set_title("Ch. {:d}".format(c))
        if a1 in ax_all[:, -1] or c == channels[-1]:
            a2.set_ylabel("IWCT counts")
        if a1 in ax_all[:, 0]:
            a1.set_ylabel("space counts")
        if a1 in ax_all[-1, :] or c==len(channels)//ncol*ncol:
            a1.set_xlabel("Calib. scanpos")
    # set remaining invisible
    for a in ax_all.ravel()[len(channels):]:
        a.set_visible(False)
    f.suptitle(title)
    f.subplots_adjust(hspace=0.5, wspace=0.5,
        right=0.75 if nrow*ncol==len(channels) else 0.9)
    pyatmlab.graphics.print_or_show(f, False, filename)

def read_and_plot_calibcount_stats(sat, from_date, to_date, channels):
    h = fcdr.which_hirs_fcdr(sat)
    M = h.read_period(from_date, to_date,
            fields=["time", "counts", h.scantype_fieldname])
    plot_calibcount_stats(h, M, channels,
        title="HIRS calibration consistency check per scanpos\n"
              "{sat:s} {from_date:%Y-%m-%d} -- {to_date:%Y-%m-%d}".format(
                **locals()),
        filename="hirs_calib_per_scanpos_{sat:s}_{from_date:%Y%m%d%H%M}-"
                 "{to_date:%Y%m%d%H%M}.png".format(**locals()))

def main():
    p = parsed_cmdline
    from_date = datetime.datetime.strptime(p.from_date, p.datefmt)
    to_date = datetime.datetime.strptime(p.to_date, p.datefmt)
    read_and_plot_calibcount_stats(p.satname, from_date, to_date,
        p.channels)
