"""Plot temperature matrix

"""

import argparse
from .. import common

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description="Plot temperature matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parse = common.add_to_argparse(parser,
        include_period=True,
        include_sat=True,
        include_channels=False,
        include_temperatures=True)

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

import matplotlib.pyplot
import matplotlib.ticker
import typhon.plots
matplotlib.pyplot.style.use(typhon.plots.styles("typhon"))
import pyatmlab.graphics

from .. import fcdr

def plot_temperature_matrix(M, temp_fields,
        title="", filename=""):
    N = len(temp_fields)
    (f, ax_all) = matplotlib.pyplot.subplots(N, N, figsize=(4+3*N, 4+3*N))

    for ((x_i, x_f), (y_i, y_f)) in itertools.product(enumerate(temp_fields), repeat=2):
        a = ax_all[y_i, x_i]

        x = M["temp_{:s}".format(x_f)]
        y = M["temp_{:s}".format(y_f)]
        while x.ndim > 1:
            x = x.mean(-1)
        while y.ndim > 1:
            y = y.mean(-1)

        rng = [scipy.stats.scoreatpercentile(x, [1, 99]),
               scipy.stats.scoreatpercentile(y, [1, 99])]
        if x_i == y_i:
            a.hist(x, bins=40, range=rng[0])
        else:
            (_, _, _, im) = a.hist2d(x, y, bins=40, range=rng, cmin=1, cmap="viridis")
            typhon.plots.plot_distribution_as_percentiles(
                a, x, y, nbins=40, color="tan",
                ptiles=[5, 25, 50, 75, 95],
                linestyles=[":", "--", "-", "--", ":"],
                linewidth=1.5)

            a.set_xlim(rng[0])
            a.set_ylim(rng[1])

        # I seem to have swapped x and y
        if y_i == len(temp_fields)-1:
            a.set_xlabel(x_f)

        if x_i == 0:
            a.set_ylabel(y_f)

        for ax in (a.xaxis, a.yaxis):
            ax.set_major_locator(
                matplotlib.ticker.MaxNLocator(nbins=4,
                prune="both"))

        #a.set_title(x_f + ", " + y_f)

    f.suptitle(title)
    f.subplots_adjust(hspace=0.5, wspace=0.5)
    pyatmlab.graphics.print_or_show(f, False, filename)

def read_and_plot_temperatures(sat, from_date, to_date, temp_fields):
    h = fcdr.which_hirs_fcdr(sat)
    M = h.read_period(from_date, to_date,
        fields=["temp_{:s}".format(t) for t in temp_fields])
    plot_temperature_matrix(M, temp_fields,
        title="HIRS temperatures {sat:s} {from_date:%Y-%m-%d} -- {to_date:%Y-%m-%d}".format(
            **locals()),
        filename="hirs_tempmat_{sat:s}_{from_date:%Y%m%d%H%M}--{to_date:%Y%m%d%H%M}.png".format(
            **locals()))

def main():
    p = parsed_cmdline
    from_date = datetime.datetime.strptime(p.from_date, p.datefmt)
    to_date = datetime.datetime.strptime(p.to_date, p.datefmt)
    read_and_plot_temperatures(p.satname, from_date, to_date, p.temp_fields)
