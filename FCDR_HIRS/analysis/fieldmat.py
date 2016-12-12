"""Plot scatter field density plots for temperature or other

"""

import argparse
from .. import common

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description="Plot field scatter density plot matrices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parse = common.add_to_argparse(parser,
        include_period=True,
        include_sat=True,
        include_channels=True,
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
import scipy.stats
import numpy

import matplotlib.pyplot
import matplotlib.ticker
import typhon.plots
matplotlib.pyplot.style.use(typhon.plots.styles("typhon"))
import typhon.plots.plots
import pyatmlab.graphics

from typhon.physics.units.common import ureg
from .. import fcdr

def plot_field_matrix(MM, ranges, title, filename, x_u, y_u):
    f = typhon.plots.plots.scatter_density_plot_matrix(
        MM,
        hist_kw={"bins": 20},
        hist2d_kw={"bins": 20, "cmin": 1, "cmap": "viridis"},
        hexbin_kw={"gridsize": 20, "mincnt": 1, "cmap": "viridis"},
        ranges=ranges,
        x_u=x_u, y_u=y_u)
    for a in f.get_axes():
        for ax in (a.xaxis, a.yaxis):
            ax.set_major_locator(
                matplotlib.ticker.MaxNLocator(nbins=4, prune="both"))
    f.suptitle(title)
    f.subplots_adjust(hspace=0.5, wspace=0.5)
    pyatmlab.graphics.print_or_show(f, False, filename)

def plot_temperature_matrix(M, temp_fields,
        title="", filename=""):
    MM = numpy.zeros(
        shape=M.shape,
        dtype=[(t, "f8") for t in temp_fields])
    for t in temp_fields:
        x = M["temp_{:s}".format(t)]
        while x.ndim > 1:
            x = x.mean(-1)
        MM[t] = x
    plot_field_matrix(MM,
        ranges=
            {fld: scipy.stats.scoreatpercentile(M["temp_{:s}".format(fld)], [1, 99])
                for fld in temp_fields},
        title=title, filename=filename,
        x_u="K", y_u="K")

def plot_noise_matrix(h, M, channels,
        noise_typ="iwt",
        title="", filename=""):

    for (i, ch) in enumerate(channels):
        (t_x, x) = h.estimate_noise(M, ch, typ=noise_typ)
        if i == 0:
            MM = ureg.Quantity(
                numpy.ma.zeros(
                    shape=x.shape,
                    dtype=[("ch{:d}".format(ch), "f8") for ch in channels]),
                x.u)
                    
        MM["ch{:d}".format(ch)] = x
        #MM["ch{:d}".format(ch)].mask = x.mask
    plot_field_matrix(MM,
        ranges=
            {"ch{:d}".format(ch): scipy.stats.scoreatpercentile(
                MM["ch{:d}".format(ch)], [1, 99]) for ch in channels},
        title=title, filename=filename,
        x_u="{:~}".format(x.u),
        y_u="{:~}".format(x.u))

def read_and_plot_field_matrices(sat, from_date, to_date, temp_fields, channels):
    h = fcdr.which_hirs_fcdr(sat)
    temp_fields_full = ["temp_{:s}".format(t) for t in temp_fields]
    M = h.read_period(from_date, to_date,
        fields=temp_fields_full + ["counts", "time"])
    title = "HIRS {{what:s}} {sat:s} {from_date:%Y-%m-%d} -- {to_date:%Y-%m-%d}".format(
        **locals())
    filename="hirs_{{what:s}}_{sat:s}_{from_date:%Y%m%d%H%M}--{to_date:%Y%m%d%H%M}".format(
        **locals())
    plot_temperature_matrix(M, temp_fields,
        title=title.format(what="temperatures"),
        filename=filename.format(what="tempmat") +
            "_T_{:s}.png".format(",".join(temp_fields)))
    for typ in ("iwt", "space"):
        plot_noise_matrix(h, M, channels, noise_typ=typ,
            title=title.format(what="{:s} noises".format(typ)),
            filename=filename.format(what="{:s}noise".format(typ)) + 
                "_ch_{:s}.png".format(','.join([str(x) for x in channels])))


def main():
    p = parsed_cmdline
    from_date = datetime.datetime.strptime(p.from_date, p.datefmt)
    to_date = datetime.datetime.strptime(p.to_date, p.datefmt)
    read_and_plot_field_matrices(p.satname, from_date, to_date,
        p.temperatures, p.channels)
