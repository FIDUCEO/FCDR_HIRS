"""Show some plots testing the self-emission model

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

import matplotlib.ticker
import scipy.stats
import datetime
import typhon.plots
import pyatmlab.graphics

from typhon.physics.units.common import radiance_units as rad_u

from .. import models
from .. import fcdr

def plot_rself_test(h, ds, temperatures, channels,
        tit, fn):
    
    model = models.RSelf(h, temperatures)

    #view_space = M[h.scantype_fieldname] == h.typ_space

    #M = M[view_space]

    N = len(channels)

    (ncol, nrow) = typhon.plots.get_subplot_arrangement(N)

    (f, ax_all) = matplotlib.pyplot.subplots(nrow, ncol,
        sharex=False, figsize=(2+3*ncol, 3+2.5*nrow))
    for (a, c) in zip(ax_all.ravel(), channels):
        model.fit(ds, c)
        (X, Y_ref, Y_pred) = model.test(ds, c)
        xloc = Y_ref.to(rad_u["ir"], "radiance").m
        yloc = (Y_pred - Y_ref).to(rad_u["ir"], "radiance").m
        rng = [scipy.stats.scoreatpercentile(
            xloc[~xloc.mask], [1, 99]),
               scipy.stats.scoreatpercentile(
            yloc[~xloc.mask], [1, 99])]
        unmasked = (~xloc.mask) & (~yloc.mask)
        a.hexbin(xloc[unmasked], yloc[unmasked],
            cmap="viridis", mincnt=1,
            gridsize=20,
            extent=[rng[0][0], rng[0][1], rng[1][0], rng[1][1]])
#        a.hist2d(xloc, yloc,
#                cmap="viridis", cmin=1,
#                bins=20, range=rng)
        typhon.plots.plot_distribution_as_percentiles(a,
            xloc[unmasked], yloc[unmasked],
            nbins=20, color="tan",
            ptiles=[5, 25, 50, 75, 95],
            linestyles=[":", "--", "-", "--", ":"])
        a.set_title("Ch. {:d}".format(c))
        a.grid(axis="both", color="white")
        #a.set_aspect("equal", "box", "C")
        a.plot(rng, [0, 0], 'k-', linewidth=0.5)
        if a in ax_all[:, 0]:
            a.set_ylabel("Calib. offset (Estimate-Reference)\n[{:~}]".format(rad_u["ir"].u))
        if a in ax_all[-1, :]:
            a.set_xlabel("Calib. offset (Reference)\n[{:~}]".format(rad_u["ir"].u))
        for ax in (a.xaxis, a.yaxis):
            ax.set_major_locator(
                matplotlib.ticker.MaxNLocator(nbins=4, prune=None))
#    for a in ax_all.ravel()[:len(channels)]:
        a.set_xlim(rng[0])
        a.set_ylim(rng[1])
    for a in ax_all.ravel()[len(channels):]:
        a.set_visible(False)

    f.suptitle(tit)
    f.subplots_adjust(hspace=0.3)
    pyatmlab.graphics.print_or_show(f, False, fn)


def read_and_plot_rself_test(sat, from_date, to_date, temperatures,
                             channels):
    h = fcdr.which_hirs_fcdr(sat)
    M = h.read_period(from_date, to_date,
        fields=["time", "counts", "lat", "lon", "bt"] +
            ["temp_{:s}".format(t) for t in temperatures])
    ds = h.as_xarray_dataset(M)
    plot_rself_test(h, ds, temperatures, channels,
        "self-emission regression performance, {sat:s}, "
        "{from_date:%Y-%m-%d} -- {to_date:%Y-%m-%d}\n"
        "using {temperatures:s}".format(
            sat=sat, from_date=from_date, to_date=to_date,
            temperatures=', '.join(temperatures)),
        "rself_regr_test_{sat:s}_{from_date:%Y%m%d%H%M}-"
        "{to_date:%Y%m%d%H%M}_"
        "ch_{ch:s}_"
        "from_{temperatures:s}.png".format(
            sat=sat, from_date=from_date, to_date=to_date,
            ch=",".join([str(c) for c in channels]),
            temperatures=",".join(temperatures)))

def main():
    import warnings
#    warnings.filterwarnings("error")
#    warnings.filterwarnings("always", category=DeprecationWarning)
    p = parsed_cmdline
    from_date = datetime.datetime.strptime(p.from_date, p.datefmt)
    to_date = datetime.datetime.strptime(p.to_date, p.datefmt)
    read_and_plot_rself_test(p.satname, from_date, to_date,
        p.temperatures, p.channels)
