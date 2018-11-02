"""Show some plots testing the self-emission model

"""

from .. import common
import argparse
import json


import logging
import numpy
import matplotlib
# matplotlib.use("Agg") # now in matplotlibrc
import pathlib
#pathlib.Path("/dev/shm/gerrit/cache").mkdir(parents=True, exist_ok=True)

import matplotlib.ticker
import scipy.stats
import datetime
import typhon.plots
import pyatmlab.graphics

from typhon.physics.units.common import ureg, radiance_units as rad_u

from .. import models
from .. import fcdr
from .. import common

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = common.add_to_argparse(parser,
        include_period=True,
        include_sat=True,
        include_channels=True,
        include_temperatures=True)

    parser.add_argument("--regression_type", action="store", type=str,
        choices=["LR", "PLSR"],
        default="PLSR",
        help="What kind of regression to use for prediction?")

    parser.add_argument("--regression_args", action="store", type=json.loads,
        default={"n_components": 2, "scale": True},
        help="Arguments to pass to regression class (as json/Python dict)")

    p = parser.parse_args()
    return p

def plot_rself_test(h, ds, temperatures, channels,
        regr_type, regr_args, tit, fn):
    
    model = models.RSelf(h, temperatures,
        (regr_type, regr_args))

    #view_space = M[h.scantype_fieldname] == h.typ_space

    #M = M[view_space]

    N = len(channels)

    (ncol, nrow) = typhon.plots.get_subplot_arrangement(N)

    (f, ax_all) = matplotlib.pyplot.subplots(nrow, ncol,
        sharex=False, figsize=(2+3*ncol, 3+2.5*nrow),
        squeeze=False)
    for (a, c) in zip(ax_all.ravel(), channels):
        model.fit(ds, c)
        (X, Y_ref, Y_pred) = model.test(ds, c)
        xloc = Y_ref.to(rad_u["ir"], "radiance")
        yloc = (Y_pred - Y_ref).to(rad_u["ir"], "radiance")
        rng = [scipy.stats.scoreatpercentile(
            xloc[~xloc.isnull()], [1, 99]),
               scipy.stats.scoreatpercentile(
            yloc[~xloc.isnull()], [1, 99])]
        unmasked = (~xloc.isnull()) & (~yloc.isnull())
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
        rmse = numpy.sqrt(((Y_pred-Y_ref).to(rad_u["ir"], "radiance")**2).mean())
        rmse = ureg.Quantity(rmse.values, rmse.attrs["units"]) 
        a.annotate(f"RMSE: {rmse.round(2):~}", xy=(.99, .99), xycoords='axes fraction', horizontalalignment='right', verticalalignment="top")
        a.set_title("Ch. {:d}".format(c))
        a.grid(axis="both", color="white")
        #a.set_aspect("equal", "box", "C")
        a.plot(rng, [0, 0], 'k-', linewidth=0.5)
        if a in ax_all[:, 0]:
            a.set_ylabel("Calib. offset (Estimate-Reference)\n[{:~}]".format(rad_u["ir"]))
        if a in ax_all[-1, :]:
            a.set_xlabel("Calib. offset (Reference)\n[{:~}]".format(rad_u["ir"]))
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
                             channels, regr_type, regr_args):
    h = fcdr.which_hirs_fcdr(sat)
    M = h.read_period(from_date, to_date,
        fields=["time", "counts", "lat", "lon", "bt", h.scantype_fieldname] +
            ["temp_{:s}".format(t) for t in temperatures] +
            (["temp_iwt"] if "iwt" not in temperatures else []))
    ds = h.as_xarray_dataset(M)
    plot_rself_test(h, ds, temperatures, channels,
        regr_type, regr_args,
        "self-emission regression performance, {sat:s}, "
        "{from_date:%Y-%m-%d} -- {to_date:%Y-%m-%d}\n"
        "using {temperatures:s}\n"
        "{regr_type:s}, {regr_args!s}".format(
            sat=sat, from_date=from_date, to_date=to_date,
            temperatures=', '.join(temperatures),
            regr_type=regr_type, regr_args=regr_args),
        "rself_regr_test_{sat:s}_{from_date:%Y%m%d%H%M}-"
        "{to_date:%Y%m%d%H%M}_"
        "ch_{ch:s}_"
        "from_{temperatures:s}_"
        "{regr_type:s}_{regr_args:s}.png".format(
            sat=sat, from_date=from_date, to_date=to_date,
            ch=",".join([str(c) for c in channels]),
            temperatures=",".join(temperatures),
            regr_type=regr_type,
            regr_args=','.join(f"{k:s}{v!s}" for (k,v) in regr_args.items())))

def main():
    import warnings
    p = parse_cmdline()
    common.set_root_logger(
        logging.DEBUG if p.verbose else logging.INFO,
        p.log)
#    warnings.filterwarnings("error")
#    warnings.filterwarnings("always", category=DeprecationWarning)
    from_date = datetime.datetime.strptime(p.from_date, p.datefmt)
    to_date = datetime.datetime.strptime(p.to_date, p.datefmt)
    read_and_plot_rself_test(p.satname, from_date, to_date,
        p.temperatures, p.channels, p.regression_type, p.regression_args)
