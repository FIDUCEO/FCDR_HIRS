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

import datetime

from .. import models
from .. import fcdr

def plot_rself_test(h, M, temperatures, channels):
    
    model = models.RSelf(temperatures)

    view_space = M[h.scantype_fieldname] == h.typ_space

    M = M[view_space]
    model.fit(M)
    (X, Y_ref, Y_pred) = model.test(M)

    N = len(channels)

    (nrow, ncol) = typhon.plots.get_subplot_arrangement(N)

    (f, ax_all) = matplotlib.pyplot.subplots(nrow, ncol,
        sharex=True, figsize=(4+3*nrow, 4+2*ncol))
    for (a1, c) in zip(ax_all.ravel(), channels):
        pass # FIXME TODO
        a1.hist2d(Y_ref, Y_pred, cmap="viridis")
    for a in ax_all.ravel()[len(channels):]:
        a.set_visible(False)


def read_and_plot_rself_test(sat, from_date, to_date, temperatures,
                             channels):
    h = fcdr.which_hirs_fcdr(sat)
    M = h.read_period(from_date, to_date,
        fields=["time", "counts"] +
            ["temp_{:s}".format(t) for t in temperatures])
    plot_rself_test(M, temperatures, channels)
