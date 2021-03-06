#!/usr/bin/env python3.5
"""Inspect HIRS matchups

This module contains routines inspecting the raw matchups as produced by
Brockmann Consult.  Normally, it is executed by the script
:ref:`inspect-hirs-matchups`.  This is distinct from
:mod:`inspect_hirs_harm_matchups`, which analyses the enhanced matchup
files produced by :ref:`combine-hirs-hirs-matchups` and friends.
"""

import argparse


import logging
                      
import datetime
import pathlib

import numpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot
import netCDF4
import scipy.stats

import typhon.plots
import matplotlib.ticker

from typhon.physics.units import radiance_units as rad_u
from typhon.datasets.tovs import HIRSHIRS

from .. import (fcdr, matchups, common, graphics)
matplotlib.pyplot.style.use(typhon.plots.styles("typhon"))
logger = logging.getLogger(__name__)

#srcfile = pathlib.Path("/group_workspaces/cems2/fiduceo/Data/Matchup_Data/HIRS_matchups/mmd05_hirs-ma_hirs-n17_2009-094_2009-102_v2.nc")
#srcfile = pathlib.Path("/group_workspaces/cems2/fiduceo/Data/Matchup_Data/HIRS_matchups/mmd05_hirs-ma_hirs-n17_2009-096_2009-102.nc")
#srcfile = pathlib.Path("/group_workspaces/cems2/fiduceo/Data/mms/mmd/mmd05/hirs_n17_n16/mmd05_hirs-n17_hirs-n16_2011-251_2011-257.nc")

hh = HIRSHIRS()

def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    choices=["ma", "mb", "n19", "n18", "n17", "n16", "n15", "n14",
        "n13", "n12", "n11", "n10", "n09", "n08", "n07", "n06", "tn"]
    choices.sort()

    parser.add_argument("prim", action="store", type=str,
        help="Primary satellite",
        choices=choices)

    parser.add_argument("sec", action="store", type=str,
        help="Secondary satellite",
        choices=choices)

    parser.add_argument("from_date", action="store", type=str,
        help="Starting date/time")

    parser.add_argument("to_date", action="store", type=str,
        help="Ending date/time")

    parser.add_argument("--datefmt", action="store",
        default="%Y%m%d", help="Date range.")

    parser.add_argument("--verbose", action="store_true",
        default=False, help="Be verbose.")

    return parser
def parse_cmdline():
    return get_parser().parse_args()

class HIRSMatchupInspector(matchups.HIRSMatchupCombiner):
    """Class to analyse sets of HIRS matchups

    This is a subclass of :class:`matchups.HIRSMatchupCombiner`, adding
    a single method :meth:`plot_channel`, for matchup analysis and visualisation.

    """
    def plot_channel(self, ch):#, prim="n17", sec="n16"):
        """Plot statistics for HIRS-HIRS matchups for channel

        For a single channel, plot 10 panels of HIRS-HIRS matchup
        statistics, the top row y vs. x, the bottom row y-x vs x:

        .. image:: /images/hirshirs.png

        Parameters
        ----------

        ch : int
            Channel number

        """
        prim = self.prim_name
        sec = self.sec_name
        xlab = "HIRS {prim:s}".format(prim=prim.upper())
        ylab = "HIRS {sec:s}".format(sec=sec.upper())
        Δylab = "HIRS {prim:s}-{sec:s}".format(prim=prim.upper(),
            sec=sec.upper())

        #v_all = ("counts", "radiance", "radiance_fid", "bt", "bt_fid")
        v_all = ("counts", "radiance", "bt") # for fid, need to take from
                                             # self.Mcp

        (f, a) = matplotlib.pyplot.subplots(2, 3, figsize=(30, 10))
        invalid = numpy.zeros(
            shape=(self.ds.dims["matchup_count"]),
            dtype="?")

        # only plot those where data are unmasked for all three variables
        x_all = []
        y_all = []
        for (i, v) in enumerate(v_all):
            if v.endswith("_fid"):
                x = numpy.ma.masked_invalid(self.Mcp[v][:, ch-1])
                y = numpy.ma.masked_invalid(self.Mcs[v][:, ch-1])
            else:
                x = numpy.ma.masked_invalid(
                    self.ds["hirs-{:s}_{:s}_ch{:02d}".format(prim, v, ch)][:, 3, 3])
                y = numpy.ma.masked_invalid(
                    self.ds["hirs-{:s}_{:s}_ch{:02d}".format(sec, v, ch)][:, 3, 3])
             # need to skip this due to #214, see
             # https://github.com/FIDUCEO/FCDR_HIRS/issues/214
            is_measurement = (
                (self.ds["hirs-{:s}_scanline_type".format(prim)][:, 3, 3] == 0) &
                (self.ds["hirs-{:s}_scanline_type".format(sec)][:, 3, 3] == 0))
            x.mask |= ~is_measurement
            y.mask |= ~is_measurement
            invalid |= x.mask
            invalid |= y.mask
            x_all.append(x)
            y_all.append(y)

        for (i, (x, y)) in enumerate(zip(x_all, y_all)):
            x = x[~invalid]
            y = y[~invalid]
            rng = numpy.asarray([scipy.stats.scoreatpercentile(x, [1, 99]),
                   scipy.stats.scoreatpercentile(y, [1, 99])])
            #a[0, i].plot(x, y, '.')
            # FIXME: use hexplot
            a[0, i].hist2d(x, y, bins=40, range=rng, cmap="viridis",
                cmin=1)
            typhon.plots.plot_distribution_as_percentiles(a[0, i], x, y,
                nbins=40, ptiles=[5, 25, 50, 75, 95], color="tan",
                label=' ')
            a[0, i].grid("on")
            a[0, i].plot(*[[rng.min(), rng.max()]]*2, 'k-',
                linewidth=3)
            a[0, i].set_aspect("equal", "box", "C")

            rng[1] = scipy.stats.scoreatpercentile(y-x, [1, 99])
            #a[1, i].plot(x, y-x, '.')
            # FIXME: use hexplot
            a[1, i].hist2d(x, y-x, bins=40, range=rng, cmap="viridis",
                cmin=1)
            typhon.plots.plot_distribution_as_percentiles(a[1, i], x, y-x,
                nbins=40, ptiles=[5, 25, 50, 75, 95], color="tan",
                label=' ')
            a[1, i].grid("on")
            a[1, i].plot([rng[0, 0], rng[0, 1]], [0, 0], 'k-',
                linewidth=3)
            a[1, i].set_aspect("equal", "box", "C")

        a[0, 0].set_xlabel(xlab + " counts")
        a[1, 0].set_xlabel(xlab + " counts")

        a[0, 0].set_ylabel(ylab + " counts")
        a[1, 0].set_ylabel(Δylab + " counts")

        a[0, 1].set_xlabel(xlab + " NOAA radiance\n[{:~}]".format(rad_u["ir"]))
        a[1, 1].set_xlabel(xlab + " NOAA radiance\n[{:~}]".format(rad_u["ir"]))

        a[0, 1].set_ylabel(ylab + " NOAA radiance\n[{:~}]".format(rad_u["ir"]))
        a[1, 1].set_ylabel(Δylab + " NOAA radiance\n[{:~}]".format(rad_u["ir"]))

#        a[0, 2].set_xlabel(xlab + "FID radiance\n[{:~}]".format(rad_u["ir"]))
#        a[1, 2].set_xlabel(xlab + "FID radiance\n[{:~}]".format(rad_u["ir"]))
#
#        a[0, 2].set_ylabel(ylab + " FID radiance\n[{:~}]".format(rad_u["ir"]))
#        a[1, 2].set_ylabel(Δylab + " FID radiance\n[{:~}]".format(rad_u["ir"]))

        a[0, 2].set_xlabel(xlab + " NOAA BT [K]")
        a[1, 2].set_xlabel(xlab + " NOAA BT [K]")

        a[0, 2].set_ylabel(ylab + " NOAA BT [K]")
        a[1, 2].set_ylabel(Δylab + " NOAA BT [K]")

#        a[0, 4].set_xlabel(xlab + " FID BT [K]")
#        a[1, 4].set_xlabel(xlab + " FID BT [K]")
#
#        a[0, 4].set_ylabel(ylab + " FID BT [K]")
#        a[1, 4].set_ylabel(Δylab + " FID BT [K]")

        a[0, 2].legend(loc="upper left", bbox_to_anchor=(1, 1))

        for ta in a.ravel():
            ta.xaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(nbins=4, prune=None))
            ta.yaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(
                    nbins=max(int(round(numpy.ptp(ta.get_ylim()) /
                                    numpy.ptp(ta.get_xlim()) * 6)),
                              3),
                    prune=None))
            # yaxis still okay

        f.subplots_adjust(hspace=0.2, wspace=0.4, right=0.8)
        f.suptitle("HIRS-HIRS {:s}-{:s} ch. {:d}\n{:%Y-%m-%d %H:%M}--{:%Y-%m-%d %H:%M}".format(
            prim, sec, ch,
            self.ds["time"][0].values.astype("M8[ms]").astype(datetime.datetime),
            self.ds["time"][-1].values.astype("M8[ms]").astype(datetime.datetime)))

        graphics.print_or_show(f, False,
            "hirshirs/{prim:s}_{sec:s}/hirshirs_{prim:s}_{sec:s}_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}_ch{ch:d}.png".format(
                prim=prim, sec=sec,
                start=self.ds["time"][0].values.astype("M8[ms]").astype(datetime.datetime),
                end=self.ds["time"][-1].values.astype("M8[ms]").astype(datetime.datetime),
                ch=ch))

def main():
    """Main function, expects commandline input.

    See module docstring and :ref:`inspect-hirs-matchups`.
    """
    p = parse_cmdline()
    common.set_logger(
        logging.DEBUG if p.verbose else logging.INFO,
        p.log,
        loggers={"FCDR_HIRS", "typhon"})
    hmi = HIRSMatchupInspector(
        datetime.datetime.strptime(p.from_date, p.datefmt),
        datetime.datetime.strptime(p.to_date, p.datefmt),
        p.prim, p.sec)
    for ch in range(1, 20):
        hmi.plot_channel(ch)#, prim="n14", sec="n12")
