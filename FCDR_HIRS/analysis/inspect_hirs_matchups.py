#!/usr/bin/env python3.5

import logging
logging.basicConfig(
    format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
             "%(lineno)s: %(message)s"),
    level=logging.DEBUG)
                      
import pathlib

import numpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot
import netCDF4

import typhon.plots
import pyatmlab.graphics

from typhon.physics.units import radiance_units as rad_u
from typhon.datasets.tovs import HIRSHIRS

#srcfile = pathlib.Path("/group_workspaces/cems2/fiduceo/Data/Matchup_Data/HIRS_matchups/mmd05_hirs-ma_hirs-n17_2009-094_2009-102_v2.nc")
#srcfile = pathlib.Path("/group_workspaces/cems2/fiduceo/Data/Matchup_Data/HIRS_matchups/mmd05_hirs-ma_hirs-n17_2009-096_2009-102.nc")
#srcfile = pathlib.Path("/group_workspaces/cems2/fiduceo/Data/mms/mmd/mmd05/hirs_n17_n16/mmd05_hirs-n17_hirs-n16_2011-251_2011-257.nc")

hh = HIRSHIRS()

class HIRSMatchupInspector:
    def __init__(self, sf):
        self.ds = netCDF4.Dataset(str(sf), "r")
        self.sf = sf

    def plot_channel(self, ch, prim="n17", sec="n16"):
 
        xlab = "HIRS {prim:s}".format(prim=prim.upper())
        ylab = "HIRS {sec:s}".format(sec=sec.upper())
        Δylab = "HIRS {prim:s}-{sec:s}".format(prim=prim.upper(),
            sec=sec.upper())

        v_all = ("counts", "radiance", "bt")

        (f, a) = matplotlib.pyplot.subplots(2, 3, figsize=(14, 10))
        invalid = numpy.zeros(
            shape=(self.ds.dimensions["matchup_count"].size),
            dtype="?")

        # only plot those where data are unmasked for all three variables
        x_all = []
        y_all = []
        for (i, v) in enumerate(v_all):
            x = numpy.ma.masked_invalid(
                self.ds.variables["hirs-{:s}_{:s}_ch{:02d}".format(prim, v, ch)][:, 3, 3])
            y = numpy.ma.masked_invalid(
                self.ds.variables["hirs-{:s}_{:s}_ch{:02d}".format(sec, v, ch)][:, 3, 3])
            is_measurement = (
                (self.ds.variables["hirs-{:s}_scanline_type".format(prim)][:, 3, 3] == 0) &
                (self.ds.variables["hirs-{:s}_scanline_type".format(sec)][:, 3, 3] == 0))
            x.mask |= ~is_measurement
            y.mask |= ~is_measurement
            invalid |= x.mask
            invalid |= y.mask
            x_all.append(x)
            y_all.append(y)

        for (i, (x, y)) in enumerate(zip(x_all, y_all)):
            x = x[~invalid]
            y = y[~invalid]
            a[0, i].plot(x, y, '.')
            typhon.plots.plot_distribution_as_percentiles(a[0, i], x, y,
                nbins=40, ptiles=[5, 25, 50, 75, 95], label="N17-MA")
            a[0, i].grid("on")

            a[1, i].plot(x, y-x, '.')
            typhon.plots.plot_distribution_as_percentiles(a[1, i], x, y-x,
                nbins=40, ptiles=[5, 25, 50, 75, 95], label="N17-MA")
            a[1, i].grid("on")

        a[0, 0].set_xlabel(xlab + " counts")
        a[1, 0].set_xlabel(xlab + " counts")

        a[0, 0].set_ylabel(ylab + " counts")
        a[1, 0].set_ylabel(Δylab + " counts")

        a[0, 1].set_xlabel(xlab + " radiance [{:~}]".format(rad_u["ir"].u))
        a[1, 1].set_xlabel(xlab + " radiance [{:~}]".format(rad_u["ir"].u))

        a[0, 1].set_ylabel(ylab + " radiance [{:~}]".format(rad_u["ir"].u))
        a[1, 1].set_ylabel(Δylab + " radiance [{:~}]".format(rad_u["ir"].u))

        a[0, 2].set_xlabel(xlab + " BT [K]")
        a[1, 2].set_xlabel(xlab + " BT [K]")

        a[0, 2].set_ylabel(ylab + " BT [K]")
        a[1, 2].set_ylabel(Δylab + " BT [K]")

        a[0, 2].legend(loc="upper left", bbox_to_anchor=(1, 1))

        f.subplots_adjust(hspace=0.2, wspace=0.3, right=0.8)
        f.suptitle("Inspecting {:s}, ch. {:d}".format(self.sf.name, ch))

        pyatmlab.graphics.print_or_show(f, False,
            "{:s}/{:s}_{:d}.png".format(self.sf.parent.name, self.sf.stem, ch),
            dump_pickle=False) # for speed

def main():
    for srcfile in hh.find_granules_sorted(prim="n14", sec="n12"):
        hmi = HIRSMatchupInspector(srcfile)
        for ch in range(1, 20):
            hmi.plot_channel(ch, prim="n14", sec="n12")
   
if __name__ == "__main__":
    main()
