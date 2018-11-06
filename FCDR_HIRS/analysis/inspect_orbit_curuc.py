"""Plots to test and inspect results of CURUC

Includes functionality for figures 3 and 4 from
Merchant, Holl, ... (submitted 2018).
"""

import matplotlib
import matplotlib.pyplot
import numpy
import xarray
import argparse
import sys
import pathlib

import pyatmlab.graphics

from ..processing.generate_fcdr import FCDRGenerator

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("path", action="store", type=pathlib.Path,
        help="Path for which to show CURUC stuff")

    parser.add_argument("-x", action="store", type=int,
        dest="x_all",
        nargs="*",
        help="List of x-coordinates for which to recalculate/show detailed info")

    parser.add_argument("-y", action="store", type=int,
        nargs="*",
        dest="y_all",
        help="List of y-coordinates for which to recalculate/show detailed info")

    parser.add_argument("--lines", action="store", type=int,
        nargs=2,
        help="Range of lines to explore in detail")

    return parser.parse_args()

def plot_curuc_for_pixels(ds, lines, x_all, y_all):
    """Plot some CURUC stats for specific pixels

    Produce the plots for figures 3 and 4 from the easyFCDR paper.
    """
    start = ds["time"].isel(y=lines[0])
    end = ds["time"].isel(y=lines[1])

    # recalculate FCDR to get CURUC specifically for this segment
    fg = FCDRGenerator(None, None, [], no_harm=False) # no storing, no period
    ds_new = fg.get_piece(start, end, reset_context=True)
    raise NotImplementedError("I'm not quite there yet")

def plot_compare_correlation_scanline(ds):
    ds5 = ds.sel(calibrated_channel=5)
    y_real = ds5["cross_line_radiance_error_correlation_length_average"]
    x_real = y_real["delta_scanline_earth"]
    Δ_l = ds5["cross_line_radiance_error_correlation_length_scale_structured_effects"]
    y_approx = numpy.exp(-numpy.abs(x_real)/Δ_l)

    (f, a) = matplotlib.pyplot.subplots(figsize=(12, 6))
    a.plot(x_real, y_real, label="The Real Thing™ (maybe)")
    a.plot(x_real, y_approx, label="Exponential approximation")

    f.legend()
    a.set_xlim([0, 500])
    a.set_xlabel(r"$\Delta$ scanline")
    a.set_ylabel("Mean correlation coefficient")
    a.set_title("Mean correlation as function of scanline interval, single orbit")

    pyatmlab.graphics.print_or_show(f, False, "orbit_curuc_test.")

def main():
    p = parse_cmdline()
    plot_curuc_for_pixels(
        xarray.open_dataset(p.path),
        lines=p.lines,
        x_all=p.x_all,
        y_all=p.y_all)

    sys.exit("The development of this script is still in progress.")
    plot_compare_correlation_scanline(xarray.open_dataset(p.path))
