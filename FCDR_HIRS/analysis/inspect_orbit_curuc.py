"""Plots to test and inspect results of CURUC
"""

import matplotlib
import matplotlib.pyplot
import numpy
import xarray
import argparse
import sys

import pyatmlab.graphics

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("path", action="store", type=str,
        help="Path for which to show CURUC stuff")

    parser.add_argument("--x", action="store", type=int,
        nargs="*",
        help="List of x-coordinates for which to recalculate/show detailed info")

    parser.add_argument("--y", action="store", type=int,
        nargs="*",
        help="List of y-coordinates for which to recalculate/show detailed info")

    parser.add_argument("--lines", action="store", type=int,
        nargs=2,
        help="Range of lines to explore in detail")

    return parser.parse_args()

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
    sys.exit("The development of this script is still in progress.")
    plot_compare_correlation_scanline(xarray.open_dataset(p.path))
