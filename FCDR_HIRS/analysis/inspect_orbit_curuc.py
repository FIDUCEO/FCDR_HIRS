"""Plots to test and inspect results of CURUC
"""

import matplotlib
import matplotlib.pyplot
import numpy
import xarray

import pyatmlab.graphics

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
    plot_compare_correlation_scanline(xarray.open_dataset("/group_workspaces/cems2/fiduceo/Data/FCDR/HIRS/v0.8pre/debug/metopa/2015/02/01/FIDUCEO_FCDR_L1C_HIRS4_metopa_20150201205557_20150201223710_debug_v0.8pre_fv0.6.nc"))
