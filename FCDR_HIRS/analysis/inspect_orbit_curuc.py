"""Plots to test and inspect results of CURUC

Includes functionality for figures 3 and 4 from
Merchant, Holl, ... (submitted 2018).
"""

import datetime
import argparse
import sys
import pathlib
import logging

import matplotlib
import matplotlib.pyplot
import numpy
import xarray

import typhon.physics.units.em
import typhon.datasets.tovs
from typhon.physics.units.common import radiance_units as rad_u

import pyatmlab.graphics

from ..processing.generate_fcdr import FCDRGenerator
from ..common import (set_logger, add_to_argparse)
from .. import metrology

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = add_to_argparse(parser,
        include_period=False,
        include_sat=False,
        include_channels=False,
        include_temperatures=False,
        include_debug=False)

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

    parser.add_argument("--channel", action="store", type=int,
        help="Channel")

    parser.add_argument("--lines", action="store", type=int,
        nargs=2,
        help="Range of lines to explore in detail")

    return parser.parse_args()

def _S_radsi_to_K(S, srf):
    S = S.values * rad_u["si"]**2
    return numpy.sign(S) * numpy.sqrt(numpy.abs(S)).to("K", "radiance", srf=srf)**2

def plot_curuc_for_pixels(ds, lines, channel, x_all, y_all):
    """Plot some CURUC stats for specific pixels

    Produce the plots for figures 3 and 4 from the easyFCDR paper.
    """
    start = ds["time"].isel(y=lines[0]).values.astype("M8[ms]").item()
    end = ds["time"].isel(y=lines[1]).values.astype("M8[ms]").item()

    # recalculate FCDR to get CURUC specifically for this segment
    fg = FCDRGenerator(ds.satellite, 
        datetime.datetime.now(),
        datetime.datetime.now(), [], no_harm=False) # no storing, dates not relevant
    (ds_new, sensRe) = fg.get_piece(start, end, return_more=True, reset_context=True)

    (Δ_l, Δ_e, R_ci, R_cs, Δ_l_full, Δ_e_full, D) = metrology.calc_corr_scale_channel(
        fg.fcdr._effects, sensRe, ds_new, flags=fg.fcdr._flags,
        robust=True, return_vectors=True, interpolate_lengths=True,
        sampling_l=1, sampling_e=1, return_locals=True)

    del sensRe # this causes pyatmlabs get_verbose_stack_description to
               # fail as pprint.pprint can't sort Relational objects

    # get centroid
    srf = typhon.physics.units.em.SRF.fromRTTOV(
        typhon.datasets.tovs.norm_tovs_name(ds.satellite, mode="RTTOV"),
        "hirs", channel)
    cntr = srf.centroid().to("µm", "sp")
    shared_tit = (f"{ds.satellite:s} channel {channel:d} ({cntr:.4~}) ")
    period_tit = f"{start:%Y-%m-%d %H:%M:%S} – {end:%H:%M:%S}"

    shared_fn = (f"{ds.satellite:s}_ch{channel:d}_{start:%Y%m%d%H%M%S}-{end:%H%M%S}")
    # cross-element error correlation function
    (f, a) = matplotlib.pyplot.subplots(1, 1, figsize=(8, 4.5))
    a.plot(Δ_e_full["Δp"], Δ_e_full.sel(n_c=channel))
    a.set_xlabel("Δe")
    a.set_ylabel("$[r_e]_{\Delta e}$")
    a.set_title("Cross-element error correlation function "
                + shared_tit + "\n" + period_tit)
    pyatmlab.graphics.print_or_show(f, False,
        "curuc/cross_element_error_correlation_function_"+shared_fn+".png")

    # cross-line error correlation function
    (f, a) = matplotlib.pyplot.subplots(1, 1, figsize=(8, 4.5))
    a.plot(Δ_l_full["Δp"], Δ_l_full.sel(n_c=channel))
    a.set_xlabel("Δl")
    a.set_ylabel("$[r_l]_{\Delta l}$")
    a.set_title("Cross-line error correlation function "
                + shared_tit + "\n" + period_tit)
    pyatmlab.graphics.print_or_show(f, False,
        "curuc/cross_line_error_correlation_function_"+shared_fn+".png")

    cmap = "magma_r"
    imshow_args = {"cmap": cmap, "interpolation": None, "origin": "upper"}
    for (x, y) in zip(x_all, y_all):
        scnlinlab = "scanline at {:%Y-%m-%d %H:%M:%S}".format(
            ds["time"].isel(y=y).values.astype("M8[ms]").item())
        # cross-element error covariance matrix for line
        (f, a) = matplotlib.pyplot.subplots(1, 1, figsize=(8, 6))
        S = D["S_esΛl"][channel-1, y-lines[0], :, :]
        S = _S_radsi_to_K(S, srf=srf)
        #p = a.pcolor(S.m, cmap=cmap)
        p = a.imshow(S.m, **imshow_args)
        cb = f.colorbar(p)
        cb.set_label("$S_{es}^l$ [K$^2$]")
        a.set_xlabel("$\Delta$e")
        a.set_ylabel(a.get_xlabel())
        a.set_title("Cross-element error covariance matrix "
            + shared_tit
            + "\n"
            + scnlinlab)
        a.set_aspect("equal")
        pyatmlab.graphics.print_or_show(f, False,
            "curuc/cross_element_S" + shared_fn +
            f"x{x:d}y{y:d}.png")

        # cross-line error covariance matrix for element
        (f, a) = matplotlib.pyplot.subplots(1, 1, figsize=(8, 6))
        S = D["S_lsΛe"][channel-1, x, :, :]
        S = _S_radsi_to_K(S, srf=srf)
        #p = a.pcolor(S.m, cmap=cmap)
        p = a.imshow(S.m, **imshow_args)
        cb = f.colorbar(p)
        cb.set_label("$S_{ls}^e$ [K$^2$]")
        a.set_xlabel("$\Delta$l")
        a.set_ylabel(a.get_xlabel())
        a.set_title("Cross-line error covariance matrix "
            + "\n"
            + shared_tit
            + f"element {x:d}")
        a.set_aspect("equal")
        pyatmlab.graphics.print_or_show(f, False,
            "curuc/cross_line_S" + shared_fn +
            f"_x{x:d}y{y:d}.png")

        # cross-channel error covariance matrix
        (f, a) = matplotlib.pyplot.subplots(1, 1, figsize=(8, 6))
        S = D["S_csΛp"].sel(n_l=y-lines[0], n_e=x)
        S = _S_radsi_to_K(S, srf=srf)
        #p = a.pcolor(S.m, cmap=cmap)
        p = a.imshow(S.m, **imshow_args)
        cb = f.colorbar(p)
        cb.set_label("$S_{cs}^p$ [K$^2$]")
        a.set_xlabel("channel")
        a.set_ylabel(a.get_xlabel())
        a.set_title("Cross-channel error covariance matrix "
            + shared_tit
            + "\n"
            + scnlinlab
            + f" element {x:d}")
        a.set_aspect("equal")
        a.set_xticks(numpy.arange(ds.dims["channel"]))
        a.set_xticklabels([str(x.item()) for x in ds["channel"]])
        a.set_yticks(numpy.arange(ds.dims["channel"]))
        a.set_yticklabels([str(x.item()) for x in ds["channel"]])
        pyatmlab.graphics.print_or_show(f, False,
            "curuc/cross_channel_S" + shared_fn +
            f"_x{x:d}y{y:d}.png")

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
    set_logger(logging.DEBUG if p.verbose else logging.INFO)
    plot_curuc_for_pixels(
        xarray.open_dataset(p.path),
        lines=p.lines,
        x_all=p.x_all,
        y_all=p.y_all,
        channel=p.channel)

#    sys.exit("The development of this script is still in progress.")
#    plot_compare_correlation_scanline(xarray.open_dataset(p.path))
