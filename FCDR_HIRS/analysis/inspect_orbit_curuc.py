"""Plots to test and inspect results of CURUC

Includes functionality for figures 3 and 4 from
Merchant, Holl, ... (submitted 2018).
"""

import datetime
import argparse
import sys
import pathlib
import logging
import numbers

import matplotlib
import matplotlib.pyplot
import numpy
import xarray

import typhon.physics.units.em
import typhon.datasets.tovs
from typhon.physics.units.common import radiance_units as rad_u, ureg
from typhon.physics.units.tools import UnitsAwareDataArray as UADA   

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

def _S_radsi_to_K(S, srf, ref_L):
    """Convert covariance matrix in radiance to one in BT

    For cross-line and cross-element covariances, this is done by adding
    sqrt(S) to ref_L, converting to BT, then subtracting ref_BT and
    squaring.

    For cross-channel covariances, it is more complicated.
    In pseudocode::

        S_bt[i, j] = (((ref_L[i] + u_L[i, j]).to("K") - ref_bt[i]) *
                      ((ref_L[j] + u_L[i, j]).to("K") - ref_bt[j]))

    Arguments:

        S

            Covariance matrix in rad_si units

        srf

            Either single typhon.physics.units.em.SRF or list
            thereoff

        ref_L

            Reference radiances in rad_si units, relative to which the
            convertion to S in BT will be performed

    """
    S_L = S.values * rad_u["si"]**2
    sgn = numpy.sign(S_L)
    u_L = numpy.sqrt(numpy.abs(S_L))
    if isinstance(ref_L, (numbers.Number, numpy.ndarray)):
        ref_L = ureg.Quantity(ref_L, rad_u["si"])
    elif isinstance(ref_L, xarray.DataArray):
        ref_L = UADA(ref_L)
        if not "units" in ref_L.attrs.keys():
            ref_L.attrs["units"] = rad_u["si"]
    else:
        raise TypeError("expected number of array for ref_L, got {!s}".format(
            type(ref_L)))

    if isinstance(srf, list): # FIXME: should be sequence more flexible
        ref_bt = xarray.zeros_like(ref_L)
        ref_bt.attrs["units"] = "K"
        ref_bt.values[...] = [L.to("K", "radiance", srf=srf[i]) for (i, L) in enumerate(ref_L)]
        u_L = UADA(
            u_L.m,
            attrs={"units":u_L.u},
            dims=("calibrated_channel", "calibrated_channel"),
            coords={"calibrated_channel": ref_L["calibrated_channel"]})
        ref_bt = ureg.Quantity(ref_bt.values, ref_bt.attrs["units"])
        S_bt = sgn * numpy.array(
            [((ureg.Quantity(ref_L.values[i]+u_L.values[i,j], rad_u["si"]).to("K", "radiance", srf=srf1)-ref_bt[i]) *
              (ureg.Quantity(ref_L.values[j]+u_L.values[i,j], rad_u["si"]).to("K", "radiance", srf=srf2)-ref_bt[j])).m
             for (i, srf1) in enumerate(srf)
             for (j, srf2) in enumerate(srf)]).reshape(
                (len(srf), len(srf))) # transpose?
        S_bt = ureg.Quantity(S_bt, ureg.K**2)
    else: # for each channel
        ref_bt = ref_L.to("K", "radiance", srf=srf)
        u_bt = (ref_L + u_L).to("K", "radiance", srf=srf)-ref_bt
        S_bt = sgn*u_bt**2

    return S_bt

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
    srf = fg.fcdr.srfs[channel-1]
    cntr = srf.centroid().to("µm", "sp")
    shared_tit = (f"{ds.satellite:s} channel {channel:d} ({cntr:.4~}) ")
    period_tit = f"{start:%Y-%m-%d %H:%M:%S} – {end:%H:%M:%S}"

    shared_fn = (f"{ds.satellite:s}_ch{channel:d}_{start:%Y%m%d%H%M%S}-{end:%H%M%S}")
    # cross-element error correlation function
    (f, a) = matplotlib.pyplot.subplots(1, 1, figsize=(8, 4.5))
    a.plot(Δ_e_full["Δp"], Δ_e_full.sel(n_c=channel))
    a.set_xlabel("separation between elements")
    a.set_ylabel("mean correlation coefficient")
    a.set_title("Cross-element error correlation function "
                + shared_tit + "\n" + period_tit)
    pyatmlab.graphics.print_or_show(f, False,
        "curuc/cross_element_error_correlation_function_"+shared_fn+".png")

    # cross-line error correlation function
    (f, a) = matplotlib.pyplot.subplots(1, 1, figsize=(8, 4.5))
    a.plot(Δ_l_full["Δp"], Δ_l_full.sel(n_c=channel))
    a.set_xlabel("separation between lines")
    a.set_ylabel("mean correlation coefficient")
    a.set_title("Cross-line error correlation function "
                + shared_tit + "\n" + period_tit)
    pyatmlab.graphics.print_or_show(f, False,
        "curuc/cross_line_error_correlation_function_"+shared_fn+".png")

    cmap = "magma_r"
    imshow_args = {"cmap": cmap, "interpolation": None, "origin": "upper"}
    for (x, y) in zip(x_all, y_all):
        y_new = abs(ds_new["time"]-ds.sel(y=y)["time"]).argmin().item()
        scnlinlab = "scanline at {:%Y-%m-%d %H:%M:%S}".format(
            ds["time"].isel(y=y).values.astype("M8[ms]").item())
        # cross-element error covariance matrix for line
        (f, a) = matplotlib.pyplot.subplots(1, 1, figsize=(8, 6))
        S = D["S_esΛl"][channel-1, y-lines[0], :, :]
        S = _S_radsi_to_K(S, srf=srf,
            ref_L=ds_new.isel(scanline_earth=y_new,scanpos=x,calibrated_channel=channel)["R_e"].item())
        #p = a.pcolor(S.m, cmap=cmap)
        p = a.imshow(S.m, **imshow_args)
        cb = f.colorbar(p)
        cb.set_label("Covariance [K$^2$]")
        a.set_xlabel("separation between elements")
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
        S = _S_radsi_to_K(S, srf=srf,
            ref_L=ds_new.isel(scanline_earth=y_new,scanpos=x,calibrated_channel=channel)["R_e"].item())
        #p = a.pcolor(S.m, cmap=cmap)
        p = a.imshow(S.m, **imshow_args)
        cb = f.colorbar(p)
        cb.set_label("Covariance [K$^2$]")
        a.set_xlabel("separation between lines")
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
        S = _S_radsi_to_K(S, srf=fg.fcdr.srfs,
            ref_L=ds_new.isel(scanline_earth=y_new,scanpos=x)["R_e"])
        #p = a.pcolor(S.m, cmap=cmap)
        p = a.imshow(S.m, vmax=sorted(S.m.flat)[-2], **imshow_args)
        cb = f.colorbar(p)
        cb.set_label("Covariance [K$^2$]")
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
