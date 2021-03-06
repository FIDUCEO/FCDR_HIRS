"""Plots to test and inspect results of CURUC

Includes functionality for figures 3 and 4 from
Merchant, Holl, ... (2019).
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

from ..processing.generate_fcdr import FCDRGenerator
from ..common import (set_logger, add_to_argparse)
from .. import metrology
from .. import graphics

def get_parser():    
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

    return parser
def parse_cmdline():
    return get_parser().parse_args()

def _S_radsi_to_K(S, u_radK):
    """Convert covariance matrix in radiance to one in BT

    This is done by first converting covariance matrices to correlation
    matrices using one set of uncertainties, then back with another.

    Parameters
    ----------

    S : xarray.DataArray
        Covariance matrix in rad_si units
    u_radK : xarray.DataArray
        Uncertainties in brightness temperature in K.

    Returns
    -------

    ureg.Quantity
        Covariance matrix in brightness temperature

    """
    S_L = ureg.Quantity(S.values, rad_u["si"]**2)
    uRe = ureg.Quantity(numpy.sqrt(numpy.diag(S_L)), rad_u["si"])
    R = S_L / (uRe[:, numpy.newaxis]*uRe[numpy.newaxis, :])  
    S_BT = R * (u_radK[:, numpy.newaxis]*u_radK[numpy.newaxis, :])
    return S_BT

def plot_curuc_for_pixels(ds, lines, channel, x_all, y_all):
    """Plot some CURUC stats for specific pixels

    Based on a segment of a L1B HIRS dataset, calculate the FCDR and
    properties of CURUC.  For one or more pixels provided by the user,
    calculate and visualise:

    The cross-element error covariance matrix:

    .. image:: /images/cross_element_Smetopb_ch7_20160227173937-174945x30y458.png

    The cross-line error covariance matrix:

    .. image:: /images/cross_line_Smetopb_ch7_20160227173937-174945_x30y458.png
    
    The cross-channel error covariance matrix:

    .. image:: /images/cross_channel_S_correlated_noise_metopb_ch7_20160227173937-174945.png

    The cross-channel error correlation matrix:

    .. image:: /images/cross_channel_R_correlated_noise_metopb_ch7_20160227173937-174945.png

    And for the overall segment, calculate and visualise:

    The cross-line error correlation function:

    .. image:: /images/cross_line_error_correlation_function_metopb_ch7_20160227173937-174945.png

    And finally, the cross-element error correlation function:

    .. image:: /images/cross_element_error_correlation_function_metopb_ch7_20160227173937-174945.png


    Parameters
    ----------

    ds : xarray.Dataset
        Dataset from which to plot CURUC

    lines : [int, int]
        Range of lines to select from ``ds``

    channel : int
        Channel for which to show CURUC
    
    x_all : array_like
        Array of integers containing x-coordinates for which to show the
        correlation and covariance matrices

    y_all : array_like
        Array of integers containing y-coordinates for which to show the
        correlation and covariance matrices

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

    del sensRe # this causes get_verbose_stack_description to
               # fail as pprint.pprint can't sort Relational objects,
               # perhaps replacing pickle by dill will help

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
    graphics.print_or_show(f, False,
        "curuc/cross_element_error_correlation_function_"+shared_fn+".")

    # cross-line error correlation function
    (f, a) = matplotlib.pyplot.subplots(1, 1, figsize=(8, 4.5))
    a.plot(Δ_l_full["Δp"], Δ_l_full.sel(n_c=channel))
    a.set_xlabel("separation between scanlines")
    a.set_ylabel("mean correlation coefficient")
    a.set_title("Cross-line error correlation function "
                + shared_tit + "\n" + period_tit)
    graphics.print_or_show(f, False,
        "curuc/cross_line_error_correlation_function_"+shared_fn+".")

    cmap = "magma_r"
    imshow_args = {"cmap": cmap, "interpolation": None, "origin": "upper"}
    for (x, y) in zip(x_all, y_all):
        y_new = abs(ds_new["scanline_earth"]-ds.sel(y=y)["time"]).argmin().item()
        scnlinlab = "scanline at {:%Y-%m-%d %H:%M:%S}".format(
            ds["time"].isel(y=y).values.astype("M8[ms]").item())
        # cross-element error covariance matrix for line
        (f, a) = matplotlib.pyplot.subplots(1, 1, figsize=(8, 6))
        S = D["S_esΛl"].sel(n_c=channel, n_l=y-lines[0])
        # WARNING FIXME: what is the correct BT to put in?
        S = _S_radsi_to_K(S,
             u_radK=ds_new.sel(calibrated_channel=channel).isel(scanline_earth=y_new)["u_T_b_nonrandom"].values)
        #p = a.pcolor(S.m, cmap=cmap)
        p = a.imshow(S.m, **imshow_args)
        cb = f.colorbar(p)
        cb.set_label("Covariance / K$^2$")
        a.set_xlabel("element")
        a.set_ylabel(a.get_xlabel())
        a.set_title("Cross-element error covariance matrix")
#            + shared_tit
#            + "\n"
#            + scnlinlab)
        a.set_aspect("equal")
        graphics.print_or_show(f, False,
            "curuc/cross_element_S" + shared_fn +
            f"x{x:d}y{y:d}.")

        # cross-line error covariance matrix for element
        (f, a) = matplotlib.pyplot.subplots(1, 1, figsize=(8, 6))
        S = D["S_lsΛe"].sel(n_c=channel, n_e=x-1)
        # WARNING FIXME: what is the correct BT to put in?
        S = _S_radsi_to_K(S,
             u_radK=ds_new.sel(calibrated_channel=channel).isel(scanpos=x-1)["u_T_b_nonrandom"].values)
        #p = a.pcolor(S.m, cmap=cmap)
        p = a.imshow(S.m, **imshow_args)
        cb = f.colorbar(p)
        cb.set_label("Covariance / K$^2$")
        a.set_xlabel("scanline")
        a.set_ylabel(a.get_xlabel())
        a.set_title("Cross-line error covariance matrix")
#            + "\n"
#            + shared_tit
#            + f"element {x:d}")
        a.set_aspect("equal")
        graphics.print_or_show(f, False,
            "curuc/cross_line_S" + shared_fn +
            f"_x{x:d}y{y:d}.")

        # cross-channel error covariance matrix
        (f, a) = matplotlib.pyplot.subplots(1, 1, figsize=(8, 6))
        S = D["S_csΛp"].sel(n_l=y-lines[0], n_e=x-1)
        # WARNING FIXME: what is the correct BT to put in?
        S = _S_radsi_to_K(S,
             u_radK=ds_new.isel(scanpos=x-1,scanline_earth=y_new)["u_T_b_nonrandom"].values)
        #p = a.pcolor(S.m, cmap=cmap)
        p = a.imshow(S.m, vmax=sorted(S.m.flat)[-2], **imshow_args)
        cb = f.colorbar(p)
        cb.set_label("Covariance / K$^2$")
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
        graphics.print_or_show(f, False,
            "curuc/cross_channel_S" + shared_fn +
            f"_x{x:d}y{y:d}.")

        # cross-channel correlation matrix for correlated noise effect only
        (f, a) = matplotlib.pyplot.subplots(1, 1, figsize=(8, 6))
        R = ds_new["channel_correlation_matrix"]
        p = a.imshow(R.values[:19, :19], vmax=1, vmin=-1, cmap="PuOr_r",
            interpolation="none", origin="upper")
        cb = f.colorbar(p)
        cb.set_label("Correlation coefficient")
        a.set_xlabel("channel")
        a.set_ylabel(a.get_xlabel())
        a.set_title(
            "Cross-channel error correlation matrix due to correlated noise\n")
#            shared_tit)
        a.set_aspect("equal")
        a.set_xticks(numpy.arange(ds.dims["channel"]))
        a.set_xticklabels([str(x.item()) for x in ds["channel"]])
        a.set_yticks(numpy.arange(ds.dims["channel"]))
        a.set_yticklabels([str(x.item()) for x in ds["channel"]])
        graphics.print_or_show(f, False,
            "curuc/cross_channel_R_correlated_noise_" + shared_fn +
            f".")

        # cross-channel covariance matrix for correlated noise effect only
        (f, a) = matplotlib.pyplot.subplots(1, 1, figsize=(8, 6))
        u_radK=ds_new.isel(scanpos=x-1,scanline_earth=y_new)["u_T_b_random"].values
        S = R.values[:, :19][:19, :] * (u_radK[:, numpy.newaxis]*u_radK[numpy.newaxis, :])
        p = a.imshow(S, vmax=sorted(S.flat)[-2], **imshow_args)
        cb = f.colorbar(p)
        cb.set_label("Covariance / K$^2$")
        a.set_xlabel("channel")
        a.set_ylabel(a.get_xlabel())
        a.set_title("Cross-channel error covariance matrix due to correlated noise\n"
            + shared_tit)
        a.set_aspect("equal")
        a.set_xticks(numpy.arange(ds.dims["channel"]))
        a.set_xticklabels([str(x.item()) for x in ds["channel"]])
        a.set_yticks(numpy.arange(ds.dims["channel"]))
        a.set_yticklabels([str(x.item()) for x in ds["channel"]])
        graphics.print_or_show(f, False,
            "curuc/cross_channel_S_correlated_noise_" + shared_fn +
            f".")

def plot_compare_correlation_scanline(ds):
    """Do not use.

    Use :func:`plot_curuc_for_pixels` instead.

    Do not use.
    """
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

    graphics.print_or_show(f, False, "orbit_curuc_test.")

def main():
    """Main function, expects cmdline input.

    See module and script documentation.
    """
    p = parse_cmdline()
    set_logger(logging.DEBUG if p.verbose else logging.INFO,
        loggers={"FCDR_HIRS", "typhon"})
    plot_curuc_for_pixels(
        xarray.open_dataset(p.path),
        lines=p.lines,
        x_all=p.x_all,
        y_all=p.y_all,
        channel=p.channel)

#    sys.exit("The development of this script is still in progress.")
#    plot_compare_correlation_scanline(xarray.open_dataset(p.path))

