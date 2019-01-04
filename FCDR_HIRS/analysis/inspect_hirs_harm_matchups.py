"""Plotting the health of harmonisation matchups

This module, which can be run as the script
:ref:`hirs-inspect-harm-matchups`, performs two main tasks:

- Plot statistics for a generated harmonisation enhance matchup file, such
  as produced by running first :ref:`combine-hirs-hirs-matchups` and then
  :ref:`merge-hirs-harmonisation`.  For a single file, it will plot
  (currently) 11 panels showing the statistics of K, Kr, Ks, measurands,
  measurand differences, and uncertainties, in various combinations.  The
  input harmonisation matchup file may either be filtered or unfiltered.
- If the input harmonisation matchup file is unfiltered, as well as
  writing out statistics, the script will also derive parameters for two
  filters: :class:`~matchups.KrFilterFeltaLKr` and
  :class:`matchups.KFilterKDeltaL`.  See
  :func:`plot_hist_with_medmad_and_fitted_normal`.
"""

import argparse
from .. import common

import logging

import sys
import math
import pathlib
import unicodedata

import numpy
import matplotlib.pyplot
import xarray
import scipy.stats

from typhon.physics.units.tools import UnitsAwareDataArray as UADA
from typhon.physics.units.common import radiance_units as rad_u
import typhon.physics.units.em
import typhon.config

from .. import matchups
from .. import fcdr
from .. import common
from .. import graphics

logger = logging.getLogger(__name__)
def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = common.add_to_argparse(parser,
        include_period=False,
        include_sat=0,
        include_channels=False,
        include_temperatures=False)

    parser.add_argument("file",
        action="store",
        type=str,
        help="Path to file containing enhanced matchups")

    parser.add_argument("--write-filters",
        action="store_true",
        help="Write filters to files") 

    return parser
def parse_cmdline():
    return get_parser().parse_args()

def plot_hist_with_medmad_and_fitted_normal(a, y, rge, xlab, ylab, tit,
        write=False, max_ratcorr=0.02, rge_fact=0.1, max_tries=30):
    """Plot histogram with fitted normal and stats and write filters

    This plots:

    - A histogram of the data in ``y``, as a bar graph, normalised such
      that it shows the probability density.
    - A fitted normal distribution, as a line graph.  The normal
      distribution is fitted such that the mean of the normal distribution
      corresponds to the median of ``y``, and the standard deviation
      of the normal distribution corresponds to the 67th percentile of
      the absolute deviation corresponds of ``y``.  This means that if
      ``y`` is normally distributed, the normal distribution will show
      exactly on top of the histogram.
    - Lines at the median and various times the median absolute
      deviation from the median.
    - The ratio of the fitted normal distribution to the probability
      density in ``y``.
    
    The ratio of the fitted normal to the probability density is also
    written to a file.  Depending on the quantity that is being plotted,
    the filters :class:`filters.KrFilterDeltaLKr` and
    :class:`filters.KFilterKDeltaL` are written to a file, so that they
    can be used by those filters.  This is designed to be useful when this
    function is used on /unfiltered/ matchups.  This output is only
    written if write=True.  The destination is determined by the
    ``harmfilterparams`` field in the ``main`` section of the
    configuration file (see :ref:`configuration`).

    Parameters
    ----------

    a : axes
        Axes object in which to place the plots.
    y : Data to be plotted and fitted to.
    rge : [float, float]
        Range over which the histogram is plotted.
    xlab : str
        xlabel on plot
    ylab : str
        ylabel on plot
    tit : str
        title on plot
    write : bool, optional
        If True, write the ratio to a file in the ``harmfilterparams``
        directory in the ``main`` section of configuration.
    max_ratcorr : float
        The maximum tolerable corrected ratio between the fitted normal
        distribution and the q9.  If, at the beginning or
        the end of the range, the ratio is larger than this value
    rge_fact : float
        Factor at which to increase range iteratively
    max_tries : int
        Maximum number of tries to increase range at
    """
    med = y.median()
    mad = abs(y-med).median()
    p67ad = scipy.stats.scoreatpercentile(abs(y-med), 68.3)
    # also plot fitted normal distribution
    # NB: how about outliers or non-normal dists?
    # "robust standard deviation"
    # extend the range until we cover sufficient
    rge = numpy.asarray(rge)
    for i in range(max_tries):
        (dens, bins) = numpy.histogram(
            y,
            bins="auto",
            range=rge,
            density=True)
        peak = dens.max()
        σ_fitting_peak = 1/(peak*math.sqrt(2*math.pi))
        midbins = (bins[1:]+bins[:-1])/2
        x = numpy.linspace(*rge, 500)
        norm_fitting_peak = scipy.stats.norm.pdf(x, med, σ_fitting_peak)
        rat = scipy.stats.norm.pdf(midbins, med, σ_fitting_peak) / dens
        ratcorr = numpy.where(rat<=1, rat, 1)
        # but for bins where histogram shows zero, I want ratcorr 0 not 1
        ratcorr = numpy.where(dens==0, 0, ratcorr)
        #if ratcorr[0] > max_ratcorr and dens[0]>0:
        if dens[0]>0:
            logger.debug(f"Extending lower range for {write:s} beyond "
                f"{ratcorr[0]:10.3e} at {rge[0]:.5f}")
            rge[0] -= rge.ptp()*rge_fact
        #elif ratcorr[-1] > max_ratcorr and dens[-1]>0:
        elif dens[-1]>0:
            logger.debug(f"Extending upper range for {write:s} beyond "
                f"{ratcorr[-1]:10.3e} at {rge[-1]:.5f}")
            rge[-1] += rge.ptp()*rge_fact
        else:
            break
    else:
        raise ValueError("Histogram falls of faster-than-exponential, "
            f"cannot derive acceptance ratio after {i:d} tries!")
    for i in range(-9, 10, 3):
        a.plot([med+i*mad]*2, [0, dens.max()], color="red")
        a.text(med+i*mad, .8*dens.max(), str(i))

    (dens, bins, patches) = a.hist(
        y,
        histtype="step",
        bins="auto",
        range=rge,
        density=True)

    a.plot(x, norm_fitting_peak, color="black", label="fitted 1")
    a2 = a.twinx()
    a2.plot(midbins, ratcorr, 'k--')
    a2.set_ylabel("filter: P(keep)")
    a2.set_ylim([0, 1])
    a.set_xlabel(xlab)
    a.set_ylabel(ylab)
    a.set_title(f"Histogram of {tit:s} with MADs from MED\n"
                f"MAD={mad.item():.3f}, MED={med.item():.3f}, P67AD={p67ad:.3f}")
    a.set_xlim(rge)
    if write:
        hfpdir = pathlib.Path(typhon.config.conf["main"]["harmfilterparams"])
        # turn into ASCII because some LOTUS nodes dislike unicode?
        write = write.replace("-1²", "-2")
        write = unicodedata.normalize("NFKC", write)
        hfpfile = (hfpdir / write).with_suffix(".nc")
        hfpfile.parent.mkdir(parents=True, exist_ok=True)
        da = xarray.DataArray(ratcorr,
            dims=("x",),
            coords={"x": midbins},
            name="y")
        logger.info(f"Writing filter to {hfpfile!s}")
        da.to_netcdf(hfpfile)

def plot_ds_summary_stats(ds, lab="", Ldb=None, write=False):
    """Plot statistics for enhanced matchup harmonisation file
    
    This function plots statistics for enhanced matchup harmonisation
    files, such as these:

    .. image:: /images/harmonisation-stats.png

    In addition, if ``write`` is ``True``, it also writes filter
    parameters --- see :func:`plot_hist_with_medmad_and_fitted_normal` for
    details.

    The resulting plot is written to a file.

    Parameters
    ----------

    ds : xarray.Dataset
        Dataset from which to plot summaries.  This dataset must
        correspond to the format as defined by Sam Hunt (W-matrix file)
        and as written out by `FCDR_HIRS.processing.analysis.merge_all`.
    lab : str, optional
        Additional ``debug`` label to describe the matchup file.  This can
        be empty for a standard plot, or have a string such as
        ``neighbours_delta_cm·mW m^-2 sr^-1``, which corresponds to the
        ``--debug`` option of the :ref:`combine-hirs-hirs-matchups`
        script.
    Ldb : xarray.Dataset
        Dataset describing the HIRS-IASI dataset model, such as
        initialised by :meth:`matchups.init_Ldb`.
    write : bool, optional
        Will be passed on to
        :func:`plot_hist_with_medmad_and_fitted_normal`; if True, write
        out filter parameters.  Defaults to False.

    """

    if lab:
        # extra cruft added to string by combine_hirs_hirs_matchups
        lab = f"other_{lab:s}_"
    
    (f, ax_all) = matplotlib.pyplot.subplots(3, 5, figsize=(30, 15))

    g = ax_all.flat

    cbs = []
    
    chan = ds["channel"].item()
    # for unit conversions
    srf1 = typhon.physics.units.em.SRF.fromArtsXML(
            typhon.datasets.tovs.norm_tovs_name(ds.sensor_1_name).upper(),
            "hirs", ds["channel"].item())
    srf2 = typhon.physics.units.em.SRF.fromArtsXML(
            typhon.datasets.tovs.norm_tovs_name(ds.sensor_2_name).upper(),
            "hirs", ds["channel"].item())

    y1 = UADA(ds["nominal_measurand1"]).to(
            ds[f"K_{lab:s}forward"].units, "radiance", srf=srf1)
    y2 = UADA(ds["nominal_measurand2"]).to(
            ds[f"K_{lab:s}forward"].units, "radiance", srf=srf2)
    yb = [y1, y2]

    plo, phi = 1, 99
    while True:
        kxrange = scipy.stats.scoreatpercentile(ds[f"K_{lab:s}forward"], [1, 99])
        kyrange = scipy.stats.scoreatpercentile(ds[f"K_{lab:s}backward"], [1, 99])
        kΔrange = scipy.stats.scoreatpercentile(ds[f"K_{lab:s}forward"]+ds[f"K_{lab:s}backward"], [1, 99])
        Lxrange = scipy.stats.scoreatpercentile(y1, [1, 99])
        Lyrange = scipy.stats.scoreatpercentile(y2, [1, 99])
        Lmax = max(Lxrange[1], Lyrange[1])
        Lmin = min(Lxrange[0], Lyrange[0])
        LΔrange = scipy.stats.scoreatpercentile(
            y2 - y1,
            [1, 99])
        if all(max(abs(rng))/min(abs(rng))<100
               for rng in (kxrange, kyrange, kΛrange,
                           Lxrange, Lyrange, LΔrange)):
            break
        else:
            plo += 4
            phi -= 4
        if not plo < phi:
            raise ValueError("Can't retrieve a reasonable range, all outliers?!")
        

    # radiance comparison
    a = next(g)
    pc = a.hexbin(
        y1,
        y2,
        extent=(Lmin, Lmax, Lmin, Lmax),
        mincnt=1)
    a.plot([Lmin, Lmax], [Lmin, Lmax], 'k--')
    a.set_xlabel("Radiance {sensor_1_name:s}".format(**ds.attrs)
        + f"[{y1.units:s}]")
    a.set_ylabel("Radiance {sensor_2_name:s}".format(**ds.attrs)
        + f"[{y2.units:s}]")
    a.set_title("Radiance comparison")
    a.set_xlim(Lmin, Lmax)
    a.set_ylim(Lmin, Lmax)
    cbs.append(f.colorbar(pc, ax=a))

    # histograms for real and simulated measurements
    a = next(g)
    sensor_names = [ds.sensor_1_name, ds.sensor_2_name]
    for i in range(2):
        (cnts, bins, patches) = a.hist(
            yb[i],
            label=f"{sensor_names[i]:s} (measured)",
            histtype="step",
            range=(Lmin, Lmax),
            density=True,
            stacked=False,
            bins=100)
    for nm in Ldb.data_vars.keys():
        (cnts, bins, patches) = a.hist(
            Ldb[nm].sel(chan=chan),
            label=f"{nm:s} (IASI-simulated)",
            histtype="step",
            range=(Lmin, Lmax),
            density=True,
            stacked=False,
            bins=100)
    a.legend()
    a.set_xlabel("Radiance " + f"[{y1.units:s}]")
    a.set_ylabel("Density per bin")
    a.set_title("Histograms of radiances")

    # K forward vs. K backward
    a = next(g)
    pc = a.hexbin(
        ds[f"K_{lab:s}forward"],
        ds[f"K_{lab:s}backward"],
       extent=numpy.concatenate([kxrange, kyrange]),
       mincnt=1)
    a.plot(kxrange, -kxrange, 'k--')
    a.set_xlabel("K forward\n[{units:s}]".format(**ds[f"K_{lab:s}forward"].attrs))  
    a.set_ylabel("K backward\n[{units:s}]".format(**ds[f"K_{lab:s}backward"].attrs))  
    a.set_title("Estimating K forward or backward, comparison")
    a.set_xlim(kxrange)
    a.set_ylim(kyrange)
    cbs.append(f.colorbar(pc, ax=a))

    # histogram of K forward / backward differences
    a = next(g)
    (cnts, bins, patches) = a.hist(
        ds[f"K_{lab:s}forward"]+ds[f"K_{lab:s}backward"],
        histtype="step",
        bins=100,
        range=kΔrange)
    a.plot([0, 0], [0, cnts.max()], 'k--')
    a.set_xlabel("Sum of K estimates [{units:s}]".format(**ds[f"K_{lab:s}forward"].attrs))
    a.set_ylabel("No. matchups in bin")
    a.set_title("Distribution of sum of K estimates")
    a.set_xlim(kΔrange)

    # Ks vs. Kforward
    a = next(g)
    pc = a.hexbin(
        ds[f"K_{lab:s}forward"],
        ds[f"K_{lab:s}forward"]+ds[f"K_{lab:s}backward"],
        extent=numpy.concatenate([kxrange, kΔrange]),
        mincnt=1)
    a.plot(kxrange, [0, 0], 'k--')
    a.set_xlabel("K forward\n[{units:s}]".format(**ds[f"K_{lab:s}forward"].attrs))  
    a.set_ylabel("Sum of K estimates [{units:s}]".format(**ds[f"K_{lab:s}forward"].attrs))
    a.set_title("K difference vs. K forward")
    a.set_xlim(kxrange)
    a.set_ylim(kΔrange)
    cbs.append(f.colorbar(pc, ax=a))

    # K vs. radiance
    a = next(g)
    pc = a.hexbin(y1,
        ds[f"K_{lab:s}forward"],
        extent=numpy.concatenate([Lxrange, kxrange]),
        mincnt=1)
    a.set_xlabel("Radiance {sensor_1_name:s}".format(**ds.attrs)
        + f"[{y1.units:s}]")
    a.set_ylabel("K forward\n[{units:s}]".format(**ds[f"K_{lab:s}forward"].attrs))  
    a.set_title("K vs. measurement")
    a.set_xlim(Lxrange)
    a.set_ylim(kxrange)
    cbs.append(f.colorbar(pc, ax=a))

    # K vs. ΔL
    a = next(g)
    extremes = [min([LΔrange[0], kxrange[0]]), max([LΔrange[1], kxrange[1]])]
    ΔL = y2-y1
    pc = a.hexbin(ΔL,
        ds[f"K_{lab:s}forward"],
        extent=numpy.concatenate([LΔrange, kxrange]),
        mincnt=1)
    a.plot(extremes, extremes, 'k--')
    a.set_xlabel("Radiance {sensor_2_name:s} - {sensor_1_name:s}".format(**ds.attrs)
        + f"[{y1.units:s}]")
    a.set_ylabel("K forward\n[{units:s}]".format(**ds[f"K_{lab:s}forward"].attrs))  
    a.set_title("K vs. measurement difference")
    a.set_xlim(LΔrange)
    a.set_ylim(kxrange)
    cbs.append(f.colorbar(pc, ax=a))

    # K - ΔL vs. radiance
    a = next(g)
    K_min_ΔL = ds[f"K_{lab:s}forward"] - ΔL
    pc = a.hexbin(y1,
        K_min_ΔL,
        extent=numpy.concatenate([[Lmin, Lmax], kxrange-LΔrange]),
        mincnt=1)
    a.plot([0, Lmax], [0, 0], 'k--')
    a.set_xlabel("Radiance {sensor_1_name:s}".format(**ds.attrs)
        + f"[{y1.units:s}]")
    a.set_ylabel(f"K - ΔL [{y1.units:s}]".format(**ds.attrs))
    a.set_xlim(Lmin, Lmax)
    a.set_ylim(sorted(kxrange-LΔrange))
    a.set_title('K "wrongness" per radiance')
    cbs.append(f.colorbar(pc, ax=a))

    # Kr / u_independent for both
    a = next(g)
    # awaiting having u_independent in files
    Kr_K = (
        ((UADA(ds["nominal_measurand1"])+UADA(ds["Kr"])).to(
            ds[f"K_{lab:s}forward"].units, "radiance", srf=srf1) -
         UADA(ds["nominal_measurand1"]).to(
            ds[f"K_{lab:s}forward"].units, "radiance", srf=srf1)))
    Kr_K99 = min(scipy.stats.scoreatpercentile(Kr_K, 99),
                 10*Kr_K.median().item())
    (cnts, bins, p1) = a.hist(
        Kr_K,
        histtype="step",
        bins=100,
        #density=True,
        range=[0, Kr_K99])
    a.set_xlabel(f"Kr [{y1.units:s}]")
    a.set_ylabel("Count")
    a.set_xlim([0, Kr_K99])
    # now with u
    u1 = ds["nominal_measurand_uncertainty_independent1"]
    u2 = ds["nominal_measurand_uncertainty_independent2"]
    # workaround, I forgot to add units
    u1.attrs["units"] = ds["nominal_measurand1"].attrs["units"]
    u2.attrs["units"] = ds["nominal_measurand2"].attrs["units"]
    u1_K = ((UADA(ds["nominal_measurand1"])+UADA(u1)).to(
            ds[f"K_{lab:s}forward"].units, "radiance", srf=srf1) -
             UADA(ds["nominal_measurand1"]).to(
            ds[f"K_{lab:s}forward"].units, "radiance", srf=srf1))
    u2_K = ((UADA(ds["nominal_measurand2"])+UADA(u2)).to(
            ds[f"K_{lab:s}forward"].units, "radiance", srf=srf2) -
             UADA(ds["nominal_measurand2"]).to(
            ds[f"K_{lab:s}forward"].units, "radiance", srf=srf2))
    uj = numpy.sqrt(u1_K**2+u2_K**2)
    uj99 = min(scipy.stats.scoreatpercentile(uj, 99),
               uj.median().item()*10)
    Kr_K_uj = Kr_K/uj
    KrKuj99 = min(scipy.stats.scoreatpercentile(Kr_K/uj, 99),
                  Kr_K_uj.median().item()*10)
    a2 = a.twiny()
    (cnts, bins, p2) = a2.hist(
        Kr_K_uj,
        histtype="step",
        bins=100,
        color="orange",
        #density=True,
        range=[0, KrKuj99])
    a2.set_xlabel("Kr / u [1]")
    a2.set_xlim([0, KrKuj99])
    a.set_title("Histogram of Kr (normalised by joint noise level)",
                y=1.11)

    a.xaxis.label.set_color(p1[0].get_edgecolor())
    a2.xaxis.label.set_color(p2[0].get_edgecolor())
    a.tick_params(axis='x', colors=p1[0].get_edgecolor())
    a2.tick_params(axis='x', colors=p2[0].get_edgecolor())

    # K-ΔL simply histogram
    a = next(g)
    plot_hist_with_medmad_and_fitted_normal(a, K_min_ΔL,
        sorted(kxrange-LΔrange), 
        f"K - ΔL [{y1.units:s}]",
        "Density",
        "K-ΔL",
        write="{sensor_1_name:s}_{sensor_2_name:s}/ch{channel:d}/{lab:s}/K_min_dL".format(
            channel=ds["channel"].item(), lab=lab, **ds.attrs)
            if write else False)

    # Kr vs. K-ΔL hexbin
    a = next(g)
    pc = a.hexbin(Kr_K,
        K_min_ΔL,
        extent=numpy.concatenate([[0, Kr_K99], sorted(kxrange-LΔrange)]),
        mincnt=1)
    a.set_xlabel(f"Kr [{y1.units:s}]")
    a.set_ylabel(f"K - ΔL [{y1.units:s}]")
    a.set_title("Joint distribution Kr and K - ΔL")
    cbs.append(f.colorbar(pc, ax=a))

    # Kr vs. uncertainty
    a = next(g)
    pc = a.hexbin(Kr_K, uj,
        extent=numpy.concatenate([[0, Kr_K99],
                                  [0, uj99]]),
        mincnt=1)
    a.set_xlabel(f"Kr [{y1.units:s}]")
    a.set_ylabel(f"joint noise level [{y1.units:s}]")
    a.set_title("Joint distribution Kr and noise")
    # with some potential filters as lines
    x = numpy.array([0, Kr_K99])
    for (ft, c, s) in ((5, "red", ":"),
                      (25, "red", "--"),
                      (150, "cyan", ":"),
                      (750, "cyan", "--")):
        a.plot(x, x/ft, color=c, linewidth=2, linestyle=s,
            label="x/{:d} (removes {:.1%})".format(ft, ((Kr_K_uj>ft).sum()/Kr_K.size).item()))
    a.legend()
    a.set_xlim([0, Kr_K99])
    a.set_ylim([0, uj99])
    cbs.append(f.colorbar(pc, ax=a))

    # ΔL/Kr, as suggested by Viju, see e-mail 2018-09-27
    a = next(g)
    plot_hist_with_medmad_and_fitted_normal(a, ΔL/Kr_K,
        scipy.stats.scoreatpercentile(ΔL/Kr_K, [1, 99]),
        f"ΔL/Kr [1]",
        "Density",
        "ΔL/Kr",
        write="{sensor_1_name:s}_{sensor_2_name:s}/ch{channel:d}/{lab:s}/dL_over_Kr".format(
            channel=ds["channel"].item(), lab=lab, **ds.attrs)
            if write else False)

    # histogram of actually chosen K uncorrected
    a = next(g)
    (cnts, bins, p) = a.hist(
        ds["K"],
        histtype="step",
        bins=100,
        density=True,
        range=scipy.stats.scoreatpercentile(ds["K"], [1, 99]))
    a.set_xlabel("K [native units]")
    a.set_ylabel("density")
    a.set_title("Histogram of chosen K")

    for cb in cbs:
        cb.set_label("No. matchups in bin")

    for a in ax_all.flat:
        a.grid(axis="both")

    try:
        chanstr = ", ".join(str(c) for c in numpy.atleast_1d(ds[f"K_{lab:s}forward"].attrs["channels_prediction"]))
    except KeyError:
        # until commit 828bd13, I inconsistently mixed "channels_prediction"
        # and "channels_used"
        chanstr = ", ".join(str(c) for c in numpy.atleast_1d(ds[f"K_{lab:s}forward"].attrs["channels_used"]))
    
    f.suptitle("K stats for pair {sensor_1_name:s}, {sensor_2_name:s}, {time_coverage:s}".format(**ds.attrs)
        + ", channel " + str(ds["channel"].item()) + ", " + lab
        + "\nchannels used to predict: " + chanstr)
    f.subplots_adjust(hspace=0.35, wspace=0.3)
    lab = lab.replace("·", "") # in LSF some nodes have ascii filesystem encoding?!

    graphics.print_or_show(f, False,
        "harmstats/{sensor_1_name:s}_{sensor_2_name:s}/ch{channel:d}/harmonisation_K_stats_{sensor_1_name:s}-{sensor_2_name:s}_ch{channel:d}_{time_coverage:s}_{lab:s}.".format(
            channel=ds["channel"].item(), lab=lab, **ds.attrs))
    
def plot_harm_input_stats(ds):
    """Plot histograms and such of harmonisation inputs

    For all inputs to the harmonisation, plot histograms along with
    statistics on the median and values that are n times the median
    absolute deviation from the median.

    Writes a plot to a file.  The plot may look something like this:

    .. image:: /images/harm-input-stats.png

    Parameters
    ----------

    ds : xarray.dataset
        Harmonisation matchup dataset for which to generate plots.
    """
    N = ds.dims["m1"]
    (f, ax_all) = matplotlib.pyplot.subplots(2, N, figsize=(5*N, 10))
    for i in range(N):
        for j in range(1, 3):
            vn = f"X{j:d}"
            dn = f"m{j:d}"
            a = ax_all[j-1, i]
            da = ds[vn][{dn:i}]
            (n, bins, patches) = a.hist(da, bins=100)
            # plot median and N*MAD from median
            med = da.median()
            mad = numpy.abs(da-med).median()
            for ext in [-10, -7, -3, 0, 3, 7, 10]:
                a.plot([med+ext*mad, med+ext*mad], [0, n.max()], color="red")
                a.text(med+ext*mad, .8*n.max(), str(ext))
            a.set_title(ds.attrs[f"sensor_{j:d}_name"] + ", " + ds[dn][i].item())
            a.set_xlabel(ds[dn][i].item())
            a.set_ylabel("Count")
    f.suptitle("harm input stats for pair {sensor_1_name:s}, {sensor_2_name:s}, {time_coverage:s}, with med+N*mad away".format(**ds.attrs)
        + ", channel " + str(ds["channel"].item()))
    graphics.print_or_show(f, False,
        "harmstats/{sensor_1_name:s}_{sensor_2_name:s}/ch{channel:d}/harmonisation_input_stats_{sensor_1_name:s}-{sensor_2_name:s}_ch{channel:d}_{time_coverage:s}_.".format(
            channel=ds["channel"].item(), **ds.attrs))

def plot_file_summary_stats(path, write=False):
    """Plot various summary statistics for harmonisation file

    Assumes it contains the extra fields that I generate for HIRS using
    :ref:`combine-hirs-hirs-matchups`.

    Parameters
    ----------

    path : str or pathlib.Path
        Path to file containing enhanced harmonisation files
    write : bool, optional
        If True, write out filter parameters
    """

    # TODO:
    #   - get extent from data percentiles
    #   - get axes labels from data-array attributes

    ds = xarray.open_dataset(path)

    # one subplot needs IASI simulations, which I will obtain from kmodel
    kmodel = matchups.KModelSRFIASIDB(
        chan_pairs="single", # not relevant, only using iasi db
        mode="standard", # idem
        units=rad_u["si"],
        debug=True, # will have one of each, makes difference for units
        prim_name=ds.attrs["sensor_1_name"],
        prim_hirs=fcdr.which_hirs_fcdr(ds.attrs["sensor_1_name"], read="L1C"),
        sec_name=ds.attrs["sensor_2_name"],
        sec_hirs=fcdr.which_hirs_fcdr(ds.attrs["sensor_2_name"], read="L1C"))
    kmodel.init_Ldb()

    plot_harm_input_stats(ds)
    others = [k.replace("K_other_", "").replace("_forward", "")
            for k in ds.data_vars.keys()
            if k.startswith("K_other_") and k.endswith("_forward")]
    plot_ds_summary_stats(ds, "", kmodel.Ldb_hirs_simul, write=write)
    if others:
        for lab in others:
            plot_ds_summary_stats(ds, lab,
                kmodel.others[lab].Ldb_hirs_simul,
                write=True)

def main():
    """Main function for script, expects commandline input.

    See module documentation and :ref:`hirs-inspect-harm-matchups`.
    """
    p = parse_cmdline()

    common.set_logger(
        logging.DEBUG if p.verbose else logging.INFO,
        p.log,
        loggers={"FCDR_HIRS", "typhon"})
        
    plot_file_summary_stats(pathlib.Path(p.file), write=p.write_filters)
