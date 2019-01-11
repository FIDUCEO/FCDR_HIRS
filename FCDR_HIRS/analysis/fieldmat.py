"""Plot scatter field density plots for temperature or other

This module contains functionality for scatter density plots and
correlation matrices between temperatures, between HIRS noise levels, and
between HIRS noise valuse.

A scatter density matrix for a series of fields makes a scatter density
plot of each pair of fields against each other, such that there will be
N x N subplots for N fields.  This functionality is built around around
:func:`typhon.plots.scatter_density_plot_matrix` and in this module
implemented in :func:`~analysis.fieldmat.plot_field_matrix`.

The module also contains functionality to calculate the FFT for space or
IWCT anomalies (for want of a better place to put it).

For the full documentation of the script, see :ref:`plot-hirs-field-matrix`.

Otherwise this module contains the functionality to plot correlation
matrices, such as between anomalies on HIRS channel calibration counts.
"""

import argparse
from .. import common

import logging

import matplotlib
# matplotlib.use("Agg") # now in matplotlibrc
import pathlib
# now in "inmyvenv"
# pathlib.Path("/dev/shm/gerrit/cache").mkdir(parents=True, exist_ok=True)


import datetime
import scipy.stats
import numpy
import itertools
import abc

import matplotlib.pyplot
import matplotlib.ticker
import matplotlib.gridspec
import typhon.plots
import typhon.plots.plots

from typhon.physics.units.common import ureg
from .. import fcdr
from typhon.datasets import tovs
from typhon.datasets.dataset import DataFileError
from .. import cached
from .. import graphics

logger = logging.getLogger(__name__)

month_pairs = dict(
    tirosn = ((1978, 11), (1979, 12)),
    noaa06 = ((1979, 7), (1983, 3)),
    noaa07 = ((1981, 8), (1984, 12)),
    noaa08 = ((1983, 5), (1984, 6)),
    noaa09 = ((1985, 2), (1988, 11)),
    noaa10 = ((1986, 11), (1991, 9)),
    noaa11 = ((1988, 11), (1998, 12)),
    noaa12 = ((1991, 9), (1998, 11)),
    noaa14 = ((1995, 1), (2005, 12)),
    noaa15 = ((1999, 1), (2009, 6)),
    noaa16 = ((2001, 1), (2014, 6)),
    noaa17 = ((2002, 7), (2013, 4)),
    noaa18 = ((2006, 11), (2011, 3)),
    noaa19 = ((2009, 4), (2013, 7)),
    metopa = ((2006, 12), (2016, 10)),
    metopb = ((2013, 2), (2017, 5)))

period_pairs = {sat:
    ((datetime.datetime(*start, 1, 0, 0), datetime.datetime(*start, 28, 23, 59)),
     (datetime.datetime(*end, 1, 0, 0), datetime.datetime(*end, 28, 23, 59)))
        for (sat, (start, end)) in month_pairs.items()}

#period_pairs = {sat:
#    ((datetime.datetime(*start, 28, 12, 0), datetime.datetime(*start, 28, 23, 59)),
#     (datetime.datetime(*end, 1, 0, 0), datetime.datetime(*end, 1, 12, 0)))
#        for (sat, (start, end)) in month_pairs.items()}

def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parse = common.add_to_argparse(parser,
        include_period=True,
        include_sat=True,
        include_channels=True,
        include_temperatures=True)

    parser.add_argument("--plot_temperature_sdm", action="store_true",
        help="Include scatter density matrix (SDM) of Digital A telemetry"
             " temperatures")

    parser.add_argument("--plot_noise_level_sdm", action="store_true",
        help="Include SDM of noise levels between channels")

    parser.add_argument("--plot_noise_value_scanpos_sdm",
        action="store_true",
        help="Include SDM of noise values between scan positions")

    parser.add_argument("--npos", action="store", type=int,
        default=[6], nargs="+",
        help="When plotting SDM of noise values or noise level between "
             "scan positions, plot this scan position")

    parser.add_argument("--plot_noise_value_channel_sdm",
        action="store_true",
        help="Include SDM of noise values between channels")

    parser.add_argument("--plot_noise_value_channel_corr",
        action="store_true",
        help="Plot correlation matrix between actual channel noise")

    parser.add_argument("--plot_noise_value_scanpos_corr",
        action="store_true",
        help="Plot correlation matrix between actual scanpos noise")

    parser.add_argument("--plot_temperature_corr",
        action="store_true",
        help="Plot correlation matrix between temperatures")

    parser.add_argument("--calibpos", action="store", type=int,
        nargs="+", default=[20],
        help="When plotting SDM of noise values or noise levels between chanels, "
             "plot this scan position")

    parser.add_argument("--noise_typ",
        action="store",
        choices=["iwt", "ict", "space"],
        default=["iwt"],
        nargs="+",
        help="What source of noise to plot for")

    parser.add_argument("--plot_all_corr", action="store_true",
        help="Plot all channel correlations for beginning and end of "
             "satellite lifetime for all satellites, with lower triangle "
             "for early in satellite life and upper triangle for late "
             "in satellite life.")

    return parser
def parse_cmdline():
    return get_parser().parse_args()

def plot_field_matrix(MM, ranges, title, filename, units):
    """Plot field matrix

    Thin layer around :func:`typhon.plots.scatter_density_plot_matrix`.
    See documentation there for details.

    Parameters
    ----------

    MM : structured ndarray
        Structured array, fields will be plotted against each other
    ranges : Dict
        For all fields, this is the range to limit the density plot to
    title : str
        Title for figure
    filename : str
        Filename for figure
    units : Dict
        Units for each field
    """
    f = typhon.plots.plots.scatter_density_plot_matrix(
        MM,
        hist_kw={"bins": 20},
        hist2d_kw={"bins": 20, "cmin": 1, "cmap": "viridis"},
        hexbin_kw={"gridsize": 20, "mincnt": 1, "cmap": "viridis"},
        ranges=ranges,
        units=units)
    for a in f.get_axes():
        for ax in (a.xaxis, a.yaxis):
            ax.set_major_locator(
                matplotlib.ticker.MaxNLocator(nbins=4, prune="both"))
    f.suptitle(title)
    f.subplots_adjust(hspace=0.5, wspace=0.5)
    graphics.print_or_show(f, False, filename)

#
class _SatPlotHelper(metaclass=abc.ABCMeta):
    """Helper for `MatrixPlotter.plot_all_sats_early_late`

    The method `MatrixPlotter.plot_all_sats_early_late` is able to plot
    various things for all satellites for an early and for a late month.
    This class implements the interface that it uses to do so.  In order
    for :meth:`~MatrixPlotter.plot_all_sats_early_late` to do its work, it will
    get passed an instance of an implementation of the
    :class:`_SatPlotHelper`, then calls, in order, the methods
    :meth:`prepare_early`, :meth:`prepare_late`, :meth:`plot_both`, and
    :meth:`finalise`.  This ensures that the surrounding loop logic of
    reading the early and late periods for each channel is only
    implemented once (in
    :meth:`~MatrixPlotter.plot_all_sats_early_late`), yet can be used for
    various purposes, one for each implementation of this class.

    The developer is not supposed to use this class or its implementations
    directly, but rather to subclass and implement it in case additional
    plots want to be made for each early and late period, satellite, and
    channel.  An instance of a subclass must be passed as the first
    argument to :meth:`MatrixPlotter.plot_all_sats_early_late`.

    """

    multichannel = None
    """bool : one plot per channel or one plot for all

    If True, put all channels together in one plot, with each channel as
    its own subplot.  If False, make one figure per channel.
    """

    @abc.abstractmethod
    def prepare_early(self, mp, **extra):
        """Method called before the early period

        Parameters
        ----------

        mp : MatrixPlotter
            Instance of :class:`MatrixPlotter`.
        **extra
            Any extra keyword arguments that specific implementations may
            desire.
        """
        ...

    @abc.abstractmethod
    def prepare_late(self, mp, **extra):
        """Method called before the late period

        Parameters
        ----------

        mp : MatrixPlotter
            Instance of :class:`MatrixPlotter`.
        **extra
            Any extra keyword arguments that specific implementations may
            desire.
        """
        ...

    @abc.abstractmethod
    def plot_both(self, mp, ax, gs, sat, r, c, ep, lp, **extra):
        """Plot both the early and lathe period to ax

        Parameters
        ----------

        mp : MatrixPlotter
            Instance of :class:`MatrixPlotter`.
        ax : matplotlib.axes._subplots.AxesSubplot
            Instance of `matplotlib.axes._subplots.AxesSubplot` in which
            the plot will be placed
        gs : matplotlib.gridspec.GridSpec
            Instance of `matplotlib.gridspec.GridSpec`
        sat : str
            Name of satellite involved
        r : int
            Row number within axes of grids
        c : int
            Column number within axes of grids
        ep : [datetime.datetime, datetime.datetime]
            List with two elements containing the beginning and ending of
            the early period
        lp : [datetime.datetime, datetime.datetime]
            List with two elements containing the beginning and ending of
            the late period
        **extra
            Any extra keyword arguments that specific implementations may
            desire.
        """
        ...

    @abc.abstractmethod
    def finalise(self, mp, f, gs, **extra):
        """Method called after everything else is done

        An implementation will probably want to finalise the figure in
        some way.

        Parameters
        ----------

        mp : MatrixPlotter
            Instance of :class:`MatrixPlotter`.
        f : matplotlib.figure.Figure
            Instance of `matplotlib.figure.Figure` that want finalisation
        gs : matplotlib.gridspec.GridSpec
            Instance of `matplotlib.gridspec.GridSpec`
        **extra
            Any extra keyword arguments that specific implementations may
            desire.
        """
        ...
#
class _SatPlotChCorrmat(_SatPlotHelper):
    """Helper class for `MatrixPlotter.plot_all_sats_early_late`

    This is a helper class for plotting a channel correlation matrix for
    all early and late periods.  Do not use directly, but pass an object
    of this class to :meth:`~MatrixPlotter.plot_all_sats_early_late`
    or :meth:`~MatrixPlotter.plot_selection_sats_early_late`.
    """

    multichannel = False
    def __init__(self, channels, noise_typ, calibpos):
        """Initialise the channel correlation matrix helper

        Parameters
        ----------

        channels : array_like
            Channels to consider
        noise_typ : str
            Where to estimate noise from: "space", "iwt", or "ict"
        calibpos : int
            What calibration position to use for correlation calculations
        """

        self.channels = numpy.asarray(channels)
        self.noise_typ = noise_typ
        self.calibpos = calibpos
    
    def _get_ch_corrmat(self, mp):
        """Get channel correlation matrix

        Parameters
        ----------

        mp : MatrixPlotter
        """
        (S, ρ, cnt) = mp._get_ch_corrmat(
            self.channels,
            self.noise_typ,
            self.calibpos)
        return (S, ρ, cnt)

    def prepare_early(self, mp):
        (S, _, ecnt) = self._get_ch_corrmat(mp)
        self.S_low = numpy.tril(S, k=-1)
        self.ecnt = ecnt

    def prepare_late(self, mp):
        (S, _, lcnt) = self._get_ch_corrmat(mp)
        self.S_hi = numpy.triu(S, k=1)
        self.lcnt = lcnt

    def plot_both(self, mp, ax, gs, sat, r, c, ep, lp, solo=False):
        im = mp._plot_ch_corrmat(
            self.S_low +
            self.S_hi +
            numpy.diag(numpy.zeros(self.channels.size)*numpy.nan),
                              ax, self.channels, add_x=r==3, add_y=c==0)
        f = ax.figure
        if solo:
            cb = f.colorbar(im)
        elif r==c==0:
            cax = f.add_subplot(gs[:-1, -1])
            cb = f.colorbar(im, cax=cax)
        else:
            cb = None

        if cb is not None:
            cb.set_label("Pearson product-moment correlation coefficient")

        ax.set_title(("HIRS noise correlations for " if solo else "") +
                     f"{sat:s}\n"
                     f"{ep[0]:%Y-%m}, {self.ecnt:d} cycles\n"
                     f"{lp[0]:%Y-%m}, {self.lcnt:d} cycles")

    def finalise(self, mp, f, gs, sat=None, solo=False):
        if solo:
            graphics.print_or_show(f, False,
                f"hirs_noise_correlations_sat{sat:s}_pos{self.calibpos:d}.")
        else:
            gs.update(wspace=0.10, hspace=0.4)
            f.suptitle("HIRS noise correlations for all HIRS, pos "
                      f"{self.calibpos:d} ", fontsize=40)
            graphics.print_or_show(f, False,
                f"hirs_noise_correlations_allsats_pos{self.calibpos:d}.")

class _SatPlotFFT(_SatPlotHelper):
    """Helper class For plotting FFT stuff

    Pass an object of this class to
    :meth:`MatrixPlotter.plot_all_sats_early_late` to get a figure
    with for each channel an FFT for anomalies for space, IWCT, and Earth
    views.

    This is a helper class for plotting a FFT for all early and late
    periods.  Do not use directly, but pass an object
    of this class to :meth:`MatrixPlotter.plot_all_sats_early_late`.
    """
    multichannel = True

    def __init__(self):#, channel):
        #self.n = 2**6
        self.n = 48
        #self.channel = channel
        self.early_spc = {}
        self.early_iwctc = {}
        self.early_ec = {}
        self.early_cnt = {}
        self.late_spc = {}
        self.late_iwctc = {}
        self.late_ec = {}
        self.late_cnt = {}

    def _extract_counts(self, mp, channel):
        """Extract space, IWCT, and Earth counts

        Parameters
        ----------

        mp : MatrixPlotter
            MatrixPlotter instance
        channel : int
            Channel for which we're extracting counts
        """
        spc = mp._get_accnt("space")[:, :, channel-1]
        spc = spc[(~spc.mask).all(1), :]
        iwctc = mp._get_accnt("iwt")[:, :, channel-1]
        iwctc = iwctc[(~iwctc.mask).all(1), :]
        ec = mp._get_accnt("Earth")[:, :, channel-1]
        ec = ec[(~ec.mask).all(1), :]
        return (spc, iwctc, ec)

    def prepare_early(self, mp, channel):
        (spc, iwctc, ec) = self._extract_counts(mp, channel)
        self.early_spc[channel] = spc
        self.early_iwctc[channel] = iwctc
        self.early_ec[channel] = ec
        self.early_cnt[channel] = spc.shape[0]

    def prepare_late(self, mp, channel):
        (spc, iwctc, ec) = self._extract_counts(mp, channel)
        self.late_spc[channel] = spc
        self.late_iwctc[channel] = iwctc
        self.late_ec[channel] = ec
        self.late_cnt[channel] = spc.shape[0]

    def plot_both(self, mp, ax, gs, sat, r, c, ep, lp, channel,
                  solo=False, c_max=3, r_max=3, custom=False):
        """Plot FFT for calibration counts and Earth views
        """

        n = self.n

        # 0.1 seconds between observations (NOAA KLM User's Guide, Appendix)
        #x = numpy.fft.fftfreq(n, d=0.1)
        # rather do pixels than seconds (feedback CM)
        x = numpy.fft.fftfreq(n, d=1)

        logging.debug("Calculating FFTs")

        fft_e_ec = abs(numpy.fft.fft(self.early_ec[channel], axis=1, n=n)[:, 1:n//2]).mean(0)
        fft_e_spc = abs(numpy.fft.fft(self.early_spc[channel], axis=1, n=n)[:, 1:n//2]).mean(0)
        fft_e_iwctc = abs(numpy.fft.fft(self.early_iwctc[channel], axis=1, n=n)[:, 1:n//2]).mean(0)
        fft_l_ec = abs(numpy.fft.fft(self.late_ec[channel], axis=1, n=n)[:, 1:n//2]).mean(0)
        fft_l_spc = abs(numpy.fft.fft(self.late_spc[channel], axis=1, n=n)[:, 1:n//2]).mean(0)
        fft_l_iwctc = abs(numpy.fft.fft(self.late_iwctc[channel], axis=1, n=n)[:, 1:n//2]).mean(0)

        logging.debug("Done calculating, now plotting FFTs")

        ax.plot(
            x[1:n//2],
            fft_e_ec,
            color="black",
            marker="*",
            markersize=9,
            linewidth=1,
            markerfacecolor="none",
            markeredgecolor="black",
            label="early, Earth")
        ax.plot(
            x[1:n//2],
            fft_e_spc,
            color="red",
            markerfacecolor="none",
            markeredgecolor="red",
            marker="^",
            markersize=9,
            linewidth=1,
            label="early, space")
        ax.plot(
            x[1:n//2],
            fft_e_iwctc,
            color="teal",
            marker="<",
            markerfacecolor="none",
            markeredgecolor="teal",
            markersize=9,
            linewidth=1,
            label="early, IWCT")

        ax.plot(
            x[1:n//2],
            fft_l_ec,
            color="black",
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=9,
            linewidth=1,
            linestyle=(0, (5,5)),
            label="late, Earth")
        ax.plot(
            x[1:n//2],
            fft_l_spc,
            color="red",
            marker="v",
            markerfacecolor="none",
            markeredgecolor="red",
            markersize=9,
            linewidth=1,
            linestyle=(0, (5,5)),
            label="late, space")
        ax.plot(
            x[1:n//2],
            fft_l_iwctc,
            color="teal",
            marker=">",
            markerfacecolor="none",
            markeredgecolor="teal",
            markersize=9,
            linewidth=1,
            linestyle=(0, (5,5)),
            label="late, IWCT")

        ax.set_xscale("log")
        ax.set_yscale("log")
        xpos = [48, 24, 12, 6, 3]
        ax.set_xticks([1/x for x in xpos])
        ax.grid(axis="both")
        if solo or r==r_max:
            ax.set_xlabel("Frequency [cycles/pixel]")
            #ax.set_xscale('log')
            #ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xticklabels([fr"$\frac{{1}}{{{x:d}}}$" for x in xpos])
        else:
            ax.set_xticklabels([])
        if solo or c==0:
            ax.set_ylabel("Amplitude [counts]")
        else:
            ax.set_yticklabels([])
        if solo or (r==0 and c==c_max):
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
            #ax.legend()
        ax.set_title((f"HIRS noise spectral analysis for {sat:s} channel "
                     f"{channel:d}\n" if solo else f"{sat:s} ch. {channel:d}\n" if custom else f"{sat:s}\n") +
                     f"{ep[0]:%Y-%m}, {self.early_cnt[channel]:d} cycles\n"
                     f"{lp[0]:%Y-%m}, {self.late_cnt[channel]:d} cycles")
        if not solo: # for solo, let automated positions decide
            ax.set_ylim([1e0, 1e4])
        logging.debug("Done plotting FFTs")

    def finalise(self, mp, f, gs, channel=None, sat=None, solo=False,
                 custom=False):
        if custom:
            graphics.print_or_show(f, False,
                f"hirs_crosstalk_fft_custom.")
        elif solo:
            graphics.print_or_show(f, False,
                f"hirs_crosstalk_fft_sat{sat:s}_ch{channel:d}.")
        else:
            gs.update(wspace=0.10, hspace=0.4)
            f.subplots_adjust(right=0.8, bottom=0.2, top=0.90)
            f.suptitle(f"Spectral analysis, channel {channel:d}")
            graphics.print_or_show(f, False,
                f"hirs_crosstalk_fft_ch{channel:d}.")

class _SatPlotAll(_SatPlotChCorrmat, _SatPlotFFT):
    """Helper class for plotting both ch corrmats and fft

    Just a wrapper to in one go plot both what is implemented in
    `_SatPlotChCorrmat` and in `_SatPlotFFT` in one go, without the need
    to read all the data twice.
    """
    multichannel = True
    def __init__(self, *args, **kwargs):
        _SatPlotChCorrmat.__init__(*args, **kwargs)
        _SatPlotFFT.__init__(*args, **kwargs)

    def prepare_early(self, *args, **kwargs):
        _SatPlotChCorrmat.prepare_early(*args, **kwargs)
        _SatPlotFFT.prepare_early(*args, **kwargs)
        
    def prepare_late(self, *args, **kwargs):
        _SatPlotChCorrmat.prepare_late(*args, **kwargs)
        _SatPlotFFT.prepare_late(*args, **kwargs)

    def plot_both(self, *args, **kwargs):
        _SatPlotChCorrmat.plot_both(*args, **kwargs)
        _SatPlotFFT.plot_both(*args, **kwargs)

    def finalise(self, *args, **kwargs):
        _SatPlotChCorrmat.finalise(*args, **kwargs)
        _SatPlotFFT.finalise(*args, **kwargs)

class MatrixPlotter:
    """Plot varous scatter density matrices and correlation matrices

    This class gathers attributes and methods to plot various scatter
    density matrices, correlation matrices, and FFTs.  The most
    interesting are probably the methods describing the correlations
    between count anomalies, the "correlated noise" problem.

    The highest level method is :meth:`plot_all_sats_early_late`, which
    makes one big figure with correlation matrices for all satellites and
    all channels, early and late.

    """

    #: List of all satellites
    all_sats = ["tirosn"] + [f"noaa{no:02d}" for no in range(6, 20) if no!=13] + ["metopa", "metopb"]
#    def __init__(self, sat, from_date, to_date):
#        self.reset(sat, from_date, to_date)

    #: ndarray : Attribute that will hold data after calling :meth:`reset`
    M = None

    def reset(self, sat, from_date, to_date):
        """Reset matrix plotter to an initial state

        This methods reads the data needed to subsequently make plots
        related to ``sat`` between the period ``from_date`` and
        ``to_date``.  Call this method before any of the plotting methods
        as well as when you want to redo any of the plotting methods for a
        different satellite and time period.

        This method returns nothing, but after calling this method, the
        data will be stored in the :attr:`M` attribute.
        Parameters
        ----------

        sat : str
            Name of satellite to read
        from_date : datetime.datetime
            The `~datetime.datetime` object representing the starting time
            to read
        to_date : datetime.datetime
            Like ``from_date`` but describing the final time to read

        """
        h = tovs.which_hirs(sat)
        self.hirs = h
        M = cached.read_tovs_hirs_period(sat, from_date, to_date, 
            fields=["temp_{:s}".format(t) for t in sorted(h.temperature_fields)] + 
                   ["counts", "time", h.scantype_fieldname])
        self.M = M
        self.start_date = from_date
        self.end_date = to_date

        self.title_sat_date = f"{sat:s} {from_date:%Y-%m-%d} -- {to_date:%Y-%m-%d}"
        self.filename_sat_date = f"{sat:s}_{from_date:%Y}/{sat:s}_{from_date:%Y%m%d%H%M}--{to_date:%Y%m%d%H%M}"

    def _get_temps(self, temp_fields):
        """Extract temperatures from data

        After :attr:`M` has been set by :meth:`reset`, this method will
        extract the desired temperature fields from it.  Possible
        fieldnames are given by the ``temperature_fields`` attribute on
        :class:`typhon.datasets.tovs.HIRS`.

        Parameters
        ----------

        temp_fields : list[str]
            List of temperature fieldnames.

        Returns
        -------

        ndarray
            Structured array containing only the temperatures.
        """
        MM = numpy.ma.zeros(
            shape=self.M.shape,
            dtype=[(t, "f8") for t in temp_fields])
        for t in temp_fields:
            x = self.M["temp_{:s}".format(t)]
            while x.ndim > 1:
                x = x.mean(-1)
            MM[t] = x
        return MM

    def plot_temperature_sdm(self, temp_fields):
        """Plot a scatter density matrix with temperatures

        Make ``len(temp_fields)`` * ``len(temp_fields)`` subplots, with
        each of those subplots a scatter density matrix between
        ``temp_fields[i]`` vs. ``temp_fields[j]``.  This is a thin wrapper
        around :func:`plot_field_matrix` using temperatures.  Valid
        temperatures are given by the ``temperature_fields`` attribute on
        :class:`typhon.datasets.tovs.HIRS`.

        Parameters
        ----------

        temp_fields : list[str]
            List of temperature fieldnames.


        See also
        --------

        plot_noise_level_sdm
            Similar SDM but for noise level
        plot_field_matrix
            Underlying function plotting a field matrix
        """
        MM = self._get_temps(temp_fields)
        plot_field_matrix(MM,
            ranges=
                {fld: scipy.stats.scoreatpercentile(MM[fld], [1, 99])
                    for fld in temp_fields},
            title="HIRS temperature matrix {:s}".format(self.title_sat_date),
            filename="hirs_temperature_sdm_{:s}_{:s}.png".format(
                self.filename_sat_date,
                ",".join(temp_fields)),
            units={fld: "K" for fld in temp_fields})

    def plot_noise_level_sdm(self, channels, noise_typ="iwt"):
        """Plot a scatter density matrix with noise levels

        Plot a scatter density matrix with ``len(channels)`` times
        ``len(channels)`` subplots, with each subplot a scatter density
        plot the noise levels between those channels.  Note that this is
        between the noise level and not the actual deviation from the
        mean, which would be in :meth:`plot_noise_value_scanpos_sdm``.

        Parameters
        ----------

        channels : list[int]
            List of channel numbers
        noise_typ : str, optional
            What type to plot the SDM for.  Can be "iwt", "ict", or "space".

        See also
        --------

        plot_noise_value_scanpos_sdm
            Plot SDM between noise values rather than noise levels.
        plot_temperature_sdm
            Similar SDM but between temperature fields
        plot_field_matrix
            Underlying function that plots the field matrix
        """
        for (i, ch) in enumerate(channels):
            (t_x, x) = self.hirs.estimate_noise(self.M, ch, typ=noise_typ)
            if i == 0:
                MM = ureg.Quantity(
                    numpy.ma.zeros(
                        shape=x.shape,
                        dtype=[("ch{:d}".format(ch), "f8") for ch in channels]),
                    x.u)
                        
            MM["ch{:d}".format(ch)] = x
            #MM["ch{:d}".format(ch)].mask = x.mask
        plot_field_matrix(MM,
            ranges=
                {"ch{:d}".format(ch): scipy.stats.scoreatpercentile(
                    MM["ch{:d}".format(ch)], [1, 99]) for ch in channels},
            title="HIRS noise level matrix {:s}, ch. {:s}".format(
                self.title_sat_date, ", ".join(str(ch) for ch in channels)),
            filename="hirs_noise_level_sdm_{:s}_{:s}.png".format(
                self.filename_sat_date, ",".join(str(ch) for ch in channels)),
            units={"ch{:d}".format(ch): x.u for ch in channels})

    def _get_accnt(self, noise_typ):
        """Extract count anomalies for calibration type

        Extract from the ndarray stored in :attr:`M` the count anomalies
        (deviations from the mean per calibration cycle) for
        ``noise_typ``.

        Parameters
        ----------

        noise_typ : str
            What calibration line to estimate the anomalies for: "iwt",
            "ict", or "space".
        """
        views = self.M[self.hirs.scantype_fieldname] == getattr(self.hirs, "typ_{:s}".format(noise_typ))
        ccnt = self.M["counts"][views, 8:, :]
        mccnt = ccnt.mean(1, keepdims=True)
        accnt = ccnt - mccnt
        return accnt

    def plot_noise_value_scanpos_sdm(self, channels,
            noise_typ="iwt",
            npos=6):
        """Plot SDM between noise values / count anomalies for scanpos

        For a particular "scan position", plot a scatter density matrix
        for the correlated noise.  This is calculated by taking the
        count anomalies per scanline using :meth:`_get_accnt`.

        Parameters
        ----------

        channels : list[int]
            Channels for which to make a scanpos-to-scanpos anomaly
            scatter density matrix
        noise_typ : str, optional
            What calibration type to use to estimate the noise: can be
            "iwt" (default), "space", or "ict" (HIRS/2 and HIRS/2I only)
        npos : int, optional
            How many scan positions to consider.  Defaults to 6, which
            results in a 6×6=36 subplot figure.

        See also
        --------

        plot_noise_level_sdm
            Similar SDM but for noise level rather than noise value.
        plot_noise_value_channel_sdm
            Similar SDM but between channels rather than between scanpos.
        plot_noise_value_scanpos_corr
            Plot correlation matrices for the same
        """

        accnt = self._get_accnt(noise_typ)

        allpos = numpy.linspace(0, 47, npos, dtype="uint8")
        
        for ch in channels:
            X = numpy.zeros(dtype=[("pos{:d}".format(d), "f4") for d in allpos],
                            shape=accnt.shape[0])
            for d in allpos:
                X["pos{:d}".format(d)] = accnt[:, d, ch-1]
            plot_field_matrix(
                X,
                ranges={"pos{:d}".format(d): scipy.stats.scoreatpercentile(
                    X["pos{:d}".format(d)], [1, 99]) for d in allpos},
            title="HIRS noises by scanpos, {:s}, ch {:d}, {:s}-{:d}".format(
                self.title_sat_date,
                ch,
                noise_typ,
                npos),
            filename="hirs_noise_by_scanpos_sdm_{:s}_ch{:d}_{:s}_{:d}.png".format(
                self.filename_sat_date,
                ch,
                noise_typ,
                npos),
            units={"pos{:d}".format(d): "counts" for d in allpos})

    def plot_noise_value_channel_sdm(self, channels,
            noise_typ="iwt",
            calibpos=20):
        """Plot SDM between noise values / count anomalies per channel

        Plot a scatter density matrix for the count anomalies, so
        essentially the actual error in space or IWCT or ICCT counts,
        between diferent channels, for a constant calibration position.
        The count anomalies are calculated with :meth:`_get_accnt`.

        Parameters
        ----------

        channels: array_like
            Channel numbers for which to plot the noise value SDM
        noise_typ: str, optional
            Where to estimate error correlations from: "space", "iwt", or
            "ict".
        calibpos : int, optional
            Calibration position to use.  Defaults to 20.

        See also
        --------

        plot_noise_value_scanpos_sdm
            Similar SDM but between scanpos rather than channel
        plot_noise_value_channel_corr
            Plot correlation matrix for the same
        """
        accnt = self._get_accnt(noise_typ)
        X = numpy.zeros(dtype=[("ch{:d}".format(ch), "f4") for ch in channels],
                        shape=accnt.shape[0])
        for ch in channels:
            X["ch{:d}".format(ch, "f4")] = accnt[:, calibpos, ch-1]
        plot_field_matrix(
            X,
            ranges={"ch{:d}".format(ch): scipy.stats.scoreatpercentile(
                X["ch{:d}".format(ch)], [1, 99]) for ch in channels},
            title="HIRS noise scatter densities between channels, "
                  "{:s}, {:s} pos {:d}".format(
                self.title_sat_date,
                noise_typ,
                calibpos),
            filename="hirs_noise_by_channel_sdm_{:s}_ch_{:s}_{:s}{:d}.png".format(
                self.filename_sat_date,
                ",".join(str(ch) for ch in channels),
                noise_typ,
                calibpos),
            units={"ch{:d}".format(ch): "counts" for ch in channels})

    def plot_noise_value_channel_corr(self, channels,
            noise_typ="iwt",
            calibpos=20):
        """Plot noise value channel correlation

        Plot two correlation matrices (Pearson and Spearman) for the
        inter-channel correlated noise due to electronics or filter wheel
        effects.  This is calculated by taking the anomaly (from
        :meth:`_get_accnt`) and then correlating those to each other,
        while keeping the calibration position fixed.

        Parameters
        ----------

        channels: array_like
            Channel numbers for which to plot the noise channel
            correlation
        noise_typ: str, optional
            Where to estimate error correlations from: "space", "iwt", or
            "ict".
        calibpos : int, optional
            Calibration position to use.  Defaults to 20.

        See also
        --------

        plot_noise_value_channel_sdm
            Plot scatter density matrix for the same
        plot_noise_value_scanpos_corr
            Correlation matrix but between scanpos for constant channel
        """
        (f, ax_all) = matplotlib.pyplot.subplots(1, 3, figsize=(16, 8),
            gridspec_kw={"width_ratios": (14, 14, 1)})
        channels = numpy.asarray(channels)
        (S, ρ, no) = self._get_ch_corrmat(channels, noise_typ, calibpos)
#        im1 = ax_all[0].imshow(S, cmap="PuOr", interpolation="none")
        im1 = self._plot_ch_corrmat(S, ax_all[0], channels)
#        im2 = ax_all[1].imshow(ρ, cmap="PuOr", interpolation="none")
        im2 = self._plot_ch_corrmat(ρ, ax_all[1], channels)
#        for (a, im) in zip(ax_all[:2], (im1, im2)):
#            im.set_clim([-1, 1])
#            a.set_xticks(numpy.arange(len(channels)))
#            a.set_yticks(numpy.arange(len(channels)))
#            a.set_xticklabels([str(ch) for ch in channels])
#            a.set_yticklabels([str(ch) for ch in channels])
#            a.set_xlabel("Channel no.")
#            a.set_ylabel("Channel no.")
        cb = f.colorbar(im2, cax=ax_all[2])
        cb.set_label("Correlation")
        ax_all[0].set_title("Pearson correlation")
        ax_all[1].set_title("Spearman correlation")
        f.suptitle("HIRS noise correlations, {:s}, {:s} pos {:d}\n"
            "({:d} cycles)".format(
            self.title_sat_date, noise_typ, calibpos, no))
        graphics.print_or_show(f, False,
                "hirs_noise_correlations_channels_{:s}_ch_{:s}_{:s}{:d}.".format(
            self.filename_sat_date,
            ",".join(str(ch) for ch in channels),
            noise_typ, calibpos))

    def _get_ch_corrmat(self, channels, noise_typ, calibpos):
        """Calculate inter-channel correlation matrix

        Calculate the channel-to-channel correlation matrix, for both the
        Pearson and Spearman correlations, using count anomalies
        calculated with :meth:`_get_accnt`.

        Parameters
        ----------

        channels: array_like
            Channel numbers for which to plot the noise value SDM
        noise_typ: str, optional
            Where to estimate error correlations from: "space", "iwt", or
            "ict".
        calibpos : int, optional
            Calibration position to use.  Defaults to 20.

        Returns
        -------

        S : (n_ch, n_ch) ndarray
            Channel correlation matrix, Pearson correlation
        ρ : ndarray
            Channel correlation matrix, Spearman correlation
        unmasked : int
            Number of lines with count anomalies from which this was
            calculated
        """
        # although there is a scipy.stats.mstats module,
        # scipy.stats.mstats.spearman can only calculate individual
        # covariances, not covariance matrices (it's not vectorised) and
        # explicit looping is too slow
        accnt = self._get_accnt(noise_typ)
        unmasked = ~(accnt[:, calibpos, :].mask.any(1))
        S = numpy.corrcoef(accnt[:, calibpos,  channels-1].T[:, unmasked])
        ρ = scipy.stats.spearmanr(accnt[:, calibpos, channels-1][unmasked, :])[0]
        return (S, ρ, unmasked.sum())

    @staticmethod
    def _plot_ch_corrmat(S, a, channels, add_x=False, add_y=False, each=2):
        """Plot channel correlation matrix, helper for plot_noise_value_channel_corr

        Small helper for :meth:`plot_noise_value_channel_corr`.  After the
        channel correlation matrices have been calculated, this method
        plots them into axes objects and sets the tick labels in the
        correct place.

        Parameters
        ----------

        S : ndarray
            Channel correlation matrix to be plotted.
        a : matplotlib.axes._subplots.AxesSubplot
            Instance of `matplotlib.axes._subplots.AxesSubplot` in which
            to plot the matrix.
        channels : array_like
            List or array of numbers representing the channels, this will
            be used to label the xticks and yticks.
        add_x : bool, optional
            If true, add x-label and tickmarks even if the axes is not the
            bottom row of the figure.
        add_y : bool, optional
            If true, add y-label and tickmarks even if the axes is not the
            leftmost column of the figure.
        each : int, optional
            Interval of channels to label in ticks.  Defaults to 2, so
            they will be labelled 2, 4, 6, ..., 18

        Returns
        -------

        image : `~matplotlib.image.AxesImage`
            Image as returned by
            `~matplotlib.axes._subplots.AxesSubplot.imshow`.
        """
        im = a.imshow(S, cmap="PuOr_r", interpolation="none", vmin=-1, vmax=1)
        im.set_clim([-1, 1])
        a.set_xticks(numpy.arange(len(channels)))
        if add_x or a.is_last_row():
            #a.set_xticks(numpy.arange(len(channels)))
            a.set_xticklabels([str(ch) if ch%each==0 else "" for ch in channels])
            a.set_xlabel("Channel no.")
        else:
            #a.set_xticks([])
            a.set_xticklabels([])
        a.set_yticks(numpy.arange(len(channels)))
        if add_y or a.is_first_col():
            #a.set_yticks(numpy.arange(len(channels)))
            a.set_yticklabels([str(ch) if ch%each==0 else "" for ch in channels])
            a.set_ylabel("Channel no.")
        else:
            #a.set_yticks([])
            a.set_yticklabels([])
        return im

    @staticmethod
    def _plot_pos_corrmat(S, a, add_x=False, add_y=False):
        """Plot position correlation matrix, helper for plot_noise_value_scanpos_corr

        Small helper for :meth:`plot_noise_value_scanpos_corr`.  After the
        scanpos correlation matrices have been calculated, this method
        plots them into axes objects and sets the tick labels in the
        correct place.

        S : ndarray
            Channel correlation matrix to be plotted.
        a : matplotlib.axes._subplots.AxesSubplot
            Instance of `matplotlib.axes._subplots.AxesSubplot` in which
            to plot the matrix.
        add_x : bool, optional
            If true, add x-label and tickmarks even if the axes is not the
            bottom row of the figure.
        add_y : bool, optional
            If true, add y-label and tickmarks even if the axes is not the
            leftmost column of the figure.

        Returns
        -------

        image : `~matplotlib.image.AxesImage`
            Image as returned by
            `~matplotlib.axes._subplots.AxesSubplot.imshow`.
        """
        im = a.imshow(S, cmap="PuOr_r", interpolation="none", vmin=-1, vmax=1)
        im.set_clim([-1, 1])
        if add_x:
            a.set_xlabel("Pos no.")
        else:
            #a.set_xticks([])
            a.set_xticklabels([])
        if add_y:
            a.set_ylabel("Pos no.")
        else:
            #a.set_yticks([])
            a.set_yticklabels([])
        return im

    def _get_pos_corrmat(self, ch, noise_typ):
        """Get correlation matrix between scan positions

        For a constant channel and a particular source of count anomalies,
        get the correlation matrix between scan positions.

        Parameters
        ----------

        ch : int
            Channel for which to get correlation matrix.
        noise_typ: str, optional
            Where to estimate error correlations from: "space", "iwt", or
            "ict".

        Returns
        -------

        S : ndarray
            Correlation matrix between positions
        p : ndarray
            p-value matrix corresponding to S
        unmasked : int
            Number of valid calibration lines over which the calibration
            matrix was calculated
       """
        accnt = self._get_accnt(noise_typ)
        unmasked = ~(accnt[:, :, ch-1].mask.any(1))
        (S, p) = typhon.math.stats.corrcoef(accnt[unmasked, :, ch-1].T)
        return (S, p, unmasked.sum())

    def plot_noise_value_scanpos_corr(self, channels,
            noise_typ="iwt"):
        """Plot noise value scanpos correlation

        PlotPearson  correlation matrix and p-value for the
        inter-position correlated noise due to electronics or filter wheel
        effects.  This is calculated by taking the anomaly (from
        :meth:`_get_accnt`) and then correlating those to each other,
        while keeping the channel fixed.

        Parameters
        ----------

        channels: array_like
            Channel numbers for which to plot the inter-position
            correlation matrix, one per channel
        noise_typ: str, optional
            Where to estimate error correlations from: "space", "iwt", or
            "ict".

        See also
        --------

        plot_noise_value_scanpos_sdm
            Plot scatter density matrix for the same
        plot_noise_value_channel_corr
            Correlation matrix but between channel for constant scanpos
        """

        #accnt = self._get_accnt(noise_typ)
        channels = numpy.asarray(channels)
        for ch in channels:
            (f, ax_all) = matplotlib.pyplot.subplots(1, 8, figsize=(24, 6),
                gridspec_kw={"width_ratios": (15, 1, 6, 15, 1, 6, 15, 1)})
            #S = numpy.corrcoef(accnt[:, :, ch].T)
            #unmasked = ~(accnt[:, :, ch].mask.any(1))
            #(S, p) = typhon.math.stats.corrcoef(accnt[unmasked, :, ch].T)
            (S, p, no) = self._get_pos_corrmat(ch, noise_typ)
            # hack to make logarithmic possible
            if (p==0).any():
                logging.warn("{:d} elements have underflow (p=0), setting "
                    "to tiny".format((p==0).sum()))
                p[p==0] = numpy.finfo("float64").tiny * numpy.finfo("float64").eps
            im1 = ax_all[0].imshow(S, cmap="PuOr_r", interpolation="none")
            im1.set_clim([-1, 1])
            cb1 = f.colorbar(im1, cax=ax_all[1])
            cb1.set_label("Correlation coefficient")
            ax_all[0].set_title("Pearson correlation")

            # choose range for S
            upto = scipy.stats.scoreatpercentile(abs(S[S<1]), 99)
            im2 = ax_all[3].imshow(S, cmap="PuOr_r", vmin=-upto, vmax=upto)
            im2.set_clim([-upto, upto])
            cb2 = f.colorbar(im2, cax=ax_all[4])
            cb2.set_label("Correlation coefficient")
            ax_all[3].set_title("Pearson correlation")

            im3 = ax_all[6].imshow(p, cmap="viridis",
                interpolation="none", norm=matplotlib.colors.LogNorm(
                    vmin=p.min(), vmax=(p-numpy.eye(p.shape[0])).max()))
            ax_all[6].set_title("Likelihood of non-correlation")
            for a in ax_all[::3]:
                a.set_xlabel("Scanpos")
                a.set_ylabel("Scanpos")
            cb3 = f.colorbar(im3, cax=ax_all[7])
            cb3.set_label("p-value")
            f.suptitle("HIRS noise correlations, {:s}, {:s} ch. {:d}\n"
                "({:d} cycles)".format(
                    self.title_sat_date, noise_typ, ch, no))
            ax_all[2].set_visible(False)
            ax_all[5].set_visible(False)
            graphics.print_or_show(f, False,
                "hirs_noise_correlations_scanpos_{:s}_ch{:d}_{:s}.png".format(
                    self.filename_sat_date, ch, noise_typ))

    def plot_temperature_corrmat(self, temp_fields):
        """Plot correlation matrix for temperatures.

        NB, this is a correlation matrix for the actual temperatures — not
        for their noises.
        
        Parameters
        ----------

        temp_fields : list[str]
            List of temperatures to use.  Valid temperatures are found as
            the ``temperature_fields`` attribute on
            the :class:`typhon.datasets.tovs.HIRS` class.
        """
        MM = self._get_temps(temp_fields)
        MMp = MM.view("<f8").reshape(MM.shape[0], -1).T
        MMp = MMp[:, (~MMp.mask).all(0)]
        S = numpy.corrcoef(MMp)
        (f, a) = matplotlib.pyplot.subplots(1, 1)
        im = a.imshow(S, cmap="PuOr_r", interpolation="none")
        im.set_clim([-1, 1])
        cb = f.colorbar(im)
        cb.set_label("Correlation coefficient")
        a.set_title("Temperature correlation matrix\n" +
                    self.title_sat_date)

        a.set_xticks(numpy.arange(len(temp_fields)))
        a.set_xticklabels(temp_fields)
        a.set_yticks(numpy.arange(len(temp_fields)))
        a.set_yticklabels(a.get_xticklabels())
        for tl in a.get_xticklabels():
            tl.set_rotation(90)

        f.subplots_adjust(left=0.3, bottom=0.3)

        graphics.print_or_show(f, False,
            f"hirs_temperature_correlation_{self.filename_sat_date:s}.png")


    def plot_all_sats_early_late(self, plotter, sats):
        """For all satellites, make plot for first and last month

        Using a :class:`_SatPlotHelper` implementation instance, repeat a
        specific type of plot for the first and last month of each
        satellite.

        The :class:`_SatPlotHelper` subclasses interface back to the
        present :class:`MatrixPlotter` class such that passing instances
        of it here, allows to repeat many of the methods defined elsewhere
        in this class to be repeated for the first and last month for each
        satellite.  The most extensive implementation is
        :class:`_SatPlotAll`.  Passing a :class:`_SatPlotAll` instance
        here will result in plotting the FFT, inter-channel, and
        inter-position correlations for all satellites, within various
        subplots, for both the beginning and ending of the period that
        this satellite was active, for all channels.

        Parameters
        ----------

        plotter : _SatPlotHelper
            Instance of implementation of :class:`_SatPlotHelper`
        sats : list[str]
            List of satellite names to include

        See also
        --------

        plot_selection_sats_early_late
            Sister method which plots this but only for a selection of 9
            satellite-channel pairs rather than all possible combinations.
        """
        if sats == "all":
            sats = self.all_sats

        gs = matplotlib.gridspec.GridSpec(20, 21)
        if plotter.multichannel:
            f_all = {}
            f_solo = {sat: {} for sat in sats}
            for ch in range(1, 20):
                f_all[ch] = matplotlib.pyplot.figure(figsize=(22, 24))
                for sat in sats:
                    f_solo[sat][ch] = matplotlib.pyplot.figure(
                        figsize=(6, 4))
        else:
            f = matplotlib.pyplot.figure(figsize=(22, 24))
            f_solo = {sat: matplotlib.pyplot.figure(figsize=(6, 4))
                        for sat in sats}

        for ((r, c), sat) in zip(itertools.product(range(4), range(4)), sats):
            logging.info(f"Processing {sat:s}")
            ep = period_pairs[sat][0]
            lp = period_pairs[sat][1]
            logging.info("Resetting to early")
            self.reset(sat, *ep)

            logging.info("Preparing early")
            if plotter.multichannel:
                for ch in range(1, 20):
                    plotter.prepare_early(self, ch)
            else:
                plotter.prepare_early(self)

            logging.info("Resetting to late")
            self.reset(sat, *lp)

            logging.info("Preparing late")
            if plotter.multichannel:
                for ch in range(1, 20):
                    plotter.prepare_late(self, ch)
                logging.info("Plotting all channels")
                for ch in range(1, 20):
                    ax = f_all[ch].add_subplot(gs[r*5:(r+1)*5-1, c*5:(c+1)*5])
                    ax_solo = f_solo[sat][ch].add_subplot(1, 1, 1)
                    plotter.plot_both(self, ax, gs, sat, r, c, ep, lp, ch,
                        c_max=3, r_max=3)
                    plotter.plot_both(self, ax_solo, None, sat, 0, 0, ep,
                                      lp, ch, solo=True, c_max=3, r_max=3)
            else:
                plotter.prepare_late(self)
                ax = f.add_subplot(gs[r*5:(r+1)*5-1, c*5:(c+1)*5])
                ax_solo = f_solo[sat].add_subplot(1, 1, 1)
                plotter.plot_both(self, ax, gs, sat, r, c, ep, lp,
                                  c_max=3, r_max=3)
                plotter.plot_both(self, ax_solo, None, sat, 0, 0, ep, lp,
                                  solo=True, c_max=3, r_max=3)

        if plotter.multichannel:
            for ch in range(1, 20):
                plotter.finalise(self, f_all[ch], gs, ch)
                for sat in sats:
                    plotter.finalise(self, f_solo[sat][ch], gs, ch,
                        sat=sat, solo=True)
        else:
            plotter.finalise(self, f, gs)
            for sat in sats:
                plotter.finalise(self, f_solo[sat], gs, sat=sat, solo=True)

    def plot_selection_sats_early_late(self, plotter, pairs):
        """Plot something a selection of sats.

        For a manual selection of 9 satellite-channel pairs, plot the
        implementation of a :class:`_SatPlotHelper`.

        The method :meth:`plot_all_sats_early_late` plots everything in
        many panels and figures, but this is overwhelming.  This method
        allows the user to make a manual selection of 9 satellites and
        channels, and plot only those.

        Parameters
        ----------

        plotter : _SatPlotHelper
            Implementation of `_SatPlotHelper`, such as `_SatPlotFFT`,
            which to plot for those 9 pairs.
        pairs : List[Tuple[str, int]]
            List of tuples with (satname, channel_number).

        See also
        --------

        plot_all_sats_early_late
            Sister method that plots all channels for all sats.
        """

        if len(pairs) != 9:
            raise ValueError(f"I want 9 pairs, not {len(pairs):d}")

        if not plotter.multichannel:
            raise ValueError(f"Cannot plot subselection for {plotter!s}")

        gs = matplotlib.gridspec.GridSpec(18, 19)
        f = matplotlib.pyplot.figure(figsize=(16, 16))
        #pairs = sorted(pairs)
        lastsat = None
        for ((r, c), (sat, ch)) in zip(
                itertools.product(range(3), range(3)), pairs):
            logging.info(f"Processing {sat:s} ch. {ch:d}")
            (ep, lp) = period_pairs[sat]

            logging.info("Resetting to early")
            self.reset(sat, *ep)

            logging.info("Preparing early")
            plotter.prepare_early(self, ch)

            logging.info("Resetting to late")
            self.reset(sat, *lp)

            logging.info("Preparing late")
            plotter.prepare_late(self, ch)

            logging.info("Plotting")
            ax = f.add_subplot(gs[r*6:(r+1)*6-1, c*6:(c+1)*6])

            gs.update(hspace=2.0)
            plotter.plot_both(self, ax, gs, sat, r, c, ep, lp, ch,
                              solo=False, c_max=2, r_max=2, custom=True)

        logging.info("Finalising figure")
        plotter.finalise(self, f, gs, custom=True)

    def plot_crosstalk_ffts_all_sats(self):
        """Plot crosstalk FFT for all satellites

        For a selection of 9 hardcoded pairs and then all satellites and
        channels, plot FFTs for both early and late period.

        Small interface to :meth:`plot_selection_sats_early_late` and
        :meth:`plot_all_sats_early_late`.
        """
        logging.info("Processing FFTs")
        self.plot_selection_sats_early_late(
            _SatPlotFFT(),
            pairs=[
                ("noaa06", 7),
                ("noaa06", 17),
                #("noaa09", 4),
                ("noaa11", 3),
                ("noaa11", 4),
                ("noaa12", 4),
                ("noaa14", 10),
                ("noaa15", 14),
                ("noaa16", 1),
                ("metopa", 12),
                ]
            )
        self.plot_all_sats_early_late(
            _SatPlotFFT(),
            sats="all")

    def plot_ch_corrmat_all_sats_b(self, channels, noise_typ, calibpos,
            sats="all"):
        """**Deprecated**. Use :meth:`plot_all_sats_early_late`.
        """

        logging.info("Processing channel correlation matrices")
        self.plot_all_sats_early_late(
            _SatPlotChCorrmat(channels, noise_typ, calibpos),
            sats="all")

    def plot_ch_corrmat_all_sats(self, channels, noise_typ, calibpos,
            sats="all"):
        """Plot channel noise correlation matrix for all sats.

        Plots lower half for first full month, upper half for last full
        month.  Result written to file.

        Parameters
        ----------

        channels : List[int]
            Channels to consider
        noise_typ : str
            Source of error determination: "iwt", "ict", "space".
        calibpos : int
            Calibration location to use for inter-channel correlation
            matrix.
        sats : List[str] or str, optional
            List of satellite names to use or (the default) "all".
        """

        channels = numpy.asarray(channels)
        if sats == "all":
            sats = self.all_sats

        f = matplotlib.pyplot.figure(figsize=(22, 24))
        gs = matplotlib.gridspec.GridSpec(20, 21)
        for ((r, c), sat) in zip(itertools.product(range(4), range(4)), sats):
#            h = tovs.which_hirs(sat)
            # early month in lower
            ep = period_pairs[sat][0]
            lp = period_pairs[sat][1]

            self.reset(sat, *ep)
            (S, ρ, ecnt) = self._get_ch_corrmat(channels, noise_typ,
                calibpos)
            S_low = numpy.tril(S, k=-1)
            self.reset(sat, *lp)
            (S, ρ, lcnt) = self._get_ch_corrmat(channels, noise_typ,
                calibpos)
            S_hi = numpy.triu(S, k=1)
            #
            ax = f.add_subplot(gs[r*5:(r+1)*5-1, c*5:(c+1)*5])
            im = self._plot_ch_corrmat(S_low+S_hi+numpy.diag(numpy.zeros(channels.size)*numpy.nan),
                                  ax, channels, add_x=r==3, add_y=c==0)
            if r==c==0:
                cax = f.add_subplot(gs[:-1, -1])
                cb = f.colorbar(im, cax=cax)
                cb.set_label("Pearson product-moment correlation coefficient")
            ax.set_title(f"{sat:s}\n"
                         f"{ep[0]:%Y-%m}, {ecnt:d} cycles\n"
                         f"{lp[0]:%Y-%m}, {lcnt:d} cycles")

        gs.update(wspace=0.10, hspace=0.4)
        f.suptitle("HIRS noise correlations for all HIRS, pos "
                  f"{calibpos:d} ", fontsize=40)
        graphics.print_or_show(f, False,
            f"hirs_noise_correlations_allsats_pos{calibpos:d}.")

    def plot_pos_corrmat_all_sats(self, noise_typ):
        """Plot set of correlation matrices for all satellites

        One set with all satellites as subplots, figure for each channel
        One set with all channels as subplots, figure for each satellite

        Parameters
        ----------

        noise_typ : str
            Source of error determination: "iwt", "ict", "space".
            
        """
        logging.info("Processing position correlation matrices")
        channels = numpy.arange(1, 21)
        sats = self.all_sats

        f_persat = {}
        f_perch = {}
        a_persat = {}
        a_perch = {}
        gs_persat = matplotlib.gridspec.GridSpec(26, 21)
        gs_perch = matplotlib.gridspec.GridSpec(21, 21)
        for ch in channels:
            # one figure per channel
            f_perch[ch] = matplotlib.pyplot.figure(figsize=(22, 24))
            a_perch[ch] = {}
            for ((r, c), sat) in zip(
                    itertools.product(range(4), range(4)), sats):
                # one panel per satellite
                a_perch[ch][sat] = f_perch[ch].add_subplot(
                    gs_perch[r*5:(r+1)*5-1, c*5:(c+1)*5])
        for sat in sats:
            # one figure per satellite
            f_persat[sat] = matplotlib.pyplot.figure(figsize=(22, 24)) 
            a_persat[sat] = {}
            for ((r, c), ch) in zip(
                    itertools.product(range(5), range(4)), channels):
                # one panel per channel
                a_persat[sat][ch] = f_persat[sat].add_subplot(
                    gs_persat[r*5:(r+1)*5-1, c*5:(c+1)*5])

        for (i, sat) in enumerate(sats):
            logging.info(f"Processing {sat:s}")
#            for ch in channels:
#                a_persat = a_persat[sat][ch]
#                a_perch = a_perch[ch][sat]
#            h = tovs.which_hirs(sat)
            # early month in lower
            ep = period_pairs[sat][0]
            lp = period_pairs[sat][1]

            logging.info("Getting early period")
            self.reset(sat, *ep)
            (S_each, _, ecnt_each) = zip(*[
                self._get_pos_corrmat(ch, noise_typ) for ch in channels])
            S_low = [numpy.tril(S, k=-1) for S in S_each]
            logging.info("Getting late period")
            self.reset(sat, *lp)
            (S_each, _, lcnt_each) = zip(*[
                self._get_pos_corrmat(ch, noise_typ) for ch in channels])
            S_hi = [numpy.triu(S, k=1) for S in S_each]
            #
            logging.info("Plotting all channels")
            for ch in channels:
                S_tot = S_low[ch-1]+S_hi[ch-1]+numpy.diag(numpy.zeros(48)*numpy.nan)

                ax = a_perch[ch][sat]
                im = self._plot_pos_corrmat(S_tot, ax,
                        add_y=i%4==0, add_x=i>=12)
                ax.set_title(f"{sat:s}\n"
                         f"{ep[0]:%Y-%m}, {ecnt_each[ch-1]:d} cycles\n"
                         f"{lp[0]:%Y-%m}, {lcnt_each[ch-1]:d} cycles")
                if i==0:
                    cax = ax.figure.add_subplot(gs_perch[:19, -1])
                    cb = ax.figure.colorbar(im, cax=cax)
                    cb.set_label("Pearson product-moment correlation coefficient")

                ax = a_persat[sat][ch]
                im = self._plot_pos_corrmat(S_tot, ax,
                        add_y=ch%4==1, add_x=ch>=17)

                if ch==1:
                    cax = ax.figure.add_subplot(gs_persat[:24, -1])
                    cb = ax.figure.colorbar(im, cax=cax)
                    cb.set_label("Pearson product-moment correlation coefficient")
                ax.set_title(f"ch. {ch:d}\n"
                             f"{ecnt_each[ch-1]:d}/{lcnt_each[ch-1]:d} cycles")
        for gs in (gs_perch, gs_persat):
            gs.update(wspace=0.10, hspace=0.4)

        for (ch, f) in f_perch.items():
            f.suptitle(f"HIRS noise correlations for all HIRS, ch. {ch:d}",
                       fontsize=40)
            graphics.print_or_show(f, False,
                f"hirs_noise_correlations_allsats_{noise_typ:s}_ch{ch:d}.")
        for (sat, f) in f_persat.items():

            f.suptitle(f"HIRS noise correlations for {sat:s}, all chans, {noise_typ:s} views\n"
                       f"{period_pairs[sat][0][0]:%Y-%m} / {period_pairs[sat][1][0]:%Y-%m}",
                        fontsize=40)
            graphics.print_or_show(f, False,
                f"hirs_noise_correlations_allchan_{sat:s}.")
#            im = self._plot_ch_corrmat(S_tot, ax, channels,
#                add_x=r==3, add_y=c==0)

 
def read_and_plot_field_matrices(p):
    """Read and plot field matrices

    Read satellite data for the beginning and ending month of each
    satellite, then make various plots for them.  See module docstring.

    Parameters
    ----------

    p : argparse.ArgumentParser
        Parsed `~argparse.ArgumentParser` instance.  The
        commandline-arguments determine the behaviour, see
        :ref:`plot-hirs-field-matrix`.
    """
#    h = fcdr.which_hirs_fcdr(sat)
        
    #temp_fields_full = ["temp_{:s}".format(t) for t in p.temp_fields]
    mp = MatrixPlotter()
#    if p.plot_all_fft:
#        mp.plot_fft()

    if p.plot_all_corr:
        mp.plot_crosstalk_ffts_all_sats()
        mp.plot_pos_corrmat_all_sats(p.noise_typ[0])
        mp.plot_ch_corrmat_all_sats_b(p.channels, p.noise_typ[0], p.calibpos[0])
        return

    from_date = datetime.datetime.strptime(p.from_date, p.datefmt)
    to_date = datetime.datetime.strptime(p.to_date, p.datefmt)
    mp.reset(p.satname, from_date, to_date)
    if p.plot_temperature_sdm:
        mp.plot_temperature_sdm(p.temperatures)

    if p.plot_temperature_corr:
        mp.plot_temperature_corrmat(sorted(p.temperatures))

    for typ in p.noise_typ:
        if p.plot_noise_level_sdm:
            mp.plot_noise_level_sdm(p.channels, typ)

        if p.plot_noise_value_scanpos_sdm:
            for npos in p.npos:
                mp.plot_noise_value_scanpos_sdm(p.channels, typ, npos)

        for calibpos in p.calibpos:
            if p.plot_noise_value_channel_sdm:
                mp.plot_noise_value_channel_sdm(p.channels, typ, calibpos)

            if p.plot_noise_value_channel_corr:
                mp.plot_noise_value_channel_corr(p.channels, typ, calibpos)
            
            if p.plot_noise_value_scanpos_corr:
                mp.plot_noise_value_scanpos_corr(p.channels, typ)


def main():
    """Main function

    Intended to be called as script.  See module docstring and
    :ref:`plot-hirs-field-matrix`.
    """
    p = parse_cmdline()
    common.set_logger(
        logging.DEBUG if p.verbose else logging.INFO,
        p.log,
        loggers={"FCDR_HIRS", "typhon"})
    matplotlib.pyplot.style.use(typhon.plots.styles("typhon"))
    read_and_plot_field_matrices(p)
