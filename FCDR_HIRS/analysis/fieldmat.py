"""Plot scatter field density plots for temperature or other

"""

import argparse
from .. import common

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description="Plot field scatter density matrices (SDM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parse = common.add_to_argparse(parser,
        include_period=True,
        include_sat=True,
        include_channels=True,
        include_temperatures=True)

    parser.add_argument("--plot_temperature_sdm", action="store_true",
        help="Include scatter density matrix (SDM) of Digital A telemetry"
             " temperatares")

    parser.add_argument("--plot_noise_level_sdm", action="store_true",
        help="Include SDM of noise levels between channels")

    parser.add_argument("--plot_noise_value_scanpos_sdm",
        action="store_true",
        help="Include SDM of noise values between scan positions")

    parser.add_argument("--npos", action="store", type=int,
        default=[6], nargs="+",
        help="When plotting SDM of noise values between scan positions, "
             "plot this number")

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
        help="When plotting SDM of noise values between chanels, "
             "plot this scan position")

    parser.add_argument("--noise_typ",
        action="store",
        choices=["iwt", "ict", "space"],
        default=["iwt"],
        nargs="+",
        help="What source of noise to plot for")

    parser.add_argument("--plot_all_corr", action="store_true",
        help="Plot all channel correlations for beginning and end of "
             "satellite lifetime for all satellites")

    p = parser.parse_args()
    return p

parsed_cmdline = parse_cmdline()

import logging
#logging.basicConfig(
#    format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
#            "%(lineno)s: %(message)s"),
#    filename=parsed_cmdline.log,
#    level=logging.DEBUG if parsed_cmdline.verbose else logging.INFO)

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
import pyatmlab.graphics

from typhon.physics.units.common import ureg
from .. import fcdr
from typhon.datasets import tovs
from typhon.datasets.dataset import DataFileError

month_pairs = dict(
    tirosn = ((1978, 11), (1979, 12)),
    noaa06 = ((1979, 7), (1983, 3)),
    noaa07 = ((1981, 8), (1984, 12)),
    noaa08 = ((1983, 5), (1984, 6)),
    noaa09 = ((1985, 2), (1988, 11)),
    noaa10 = ((1986, 11), (1991, 9)),
    noaa11 = ((1988, 11), (1998, 12)),
    noaa12 = ((1991, 9), (1998, 12)),
    noaa14 = ((1995, 1), (2005, 12)),
    noaa15 = ((1999, 1), (2009, 6)),
    noaa16 = ((2001, 1), (2014, 6)),
    noaa17 = ((2002, 7), (2013, 4)),
    noaa18 = ((2006, 11), (2011, 3)),
    noaa19 = ((2009, 4), (2013, 7)),
    metopa = ((2006, 12), (2016, 10)),
    metopb = ((2013, 2), (2017, 5)))

period_pairs = {sat:
    ((datetime.datetime(*start, 1), datetime.datetime(*start, 28, 23, 59)),
     (datetime.datetime(*end, 1), datetime.datetime(*end, 28, 23, 59)))
        for (sat, (start, end)) in month_pairs.items()}

def plot_field_matrix(MM, ranges, title, filename, units):
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
    pyatmlab.graphics.print_or_show(f, False, filename)

#
class _SatPlotHelper(metaclass=abc.ABCMeta):
    """Helper for MatrixPlotter.plot_all_sats_early_late
    """

    @abc.abstractmethod
    def prepare_early(self, mp, **extra):
        ...

    @abc.abstractmethod
    def prepare_late(self, mp, **extra):
        ...

    @abc.abstractmethod
    def plot_both(self, mp, ax, gs, sat, r, c, ep, lp, **extra):
        ...

    @abc.abstractmethod
    def finalise(self, mp, f, gs, **extra):
        ...
#
class _SatPlotChCorrmat(_SatPlotHelper):
    def __init__(self, channels, noise_typ, calibpos):
        self.channels = numpy.asarray(channels)
        self.noise_typ = noise_typ
        self.calibpos = calibpos
    
    def _get_ch_corrmat(self, mp):
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

    def plot_both(self, mp, ax, gs, sat, r, c, ep, lp):
        im = mp._plot_ch_corrmat(
            self.S_low +
            self.S_hi +
            numpy.diag(numpy.zeros(self.channels.size)*numpy.nan),
                              ax, self.channels, add_x=r==3, add_y=c==0)
        if r==c==0:
            f = ax.figure
            cax = f.add_subplot(gs[:-1, -1])
            cb = f.colorbar(im, cax=cax)
            cb.set_label("Pearson product-moment correlation coefficient")
        ax.set_title(f"{sat:s}\n"
                     f"{ep[0]:%Y-%m}, {self.ecnt:d} cycles\n"
                     f"{lp[0]:%Y-%m}, {self.lcnt:d} cycles")

    def finalise(self, mp, f, gs):
        gs.update(wspace=0.10, hspace=0.4)
        f.suptitle("HIRS noise correlations for all HIRS, pos "
                  f"{self.calibpos:d} ", fontsize=40)
        pyatmlab.graphics.print_or_show(f, False,
            f"hirs_noise_correlations_allsats_pos{self.calibpos:d}.")
#
class _SatPlotFFT(_SatPlotHelper):
    """For plotting FFT stuff
    """

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

    def plot_both(self, mp, ax, gs, sat, r, c, ep, lp, channel):
        """Plot FFT for calibration counts and Earth views
        """

        n = self.n

        # 0.1 seconds between observations (NOAA KLM User's Guide, Appendix)
        #x = numpy.fft.fftfreq(n, d=0.1)
        # rather do pixels than seconds (feedback CM)
        x = numpy.fft.fftfreq(n, d=1)

        ax.plot(
            x[1:n//2],
            abs(numpy.fft.fft(self.early_spc[channel], axis=1, n=n)[:, 1:n//2]).mean(0),
            color="cyan",
            label="early, space")
        ax.plot(
            x[1:n//2],
            abs(numpy.fft.fft(self.early_ec[channel], axis=1, n=n)[:, 1:n//2]).mean(0),
            color="tan",
            label="early, Earth")
        ax.plot(
            x[1:n//2],
            abs(numpy.fft.fft(self.early_iwctc[channel], axis=1, n=n)[:, 1:n//2]).mean(0),
            color="red",
            label="early, IWCT")

        ax.plot(
            x[1:n//2],
            abs(numpy.fft.fft(self.late_spc[channel], axis=1, n=n)[:, 1:n//2]).mean(0),
            color="cyan",
            linestyle="--",
            label="late, space")
        ax.plot(
            x[1:n//2],
            abs(numpy.fft.fft(self.late_ec[channel], axis=1, n=n)[:, 1:n//2]).mean(0),
            color="tan",
            linestyle="--",
            label="late, Earth")
        ax.plot(
            x[1:n//2],
            abs(numpy.fft.fft(self.late_iwctc[channel], axis=1, n=n)[:, 1:n//2]).mean(0),
            color="red",
            linestyle="--",
            label="late, IWCT")

        ax.set_xscale("log")
        ax.set_yscale("log")
        xpos = [48, 24, 12, 6, 3]
        ax.set_xticks([1/x for x in xpos])
        ax.grid(axis="both")
        if r==3:
            ax.set_xlabel("Frequency [cycles/pixel]")
            #ax.set_xscale('log')
            #ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xticklabels([fr"$\frac{{1}}{{{x:d}}}$" for x in xpos])
        else:
            ax.set_xticklabels([])
        if c==0:
            ax.set_ylabel("Amplitude [counts]")
        else:
            ax.set_yticklabels([])
        if r==0 and c==3:
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
            #ax.legend()
        ax.set_title(f"{sat:s}\n"
                     f"{ep[0]:%Y-%m}, {self.early_cnt[channel]:d} cycles\n"
                     f"{lp[0]:%Y-%m}, {self.late_cnt[channel]:d} cycles")
        ax.set_ylim([1e0, 1e4])

    def finalise(self, mp, f, gs, channel):
        gs.update(wspace=0.10, hspace=0.4)
        f.subplots_adjust(right=0.8, bottom=0.2, top=0.90)
        f.suptitle(f"Spectral analysis, channel {channel:d}")
        pyatmlab.graphics.print_or_show(f, False,
            f"hirs_crosstalk_fft_ch{channel:d}.")

class _SatPlotAll(_SatPlotChCorrmat, _SatPlotFFT):
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

    """

    all_sats = ["tirosn"] + [f"noaa{no:02d}" for no in range(6, 20) if no!=13] + ["metopa", "metopb"]
#    def __init__(self, sat, from_date, to_date):
#        self.reset(sat, from_date, to_date)

    def reset(self, sat, from_date, to_date):
        h = tovs.which_hirs(sat)
        self.hirs = h
        M = h.read_period(from_date, to_date,
            fields=["temp_{:s}".format(t) for t in h.temperature_fields] + 
                   ["counts", "time", h.scantype_fieldname])
        self.M = M
        self.start_date = from_date
        self.end_date = to_date

        self.title_sat_date = "{sat:s} {from_date:%Y-%m-%d} -- {to_date:%Y-%m-%d}".format(
            **locals())
        self.filename_sat_date = "{sat:s}_{from_date:%Y}/{sat:s}_{from_date:%Y%m%d%H%M}--{to_date:%Y%m%d%H%M}".format(
            **locals())

    def _get_temps(self, temp_fields):
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
        views = self.M[self.hirs.scantype_fieldname] == getattr(self.hirs, "typ_{:s}".format(noise_typ))
        ccnt = self.M["counts"][views, 8:, :]
        mccnt = ccnt.mean(1, keepdims=True)
        accnt = ccnt - mccnt
        return accnt

    def plot_noise_value_scanpos_sdm(self, channels,
            noise_typ="iwt",
            npos=6):

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

        For channels, noise_typ (iwt, ict, space), and calibration
        position.

        No return; writes file.
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
        pyatmlab.graphics.print_or_show(f, False,
                "hirs_noise_correlations_channels_{:s}_ch_{:s}_{:s}{:d}.".format(
            self.filename_sat_date,
            ",".join(str(ch) for ch in channels),
            noise_typ, calibpos))

    def _get_ch_corrmat(self, channels, noise_typ, calibpos):
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
        """Helper for plot_noise_value_channel_corr
        """
        im = a.imshow(S, cmap="PuOr", interpolation="none", vmin=-1, vmax=1)
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
        im = a.imshow(S, cmap="PuOr", interpolation="none", vmin=-1, vmax=1)
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
        accnt = self._get_accnt(noise_typ)
        unmasked = ~(accnt[:, :, ch-1].mask.any(1))
        (S, p) = typhon.math.stats.corrcoef(accnt[unmasked, :, ch-1].T)
        return (S, p, unmasked.sum())

    def plot_noise_value_scanpos_corr(self, channels,
            noise_typ="iwt"):

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
            im1 = ax_all[0].imshow(S, cmap="PuOr", interpolation="none")
            im1.set_clim([-1, 1])
            cb1 = f.colorbar(im1, cax=ax_all[1])
            cb1.set_label("Correlation coefficient")
            ax_all[0].set_title("Pearson correlation")

            # choose range for S
            upto = scipy.stats.scoreatpercentile(abs(S[S<1]), 99)
            im2 = ax_all[3].imshow(S, cmap="PuOr", vmin=-upto, vmax=upto)
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
            pyatmlab.graphics.print_or_show(f, False,
                "hirs_noise_correlations_scanpos_{:s}_ch{:d}_{:s}.png".format(
                    self.filename_sat_date, ch, noise_typ))

    def plot_temperature_corrmat(self, temp_fields):
        """Plot correlation matrix for temperatures.

        NB, this is a correlation matrix for the actual temperatures — not
        for their noises.
        """
        MM = self._get_temps(temp_fields)
        MMp = MM.view("<f8").reshape(MM.shape[0], -1).T
        MMp = MMp[:, (~MMp.mask).all(0)]
        S = numpy.corrcoef(MMp)
        (f, a) = matplotlib.pyplot.subplots(1, 1)
        im = a.imshow(S, cmap="PuOr", interpolation="none")
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

        pyatmlab.graphics.print_or_show(f, False,
            f"hirs_temperature_correlation_{self.filename_sat_date:s}.png")


    def plot_all_sats_early_late(self, plotter, sats):
        if sats == "all":
            sats = self.all_sats

        f_all = {}
        for ch in range(1, 20):
            f_all[ch] = matplotlib.pyplot.figure(figsize=(22, 24))
        gs = matplotlib.gridspec.GridSpec(20, 21)
        for ((r, c), sat) in zip(itertools.product(range(4), range(4)), sats):
            ep = period_pairs[sat][0]
            lp = period_pairs[sat][1]
            self.reset(sat, *ep)
            for ch in range(1, 20):
                plotter.prepare_early(self, ch)
            self.reset(sat, *lp)
            for ch in range(1, 20):
                plotter.prepare_late(self, ch)
            for ch in range(1, 20):
                ax = f_all[ch].add_subplot(gs[r*5:(r+1)*5-1, c*5:(c+1)*5])
                plotter.plot_both(self, ax, gs, sat, r, c, ep, lp, ch)
        for ch in range(1, 20):
            plotter.finalise(self, f_all[ch], gs, ch)

    def plot_crosstalk_ffts_all_sats(self):
        self.plot_all_sats_early_late(
            _SatPlotFFT(),
            sats="all")

    def plot_ch_corrmat_all_sats_b(self, channels, noise_typ, calibpos,
        sats="all"):

        self.plot_all_sats_early_late(
            _SatPlotChCorrmat(channels, noise_typ, calibpos),
            sats="all")

    def plot_ch_corrmat_all_sats(self, channels, noise_typ, calibpos,
            sats="all"):
        """Plot channel noise covariance matrix for all sats.

        Plots lower half for first full month, upper half for last full
        month.
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
        pyatmlab.graphics.print_or_show(f, False,
            f"hirs_noise_correlations_allsats_pos{calibpos:d}.")

    def plot_pos_corrmat_all_sats(self, noise_typ):
        """Plot set of correlation matrices for all satellites

        One set with all satellites as subplots, figure for each channel
        One set with all channels as subplots, figure for each satellite
        """
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
#            for ch in channels:
#                a_persat = a_persat[sat][ch]
#                a_perch = a_perch[ch][sat]
#            h = tovs.which_hirs(sat)
            # early month in lower
            ep = period_pairs[sat][0]
            lp = period_pairs[sat][1]

            self.reset(sat, *ep)
            (S_each, _, ecnt_each) = zip(*[
                self._get_pos_corrmat(ch, noise_typ) for ch in channels])
            S_low = [numpy.tril(S, k=-1) for S in S_each]
            self.reset(sat, *lp)
            (S_each, _, lcnt_each) = zip(*[
                self._get_pos_corrmat(ch, noise_typ) for ch in channels])
            S_hi = [numpy.triu(S, k=1) for S in S_each]
            #
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
            pyatmlab.graphics.print_or_show(f, False,
                f"hirs_noise_correlations_allsats_{noise_typ:s}_ch{ch:d}.")
        for (sat, f) in f_persat.items():

            f.suptitle(f"HIRS noise correlations for {sat:s}, all chans, {noise_typ:s} views\n"
                       f"{period_pairs[sat][0][0]:%Y-%m} / {period_pairs[sat][1][0]:%Y-%m}",
                        fontsize=40)
            pyatmlab.graphics.print_or_show(f, False,
                f"hirs_noise_correlations_allchan_{sat:s}.")
#            im = self._plot_ch_corrmat(S_tot, ax, channels,
#                add_x=r==3, add_y=c==0)

 
def read_and_plot_field_matrices():
#    h = fcdr.which_hirs_fcdr(sat)
    p = parsed_cmdline
        
    #temp_fields_full = ["temp_{:s}".format(t) for t in p.temp_fields]
    mp = MatrixPlotter()
#    if p.plot_all_fft:
#        mp.plot_fft()

    if p.plot_all_corr:
#        mp.plot_pos_corrmat_all_sats(p.noise_typ[0])
#        mp.plot_ch_corrmat_all_sats_b(p.channels, p.noise_typ[0], p.calibpos[0])
        mp.plot_crosstalk_ffts_all_sats()
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
    logging.basicConfig(
        format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
                "%(lineno)s: %(message)s"),
        filename=parsed_cmdline.log,
        level=logging.DEBUG if parsed_cmdline.verbose else logging.INFO)
    matplotlib.pyplot.style.use(typhon.plots.styles("typhon"))
    read_and_plot_field_matrices()
