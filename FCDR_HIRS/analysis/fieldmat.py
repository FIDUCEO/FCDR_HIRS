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

    p = parser.parse_args()
    return p

parsed_cmdline = parse_cmdline()

import logging
logging.basicConfig(
    format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
            "%(lineno)s: %(message)s"),
    filename=parsed_cmdline.log,
    level=logging.DEBUG if parsed_cmdline.verbose else logging.INFO)

import matplotlib
matplotlib.use("Agg")
import pathlib
pathlib.Path("/dev/shm/gerrit/cache").mkdir(parents=True, exist_ok=True)

import datetime
import scipy.stats
import numpy

import matplotlib.pyplot
import matplotlib.ticker
import matplotlib.gridspec
import typhon.plots
matplotlib.pyplot.style.use(typhon.plots.styles("typhon"))
import typhon.plots.plots
import pyatmlab.graphics

from typhon.physics.units.common import ureg
from .. import fcdr

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

class MatrixPlotter:
    """Plot varous scatter density matrices and correlation matrices

    """

    def __init__(self, sat, from_date, to_date):
        h = fcdr.which_hirs_fcdr(sat)
        self.hirs = h
        M = h.read_period(from_date, to_date,
            fields=["temp_{:s}".format(t) for t in h.temperature_fields] + 
                   ["counts", "time"])
        self.M = M
        self.start_date = from_date
        self.end_date = to_date

        self.title_sat_date = "{sat:s} {from_date:%Y-%m-%d} -- {to_date:%Y-%m-%d}".format(
            **locals())
        self.filename_sat_date = "{sat:s}_{from_date:%Y}/{sat:s}_{from_date:%Y%m%d%H%M}--{to_date:%Y%m%d%H%M}".format(
            **locals())

    def plot_temperature_sdm(self, temp_fields):
        MM = numpy.zeros(
            shape=self.M.shape,
            dtype=[(t, "f8") for t in temp_fields])
        for t in temp_fields:
            x = self.M["temp_{:s}".format(t)]
            while x.ndim > 1:
                x = x.mean(-1)
            MM[t] = x
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
        accnt = self._get_accnt(noise_typ)
        (f, ax_all) = matplotlib.pyplot.subplots(1, 3, figsize=(16, 8),
            gridspec_kw={"width_ratios": (14, 14, 1)})
        channels = numpy.asarray(channels)
        # although there is a scipy.stats.mstats module,
        # scipy.stats.mstats.spearman can only calculate individual
        # covariances, not covariance matrices (it's not vectorised) and
        # explicit looping is too slow
        unmasked = ~(accnt[:, calibpos, :].mask.any(1))
        S = numpy.corrcoef(accnt[:, calibpos,  channels-1].T[:, unmasked])
        im1 = ax_all[0].imshow(S, cmap="PuOr", interpolation="none")
        ρ = scipy.stats.spearmanr(accnt[:, calibpos, channels-1][unmasked, :])[0]
        im2 = ax_all[1].imshow(ρ, cmap="PuOr", interpolation="none")
        for (a, im) in zip(ax_all[:2], (im1, im2)):
            im.set_clim([-1, 1])
            a.set_xticks(numpy.arange(len(channels)))
            a.set_yticks(numpy.arange(len(channels)))
            a.set_xticklabels([str(ch) for ch in channels])
            a.set_yticklabels([str(ch) for ch in channels])
            a.set_xlabel("Channel no.")
            a.set_ylabel("Channel no.")
        cb = f.colorbar(im2, cax=ax_all[2])
        cb.set_label("Correlation")
        ax_all[0].set_title("Pearson correlation")
        ax_all[1].set_title("Spearman correlation")
        f.suptitle("HIRS noise correlations, {:s}, {:s} pos {:d}\n"
            "({:d} cycles)".format(
            self.title_sat_date, noise_typ, calibpos, unmasked.sum()))
        pyatmlab.graphics.print_or_show(f, False,
                "hirs_noise_correlations_{:s}_ch_{:s}_{:s}{:d}.png".format(
            self.filename_sat_date,
            ",".join(str(ch) for ch in channels),
            noise_typ, calibpos))

def read_and_plot_field_matrices():
#    h = fcdr.which_hirs_fcdr(sat)
    p = parsed_cmdline
    from_date = datetime.datetime.strptime(p.from_date, p.datefmt)
    to_date = datetime.datetime.strptime(p.to_date, p.datefmt)
    #temp_fields_full = ["temp_{:s}".format(t) for t in p.temp_fields]
    mp = MatrixPlotter(p.satname, from_date, to_date)
    if p.plot_temperature_sdm:
        mp.plot_temperature_sdm(p.temperatures)

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

#    M = h.read_period(from_date, to_date,
#        fields=temp_fields_full + ["counts", "time"])
#    title = "HIRS {{what:s}} {sat:s} {from_date:%Y-%m-%d} -- {to_date:%Y-%m-%d}".format(
#        **locals())
#    filename="hirs_{{what:s}}_{sat:s}_{from_date:%Y%m%d%H%M}--{to_date:%Y%m%d%H%M}".format(
#        **locals())
#    plot_temperature_matrix(M, temp_fields,
#        title=title.format(what="temperatures"),
#        filename=filename.format(what="tempmat") +
#            "_T_{:s}.png".format(",".join(temp_fields)))
#    for typ in ("iwt", "space"):
#        plot_noise_level_matrix(h, M, channels, noise_typ=typ,
#            title=title.format(what="{:s} noise levels".format(typ)),
#            filename=filename.format(what="{:s}noiselevel".format(typ)) + 
#                "_ch_{:s}.png".format(','.join([str(x) for x in channels])))
#        plot_noise_value_matrix(h, M, channels, noise_typ=typ,
#            title=title.format(what="{:s} noise values".format(typ)) +
#                " ch. {chstr:s}",
#            filename=filename.format(what="{:s}noisevalue".format(typ)) + 
#                "_ch_{chstr:s}.png",
#            npos=6)


def main():
    read_and_plot_field_matrices()
