"""Plot some monitoring info on FCDR
"""

import matplotlib
matplotlib.use("Agg")
from .. import common
import argparse

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = common.add_to_argparse(parser,
        include_period=True,
        include_sat=True,
        include_channels=True,
        include_temperatures=False)

    p = parser.parse_args()
    return p
parsed_cmdline = parse_cmdline()

import logging
logging.basicConfig(
    format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
             "%(lineno)s: %(message)s"),
    filename=parsed_cmdline.log,
    level=logging.DEBUG if parsed_cmdline.verbose else logging.INFO)

import pathlib
import itertools
import datetime
import xarray
#pathlib.Path("/dev/shm/gerrit/cache").mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot
import matplotlib.gridspec
import numpy

from typhon.physics.units.common import ureg, radiance_units as rad_u
from typhon.physics.units.tools import UnitsAwareDataArray as UADA
import pyatmlab.graphics
from .. import fcdr
from .. import _fcdr_defs

# NB: https://github.com/pydata/xarray/issues/1661#issuecomment-339525582
from pandas.tseries import converter
converter.register()

class FCDRMonitor:
    figname = ("fcdr_perf/{self.satname:s}_{tb:%Y}/ch{ch:d}/"
               "fcdr_perf_{self.satname:s}_hirs_{self.version:s}_ch{ch:d}_{tb:%Y%m%d%H%M}"
               "-{te:%Y%m%d%H%M}.png")
    figtit = ("HIRS FCDR v{self.version:s} with uncertainties {self.satname:s} ch. {ch:d}, "
              "{tb:%Y-%m-%d %H:%M}--{te:%Y-%m-%d %H:%M} (scanpos {sp:d})")
    fields=["T_b", "u_T_b_random", "u_T_b_nonrandom",
        "R_e", "u_R_Earth_random", "u_R_Earth_nonrandom",
        'u_from_R_selfE', 'u_from_a_0', 'u_from_a_2', 'u_from_a_1',
        'u_from_C_s', 'u_from_C_IWCT', 'u_from_R_IWCT', 'u_from_B',
        'u_from_Tstar', 'u_from_β', 'u_from_α', 'u_from_T_IWCT',
        'u_from_O_TIWCT', 'u_from_fstar', 'u_from_R_refl', 'u_from_C_E',
        "quality_scanline_bitmask", "quality_channel_bitmask",
        "quality_minorframe_bitmask", "quality_pixel_bitmask"]

    def __init__(self, start_date, end_date, satname,
            version="0.8pre",):
        self.hirsfcdr = fcdr.which_hirs_fcdr(satname, read="L1C")
        self.version = version
        self.ds = self.hirsfcdr.read_period(
            start_date,
            end_date,
            locator_args={"data_version": version, "fcdr_type": "debug",
                          "format_version": "0.6"},
            fields=self.fields)
        self.satname = satname

    def plot_timeseries(self, ch, sp=28):
        counter = itertools.count()
        ds = self.ds.sel(calibrated_channel=ch, scanpos=sp, minor_frame=sp)
        nrow = 7
        gs = matplotlib.gridspec.GridSpec(nrow, 4)
        fig = matplotlib.pyplot.figure(figsize=(18, 3*nrow))
#        (fig, axes) = matplotlib.pyplot.subplots(nrow, 2,
#            gridspec_kw={"width_ratios": [3, 1], "hspace": 1},
#            figsize=(18, 3*nrow))

#        bad = (2*ds["u_R_Earth_nonrandom"] > ds["R_e"])
#        for v in self.fields:
#            ds[v][bad] = numpy.nan 

        if not numpy.isfinite(ds["T_b"]).any():
            logging.warning("Found no valid BTs for "
                f"channel {ch:d}, skipping")
            return

        c = next(counter)
        a_tb = fig.add_subplot(gs[c, :3])
        a_tb_h = fig.add_subplot(gs[c, 3])

        c = next(counter)
        a_tb_u = fig.add_subplot(gs[c, :3])
        a_tb_u_h = fig.add_subplot(gs[c, 3])

        # components
        c = next(counter)
        a_tb_ucmp = fig.add_subplot(gs[c, :3])
        a_tb_ucmp_h = fig.add_subplot(gs[c, 3])
        
        bad = (
            (ds["quality_scanline_bitmask"]&_fcdr_defs.FlagsScanline.DO_NOT_USE) |
            (ds["quality_channel_bitmask"]&_fcdr_defs.FlagsChannel.DO_NOT_USE) |
            (ds["quality_pixel_bitmask"]&_fcdr_defs.FlagsPixel.DO_NOT_USE)
            )!=0
        
        # This doesn't work
        # ds[["T_b","u_T_b_random","u_T_b_nonrandom"]][{"scanline_earth": bad}] = numpy.nan
        for fld in {f for f in self.fields
                    if f.startswith("u_")
                    or f in {"T_b", "R_e"}}:
            ds[fld].loc[{"scanline_earth": bad}] = numpy.nan

        self._plot_var_with_unc(
            ds["T_b"],
            ds["u_T_b_random"],
            ds["u_T_b_nonrandom"],
            a_tb, a_tb_h, a_tb_u, a_tb_u_h)

        dsu = ds[[x for x in ds.data_vars.keys() if x.startswith("u_from_")]]
        self._plot_unc_comps(dsu, a_tb_ucmp, a_tb_ucmp_h)

        # flags
        c = next(counter)
        # although exact same width as other time series would be
        # desirable, the colourbar currently messes this up /anyway/, so
        # we might as well take the full width
        a_flags = fig.add_subplot(gs[c, :])
        perc_all = []
        labels = []
        period = ("5min" if
            (ds["time"][-1]-ds["time"][0]).values.astype("m8[s]") < numpy.timedelta64(2, 'h')
            else "1H")
        for f in ("scanline", "channel", "minorframe", "pixel"):
            da = ds[f"quality_{f:s}_bitmask"]
            (perc, meanings) = common.sample_flags(da, period, "scanline_earth")
            perc_all.append(perc)
            labels.extend(f"{f:s}_{mean:s}" for mean in meanings)
        perc = xarray.concat(perc_all, dim="flag")
        # this causes trouble when all values become nan (no flags set
        # during plotting period); and there are real nans (no data in
        # period)
        #perc.values[perc.values==0] = numpy.nan
        my_cmap = matplotlib.cm.get_cmap('cool')
        my_cmap.set_under("white")
        im = perc.T.plot.pcolormesh(ax=a_flags, cmap=my_cmap, vmin=0.0001)
        im.set_clim([0.0001, 1])
        a_flags.set_yticks(numpy.arange(len(labels)))
        a_flags.set_yticklabels(labels)
        a_flags.set_title("Percentage of flag set per hour")
#            "{:s} {:%Y%m%d}-{:%Y%m%d}".format(self.satname, start, end))
        a_flags.grid(axis="x")

#        a_tb_u_h = fig.add_subplot(gs[c, 3])

        c = next(counter)
        a_L = fig.add_subplot(gs[c, :3])
        a_L_h = fig.add_subplot(gs[c, 3])

        c = next(counter)
        a_L_u = fig.add_subplot(gs[c, :3])
        a_L_u_h = fig.add_subplot(gs[c, 3])

        self._plot_var_with_unc(
            UADA(ds["R_e"]).to(rad_u["ir"], "radiance"),
            UADA(ds["u_R_Earth_random"]).to(rad_u["ir"], "radiance"),
            UADA(ds["u_R_Earth_nonrandom"]).to(rad_u["ir"], "radiance"),
            a_L, a_L_h, a_L_u, a_L_u_h)

        c = next(counter)
        gridsize = 50
        cmap = "viridis"
        self._plot_hexbin(
            ds["T_b"], ds["u_T_b_random"],
            fig.add_subplot(gs[c, 0]))
        self._plot_hexbin(
            ds["T_b"], ds["u_T_b_nonrandom"],
            fig.add_subplot(gs[c, 1]))
        self._plot_hexbin(
            UADA(ds["R_e"]).to(rad_u["ir"], "radiance"),
            UADA(ds["u_R_Earth_random"]).to(rad_u["ir"], "radiance"),
            fig.add_subplot(gs[c, 2]))
        hb = self._plot_hexbin(
            UADA(ds["R_e"]).to(rad_u["ir"], "radiance"),
            UADA(ds["u_R_Earth_nonrandom"]).to(rad_u["ir"], "radiance"),
            fig.add_subplot(gs[c, 3]))
        # todo: colorbar

        fig.subplots_adjust(right=0.8, bottom=0.2, top=0.9, hspace=1.0,
                            wspace=0.4)

        for ax in fig.get_axes():
            for lab in ax.get_xticklabels():
                lab.set_visible(True)
                if ax.is_last_col() or ax.is_last_row():
                    # workarounds for
                    # https://github.com/matplotlib/matplotlib/issues/8509
                    # as I don't want any histogram to lose its x-axis or
                    # have rotated ticks
#                    if ax.is_last_row():
                    lab.set_ha("center")
                    lab.set_rotation(0)
#                    else:
                else:
                    lab.set_rotation(30)
            if not ax.is_last_col() and not ax.is_last_row():
                ax.set_xlabel("Time")
            ax.grid(axis="both")
        a_tb_h.set_xlabel(a_tb.get_ylabel())
        a_tb_u_h.set_xlabel(a_tb.get_ylabel())
        a_L_h.set_xlabel(a_L.get_ylabel())
        a_L_u_h.set_xlabel(a_L_u.get_ylabel())

        tb = ds["time"].values[0].astype("M8[s]").astype(datetime.datetime)
        te = ds["time"].values[-1].astype("M8[s]").astype(datetime.datetime)
        fig.suptitle(self.figtit.format(tb=tb, te=te,
            self=self, ch=ch, sp=sp))

        pyatmlab.graphics.print_or_show(fig, False,
            self.figname.format(tb=tb, te=te, self=self, ch=ch))

    def _plot_var_with_unc(self, da, da_rand, da_nonrand, a, a_h, a_u, a_u_h):
        unit = ureg(da.units).u
        name = getattr(da.attrs, "long_name", da.name)
        if da.isnull().all():
            logging.error("All nans :(")
            return
        da.plot(ax=a)
        da.plot.hist(ax=a_h)
        a.set_xlabel("Time")
        a.set_ylabel("{name:s}\n[{unit:~}]".format(
            name=name, unit=unit))
        a.set_title("Timeseries of {:s}".format(name))
        a_h.set_title("Histogram of {:s}".format(name))

        for d in (da_rand, da_nonrand):
            d.plot(ax=a_u, label=d.name)
            d.plot.hist(ax=a_u_h, label=d.name, histtype="step")
        
        a_u_h.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        a_u.set_title("Uncertainty timeseries of {:s}".format(name))
        a_u.set_ylabel(r"$\Delta$ " + a.get_ylabel())
        a_u_h.set_title("Uncertainty histogram of {:s}".format(name))

    def _plot_unc_comps(self, ds, a, a_h, n=8):
        # take 8 largest
        for k in [x[-1] for x in sorted([(x.mean(), k) for (k, x) in ds.data_vars.items()])[-1:-n-1:-1]]:
            da = ds[k]
            if da.isnull().all():
                continue
            da.plot(ax=a, label=k[7:])
            da.plot.hist(ax=a_h, label=k[7:], histtype="step")

        a_h.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        a.set_title("Uncertainty components")
        name = getattr(da.attrs, "long_name", da.name)
        #unit = ureg(da.units).u
        a.set_ylabel(rf"$\Delta$ {name:s} [K]")#f"\n[{unit:~}]")
        a.set_xlabel("Time")
        a_h.set_title("Uncertainty hists")

    def _plot_hexbin(self, da, Δda, a):
        unit = ureg(da.units).u
        name = getattr(da.attrs, "long_name", da.name)
        if da.isnull().all():
            logging.error("still all nans ☹")
            return
        hb = a.hexbin(da, Δda, gridsize=50, cmap="viridis", mincnt=1,
            marginals=False)
        a.set_xlabel("{name:s}\n[{unit:~}]".format(name=name, unit=unit))
        a.set_ylabel(Δda.name.split("_")[-1] + r" $\Delta$" + a.get_xlabel())
        a.set_title("Joint distribution for {name:s}".format(name=name))
        return hb

def plot():
    p = parsed_cmdline
    fm = FCDRMonitor(
        datetime.datetime.strptime(p.from_date, p.datefmt),
        datetime.datetime.strptime(p.to_date, p.datefmt),
        p.satname)

    for ch in p.channels:
        fm.plot_timeseries(ch)

def main():
    plot()
