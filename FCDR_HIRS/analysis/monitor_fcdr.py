"""Plot some monitoring info on FCDR
"""

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

import itertools
import datetime
import xarray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot

from typhon.physics.units.common import ureg, radiance_units as rad_u
from typhon.physics.units.tools import UnitsAwareDataArray as UADA
import pyatmlab.graphics
from .. import fcdr

class FCDRMonitor:
    figname = ("fcdr_perf/{self.satname:s}_{tb:%Y}/ch{ch:d}/"
               "fcdr_perf_{self.satname:s}_hirs_ch{ch:d}_{tb:%Y_%m_%d_%H%M}_"
               "-{te:%Y%m%d%H%M}.png")
    figtit = ("HIRS FCDR with uncertainties {self.satname:s} ch. {ch:d}, "
              "{tb:%Y-%m-%d %H:%M}--{te:%Y-%m-%d %H:%M} (scanpos {sp:d})")
    def __init__(self, start_date, end_date, satname,
            version="0.4",):
        self.hirsfcdr = fcdr.which_hirs_fcdr(satname, read="L1C")
        self.ds = self.hirsfcdr.read_period(
            start_date,
            end_date,
            locator_args={"fcdr_version": version, "fcdr_type": "debug"},
            fields=["T_b", "u_T_b_random", "u_T_b_nonrandom",
                "R_e", "u_R_Earth_random", "u_R_Earth_nonrandom"])
        self.satname = satname

    def plot_timeseries(self, ch, sp=28):
        counter = itertools.count()
        ds = self.ds.sel(calibrated_channel=ch, scanpos=sp)
        nrow = 4
        (fig, axes) = matplotlib.pyplot.subplots(nrow, 2,
            gridspec_kw={"width_ratios": [3, 1], "hspace": 1},
            figsize=(18, 3*nrow))

        c = next(counter)
        a_tb = axes[c, 0]
        a_tb_h = axes[c, 1]

        c = next(counter)
        a_tb_u = axes[c, 0]
        a_tb_u_h = axes[c, 1]

        self._plot_var_with_unc(
            ds["T_b"],
            ds["u_T_b_random"],
            ds["u_T_b_nonrandom"],
            a_tb, a_tb_h, a_tb_u, a_tb_u_h)

        c = next(counter)
        a_L = axes[c, 0]
        a_L_h = axes[c, 1]

        c = next(counter)
        a_L_u = axes[c, 0]
        a_L_u_h = axes[c, 1]

        self._plot_var_with_unc(
            UADA(ds["R_e"]).to(rad_u["ir"], "radiance"),
            UADA(ds["u_R_Earth_random"]).to(rad_u["ir"], "radiance"),
            UADA(ds["u_R_Earth_nonrandom"]).to(rad_u["ir"], "radiance"),
            a_L, a_L_h, a_L_u, a_L_u_h)

        fig.subplots_adjust(right=0.8, bottom=0.2, top=0.8, hspace=0.1)

        for ax in axes.ravel():
            for lab in ax.get_xticklabels():
                if ax.is_last_col():
                    # workarounds for
                    # https://github.com/matplotlib/matplotlib/issues/8509
                    # as I don't want any histogram to lose its x-axis or
                    # have rotated ticks
                    if ax.is_last_row():
                        lab.set_ha("center")
                        lab.set_rotation(0)
                    else:
                        lab.set_visible(True)
            ax.grid(axis="both")

#        a_tb.set_title("Brightness temperature")
#        a_tb_h.set_xlabel("T_b [K]")
#        a_tb_u.set_title("Uncertainties on brightness temperature")
#        a_tb_u.set_ylabel("Uncertainty [K]")
#        a_tb_u.set_xlabel("Time")
#        a_tb_u_h.set_title("Histogram")
#        a_tb_u_h.set_xlabel("Uncertainty [K]")
#        for a in axes.ravel():

        tb = ds["time"].values[0].astype("M8[s]").astype(datetime.datetime)
        te = ds["time"].values[-1].astype("M8[s]").astype(datetime.datetime)
        fig.suptitle(self.figtit.format(tb=tb, te=te,
            self=self, ch=ch, sp=sp))

        pyatmlab.graphics.print_or_show(fig, False,
            self.figname.format(tb=tb, te=te, self=self, ch=ch))

    def _plot_var_with_unc(self, da, da_rand, da_nonrand, a, a_h, a_u, a_u_h):
        unit = ureg(da.units).u
        name = getattr(da.attrs, "long_name", da.name)
        da.plot(ax=a)
        da.plot.hist(ax=a_h)
        a.set_xlabel("Time")
        a.set_ylabel("{name} [{unit:~}]".format(
            name=name, unit=unit))
        a.set_title("Timeseries of {:s}".format(name))
        a_h.set_title("Histogram of {:s}".format(name))

        for d in (da_rand, da_nonrand):
            d.plot(ax=a_u, label=d.name)
            d.plot.hist(ax=a_u_h, label=d.name, histtype="step")
        
        a_u_h.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        a_u.set_title("Uncertainty timeseries of {:s}".format(name))
        a_u_h.set_title("Uncertainty histogram of {:s}".format(name))

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
