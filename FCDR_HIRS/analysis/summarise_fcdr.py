"""Summarise FCDR for period, extracting statistics or plot from them
"""

import matplotlib
#matplotlib.use("Agg") # now in matplotlibrc
from .. import common
import argparse

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = common.add_to_argparse(parser,
        include_period=True,
        include_sat=True,
        include_channels=False, # False for summarise, True for plot?
        include_temperatures=False)

    parser.add_argument("--mode", action="store", type=str,
        choices=("summarise", "plot"),
        default="summarise",
        help="Mode to use: Summarise into daily summaries, "
            "or plot from them")

    parser.add_argument("--version", action="store", type=str,
        default="0.6",
        help="Version to use.  Needs debug version stored.")
#            "or plot from them")

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
#import itertools
import datetime
import xarray
#import matplotlib
#matplotlib.use("Agg")
#pathlib.Path("/dev/shm/gerrit/cache").mkdir(parents=True, exist_ok=True)
#import matplotlib.pyplot
#import matplotlib.gridspec
import numpy
import pandas
import scipy.stats

from typhon.datasets.dataset import (DataFileError, HomemadeDataset)
from typhon.physics.units.common import ureg, radiance_units as rad_u
from typhon.physics.units.tools import UnitsAwareDataArray as UADA
import pyatmlab.graphics
from .. import fcdr

labels = dict(
    u_C_Earth = "noise [counts]")

titles = dict(
    u_C_Earth = "counts noise")

class FCDRSummary(HomemadeDataset):
    name = section = "fcdr_hirs_summary"

    stored_name = ("fcdr_hirs_summary_{satname:s}_v{fcdr_version:s}_"
        "{year:04d}{month:02d}{day:02d}_"
        "{year_end:04d}{month_end:02d}{day_end:02d}.nc")

    re = (r"fcdr_hirs_summary_(?P<satname>.{6})_"
          r"(?P<data_version>.+)_"
          r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_'
          r'(?P<year_end>\d{4})(?P<month_end>\d{2})(?P<day_end>\d{2})\.nc')

    satname = None

    hirs = None

    ptiles = numpy.linspace(0, 100, 101, dtype="u1")
    data_version = "0.4"
    time_field = "date"
    read_returns = "xarray"
    plot_file = ("hirs_summary/"
        "FCDR_hirs_summary_{satname:s}_ch{channel:d}_{start:%Y%m%d}-{end:%Y%m%d}"
        "v{data_version:s}.")

    def __init__(self, *args, satname, **kwargs):
        super().__init__(*args, satname=satname, **kwargs)

        self.hirs = fcdr.which_hirs_fcdr(satname, read="L1C")
        self.start_date = self.hirs.start_date
        self.end_date = self.hirs.end_date

    def create_summary(self, start_date, end_date):
        dates = pandas.date_range(start_date, end_date+datetime.timedelta(days=1), freq="D")
        fields=["T_b", "u_T_b_random", "u_T_b_nonrandom",
                "R_e", "u_R_Earth_random", "u_R_Earth_nonrandom",
                "u_C_Earth"]
        channels = numpy.arange(1, 20)
        summary = xarray.Dataset(
            {field: 
                  (("date", "ptile", "channel"),
                    numpy.zeros((dates.size-1, self.ptiles.size,
                                 channels.size), dtype="f4")*numpy.nan)
                for field in fields},
            coords={"date": dates[:-1], "ptile": self.ptiles, "channel": channels}
            )
        for (sd, ed) in zip(dates[:-1], dates[1:]):
            try:
                ds = self.hirs.read_period(sd, ed,
                    locator_args={"data_version": self.data_version,
                                  "fcdr_type": "debug"},
                    fields=fields)
            except DataFileError:
                continue
            bad = (2*ds["u_R_Earth_nonrandom"] > ds["R_e"])
            for v in {"T_b", "u_T_b_random", "u_T_b_nonrandom", "R_e",
                      "u_R_Earth_random", "u_R_Earth_nonrandom"}:
                ds[v].values[bad.values] = numpy.nan 
            for field in fields:
                if "hertz" in ds[field].units:
                    da = UADA(ds[field]).to(rad_u["ir"], "radiance")
                else:
                    da = ds[field]
                # cannot apply limits here https://github.com/scipy/scipy/issues/7342
                pt = scipy.stats.scoreatpercentile(
                        da.values.reshape(channels.size, -1),
                        self.ptiles, axis=1)
                summary[field].loc[{"date":sd}] = pt

        of = pathlib.Path(self.basedir) / self.subdir / self.stored_name
        of = pathlib.Path(str(of).format(
            satname=self.satname, year=dates[0].year,
            month=dates[0].month, day=dates[0].day,
            year_end=dates[-2].year,
            month_end=dates[-2].month,
            day_end=dates[-2].day,
            fcdr_version=self.data_version))
        of.parent.mkdir(parents=True, exist_ok=True)
        summary.to_netcdf(str(of))

    def plot_period(self, start, end, channel, fields,
            ptiles=[5, 25, 50, 75, 95],
            pstyles=[":", "--", "-", "--", ":"]):
        (f, a_all) = matplotlib.pyplot.subplots(figsize=(10, 3*len(fields)),
            nrows=len(fields), ncols=1, squeeze=False)
        summary = self.read_period(start, end, locator_args={"data_version": "v"+self.data_version})
        #fields = ["T_b", "u_T_b_random", "u_T_b_nonrandom"]
        total_title = (f"HIRS {self.satname:s} ch {channel:d} "
                       f"{start:%Y-%m-%d}--{end:%Y-%m-%d}")

        for field in ("u_T_b_random", "u_R_Earth_random", "u_C_Earth"):
            summary[field] *= numpy.sqrt(48) # workaround #125, remove after fix
        for (i, (fld, a)) in enumerate(zip(fields, a_all.ravel())):
            summary[fld].values[summary[fld]==0] = numpy.nan # workaround #126, redundant after fix
            for (ptile, ls) in zip(ptiles, pstyles):
                summary[fld].sel(channel=channel).sel(ptile=ptile).plot(
                    ax=a, label=str(ptile), linestyle=ls, color="black")
#            summary[fld].sel(channel=channel).T.plot.pcolormesh(
#                ax=a, vmin=200 if fld=="T_b" else 0)
            if len(fields) > 1: # more than one subplot
                a.set_title(titles.get(fld, f"{fld:s}"))
            else:
                if fld in titles.keys():
                    a.set_title(total_title + ", " + titles[fld])
                else:
                    a.set_title(total_title)
            if fld in labels.keys():
                a.set_ylabel(labels[fld])
            if len(ptiles) > 1:
                a.legend([f"p-{p:d}" for f in ptiles])
        if len(fields) > 1:
            f.suptitle(total_title)
        f.autofmt_xdate()
        pyatmlab.graphics.print_or_show(f, None, 
            self.plot_file.format(satname=self.satname, start=start,
            end=end, channel=channel, data_version=self.data_version))

def summarise():
    p = parsed_cmdline
#    if p.mode != "summarise":
#        raise NotImplementedError("Only summarising implemented yet")
    summary = FCDRSummary(satname=p.satname, data_version=p.version)
    start = datetime.datetime.strptime(p.from_date, p.datefmt)
    end = datetime.datetime.strptime(p.to_date, p.datefmt)
    if p.mode == "summarise":
        summary.create_summary(start, end)
    elif p.mode == "plot":
        sumdat = summary.plot_period(start, end, 5, fields=["u_C_Earth"],
            ptiles=[50], pstyles=["-"])
