"""Summarise FCDR for period, extracting statistics or plot from them
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
        include_channels=False,
        include_temperatures=False)

    parser.add_argument("--mode", action="store", type=str,
        choices=("summarise", "plot"),
        default="summarise",
        help="Mode to use: Summarise into daily summaries, "
            "or plot from them")

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
#import pyatmlab.graphics
from .. import fcdr

class FCDRSummary(HomemadeDataset):
    name = section = "fcdr_hirs_summary"

    stored_name = ("fcdr_hirs_summary_{satname:s}_v{fcdr_version:s}_"
        "{year:04d}{month:02d}{day:02d}_"
        "{year_end:04d}{month_end:02d}{day_end:02d}.nc")

    re = (r"fcdr_hirs_summary_(?P<satname>.{6})_"
          r"(?P<fcdr_version>.+)_"
          r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_'
          r'(?P<year_end>\d{4})(?P<month_end>\d{2})(?P<day_end>\d{2})\.nc')

    satname = None

    hirs = None

    ptiles = numpy.linspace(0, 100, 101, dtype="u1")
    version = "0.4"

    def __init__(self, *args, satname, **kwargs):
        super().__init__(*args, satname=satname, **kwargs)

        self.hirs = fcdr.which_hirs_fcdr(satname, read="L1C")

    def create_summary(self, start_date, end_date):
        dates = pandas.date_range(start_date, end_date+datetime.timedelta(days=1), freq="D")
        fields=["T_b", "u_T_b_random", "u_T_b_nonrandom",
                "R_e", "u_R_Earth_random", "u_R_Earth_nonrandom"]
        channels = numpy.arange(1, 20)
        summary = xarray.Dataset(
            {field: 
                  (("date", "ptile", "channel"),
                    numpy.zeros((dates.size-1, self.ptiles.size,
                                 channels.size), dtype="f4"))
                for field in fields},
            coords={"date": dates[:-1], "ptile": self.ptiles, "channel": channels}
            )
        for (sd, ed) in zip(dates[:-1], dates[1:]):
            try:
                ds = self.hirs.read_period(sd, ed,
                    locator_args={"fcdr_version": self.version,
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
            fcdr_version=self.version))
        of.parent.mkdir(parents=True, exist_ok=True)
        summary.to_netcdf(str(of))

def summarise():
    p = parsed_cmdline
    if p.mode != "summarise":
        raise NotImplementedError("Only summarising implemented yet")
    summary = FCDRSummary(satname=p.satname)
    summary.create_summary(
        datetime.datetime.strptime(p.from_date, p.datefmt),
        datetime.datetime.strptime(p.to_date, p.datefmt))
