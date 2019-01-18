"""Plot flags.
"""

import matplotlib
#matplotlib.use("Agg")
import pathlib
#pathlib.Path("/dev/shm/gerrit/cache").mkdir(parents=True, exist_ok=True)
import argparse
from .. import common

import logging

import datetime
import xarray
import numpy
import matplotlib.pyplot
import typhon.datasets.tovs
import typhon.datasets.filters
from .. import graphics

logger = logging.getLogger(__name__)
def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = common.add_to_argparse(parser,
        include_period=True,
        include_sat=True,
        include_channels=False,
        include_temperatures=False)

    return parser
def parse_cmdline():
    return get_parser().parse_args()

def plot(sat, start, end):
    h = typhon.datasets.tovs.which_hirs(sat)
    #h15 = typhon.datasets.tovs.HIRS3(satname="noaa15")
    M15 = h.read_period(
        start, end,
        orbit_filters=[f for f in h.default_orbit_filters
            if not isinstance(f,
            (typhon.datasets.filters.HIRSCalibCountFilter,
             typhon.datasets.filters.HIRSPRTTempFilter,
             typhon.datasets.filters.HIRSFlagger))],
        reader_args={"apply_calibration": False},
        fields=["hrs_qualind", "hrs_linqualflgs", "hrs_chqualflg",
                "hrs_mnfrqual", "time"])#, "lat", "lon"])

    ds = h.as_xarray_dataset(M15)

    perc_all = []
    labels = []
    for fld in ("quality_flags_bitfield", "line_quality_flags_bitfield",
                "channel_quality_flags_bitfield",
                "minorframe_quality_flags_bitfield"):
        da = ds[fld]
#        flags = da & xarray.DataArray(da.flag_masks, dims=("flag",))
#        perc = (100*(flags!=0)).resample("1H", dim="time", how="mean")
#        for d in set(perc.dims) - {"time", "flag"}:
#            perc = perc.mean(dim=d)
#        perc_all.append(perc)
        (perc, meanings) = common.sample_flags(da, "1H", "time")
#        labels.extend(da.flag_meanings.split())
        perc_all.append(perc)
        labels.extend(meanings)

    (f, a) = matplotlib.pyplot.subplots(1, 1, figsize=(14, 9))

    perc = xarray.concat(perc_all, dim="flag")
    # I want 0 distinct from >0
    perc.values[perc.values==0] = numpy.nan
    perc.T.plot.pcolormesh(ax=a)
    a.set_yticks(numpy.arange(len(labels)))
    a.set_yticklabels(labels)
    a.set_title("Percentage of flag set per hour "
        "{:s} {:%Y%m%d}-{:%Y%m%d}".format(sat, start, end))
    a.grid(axis="x")
    #f.subplots_adjust(left=0.2)

    graphics.print_or_show(f, False,
        "hirs_flags/{sat:s}_{start:%Y}/hirs_flags_set_{sat:s}_{start:%Y%m%d%H%M}-{end:%Y%m%d%H%M}.png".format(
            sat=sat, start=start, end=end))

def main():
    p = parse_cmdline()
    common.set_logger(
        logging.DEBUG if p.verbose else logging.INFO,
        p.log,
        loggers={"FCDR_HIRS", "typhon"})
        
    sat = p.satname
    start = datetime.datetime.strptime(p.from_date, p.datefmt)
    end = datetime.datetime.strptime(p.to_date, p.datefmt)
    plot(sat, start, end)
