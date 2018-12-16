#!/usr/bin/env python3.5

"""Convert HIRS l1b to NetCDF-4, granule per granule

"""

import datetime
import itertools
import os
import pathlib
import logging
import argparse

import numpy
p = pathlib.Path("/home/users/gholl/.cache/matplotlib/tex.cache")
if p.is_symlink() and not p.exists():
    (p.parent / os.readlink(str(p))).mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot
import matplotlib.dates
import scipy.stats
import netCDF4
import progressbar

import typhon.datasets.dataset
from typhon.datasets import filters

from .. import common
from .. import fcdr

logger = logging.getLogger(__name__)

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__.strip(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = common.add_to_argparse(parser,
        include_period=True,
        include_sat=True,
        include_channels=False,
        include_temperatures=False)
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="Write destination file/s even when they exist",
                        default=False)
    p = parser.parse_args()
    return p

outdir = pathlib.Path(typhon.config.conf["main"]["fiddatadir"],
    "HIRS_L1C_NC", "{sat:s}", "{year:04d}", "{month:02d}", "{day:02d}")

def convert_granule(h, satname, dt, gran, orbit_filters, overwrite=False):
    """Reads granule and writes NetCDF file with same contents

    Parameters
    ----------

    h : typhon.datasets.tovs.HIRS
        Relevant HIRS-object (HIRS2, HIRS3, HIRS4)
    satname : str
        Name of satellite
    dt : datetime.datetime
        Corresponding datetime for granule
    gran : pathlib.Path
        Full path to granule

    """

    (lines, extra) = h.read(gran, 
        apply_scale_factors=True,
        apply_calibration=True, radiance_units="classic")
    head = extra["header"]
    for of in orbit_filters:
        lines = of.filter(lines, **extra) # this is where flags are applied now
    if lines.size == 0:
        logger.error("Apparently empty: {!s}".format(gran))
        return

    outfile = pathlib.Path(str((outdir / gran.name).with_suffix(".nc")).format(
            sat=satname, year=dt.year, month=dt.month, day=dt.day))

    if outfile.exists() and not overwrite:
        logger.info("Already exists: {!s}".format(outfile))
        return

    logger.debug("{!s} → {!s}".format(gran, outfile))
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(outfile), mode="w", clobber=True, 
            format="NETCDF4") as ds:
        ds.description = "HIRS L1C"
        ds.history = "Converted from native HIRS L1B {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now())
        ds.author = "Gerrit Holl <g.holl@reading.ac.uk>"
        ds.dataname = head["dataname"]
            # don't use hrs_h_scnlin; I want to avoid duplicates
        scanlines = ds.createDimension("scanlines",
                                       lines.shape[0])
                                       #head["hrs_h_scnlin"][0])
        channels = ds.createDimension("channels_all", 20)
        calib_channels = ds.createDimension("channels_calib", 19)
        scanpos = ds.createDimension("scanpos", 56)

        time = ds.createVariable("time", "u4", ("scanlines",), zlib=True)
        time.units = "seconds since 1970-01-01"
        time.calendar = "gregorian"

        lat = ds.createVariable("lat", "f8",
            ("scanlines", "scanpos"), zlib=True,
            fill_value=-999.)#lines["lat"].fill_value)
        lat.units = "degrees north"

        lon = ds.createVariable("lon", "f8",
            ("scanlines", "scanpos"), zlib=True,
            fill_value=-999.)#lines["lon"].fill_value)
        lon.units = "degrees east"

        bt = ds.createVariable("bt", "f4",
            ("scanlines", "scanpos", "channels_calib"), zlib=True,
            fill_value=-999.)#lines["bt"].fill_value)
        bt.units = "K"

        if "sat_za" in lines.dtype.names:
            # Only on HIRS/3 and HIRS/4
            lza = ds.createVariable("lza", "f4",
                ("scanlines", "scanpos"), zlib=True,
                fill_value=-999.)#lines["sat_za"].fill_value)
            lza.units = "degrees"
        elif "lza_approx" in lines.dtype.names:
            lza = ds.createVariable("lza", "f4",
                ("scanlines", "scanpos"), zlib=True,
                fill_value=-999.)#lines["lza_approx"].fill_value)
            lza.units = "degrees"
            lza.note = ("Values are approximate. "
                "Original data include LZA only for outermost "
                "footprint.  Remaining estimated using a full "
                "scanline of LZAs from MetOpA scaled by the ratio "
                "of the outermost footprint LZAs.")

        rad = ds.createVariable("radiance", "f8", ("scanlines", "scanpos",
                                                   "channels_all"),
                                zlib=True,
                                fill_value=-999.)#lines["radiance"].fill_value)

        #rad.units = "W m^-2 sr^-1 Hz^-1"
        rad.units = "mW m^-2 sr^-1 cm"

        counts = ds.createVariable("counts", "i4",
            ("scanlines", "scanpos", "channels_all"), zlib=True,
            fill_value=99999)#lines["counts"].fill_value)
        counts.units = "counts"

        scanline = ds.createVariable("scanline", "i2",
            ("scanlines",), zlib=True,
            fill_value=numpy.iinfo("i2").max)#lines["hrs_scnlin"].fill_value)
        scanline.units = "number"

        scanpos = ds.createVariable("scanpos", "i1",
            ("scanpos",), zlib=True,
            fill_value=0)#numpy.iinfo("i1").max)
        scanpos.units = "number"

        scantype = ds.createVariable("scanline_type", "i1",
            ("scanlines",), zlib=True,
            fill_value=9)#lines[h.scantype_fieldname].fill_value)
        scantype.units = "flag"
        scantype.note = ("0 = Earth view  1 = space view  "
                         "2 = ICCT view  3 = IWCT view")

        ds.set_auto_mask(True)

        time[:] = netCDF4.date2num(lines["time"].astype(datetime.datetime),
            units=time.units, calendar=time.calendar)

        lat[:, :] = lines["lat"]

        lon[:, :] = lines["lon"]

        bt[:, :, :] = lines["bt"]
        
        if "sat_za" in lines.dtype.names:
            # Only on HIRS/3 and HIRS/4
            lza[:, :] = lines["sat_za"]
        elif "lza_approx" in lines.dtype.names:
            lza[:, :] = lines["lza_approx"]

        rad[:] = lines["radiance"]

        counts[:, :, :] = lines["counts"]

        scanline[:] = lines["hrs_scnlin"]

        scanpos[:] = range(56)

        scantype[:] = lines[h.scantype_fieldname]


def convert_period(h, sat, start_date, end_date, **kwargs):
    logger.info("Converting NOAA to NetCDF, {:s} "
        "{:%Y-%m-%d %H:%M:%S}–{:%Y-%m-%d %H:%M:%S}".format(
            sat, start_date, end_date))
    orbit_filters = (
        filters.FirstlineDBFilter(h, h.granules_firstline_file),
        filters.TimeMaskFilter(h),
        filters.HIRSTimeSequenceDuplicateFilter(),
        filters.HIRSFlagger(h, max_flagged=0.5),
            )
    bar = progressbar.ProgressBar(maxval=1,
        widgets=common.my_pb_widget)
    bar.start()
    bar.update(0)
    for of in orbit_filters:
        of.reset()
    h.my_pseudo_fields.clear() # currently not working because filters
                               # done after reading yet some pseudo_fields
                               # require filtering to be already done!
    for (dt, gran) in h.find_granules_sorted(start_date, end_date,
            return_time=True, satname=sat):
        try:
            for of in orbit_filters:
                of.reset()
            convert_granule(h, sat, dt, gran, orbit_filters, **kwargs)
        except (typhon.datasets.dataset.InvalidDataError,
                typhon.datasets.dataset.InvalidFileError) as exc:
            logger.error("Unable to process {!s}: {:s}: {!s}".format(
                gran, type(exc).__name__, exc))
        bar.update((dt-start_date)/(end_date-start_date))
    # do not finalise filters, I don't have an overall array to put in!
    bar.update(1)
    bar.finish()

def main():
    p = parse_cmdline()
    common.set_logger(logging.INFO,
        loggers={"FCDR_HIRS", "typhon"})
    h = fcdr.which_hirs_fcdr(p.satname)
    convert_period(h, p.satname, 
            datetime.datetime.strptime(p.from_date, p.datefmt),
            datetime.datetime.strptime(p.to_date, p.datefmt),
            overwrite=p.overwrite)
