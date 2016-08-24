#!/usr/bin/env python3.5

"""Convert HIRS l1b to NetCDF-4, granule per granule

"""

import datetime
import itertools
import pathlib
import logging
import argparse
if __name__ == "__main__":
    logging.basicConfig(
        format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
                 "%(lineno)s: %(message)s"),
        level=logging.INFO)

import numpy
import matplotlib.pyplot
import matplotlib.dates
import scipy.stats
import netCDF4
import progressbar

import typhon.datasets.dataset

import pyatmlab.stats
import pyatmlab.config
import pyatmlab.graphics
import pyatmlab.datasets.tovs
import pyatmlab.tools

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__.strip(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("satellite", help="Satellite to use", type=str,
        choices=sorted(pyatmlab.datasets.tovs.HIRS2.satellites |
                 pyatmlab.datasets.tovs.HIRS3.satellites |
                 pyatmlab.datasets.tovs.HIRS4.satellites))
    parser.add_argument("begin_year", help="Year to start", type=int)
    parser.add_argument("begin_month", help="Month to start", type=int)
    parser.add_argument("begin_day", help="Day to start", type=int)
    parser.add_argument("end_year", help="Year to end", type=int)
    parser.add_argument("end_month", help="Month to end", type=int)
    parser.add_argument("end_day", help="Day to end", type=int)
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="Write destination file/s even when they exist",
                        default=False)
    p = parser.parse_args()
    return p

outdir = pathlib.Path(pyatmlab.config.conf["main"]["fiddatadir"],
    "HIRS_L1C_NC", "{sat:s}", "{year:04d}", "{month:02d}", "{day:02d}")

def convert_granule(h, satname, dt, gran, overwrite=False):
    """Reads granule and writes NetCDF file with same contents

    Arguments:

        h (pyatmlab.datasets.tovs.HIRS): Relevant HIRS-object (HIRS2,
            HIRS3, HIRS4)
        satname (str): Name of satellite
        dt (datetime.datetime): Corresponding datetime for granule
        gran (pathlib.Path): Full path to granule
    """
    (head, lines) = h.read(gran, return_header=True,
        filter_firstline=True, apply_scale_factors=True,
        calibrate=True, apply_flags=True, radiance_units="classic")
    if lines.size == 0:
        logging.error("Apparently empty: {!s}".format(gran))
        return

    outfile = pathlib.Path(str((outdir / gran.name).with_suffix(".nc")).format(
            sat=satname, year=dt.year, month=dt.month, day=dt.day))

    if outfile.exists() and not overwrite:
        logging.info("Already exists: {!s}".format(outfile))
        return

    logging.debug("{!s} → {!s}".format(gran, outfile))
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
    logging.info("Converting NOAA to NetCDF, {:s} "
        "{:%Y-%m-%d %H:%M:%S}–{:%Y-%m-%d %H:%M:%S}".format(
            sat, start_date, end_date))
    bar = progressbar.ProgressBar(maxval=1,
        widgets=pyatmlab.tools.my_pb_widget)
    bar.start()
    bar.update(0)
    for (dt, gran) in h.find_granules_sorted(start_date, end_date,
            return_time=True, satname=sat):
        try:
            convert_granule(h, sat, dt, gran, **kwargs)
        except (typhon.datasets.dataset.InvalidDataError,
                typhon.datasets.dataset.InvalidFileError) as exc:
            logging.error("Unable to process {!s}: {:s}: {!s}".format(
                gran, type(exc).__name__, exc))
        bar.update((dt-start_date)/(end_date-start_date))
    bar.update(1)
    bar.finish()

def main():
    p = parse_cmdline()
    for h in (pyatmlab.datasets.tovs.HIRS2(),
              pyatmlab.datasets.tovs.HIRS3(),
              pyatmlab.datasets.tovs.HIRS4()):
        if p.satellite in h.satellites:
            break
    else:
        raise ValueError("Unknown satellite: {:s}".format(p.satellite))
    convert_period(h, p.satellite, 
        datetime.datetime(p.begin_year, p.begin_month, p.begin_day),
        datetime.datetime(p.end_year, p.end_month, p.end_day), overwrite=p.overwrite)

if __name__ == "__main__":
    main()
