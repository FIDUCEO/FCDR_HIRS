"""Summarise FCDR for period, extracting statistics or plot from them

This module and script fulfills two roles:

- Either it can summarise any numerical time-varying field from the easy
  or the debug FCDR.  Here, summarising means that it stores for each day
  a non-normalised histogram (total counts per bin) as well as percentiles
  for the field.  The location that the summaries are written to is
  determined by the ``fcdr_hirs_summary`` field in the configuration file
  (see :ref:`configuration`).

- Or it can create a plot based on the information contained within the
  summary files.  When in plotting mode, can choose either one satellite or "all"
  satellites (within period), and make a plot of:

    - Total histogram of field values, split in subplots where each subplot shows
      several channels, and the channels are sorted by the typical magnitude
      of the quantity that the histogram belongs to.  For example, if several
      channels have typical uncertainties of 0.1 K and others have typical
      uncertainties of 2 K, those channels will be grouped in different panels
      so that all histograms are more or less optimised to the dynamic range.
      It's always 2×2 subplots arranged in a square.  One figure per channel.

    - Time series of percentiles per field, one subplot per field, all
      subplots take the full width of the figure and are plotted below each
      other, with the size of the figure adapted to the number of panels.
      If all satellites are plotted, the colours are fixed per satellite, even
      if plotting a different period in which a different subset of satellites
      is active.  Two figures are written per channel: one with the y-axis
      optimised for most of the range and another with the y-axes optimised
      for the central part of the range.

"""

import matplotlib
import math
import itertools
from .. import common
import argparse
import inspect
import functools
import errno

import logging

import pathlib
import itertools
import datetime
import xarray
import numpy
import pandas
import scipy.stats
import scipy.stats.mstats

# see https://github.com/pydata/xarray/issues/1661#issuecomment-339525582
from pandas.tseries import converter
converter.register()

from typhon.datasets.dataset import (DataFileError, HomemadeDataset)
from typhon.physics.units.common import ureg, radiance_units as rad_u
from typhon.physics.units.tools import UnitsAwareDataArray as UADA
from typhon.datasets.tovs import norm_tovs_name
from .. import graphics
from .. import fcdr

logger = logging.getLogger(__name__)

labels = dict(
    u_C_Earth = "noise [counts]",
    bt = "Brightness temperature [K]",
    u_independent = "Independent $\Delta$ BT [K]",
    u_structured = "Structured $\Delta$ BT [K]")

titles = dict(
    u_C_Earth = "counts noise",
    bt = labels["bt"][:-3],
    u_independent = "Independent brightness temperature uncertainty [K]",
    u_structured = "Structured brightness temperature uncertainty [K]")

class StoreNameRangePair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values)%3 != 0:
            raise ValueError("Name-range pairs must be muliple of 3")
        ivals = iter(values)
        setattr(namespace, self.dest, {})
        D = {}
        for i in range(len(values)//3):
            (name, start, end) = itertools.islice(ivals, 0, 3)
            D[name] = (float(start), float(end))
        setattr(namespace, self.dest, D)

def get_parser():
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

    parser.add_argument("--type", action="store", type=str,
        choices=("debug", "easy"),
        default="debug",
        help="Type of FCDR to consider.")

    parser.add_argument("--fields", action="store", type=str,
        nargs="+",
        default=["default"],
        help="Fields to summarise or plot.  For example: "
            "bt u_independent u_structured u_common. "
            "Passing 'default' will choose a default depending "
            "on the type.")

    parser.add_argument("--field-ranges", action=StoreNameRangePair,
        nargs="*",
        default={},
        help="Applicable when mode=store.  For the histograms for different "
             "fields, define what range is used for estimating the "
             "histogram.  Must pass a multiple of three in number of "
             "arguments, with a format: FIELD lower high FIELD lower "
             "higher.  This argument does not support channel-dependence.")

    parser.add_argument("--version", action="store", type=str,
        default="0.8pre",
        help="Version to use.")

    parser.add_argument("--format-version", action="store", type=str,
        default="0.7",
        help="Format version to use.")

    parser.add_argument("--ptiles", action="store", type=int,
        nargs="*",
        default=[5, 25, 50, 75, 95],
        help="Percentiles to plot.  Recommended to plot at least always 50.")

    parser.add_argument("--pstyles", action="store", type=str,
        default="- -- - -- -",
        help="Style for percentiles.  Should be single string argument "
            "to prevent styles misinterpreted as flag hyphens.")
    return parser
def parse_cmdline():
    return get_parser().parse_args()

import pathlib
import datetime
import xarray
import numpy
import pandas
import scipy.stats
import scipy.stats.mstats

# see https://github.com/pydata/xarray/issues/1661#issuecomment-339525582
from pandas.tseries import converter
converter.register()

from typhon.datasets.dataset import (DataFileError, HomemadeDataset)
from typhon.physics.units.common import ureg, radiance_units as rad_u
from typhon.physics.units.tools import UnitsAwareDataArray as UADA
from typhon.datasets.tovs import norm_tovs_name
from .. import fcdr

labels = dict(
    u_C_Earth = "noise [counts]",
    bt = "Brightness temperature [K]",
    u_independent = "Independent $\Delta$ BT [K]",
    u_structured = "Structured $\Delta$ BT [K]")

titles = dict(
    u_C_Earth = "counts noise",
    bt = labels["bt"][:-3],
    u_independent = "Independent brightness temperature uncertainty [K]",
    u_structured = "Structured brightness temperature uncertainty [K]")

class FCDRSummary(HomemadeDataset):
    name = section = "fcdr_hirs_summary"

    stored_name = ("fcdr_hirs_summary_{satname:s}_v{fcdr_version:s}_"
        "fv{format_version:s}_{fcdr_type:s}_"
        "{year:04d}{month:02d}{day:02d}_"
        "{year_end:04d}{month_end:02d}{day_end:02d}.nc")

    re = (r"fcdr_hirs_summary_(?P<satname>.{6})_"
          r"(?P<data_version>.+)_"
          r"(?P<format_version>.+)_"
          r"(?P<fcdr_type>.+)_"
          r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_'
          r'(?P<year_end>\d{4})(?P<month_end>\d{2})(?P<day_end>\d{2})\.nc')

    satname = None

    hirs = None

    ptiles = numpy.linspace(0, 100, 101, dtype="u1")
    data_version = "0.8"
    format_version = "0.7"
    time_field = "date"
    read_returns = "xarray"
    plot_file = ("hirs_summary/"
        "FCDR_hirs_summary_{satname:s}_ch{channel:d}_{start:%Y%m%d}-{end:%Y%m%d}_p{ptilestr:s}"
        "f{fieldstr:s}_tp{type:s}_"
        "v{data_version:s}_fv{format_version:s}.")
    plot_hist_file = ("hirs_summary/"
        "FCDR_hirs_hist_{satname:s}_{start:%Y%m%d}-{end:%Y%m%d}"
        "v{data_version:s}_fv{format_version:s}.")

    fields = {
        "debug":
            ["T_b", "u_T_b_random", "u_T_b_nonrandom", 
            "R_e", "u_R_Earth_random", "u_R_Earth_nonrandom",
            "u_C_Earth"],
        "easy":
            ["bt", "u_independent", "u_structured", "u_common"]
          }
    
    # extra fields needed in analysis but not summarised
    extra_fields = {
        "debug":
            {"quality_scanline_bitmask", "quality_channel_bitmask",
             "quality_pixel_bitmask"},
        "easy":
            {"quality_scanline_bitmask", "quality_channel_bitmask"},
        }

    hist_range = xarray.Dataset(
        {
        **{field: (("edges",), [170, 320]) for field in ("T_b", "bt", "T_IWCT")},
        **{field: (("edges",), [0, 50]) for field in 
            ["u_T_b_random", "u_T_b_nonrandom", "u_T_b_harm",
             "u_independent", "u_structured", "u_common",
             "toa_brightness_temperature"]},
        **{field: (("channel", "edges"),
                   [[0, 200]]*10+[[0, 100]]*2+[[0,10]]*7)
            for field in ("R_e", "u_R_Earth",
                          "u_R_Earth_random",
                          "u_R_Earth_nonrandom", "u_R_Earth_harm",
                          "R_selfE", "u_Rself", 
                          "R_IWCT", 'rad_wn_nooffset', 'rad_wn_norself',
                          'rad_wn_norselfnooffset', 'rad_wn_linear',
                          'rad_wn_linearnooffset', 'rad_wn_linearnorself',
                          'rad_wn_linearnorselfnooffset',
                          'rad_wn_noεcorr', 'rad_wn_nooffsetnoεcorr',
                          'rad_wn_norselfnoεcorr',
                          'rad_wn_norselfnooffsetnoεcorr',
                          'rad_wn_linearnoεcorr',
                          'rad_wn_linearnooffsetnoεcorr',
                          'rad_wn_linearnorselfnoεcorr',
                          'rad_wn_linearnorselfnooffsetnoεcorr',
                          "toa_outgoing_radiance_per_unit_frequency"
                          )},
        **{field: (("edges",), [-4097, 4098]) for field in
            ["C_E", "C_IWCT", "C_s", "u_C_Earth", "u_C_space",
            "u_C_IWCT"]},
        },
        coords={"channel": numpy.arange(1, 20)})
    nbins = 2000

    def __init__(self, *args, satname, **kwargs):
        super().__init__(*args, satname=satname, **kwargs)

        if [int(d) for d in self.format_version.split(".")] > [0, 7]:
            self.fields = self.fields.copy() # avoid sharing between instances
            self.fields["debug"].extend(["u_T_b_harm", "u_R_Earth_harm"])
        # special value 'all' used in summary plotting
        if satname == "all":
            self.start_date = datetime.datetime(1978, 1, 1)
            self.end_date = datetime.datetime(2018, 1, 1)
        else:
            self.sethirs(satname)

            self.start_date = self.hirs.start_date
            self.end_date = self.hirs.end_date

    def sethirs(self, satname):
        self.satname = satname
        self.hirs = fcdr.which_hirs_fcdr(satname, read="L1C")

    def create_summary(self, start_date, end_date,
            fields=None,
            fcdr_type="debug",
            field_ranges=None):
        dates = pandas.date_range(start_date, end_date+datetime.timedelta(days=1),
            freq="D")
        fields = fields if fields is not None else []
        fields.extend([f for f in self.fields[fcdr_type] if f not in fields])
        logging.debug("Summarising fields: " + " ".join(fields))
        if field_ranges is None:
            field_ranges = {}
        chandim = "channel" if fcdr_type=="easy" else "calibrated_channel"
        channels = numpy.arange(1, 20)

        hist_range = self.hist_range.copy()

        for (field, (lo, hi)) in field_ranges.items():
            hist_range[field] = ("edges", [lo, hi])

        #bins = numpy.linspace(self.hist_range, self.nbins)
        summary = xarray.Dataset(
            {
            **{field: 
                  (("date", "ptile", "channel"),
                    numpy.zeros((dates.size-1, self.ptiles.size,
                                 channels.size), dtype="f4")*numpy.nan)
                for field in fields},
            **{f"hist_{field:s}":
                  (("date", "bin_index", "channel"),
                    numpy.zeros((dates.size-1, self.nbins+1,
                                 channels.size), dtype="u4"))
                for field in fields},
            **{f"bins_{field:s}":
                (("channel", "bin_edges"),
                # numpy.concatenate([[numpy.concatenate([[0],
                # numpy.linspace(170, 320, 100), [1000, 10000]])] for i in
                # range(5)]) 
                    numpy.concatenate([[
                    numpy.concatenate(
                        [[min(
                            hist_range.sel(channel=ch,edges=0)[field]-1,
                            0)],
                          numpy.linspace(
                            hist_range.sel(
                                channel=ch,
                                edges=0)[field],
                            hist_range.sel(
                                channel=ch,
                                edges=1)[field], self.nbins,
                                dtype="f4"),
                        [max(
                            hist_range.sel(channel=ch,edges=1)[field]+1,
                            1000)]])]
                    for ch in channels]))
                for field in fields},
            },
            coords={"date": dates[:-1], "ptile": self.ptiles, "channel": channels}
        )

        for (sd, ed) in zip(dates[:-1], dates[1:]):
            try:
                ds = self.hirs.read_period(sd, ed,
                    onerror="skip",
                    excs=inspect.signature(self.hirs.read_period).parameters["excs"].default + (KeyError, OSError),
                    locator_args={"data_version": self.data_version,
                                  "format_version": self.format_version,
                                  "fcdr_type": fcdr_type},
                    fields=fields+[f for f in self.extra_fields[fcdr_type] if f not in fields])
                if fcdr_type=="easy" and ds["u_structured"].dims == ():
                    raise DataFileError("See https://github.com/FIDUCEO/FCDR_HIRS/issues/171")
            #except (DataFileError, KeyError) as e:
            except DataFileError as e:
                logger.warning("Could not read "
                    f"{sd:%Y-%m-%d}--{ed:%Y-%m-%d}: {e!r}: {e.args[0]:s}")
                continue
            if fcdr_type == "debug":
                bad = ((2*ds["u_R_Earth_nonrandom"] > ds["R_e"]) |
                        ((ds["quality_scanline_bitmask"] & 1)!=0) |
                        ((ds["quality_channel_bitmask"] & 1)!=0))
            else: # should be "easy"
                bad = ((2*ds["u_structured"] > ds["bt"]) |
                       ((ds["quality_scanline_bitmask"].astype("uint8") & 1)!=0) |
                       ((ds["quality_channel_bitmask"].astype("uint8") & 1)!=0))
            for field in fields:
                if field != "u_C_Earth":
                    # workaround for https://github.com/FIDUCEO/FCDR_HIRS/issues/152
                    try:
                        ds[field].values[bad.transpose(*ds[field].dims).values] = numpy.nan 
                    except ValueError:
                        # I seem to be unabel to mask this field
                        pass
            for field in fields:
                if "hertz" in ds[field].units:
                    da = UADA(ds[field]).to(rad_u["ir"], "radiance")
                else:
                    da = ds[field]
                if not da.notnull().any():
                    # hopeless
                    logger.warning(f"All bad data for {self.satname:s} "
                        f"{sd.year:d}-{sd.month:d}-{sd.day:d}–{ed.year:d}-{ed.month:d}-{ed.day}, not "
                        f"summarising {field:s}.")
                    continue
                # cannot apply limits here https://github.com/scipy/scipy/issues/7342
                # and need to mask nans, see
                # https://github.com/scipy/scipy/issues/2178
#                pt = scipy.stats.scoreatpercentile(
#                        da.values.reshape(channels.size, -1),
#                        self.ptiles, axis=1)
                # take transpose as workaround for
                # https://github.com/FIDUCEO/FCDR_HIRS/issues/152
                # make sure we always reshape the same way... this causes
                # both #172 and #173
                pt = scipy.stats.mstats.mquantiles(
                    numpy.ma.masked_invalid(
                        (da.transpose("channel", "x", "y")
                         if fcdr_type == "easy"
                         else da).values.reshape(channels.size, -1)),
                        prob=self.ptiles/100, axis=1,
                        alphap=0, betap=1).T
                summary[field].loc[{"date":sd}] = pt

                for ch in range(1, 20):
                    summary[f"hist_{field:s}"].loc[
                        {"date": sd, "channel": ch}] = numpy.histogram(
                            da.loc[{chandim:ch}],
                            bins=summary[f"bins_{field:s}"].sel(channel=ch),
                            range=(da.min(), da.max()))[0]

        of = pathlib.Path(self.basedir) / self.subdir / self.stored_name
        of = pathlib.Path(str(of).format(
            satname=self.satname, year=dates[0].year,
            month=dates[0].month, day=dates[0].day,
            year_end=dates[-2].year,
            month_end=dates[-2].month,
            day_end=dates[-2].day,
            fcdr_version=self.data_version,
            format_version=self.format_version,
            fcdr_type=fcdr_type))
        of.parent.mkdir(parents=True, exist_ok=True)

        for field in fields:
            summary[field].encoding.update({
#                "scale_factor": 0.001,
#                "_FillValue": numpy.iinfo("int32").min,
                "zlib": True,
#                "dtype": "int32",
                "complevel": 4})
            summary["hist_"+field].encoding.update({
                "zlib": True,
                "complevel": 4})
            summary["bins_"+field].encoding.update({
                "zlib": True,
                "complevel": 4,
#                "dtype": "int32",
#                "_FillValue": numpy.iinfo("int32").min,
#                "scale_factor": 0.001})
                })

        logger.info(f"Writing {of!s}")
        summary.to_netcdf(str(of))

    def plot_period_ptiles(self, start, end, fields,
            ptiles=[5, 25, 50, 75, 95],
            pstyles=[":", "--", "-", "--", ":"],
            fcdr_type="debug",
            sats=None):
        if sats is None:
            sats = self.satname

        allsats = fcdr.list_all_satellites_chronologically()
        if sats == "all":
            sats = allsats
            satlabel = "all"
        else:
            sats = [sats]
            satlabel = self.satname

        if fields is None:
            fields = self.fields[fcdr_type]
        
        figs = {}
        for channel in range(1, 20):
            figs[channel] = matplotlib.pyplot.subplots(figsize=(20, 4.5*len(fields)),
                nrows=len(fields), ncols=1, squeeze=False)
                
        ranges = xarray.DataArray(
            numpy.full((len(sats), 19, len(fields), 4), numpy.nan, dtype="f4"),
            dims=("satname", "channel", "field", "extremum"),
            coords={"satname": sats, "channel": range(1, 20),
                "field": fields,
                "extremum": ["lo", "midlo", "midhi", "hi"]})

        oldsatname = self.satname
        sc = itertools.count()
        for sat in sats:
            if sat != self.satname:
                self.satname = sat
            try:
                summary = self.read_period(start, end,
                    locator_args={"data_version": "v"+self.data_version,
                        "format_version": "fv" + self.format_version,
                        "fcdr_type": fcdr_type,
                        "satname": sat},
                    NO_CACHE=True)
                if all(summary.isnull().all()[
                        [f for f in fields if f in summary.data_vars.keys()]].all().variables.values()):
                    raise DataFileError(f"All data invalid for {sat:s}!")
            except DataFileError:
                continue
            else:
                si = next(sc)
            
            np = len(ptiles)
            if len(sats) == 1:
                pcolors = [f"C{i:d}" for i in range(np)]
            else:
                pcolors = [f"C{allsats.index(sat)%10:d}"]*np

            for channel in range(1, 20):
                total_title = (f"HIRS {satlabel:s} ch. {channel:d} "
                               f"{start:%Y-%m-%d}--{end:%Y-%m-%d}")
                (f, a_all) = figs[channel]
                for (i, (fld, a)) in enumerate(zip(fields, a_all.ravel())):
                    try:
                        if not numpy.isfinite(summary[fld].sel(channel=channel)).any():
                            logger.error(f"All nans for channel {channel:d} "
                                f"{fld:s}, skipping")
                            continue
                    except KeyError as e:
                        logger.error("Can't plot ptiles for "
                            f"channel {channel:d} {fld:s}, no {e.args[0]:s}")
                        continue
                    #summary[fld].values[summary[fld]==0] = numpy.nan # workaround #126, redundant after fix
                    for (ptile, ls, color) in zip(sorted(ptiles), pstyles, pcolors):
                        if i!=0:
                            label = ""
                        elif len(sats)==1:
                            label = f"p-{ptile:d}"
                        elif si==0:
                            label = f"{sat:s} p-{ptile:d}"
                        elif ptile==50:
                            label = sat
                        else:
                            label = "" # https://stackoverflow.com/a/50068622/974555
                        # Do I need to avoid xarray.DataArray.plot if I
                        # want to suppress the label?
                        # https://stackoverflow.com/q/50068423/974555
                        summary[fld].sel(channel=channel).sel(ptile=ptile).plot(
                            ax=a, label=label, linestyle=ls, color=color)
                    if len(fields) > 1: # more than one subplot
                        a.set_title(titles.get(fld, f"{fld:s}"))
                    else:
                        if fld in titles.keys():
                            a.set_title(total_title + ", " + titles[fld])
                        else:
                            a.set_title(total_title)
                    if fld in labels.keys():
                        a.set_ylabel(labels[fld])
                    if i==0 and (len(ptiles)>1 or len(sats)>1):
                        a.legend(loc="upper left", bbox_to_anchor=(1, 1))
                    a.grid(True, axis="both")
                    a.set_xlim(start, end)
                # prepare some info for later, with zoomed-in y-axes or # regular
                for (fld, a) in zip(fields, a_all.ravel()):
                    if not fld in summary.data_vars.keys():
                        continue
                    lo = scipy.stats.mstats.mquantiles(
                        numpy.ma.masked_invalid(
                            summary[fld].sel(channel=channel).sel(ptile=5).values),
                        prob=0.01,
                        alphap=0, betap=0)
                    midlo = scipy.stats.mstats.mquantiles(
                        numpy.ma.masked_invalid(
                            summary[fld].sel(channel=channel).sel(ptile=25).values),
                        prob=0.05,
                        alphap=0, betap=0)
                    midhi = scipy.stats.mstats.mquantiles(
                        numpy.ma.masked_invalid(
                            summary[fld].sel(channel=channel).sel(ptile=75).values),
                        prob=.95,
                        alphap=0, betap=0)
                    hi = scipy.stats.mstats.mquantiles(
                        numpy.ma.masked_invalid(
                            summary[fld].sel(channel=channel).sel(ptile=95).values),
                        prob=.99,
                        alphap=0, betap=0)
                    ranges.loc[{"satname": sat, "channel": channel,
                                "field": fld}].values[...] = [
                                    lo, midlo, midhi, hi]
                if len(fields) > 1:
                    f.suptitle(total_title)
                f.autofmt_xdate()
            summary.close()
            del summary
        
        
        form = functools.partial(
            self.plot_file.format,
            satname=satlabel, start=start,
            end=end, data_version=self.data_version,
            format_version=self.format_version,
            type=fcdr_type, ptilestr=','.join(str(p) for p in ptiles))
        fieldstr=",".join(fields)
        fieldstr_compact=",".join(nm[:2]+nm[-2:] for nm in fields)

        for channel in range(1, 20):
            (f, a_all) = figs[channel]

            for (fld, a) in zip(fields, a_all.ravel()):
                lo = ranges.loc[
                    {"channel": channel, "field": fld,
                     "extremum": "lo"}].min()
                hi = ranges.loc[
                    {"channel": channel, "field": fld,
                     "extremum": "hi"}].max()
                a.set_ylim([lo*.9, hi*1.1])
            try:
                graphics.print_or_show(f, False, 
                    form(channel=channel, fieldstr=fieldstr))
            except OSError as e:
                if e.errno == errno.ENAMETOOLONG:
                    graphics.print_or_show(f, False, 
                        form(channel=channel, fieldstr=fieldstr_compact))
                else:
                    raise
        # another set with zoomed-in y-axes
        for channel in range(1, 20):
            (f, a_all) = figs[channel]
            for (fld, a) in zip(fields, a_all.ravel()):
                lo = ranges.loc[
                    {"channel": channel, "field": fld,
                     "extremum": "midlo"}].min()
                hi = ranges.loc[
                    {"channel": channel, "field": fld,
                     "extremum": "midhi"}].max()
                a.set_ylim([lo*.9, hi*1.1])
            try:
                graphics.print_or_show(
                    f, False, 
                    form(channel=channel, fieldstr=fieldstr)[:-1]+"_zoom.")
            except OSError as e:
                if e.errno == errno.ENAMETOOLONG:
                    graphics.print_or_show(
                        f, False,
                        form(channel=channel, fieldstr=fieldstr_compact)[:-1]+"_zoom.")
                else:
                    raise
        self.satname = oldsatname


    def plot_period_hists(self, start, end, fcdr_type="easy"):
        (f, a_all) = matplotlib.pyplot.subplots(figsize=(20, 12),
            nrows=2, ncols=2, squeeze=False)
        summary = self.read_period(start, end,
            locator_args={"data_version": "v"+self.data_version,
                "format_version": "fv" + self.format_version,
                "fcdr_type": fcdr_type})
        
        tit = f"HIRS {self.satname:s} {start:%Y-%m-%d}--{end:%Y-%m-%d}"

        if fcdr_type == "easy":
            lab_struc = "structured"
            lab_indy = "independent"
            lab_comm = "common"
        else:
            lab_struc = "T_b_nonrandom"
            lab_indy = "T_b_random"
            lab_comm = "T_b_harm"
        # The dynamic range of structured uncertainties varies a lot
        # between channels.  Plotting together channels with large
        # differences in dynamic range of structured uncertainty may lead
        # to some of them being squeezed.  So I want to sort the channels
        # according to how much of the x-axis they need in their
        # histograms.
        idx_p95 = (summary.dims["bin_index"] -
                  ((summary[f"hist_u_{lab_struc:s}"].sum("date").cumsum("bin_index") /
                    summary[f"hist_u_{lab_struc:s}"].sum("date").sum("bin_index")) >
                        .95).sum("bin_index"))
        ch_order = idx_p95.channel[idx_p95.values.argsort()]
        
        for (i, a) in enumerate(a_all.ravel()):
            chs = sorted(ch_order[i*5:(i+1)*5].values)
            idx_hi = int(idx_p95.sel(channel=chs).max())
            for (k, ch) in enumerate(chs):
                # take off last value because bins are the edges so
                # this array is one longer than the corresponding hist
                # values
                x = summary[f"bins_u_{lab_indy:s}"].sel(channel=ch)[:-1]
                y = summary[f"hist_u_{lab_indy:s}"].sel(channel=ch).sum("date")
                a.plot(x, y, label=f"Ch. {ch:d}, {lab_indy:s}",
                    color=f"C{k:d}", linestyle="--")
                y = summary[f"hist_u_{lab_struc:s}"].sel(channel=ch).sum("date")
                a.plot(x, y, label=f"Ch. {ch:d}, {lab_struc:s}",
                    color=f"C{k:d}", linestyle="-")
                try:
                    y = summary[f"hist_u_{lab_comm:s}"].sel(channel=ch).sum("date")
                except KeyError as e:
                    logger.error(f"Cannot plot hist for channel {ch:d}: " + e.args[0])
                else:
                    a.plot(x, y, label=f"Ch. {ch:d}, {lab_comm:s}",
                        color=f"C{k:d}", linestyle=":")
            a.legend()
            a.set_xlim([0,
                float(summary[f"bins_u_{lab_struc:s}"].sel(channel=chs[0]).isel(bin_edges=idx_hi))])
            a.set_xlabel("Brightness temperature uncertainty [K]")
            a.set_ylabel("Total number of pixels")
        f.suptitle(tit)
        graphics.print_or_show(f, None, 
            self.plot_hist_file.format(satname=self.satname, start=start,
            end=end, data_version=self.data_version,
            format_version=self.format_version))

def summarise():
    p = parse_cmdline()
    common.set_logger(
        logging.DEBUG if p.verbose else logging.INFO,
        filename=p.log,
        loggers={"FCDR_HIRS", "typhon"})
#    if p.mode != "summarise":
#        raise NotImplementedError("Only summarising implemented yet")
    summary = FCDRSummary(satname=p.satname, data_version=p.version,
        format_version=p.format_version)
    start = datetime.datetime.strptime(p.from_date, p.datefmt)
    end = datetime.datetime.strptime(p.to_date, p.datefmt)
    fields = p.fields
    if fields == ["default"]:
        fields = None
    if p.mode == "summarise":
        summary.create_summary(start, end, fcdr_type=p.type,
            fields=fields, field_ranges=p.field_ranges)
    elif p.mode == "plot":
#        sumdat = summary.plot_period(start, end, 5, fields=["u_C_Earth"],
#            ptiles=[50], pstyles=["-"], fcdr_type=p.type)
        if p.satname != "all":
            # plotting "all" not supported for hists
            summary.plot_period_hists(start, end, p.type)
#            fields=["bt", "u_independent", "u_structured"],
#            fcdr_type="easy")
#        fields = ["bt", "u_independent", "u_structured"]
#        if float(p.format_version) >= 0.7:
#            fields.append("u_common")
        summary.plot_period_ptiles(start, end,
            fields=fields,
            fcdr_type=p.type,
            ptiles=p.ptiles,
            pstyles=p.pstyles.split())
