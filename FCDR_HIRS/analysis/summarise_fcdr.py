"""Summarise FCDR for period, extracting statistics or plot from them
"""

import matplotlib
from .. import common
import argparse


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
import pyatmlab.graphics
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

    parser.add_argument("--type", action="store", type=str,
        choices=("debug", "easy"),
        default="debug",
        help="Type of FCDR to consider.  For now, fields are "
             "hardcoded by type.")

    parser.add_argument("--version", action="store", type=str,
        default="0.8pre",
        help="Version to use.")
#            "or plot from them")

    parser.add_argument("--ptiles", action="store", type=int,
        nargs="*",
        default=[5, 25, 50, 75, 95],
        help="Percentiles to plot.  Recommended to plot at least always 50.")

    parser.add_argument("--pstyles", action="store", type=str,
        default=": -- - -- :",
        help="Style for percentiles.  Should be single string argument "
            "to prevent styles misinterpreted as flag hyphens.")
    p = parser.parse_args()
    return p

class FCDRSummary(HomemadeDataset):
    name = section = "fcdr_hirs_summary"

    stored_name = ("fcdr_hirs_summary_{satname:s}_v{fcdr_version:s}_{fcdr_type:s}_"
        "{year:04d}{month:02d}{day:02d}_"
        "{year_end:04d}{month_end:02d}{day_end:02d}.nc")

    re = (r"fcdr_hirs_summary_(?P<satname>.{6})_"
          r"(?P<data_version>.+)_"
          r"(?P<fcdr_type>.+)_"
          r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_'
          r'(?P<year_end>\d{4})(?P<month_end>\d{2})(?P<day_end>\d{2})\.nc')

    satname = None

    hirs = None

    ptiles = numpy.linspace(0, 100, 101, dtype="u1")
    data_version = "0.4"
    time_field = "date"
    read_returns = "xarray"
    plot_file = ("hirs_summary/"
        "FCDR_hirs_summary_{satname:s}_ch{channel:d}_{start:%Y%m%d}-{end:%Y%m%d}_p{ptilestr:s}"
        "v{data_version:s}.")
    plot_hist_file = ("hirs_summary/"
        "FCDR_hirs_hist_{satname:s}_{start:%Y%m%d}-{end:%Y%m%d}"
        "v{data_version:s}.")

    fields = {
        "debug":
            ["T_b", "u_T_b_random", "u_T_b_nonrandom",
            "R_e", "u_R_Earth_random", "u_R_Earth_nonrandom",
            "u_C_Earth"],
        "easy":
            ["bt", "u_independent", "u_structured"],
          }
    
    # extra fields needed in analysis but not summarised
    extra_fields = {
        "debug":
            ["quality_scanline_bitmask", "quality_channel_bitmask",
             "quality_pixel_bitmask"],
        "easy":
            ["quality_scanline_bitmask", "quality_channel_bitmask"],
        }

    hist_range = xarray.Dataset(
        {
        **{field: (("edges",), [170, 320]) for field in ("T_b", "bt")},
        **{field: (("edges",), [0, 50]) for field in 
            ["u_T_b_random", "u_T_b_nonrandom",
             "u_independent", "u_structured"]},
        **{field: (("channel", "edges"),
                   [[0, 200]]*10+[[0, 100]]*2+[[0,10]]*7)
            for field in ("R_e", "u_R_Earth_random", "u_R_Earth_nonrandom")},
        "u_C_Earth": (("edges",), [-4097, 4098]),
        },
        coords={"channel": numpy.arange(1, 20)})
    nbins = 2000

    def __init__(self, *args, satname, **kwargs):
        super().__init__(*args, satname=satname, **kwargs)

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
            fcdr_type="debug"):
        dates = pandas.date_range(start_date, end_date+datetime.timedelta(days=1),
            freq="D")
        if fields is None:
            fields = self.fields
        chandim = "channel" if fcdr_type=="easy" else "calibrated_channel"
        channels = numpy.arange(1, 20)
        #bins = numpy.linspace(self.hist_range, self.nbins)
        summary = xarray.Dataset(
            {
            **{field: 
                  (("date", "ptile", "channel"),
                    numpy.zeros((dates.size-1, self.ptiles.size,
                                 channels.size), dtype="f4")*numpy.nan)
                for field in fields[fcdr_type]},
            **{f"hist_{field:s}":
                  (("date", "bin_index", "channel"),
                    numpy.zeros((dates.size-1, self.nbins+1,
                                 channels.size), dtype="u4"))
                for field in fields[fcdr_type]},
            **{f"bins_{field:s}":
                (("channel", "bin_edges"),
                # numpy.concatenate([[numpy.concatenate([[0],
                # numpy.linspace(170, 320, 100), [1000, 10000]])] for i in
                # range(5)]) 
                    numpy.concatenate([[
                    numpy.concatenate(
                        [[min(
                            self.hist_range.sel(channel=ch,edges=0)[field]-1,
                            0)],
                          numpy.linspace(
                            self.hist_range.sel(
                                channel=ch,
                                edges=0)[field],
                            self.hist_range.sel(
                                channel=ch,
                                edges=1)[field], self.nbins,
                                dtype="f4"),
                        [max(
                            self.hist_range.sel(channel=ch,edges=1)[field]+1,
                            1000)]])]
                    for ch in channels]))
                for field in fields[fcdr_type]},
            },
            coords={"date": dates[:-1], "ptile": self.ptiles, "channel": channels}
        )

        for (sd, ed) in zip(dates[:-1], dates[1:]):
            try:
                ds = self.hirs.read_period(sd, ed,
                    locator_args={"data_version": self.data_version,
                                  "fcdr_type": fcdr_type},
                    fields=fields[fcdr_type]+self.extra_fields[fcdr_type])
                if fcdr_type=="easy" and ds["u_structured"].dims == ():
                    raise DataFileError("See https://github.com/FIDUCEO/FCDR_HIRS/issues/171")
            except DataFileError:
                continue
            if fcdr_type == "debug":
                bad = ((2*ds["u_R_Earth_nonrandom"] > ds["R_e"]) |
                        ((ds["quality_scanline_bitmask"] & 1)!=0) |
                        ((ds["quality_channel_bitmask"] & 1)!=0))
            else: # should be "easy"
                bad = ((2*ds["u_structured"] > ds["bt"]) |
                       ((ds["quality_scanline_bitmask"].astype("uint8") & 1)!=0) |
                       ((ds["quality_channel_bitmask"].astype("uint8") & 1)!=0))
            for field in fields[fcdr_type]:
                if field != "u_C_Earth":
                    # workaround for https://github.com/FIDUCEO/FCDR_HIRS/issues/152
                    ds[field].values[bad.transpose(*ds[field].dims).values] = numpy.nan 
            for field in fields[fcdr_type]:
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
            fcdr_type=fcdr_type))
        of.parent.mkdir(parents=True, exist_ok=True)

        for field in fields[fcdr_type]:
            summary[field].encoding.update({
                "scale_factor": 0.001,
                "_FillValue": numpy.iinfo("int32").min,
                "zlib": True,
                "dtype": "int32",
                "complevel": 4})
            summary["hist_"+field].encoding.update({
                "zlib": True,
                "complevel": 4})
            summary["bins_"+field].encoding.update({
                "zlib": True,
                "complevel": 4,
                "dtype": "int32",
                "_FillValue": numpy.iinfo("int32").min,
                "scale_factor": 0.001})

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
            satlabel = ""
        else:
            sats = [sats]
            satlabel = self.satname + " "
        
        figs = {}
        for channel in range(1, 20):
            figs[channel] = matplotlib.pyplot.subplots(figsize=(20, 4.5*len(fields)),
                nrows=len(fields), ncols=1, squeeze=False)
                
        ranges = xarray.DataArray(
            numpy.full((len(sats), 19, len(fields), 2), numpy.nan, dtype="f4"),
            dims=("satname", "channel", "field", "extremum"),
            coords={"satname": sats, "channel": range(1, 20),
                "field": fields,
                "extremum": ["lo", "hi"]})

        oldsatname = self.satname
        sc = itertools.count()
        for sat in sats:
            if sat != self.satname:
                self.satname = sat
            try:
                summary = self.read_period(start, end,
                    locator_args={"data_version": "v"+self.data_version,
                        "fcdr_type": fcdr_type,
                        "satname": sat},
                    NO_CACHE=True)
                if all(summary.isnull().all()[fields].all().variables.values()):
                    raise DataFileError(f"All data invalid for {sat:s}!")
            except DataFileError:
                continue
            else:
                si = next(sc)
            
            np = len(ptiles)
            if len(sats) == 1:
                pcolors = [f"C{i:d}" for i in
                    range(math.ceil(-np/2), math.ceil(np/2))]
            else:
                pcolors = [f"C{allsats.index(sat)%10:d}"]*np

            for channel in range(1, 20):
                total_title = (f"HIRS {satlabel:s}ch. {channel:d} "
                               f"{start:%Y-%m-%d}--{end:%Y-%m-%d}")
                (f, a_all) = figs[channel]
                for (i, (fld, a)) in enumerate(zip(fields, a_all.ravel())):
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
                # prepare some info for later, with zoomed-in y-axes
                for (fld, a) in zip(fields, a_all.ravel()):
                    lo = scipy.stats.mstats.mquantiles(
                        numpy.ma.masked_invalid(
                            summary[fld].sel(channel=channel).sel(ptile=25).values),
                        prob=0.05,
                        alphap=0, betap=0)
                    hi = scipy.stats.mstats.mquantiles(
                        numpy.ma.masked_invalid(
                            summary[fld].sel(channel=channel).sel(ptile=75).values),
                        prob=.95,
                        alphap=0, betap=0)
                    ranges.loc[{"satname": sat, "channel": channel,
                                "field": fld}].values[...] = [lo, hi]
                if len(fields) > 1:
                    f.suptitle(total_title)
                f.autofmt_xdate()
            summary.close()
            del summary
        for channel in range(1, 20):
            (f, a_all) = figs[channel]
            pyatmlab.graphics.print_or_show(f, None, 
                self.plot_file.format(satname=satlabel, start=start,
                end=end, channel=channel, data_version=self.data_version,
                ptilestr=','.join(str(p) for p in ptiles)))
        # another set with zoomed-in y-axes
        for channel in range(1, 20):
            (f, a_all) = figs[channel]
            for (fld, a) in zip(fields, a_all.ravel()):
                lo = ranges.loc[{"channel": channel, "field": fld, "extremum": "lo"}].min()
                hi = ranges.loc[{"channel": channel, "field": fld, "extremum": "hi"}].max()
                a.set_ylim([lo, hi])
            pyatmlab.graphics.print_or_show(f, None, 
                self.plot_file.format(satname=satlabel, start=start,
                    end=end, channel=channel,
                    data_version=self.data_version,
                    ptilestr=','.join(str(p) for p in ptiles))[:-1] + "_zoom.")
        self.satname = oldsatname


    def plot_period_hists(self, start, end, fcdr_type="easy"):
        (f, a_all) = matplotlib.pyplot.subplots(figsize=(20, 12),
            nrows=2, ncols=2, squeeze=False)
        summary = self.read_period(start, end,
            locator_args={"data_version": "v"+self.data_version,
                "fcdr_type": fcdr_type})
        
        tit = f"HIRS {self.satname:s} {start:%Y-%m-%d}--{end:%Y-%m-%d}"

        if fcdr_type == "easy":
            lab_struc = "structured"
            lab_indy = "independent"
        else:
            lab_struc = "T_b_nonrandom"
            lab_indy = "T_b_random"
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
            a.legend()
            a.set_xlim([0,
                float(summary[f"bins_u_{lab_struc:s}"].sel(channel=chs[0]).isel(bin_edges=idx_hi))])
            a.set_xlabel("Brightness temperature uncertainty [K]")
            a.set_ylabel("Total number of pixels")
        f.suptitle(tit)
        pyatmlab.graphics.print_or_show(f, None, 
            self.plot_hist_file.format(satname=self.satname, start=start,
            end=end, data_version=self.data_version))

def summarise():
    p = parse_cmdline()
    common.set_logger(
        logging.DEBUG if p.verbose else logging.INFO,
        filename=p.log)
#    if p.mode != "summarise":
#        raise NotImplementedError("Only summarising implemented yet")
    summary = FCDRSummary(satname=p.satname, data_version=p.version)
    start = datetime.datetime.strptime(p.from_date, p.datefmt)
    end = datetime.datetime.strptime(p.to_date, p.datefmt)
    if p.mode == "summarise":
        summary.create_summary(start, end, fcdr_type=p.type)
    elif p.mode == "plot":
#        sumdat = summary.plot_period(start, end, 5, fields=["u_C_Earth"],
#            ptiles=[50], pstyles=["-"], fcdr_type=p.type)
        if p.satname != "all":
            # plotting "all" not supported for hists
            summary.plot_period_hists(start, end, p.type)
#            fields=["bt", "u_independent", "u_structured"],
#            fcdr_type="easy")
        summary.plot_period_ptiles(start, end,
            fields=["bt", "u_independent", "u_structured"],
            fcdr_type="easy",
            ptiles=p.ptiles,
            pstyles=p.pstyles.split())
