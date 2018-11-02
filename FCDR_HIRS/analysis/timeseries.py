#!/usr/bin/env python3.5

"""Plot various timeseries for HIRS

Anomalies averaged per orbit.
"""

import argparse

import logging
import datetime
import itertools
import pathlib
import math

import numpy
import matplotlib
#matplotlib.use("Agg")
# Source: http://stackoverflow.com/a/20709149/974555
#if parsed_cmdline.plot_noise_with_other:
#    matplotlib.rc('text', usetex=True)
#    matplotlib.rcParams['text.latex.preamble'] = [
#           r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#           r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
#           #r'\usepackage{helvet}',    # set the normal font here
#           #r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
#           #r'\sansmath',              # <- tricky! -- gotta actually tell tex to use!
#           r'\DeclareSIUnit\count{count}'  # siunitx doesn't know this one
    ]
# this too must be before importing matplotlib.pyplot
#pathlib.Path("/dev/shm/gerrit/cache").mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot
import matplotlib.gridspec
import matplotlib.dates
import matplotlib.ticker

import pandas
from pandas.core.indexes.base import InvalidIndexError
import xarray

#from memory_profiler import profile

import typhon.plots
import typhon.config
#try:
#    typhon.plots.install_mplstyles() # seems to be needed to run in queueing system
#except FileExistsError:
#    pass
#matplotlib.pyplot.style.use("typhon")
matplotlib.pyplot.style.use(typhon.plots.styles("typhon"))
colours = ["blue", "green", "red", "purple", "cyan", "tan", "black", "orange", "brown"]
import scipy.stats

import typhon.math.stats
import typhon.datasets.dataset
from typhon.physics.units.common import ureg

import pyatmlab.stats
#import pyatmlab.config
import pyatmlab.graphics
#import pyatmlab.datasets.tovs
#from pyatmlab.units import ureg

from .. import common
from .. import fcdr

srcfile_temp_iwt = pathlib.Path(typhon.config.conf["main"]["myscratchdir"],
                       "hirs_{sat:s}_{year:d}_temp_iwt.npz")
logger = logging.getLogger(__name__)

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description="Study noise and temperature over time",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--plot_iwt_anomaly", action="store_true",
        #type=bool,
        default=False, help="Plot IWT anomaly for full HIRS period. "
        "This is based on an intermediate file created with "
        "extract_field_from_hirs.py.")

    parser.add_argument("--plot_noise", action="store_true", #type=bool,
        default=False, help="Plot various noise characteristics.")

    parser.add_argument("--plot_noise_with_other", action="store_true", #type=bool,
        default=False, help="Plot various other noise characteristics.")

    parser.add_argument("--plot_noise_map", action="store_true", #type=bool,
        default=False, help="Plot orbit with counts and noise.")

    parser.add_argument("--plot_noise_correlation_timeseries",
        action="store_true", default=False,
        help="Plot noise correlation timeseries")

    parser.add_argument("--from_date", action="store", type=str,
        help="Starting date for plotting period. ")

    parser.add_argument("--to_date", action="store", type=str,
        help="Ending date for plotting period.")
    
    parser.add_argument("--datefmt", action="store", type=str,
        help="Date format for interpreting from and to dates",
        default="%Y-%m-%d")

    parser.add_argument("--temp_fields", action="store", type=str,
        nargs="*", 
        choices=['an_el', 'patch_exp', 'an_scnm', 'fwm', 'scanmotor',
            'iwt', 'sectlscp', 'primtlscp', 'elec', 'baseplate', 'an_rd',
            'an_baseplate', 'ch', 'an_fwm', 'ict', 'an_pch', 'scanmirror',
            'fwh', 'patch_full', 'fsr'],
        default=["iwt", "fwm"],
        help="Temperature fields to show in 3rd subplot. "
        "May not be correct for some satellites and fields.")

    parser.add_argument("--count_fields", action="store", type=str,
        nargs="*",
        choices=["space", "ict", "iwt", "Earth"],
        default=["space", "iwt"],
        help="Count fields to show, including noise levels.  Note that ICT "
             "only exists on HIRS/2.")

    parser.add_argument("--with_gain", dest="include_gain", action="store_true",
        help="Include subplot with gain.  Only applicable "
                           "for plot_noise_with_other.")
    parser.add_argument("--without_gain", dest="include_gain",
        action="store_false")

    parser.add_argument("--with_rself", dest="include_rself", action="store_true",
        help="Include two subplots investigating self-emission.  Only "
             "applicable with plot_noise_with_other.")
    parser.add_argument("--without_rself", dest="include_rself",
        action="store_false")

    parser.add_argument("--with_corr", dest="include_corr",
        action="store", type=str,
        choices=["min_mean", "max_mean", "min_std", "max_std", "above",
                 "choose"],
        nargs="*",
        default=[],
        help="Include time series of channel pairs with extremely "
        "high/low correlations.  Only with --plot_noise_with_other. "
        "'min_mean' means pairs with lowest (negative) correlations; "
        "'max_mean' means pairs with highest correlations; "
        "'min_std' means rather constant pairs; "
        "'max_std' means rather varying pairs; "
        "'above' means N channels 'above' the present one are chosen; "
        "'choose' means user chooses pairs; need to provide -corr_pairs. "
        "Multiple alternatives possible. "
        "--corr_perc controls the rank to select. "
        "To be implemented")

    parser.add_argument("--corr_pairs", action="store",
        type=int, choices=range(1, 20),
        nargs="*", 
        help="For sure with --with_corr only, specify exactly "
             "which channel pairs to plot.  Even numbers.")

    parser.add_argument("--corr_perc", action="store",
        type=float, default=100.0,
        choices=range(0, 101),
        help="Rank (percentile) to select for extreme cases for use "
             "with --with_corr.  To be implemented, now always 100.")

    parser.add_argument("--corr_count", action="store",
        type=int, default=2,
        help="How many correlation cases to plot per selected choice. "
             "Use with --with_corrs.")
    
    parser.add_argument("--corr_calibpos", action="store",
        type=int, default=20,
        help="What calibration position to use for correlation calculations")

    parser.add_argument("--corr_timeres", action="store",
        type=str, default="12H",
        help="What time resolution to use for correlation timeseries. "
             "Valid values at "
             "http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases "
             "or string 'per_cycle' which will calculate one correlation "
             "for each calibration cycle, by using all positions.")

    parser.add_argument("--hiasi_mode", action="store", type=str,
        choices=["perc", "hist"], default="perc",
        help="For HIASI anomalies, show as PERCentiles or as HIST2d")

    parser.add_argument("--sat", action="store", type=str,
        help="Satellite to use.")

    parser.add_argument("--channel", action="store", type=int,
        nargs="+", help="Channels to consider")

    parser.add_argument("--log", action="store", type=str,
        help="Logfile to write to")

    parser.add_argument("--verbose", action="store_true",
        help="Be verbose", default=False)

    parser.add_argument("--store_only", action="store_true",
        help="Only store to NetCDF, do not plot.  Name calculate automatically.", default=False)

    parser.add_argument("--width_factor", action="store", type=float,
        default=1,
        help="Make plot a factor x wider (for noise_with_other only).")

    parser.add_argument("--write_figs", action="store_true",
        help="For plots, write not only .png, but also .pdf and figure "
             "pickle.  This is useful for later opening and editing plots. "
             "However, it can be very slow and time consuming.")

    parser.set_defaults(include_gain=True, include_rself=True)
    
    p = parser.parse_args()
    return p

def get_timeseries_temp_iwt_anomaly(sat, year_start=2005, year_end=2017):
    L = []
    for year in range(year_start, year_end):
        try:
            logger.debug("Reading {:d}".format(year))
            D = numpy.load(str(srcfile_temp_iwt).format(sat=sat, year=year))
            L.extend(D["selection"])
        except FileNotFoundError:
            pass
    logger.info("Processing")
    dts = numpy.array([x[0] for x in L], "M8[s]")
    anomalies = numpy.concatenate(
        [(x[1].mean(0).mean(1) - x[1].mean())[:, numpy.newaxis] for x in L], 1).T
    
    return (dts, anomalies)

def plot_timeseries_temp_iwt_anomaly(sat, nrow=4):
    (dts, anomalies) = get_timeseries_temp_iwt_anomaly(sat)
    dts = dts.astype("M8[s]").astype(datetime.datetime)
    (f, ax_all) = matplotlib.pyplot.subplots(nrow, 1)

    (lo, hi) = scipy.stats.scoreatpercentile(anomalies, [1, 99])
    for (i, ax) in enumerate(ax_all):
        ix_st = i * dts.shape[0]//nrow
        ix_end = min((i+1) * dts.shape[0]//nrow, dts.shape[0]-1)
        logger.info("Plotting part {:d}/{:d}".format(i+1, nrow))
        ax.plot(dts[ix_st:ix_end], anomalies[ix_st:ix_end, :])
        ax.set_ylabel(r"$\Delta$ T [K]")
        ax.locator_params(axis="x", tight=True, nbins=4)
        ax.set_ylim(1.2 * lo, 1.2 * hi)
        ax.grid(axis="both")
    ax_all[-1].set_xlabel("Date")
    f.suptitle("IWT PRT orbit-averaged anomalies, {:s}".format(sat))
    f.subplots_adjust(hspace=0.25)
    pyatmlab.graphics.print_or_show(f, False,
        "timeseries_{:s}_temp_iwp.".format(sat))

def extract_timeseries_per_day_iwt_anomaly_period(sat, start_date, end_date):
    """For satellite, extract timeseries per day

    Approximations:
    - Takes whole orbits per day
    - Does not remove duplicate scanlines
    """
    (dts, anomalies) = get_timeseries_temp_iwt_anomaly(sat,
        start_date.year, end_date.year+1)
    bins = list(itertools.takewhile(lambda x: x<=end_date,
                (start_date + datetime.timedelta(days=i) for i in itertools.count())))
    bins = numpy.array(bins, "M8[s]")

    # Binning doesn't work with a time-axis.  Convert to float, but be
    # sure that float relates to the same time-format first
    binned = pyatmlab.stats.bin(dts.astype("M8[s]").astype("f8"),
                                anomalies,
                                bins.astype("M8[s]").astype("f8"))

    anom_per_day = numpy.array([b.mean(0) if b.size>0 else [numpy.nan]*5 for b in binned])

    # Due to the way binning works, there is an off-by-one error.
    # so anything on 2005-10-10 is binned in the bin that says
    # 2005-10-11.  Sorry.
    Y = list(zip(bins, anom_per_day[1:])) # convert to list as I need to
                                          # loop through twice

    X = numpy.zeros(shape=(anom_per_day.shape[0]-1,), dtype=[("date", "M8[s]"), ("anomalies", "f8", (5,))])
    X["date"] = [y[0] for y in Y]
    X["anomalies"] = [y[1] for y in Y]
    return X

def write_timeseries_per_day_iwt_anomaly_period(sat, start_date, end_date):
    X = extract_timeseries_per_day_iwt_anomaly_period(sat, start_date, end_date)
    dest = pathlib.Path(typhon.config.conf["main"]["myscratchdir"],
        "hirs_iwt_anom_{:s}_{:%Y%m%d}-{:%Y%m%d}".format(sat, start_date, end_date))
    logger.info("Writing {!s}".format(dest))
    with dest.open("wt", encoding="ascii") as fp:
        fp.writelines([("{:%Y-%m-%d}" + 5*" {:.5f}" + "\n").format(
                x["date"].astype("M8[s]").astype(datetime.datetime), *x["anomalies"])
                    for x in X])
            

def plot_timeseries_temp_iwt_anomaly_all_sats():
    for sat in {"noaa18", "noaa19", "metopa", "metopb"}:
        logger.info("Plotting {:s}".format(sat))
        plot_timeseries_temp_iwt_anomaly(sat)
        

class NoiseAnalyser:
    ptiles = (5, 25, 50, 75, 95)
    linestyles = (":", "--", "-", "--", ":")
    fte = 0.67
    fhs = 0.73
#@profile

    def __init__(self, start_date, end_date, satname, temp_fields={"iwt",
                        "fwh", "fwm"}, writefig=False):
        self.hirs = fcdr.which_hirs_fcdr(satname)
        self.satname = satname
        hrsargs=dict(
                fields=["hrs_scnlin", self.hirs.scantype_fieldname, "time",
                        "counts", "calcof_sorted", "radiance",
                        "bt"] + 
                        ["radiance_fid_naive"] +
                       ["temp_{:s}".format(f) for f in
                       set(temp_fields) | {"iwt"}],
                locator_args=dict(satname=self.satname),
                orbit_filters=
                    [typhon.datasets.filters.HIRSBestLineFilter(self.hirs),
                     typhon.datasets.filters.TimeMaskFilter(self.hirs),
                     typhon.datasets.filters.HIRSTimeSequenceDuplicateFilter(),
                     typhon.datasets.filters.HIRSFlagger(self.hirs, max_flagged=0.9),
                     typhon.datasets.filters.HIRSCalibCountFilter(self.hirs, self.hirs.filter_calibcounts),
                     ],
                excs=(typhon.datasets.dataset.DataFileError, typhon.datasets.filters.FilterError, InvalidIndexError))
        # those need to be read before combining with HIASI, because
        # afterward, I lose the calibration rounds.  But doing it after
        # read a full month (or more) of data takes too much RAM as I will
        # need to copy the entire period; therefore, add it on-the-fly
        self.hirs.maxsize = 200*2**30 # tolerate 200 GB
        Mhrsall = self.hirs.read_period(start_date, end_date,
            pseudo_fields=
                {"tsc": self.hirs.calc_time_since_last_calib,
                 "lsc": self.hirs.count_lines_since_last_calib},
            NO_CACHE=True, **hrsargs)
        self.Mhrsall = Mhrsall
        self.hiasi = typhon.datasets.tovs.HIASI()
        # Split IASI reading in smaller parts to save memory
        dt = start_date
        found_iasi = False
        Mhrscmb = []
        if satname == "metopa":
            Lhiasi = []
            while dt < end_date: 
                step = datetime.timedelta(days=1)
                try:
                    Miasi = self.hiasi.read_period(dt, dt+step,
                        NO_CACHE=True,
                        enforce_no_duplicates=False)
                except typhon.datasets.dataset.DataFileError:
                    logger.info("No IASI found in "
                        "[{:%Y-%m-%d %H:%M}-{:%Y-%m-%d %H:%M}]".format(
                            dt, dt+step))
                    #self.Miasi = None
                    pass
                else:
                    found_iasi = True
                    logger.info("Combining HIASI with HIRS…")
                    Mhrscmb.append(
                            self.hiasi.combine(Miasi, self.hirs, Mhrsall,
                            other_args=hrsargs, trans={"time": "time"},
                            timetol=numpy.timedelta64(3,
                            's')))
                    Lhiasi.append(self.calc_hiasi(Miasi["ref_radiance"]))
                    del Miasi
                finally:
                    dt += step
        else:
            found_iasi = False

            #self.Miasi = Miasi
        if found_iasi:
            self.Mhrscmb = numpy.ma.concatenate(Mhrscmb)
            self.Lhiasi = ureg.Quantity(
                numpy.ma.masked_equal(numpy.concatenate(Lhiasi, 1).T, 0), 
                typhon.physics.units.common.radiance_units["ir"])
        else:
            self.Mhrscmb = None
            self.Lhiasi = None
            self.tsc = None

        self.start_date = start_date
        self.end_date = end_date

        self.writefig = writefig

    def calc_hiasi(self, rad):
        """Calculate IASI-simulated HIRS
        """
        freq = ureg.Quantity(numpy.loadtxt(self.hiasi.freqfile), ureg.Hz)
        #rad = self.Miasi["ref_radiance"]
        # not sure why it's sometimes zero, but it is
        L = ureg.Quantity(
            numpy.ma.masked_where(rad==0, rad),
            typhon.physics.units.common.radiance_units["ir"])
        return numpy.vstack([
            srf.integrate_radiances(freq, L) for srf in self.hirs.srfs])

    @staticmethod
    def set_ax_loc(a, t, N=1):
        """Set N≠1 if you spread a time axis over multiple rows
        """
        Δt = t[-1] - t[0]
        Δdays = Δt.days + Δt.seconds/86444
        if Δt < datetime.timedelta(hours=(N*2)): # <2 hours
            a.xaxis.set_major_locator(
                matplotlib.dates.MinuteLocator(
                    byminute=range(0, 60, 10)))
            a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        elif Δt < datetime.timedelta(days=(N*2)): # <2 days
            a.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=
                    math.ceil((24*Δdays)//(N*10)+1)))
            a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
            # FIXME: minor locator
        else: # >2 days
            a.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=
                    int(Δdays//(N*10))+1))
            a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %-d"))
    
            if Δt < datetime.timedelta(days=(N*30)):
                a.xaxis.set_minor_locator(matplotlib.dates.HourLocator(
                    interval=[1, 2, 3, 4, 6, 8, 12][int(Δdays//(N*5))]))
            else:
                a.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
            

    def plot_noise(self, nrow=4):
        M = self.Mhrsall
        start_date = self.start_date
        end_date = self.end_date
        if "scantype" in M.dtype.names:
            # HIRS/2
            views_iwt = H["scantype"] = self.hirs.typ_iwt
            views_ict = H["scantype"] = self.hirs.typ_ict
            views_space = H["scantype"] = self.hirs.typ_space
        else:
            # HIRS/3, HIRS/4
            views_iwt = M["hrs_scntyp"] == self.hirs.typ_iwt
            views_space = M["hrs_scntyp"] == self.hirs.typ_space
        (f, ax_all) = matplotlib.pyplot.subplots(nrow, 1)
        t = M[views_space]["time"].astype("M8[s]").astype(datetime.datetime)
        # standard deviation...
        # want to have version with regular and with std.dev.
        #y_space = M[views_space]["counts"][:, 8:, :].std(1)
        y_space = typhon.math.stats.adev(M[views_space]["counts"][:, 8:, :], 1)
        #y_iwt = M[views_iwt]["counts"][:, 8:, :].std(1)
        y_iwt = typhon.math.stats.adev(M[views_iwt]["counts"][:, 8:, :], 1)
        if "scantype" in M.dtype.names:
            #y_ict = M[views_ict]["counts"][:, 8:, :].std(1)
            y_ict = typhon.math.stats.adev(M[views_ict]["counts"][:, 8:, :], 1)
        for (i, a) in enumerate(ax_all):
            ix_st = i * t.shape[0]//nrow
            ix_end = min((i+1) * t.shape[0]//nrow, t.shape[0]-1)
            for k in (1, 2, 8, 19):
                a.plot_date(t[ix_st:ix_end], y_space[ix_st:ix_end, k-1], '.',
                    label="Ch. {:d}".format(k),
                    markersize=5)
            self.set_ax_loc(a, t, N=nrow)
            a.grid(axis="both")
            a.autoscale_view()
            a.set_xlabel("Time")
            a.set_ylabel("Space counts Allan deviation")
            if i == 0:
                a.legend()
        f.suptitle("Space counts Allan deviation {:s} HIRS {:%Y-%m-%d}--{:%Y-%m-%d}".format(
            self.satname, t[0], t[-1]))
        f.subplots_adjust(hspace=0.25)
        pyatmlab.graphics.print_or_show(f, False,
            "hirs_{:s}_space_counts_adev_{:%Y%m%d}-{:%Y%m%d}.".format(
                self.satname, t[0], t[-1])
                + "" if self.writefig else "png")

    def get_gain(self, M, ch):
        (t_slope, _, slope, _) = self.hirs.calculate_offset_and_slope(M,
                ch, accept_nan_for_nan=True)
        # most of this module was written before the migration to xarray;
        # migrate back for now for the sake of legacy code
        t_slope = numpy.ma.MaskedArray(t_slope.values, mask=numpy.zeros_like(t_slope.values))
        slope = ureg.Quantity(numpy.ma.masked_invalid(slope.values), slope.attrs["units"])
        slope = slope.to(ureg.mW/(ureg.m**2 * ureg.sr *
            1/ureg.cm * ureg.counts), "radiance")
        u_slope = self.hirs.calc_uslope(M, ch)
        gain = 1/slope
        u_gain = (numpy.sqrt((-1/slope**2)**2 
                  * (u_slope.to(slope.u, "radiance"))**2))
        med_gain = ureg.Quantity(
            numpy.ma.median(gain.m[:, :], 1),
            gain.u)
        if numpy.isscalar(med_gain.mask):
            med_gain.mask = (numpy.ones if med_gain.mask
                                      else numpy.zeros)(
                shape=med_gain.shape, dtype="?")
        # http://physics.stackexchange.com/a/292884/6319
        u_med_gain = u_gain.mean(1) * numpy.sqrt(numpy.pi*(2*48+1)/(4*48))

        return (t_slope, med_gain, u_med_gain)


    #@profile
    def plot_noise_with_other(self, 
            ch,
            all_tp=["space", "iwt"], temperatures=["iwt"],
            include_gain=True,
            include_rself=True,
            include_corr=(),
            corr_info={},
            hiasi_mode="perc",
            width_factor=1):
        logger.info("Channel {:d}".format(ch))
        M = self.Mhrsall
#        ch = self.ch
        start_date = self.start_date
        end_date = self.end_date
        # need SRF for calculating gain
#        hconf = pyatmlab.config.conf["hirs"]
#        (centres, srfs) = pyatmlab.io.read_arts_srf(
#            hconf["srf_backend_f"].format(sat=self.satname.upper().replace("NOAA0","NOAA")),
#            hconf["srf_backend_response"].format(sat=self.satname.upper().replace("NOAA0","NOAA")))
#        srf = pyatmlab.physics.SRF(*srfs[ch-1])

        # cycle manually as I plot many at once
        styles = list(matplotlib.pyplot.style.library["typhon"]["axes.prop_cycle"])
        N = (2*(len(all_tp)>0) + (len(temperatures)>0) + 2*include_gain +
             3*include_rself + (len(include_corr)>0)*len(all_tp))
        #k = int(numpy.ceil(len(temperatures)/2)*2)
        Ntemps = len(temperatures)
        fact = 16*width_factor
        k = int((max(Ntemps,1)+(2/fact))*fact)
        rshift = (1-(1-self.fhs)/width_factor)-self.fhs # compensate for wider
        lshift = rshift + (self.fhs-self.fte)-(self.fhs-self.fte)/width_factor
        self.ifte = int((self.fte+lshift)*k)
        self.ifhs = int((self.fhs+rshift)*k)
        self.gridspec = matplotlib.gridspec.GridSpec(N, k)
        self.fig = matplotlib.pyplot.figure(figsize=(18*width_factor, 3*N))
        #(f, ax) = matplotlib.pyplot.subplots(N, 1, figsize=(16, 3*N))
        #itax = iter(ax)
        logger.info("Plotting calibration counts + noise")
        self.counter = itertools.count()
        a_cc = ah_cc = a_ccn = ah_ccn = None
        if len(all_tp) > 0:
            (a_cc, ah_cc, a_ccn, ah_ccn) = self.plot_calib_counts_noise(M=M, ch=ch, all_tp=all_tp, styles=styles)

        if include_gain or include_rself:
            (t_slope, med_gain, u_med_gain) = self.get_gain(M, ch)

        a_gain = ah_gain = None
        if include_gain:
            (a_gain, ah_gain) = self.plot_gain(t_slope=t_slope, med_gain=med_gain,
                u_med_gain=u_med_gain)

        a_corr = []
        if len(include_corr)>0:
            for typ in all_tp:
                a_corr.append(self.plot_corr(ch, typ,
                    calibpos=corr_info.get("calibpos", 20),
                    timeres=corr_info.get("timeres", "3H"),
                    N=corr_info.get("count", 2),
                    corr_types=include_corr))
                
        a_temp = ah_temp = None
        if len(temperatures) > 0:
            (ax2lims, at, ath) = self.plot_temperatures(M=M, temperatures=temperatures)

        # make correlation axlimits consistent with others
        for ac in a_corr:
            for aref in (a_cc, a_ccn, a_gain, a_temp):
                if aref:
                    ac.set_xlim(aref.get_xlim())
                    break

        if include_rself:
            if not temperatures:
                raise RuntimeError("Due to implementation limitation, "
                    "can only plot Rself if also plotting a temperature")
            allcb = self.plot_rself(
                M=M, t_slope=t_slope, med_gain=med_gain, ch=ch,
                temperatures=temperatures, k=k,
                ax2lims=ax2lims, styles=styles,
                hiasi_mode=hiasi_mode,
                include_gain=include_gain)
            # some self-emission characteristics

        logger.info("Finalising")

        if include_rself:
            for cb in allcb:
                cb.set_label("No.")
                cb.ax.yaxis.set_major_locator(
                    matplotlib.ticker.MaxNLocator(nbins=5, prune="both"))
                cb.update_ticks()
                    
        t = M["time"].astype("M8[s]").astype(datetime.datetime)
        for a in self.fig.axes:
            if not "hist" in a.get_title():
                a.autoscale_view()
            a.grid("on")
            if isinstance(a.xaxis.major.formatter, matplotlib.dates.AutoDateFormatter):
                self.set_ax_loc(a, t, N=1)
        self.fig.suptitle("Characteristics for {:s} HIRS ch. {:d}, "
                   "{:%Y-%m-%d}--{:%Y-%m-%d}".format(
                        self.satname, ch, start_date, end_date),
                y=1.05,
                fontsize=26)
        self.fig.subplots_adjust(hspace=0.5, top=0.95)
        logger.info("Writing out")
        # Write only PNG, the rest is too slow / memory-intensive
        # For some reason, sometimes it still fails to use the LaTeX
        # cache.  Make sure we create it /again/ ?!
        pathlib.Path("/dev/shm/gerrit/cache").mkdir(parents=True, exist_ok=True)
        pyatmlab.graphics.print_or_show(self.fig, False,
            "hirs_noise/{self.satname:s}_{tb:%Y}/ch{ch:d}/disect_{self.satname:s}_hrs_ch{ch:d}_{alltyp:s}_{alltemp:s}_{tb:%Y%m%d%H%M}-{te:%Y%m%d%H%M}{corrinfo:s}.".format(
                self=self, ch=ch, alltyp='_'.join(all_tp),
                alltemp='_'.join(temperatures), tb=t[0], te=t[-1],
                corrinfo=(f"_corr_{corr_info.get('count', 2):d}"
                          f"_{corr_info.get('timeres', '3H'):s}") if include_corr else "")
                    + "" if self.writefig else "png")

    def get_calib_counts_noise(self, M, ch, all_tp):
        D = {}
        for (i, tp) in enumerate(all_tp): 
            views = M[self.hirs.scantype_fieldname
                        ] == getattr(self.hirs, "typ_"+tp)
            t = M[views]["time"].astype("M8[s]").astype(datetime.datetime)
            C = M[views]["counts"][:, 8:, ch-1]
            adv = typhon.math.stats.adev(C, 1)
            D[tp] = (t, C, adv)
        return D

    def plot_calib_counts_noise(self, M, ch, all_tp, styles):
        """Add two rows with calibration counts + noise timeseries
        """
        C = next(self.counter)
        a0 = self.fig.add_subplot(self.gridspec[C, :self.ifte])
        a0h = self.fig.add_subplot(self.gridspec[C, self.ifhs:])
        C = next(self.counter)
        a1 = self.fig.add_subplot(self.gridspec[C, :self.ifte])
        a1h = self.fig.add_subplot(self.gridspec[C, self.ifhs:])
        L0 = []
        L1 = []
        D = self.get_calib_counts_noise(M, ch, all_tp)
        success = False
        for (i, tp) in enumerate(all_tp):
#            views = M[self.hirs.scantype_fieldname
#                        ] == getattr(self.hirs, "typ_"+tp)
#
#            t = M[views]["time"].astype(datetime.datetime)
#            x = M[views]["counts"][:, 8:, ch-1]
            (t, x, adv) = D[tp]
            nok = (~x.mask).any(1).sum()
            if nok < 3:
                logger.warning("Found only {:d} valid timestamps with "
                    "{:s} counts, not plotting".format(nok, tp))
                continue
            success = True
            L0.append(a0.plot_date(t, x, marker='.', markersize=5,
                    color=colours[i]))
            a0h.hist(x.ravel()[~x.ravel().mask], bins=40, normed=False, histtype="step",
                color=colours[i])
            #ax[1].plot_date(t, x.std(1), '.', color="black")
            adv = typhon.math.stats.adev(x, 1)
            L1.append(a1.plot_date(t, adv,
                            linestyle="none", marker=".",
                            color=colours[i],
                            alpha=0.5,
                            markersize=5))
            a1h.hist(adv[~adv.mask], bins=40, normed=False, histtype="step",
                color=colours[i])
        a0.set_xlabel("Date / time")
        a0.set_ylabel("Counts")
        a0h.set_xlabel(a0.get_ylabel().replace("\n", " "))
        a0h.set_ylabel("Number")
        a0.set_title("Calibration counts over time for "
            + "and".join(all_tp) + " views")
        a0h.set_title("Calib. counts hist.")
        a0h.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=6))
        a1.set_title("Calibration noise (Allan deviation) for "
            + "and".join(all_tp) + " views")
        a1h.set_title("Calib. noise hist.")
        a1.set_xlabel("Date / time")
        a1.set_ylabel("Allan deviation\n[counts]")
        a1h.set_xlabel(a1.get_ylabel().replace("\n", " "))
        a1h.set_ylabel("Number")
        if success and len(all_tp)>1:
            a0h.legend([x[0] for x in L0], all_tp, loc="upper left",
                bbox_to_anchor=(1.0, 1.0))
            a1h.legend([x[0] for x in L1], all_tp, loc="upper left",
                bbox_to_anchor=(1.0, 1.0))

        return (a0, a0h, a1, a1h)

    def plot_gain(self, t_slope, med_gain, u_med_gain):
        """Add gain time series plot + hist to next row of figure
        """
        C = next(self.counter)
        a = self.fig.add_subplot(self.gridspec[C, :self.ifte])
        ah = self.fig.add_subplot(self.gridspec[C, self.ifhs:])
        logger.info("Plotting gain")
        if ((~med_gain.mask).sum()) < 3:
            logger.warning("Not enough valid gain values found, skipping")
            return (None, None)
#        a.plot_date(t_slope.astype(datetime.datetime),
        a.xaxis_date()
        # keeping masked-array as-is leads to either 'UserWarning:
        # Warning: converting a masked element to nan' or to
        # 'ValueError: setting an array element with a sequence', see also
        # https://github.com/numpy/numpy/issues/8461
        OK = (~t_slope.mask)&(~med_gain.mask)
        a.errorbar(t_slope.astype("M8[s]").astype(datetime.datetime).data[OK],
                    med_gain.m.data[OK], 
                    xerr=None,
                    yerr=u_med_gain.m.data[OK],
                    fmt='.',
                        color="black",
                        markersize=5)
        a.set_xlabel("Date / time")
        a.set_ylabel("Gain\n" + "[{:Lx}]".format(med_gain.u))
        valid = (~t_slope.mask) & (~med_gain.mask)
        a.set_ylim(
            scipy.stats.scoreatpercentile(med_gain[valid]-u_med_gain[valid], 1),
            scipy.stats.scoreatpercentile(med_gain[valid]+u_med_gain[valid], 99),
            )
        a.set_title("Gain development over time")

        # Due to https://github.com/numpy/numpy/issues/8123 must
        # convert range to same type as med_gain
        ah.hist(med_gain, bins=40, normed=False, histtype="bar",
            color="black",
            range=numpy.asarray(a.get_ylim(), med_gain.dtype))
        ah.set_xlabel(a.get_ylabel().replace("\n", " "))
        ah.set_ylabel("Number")
        ah.set_title("Gain hist.")
        ah.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=6))
        return (a, ah)
    
    def plot_corr(self, ch, typ, calibpos, timeres, N, corr_types):
        """Add correlation timeseries plot to next row of figure
        """
        C = next(self.counter)
        a = self.fig.add_subplot(self.gridspec[C, :self.ifte])
        logger.info("Plotting correlation timeseries ({:s})".format(typ))
        correlations = self.get_correlations(timeres, typ, calibpos).sel(cha=ch)
        # in case I need to select based on max_std or min_std I need to
        # calculate all correlations anyway
#        mcorr = correlations.mean(dim="time")
#        stdcorr = correlations.std(dim="time")
        chpairs = set()
        if any(x in corr_types for x in {"min_std", "max_std", "min_mean",
                "max_mean"}):
        # gather min_mean, max_mean, min_std, max_std
            for extr_name in ("min", "max"):
                for reduc_name in ("mean", "std"):
                    rcorr = getattr(correlations, reduc_name)(dim="time")
                    rcorr = rcorr.sel(chb=numpy.setdiff1d(rcorr.chb, rcorr.cha))
                    extremum = getattr(rcorr, extr_name)()
                    ch_sec = rcorr.chb[rcorr.argmax()]
                    asrt = rcorr.argsort()
                    if extr_name == "max":
                        asrt = asrt[::-1]
                    for ch_sec in rcorr.chb[asrt[:N]]:
                        t = (ch, int(ch_sec))
                        if not t in chpairs:
                            chpairs.add(t)
        if "above" in corr_types:
            chpairs |= {(ch, chb+1) for chb in
                            numpy.arange(ch, ch+N) % 19}
        # plot all those pairs
        success = False
        for (cha, chb) in sorted(chpairs):
            if correlations.sel(chb=chb).shape[0] < 3:
                logger.warning("Found only {:d} valid values for "
                    "({:d}, {:d}), skipping".format(
                        correlations.sel(chb=chb).shape[0], cha, chb))
                continue
            success = True
            a.plot_date(correlations["time"].values.astype("M8[s]").astype(object),
                correlations.sel(chb=chb),
                markersize=1,
                linestyle="-",
                linewidth=1,
                label="ch. {:d}".format(chb))
        if success:
            a.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        a.set_title("{:s} error correlations".format(typ))
        a.set_ylabel("error correlation")
        a.set_xlabel("Date / time")
        return a
        #correlations.sel(cha=ch_a, chb=ch_b)
        # FIXME: histogram
        #ah = self.fig.add_subplot(self.gridspec[C, self.ifhs:])

    def plot_temperatures(self, M, temperatures):
        """Add temperatures timeseries plot + hist to next row of figure
        """
        C = next(self.counter)
        a = self.fig.add_subplot(self.gridspec[C, :self.ifte])
        ah = self.fig.add_subplot(self.gridspec[C, self.ifhs:])
        logger.info("Plotting temperatures")
        ax2lims = None
        t = M["time"].astype("M8[s]").astype(datetime.datetime)
        for (i, (tmpfld, xt)) in enumerate(self.loop_through_temps(M, temperatures)):
            a.plot_date(t, xt, '.', label=tmpfld.replace("_", ""),
                        color=colours[i],
                        markersize=5)
            lims = scipy.stats.scoreatpercentile(xt, [1, 99])
            if ax2lims is None:
                ax2lims = lims
            else:
                ax2lims[0] = min(lims[0], ax2lims[0])
                ax2lims[1] = max(lims[1], ax2lims[1])
        a.set_title("Temperature of various components")
        a.set_xlabel("Date / time")
        a.set_ylabel("Temperature\n[K]")
        a.set_ylim(ax2lims)
        for (i, (tmpfld, xt)) in enumerate(self.loop_through_temps(M, temperatures)):
            ah.hist(xt, bins=40, normed=False, histtype="step",
                label=tmpfld.replace("_", ""), range=ax2lims,
                color=colours[i], linestyle="solid")
        ah.legend(loc="upper left", bbox_to_anchor=(1, 1.0))
        ah.set_title("Temp. hist")
        ah.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=6))
        ah.set_xlabel(a.get_ylabel().replace("\n", " "))
        ah.set_ylabel("Number")
        return ax2lims, a, ah

    def plot_rself(self, M, t_slope, med_gain, ch, temperatures, k,
                    ax2lims, styles, hiasi_mode, include_gain):
        """Add self-emission characteristics to next rows of figure

        Adds up to five rows:
        - space counts vs. temperatures and ΔRspace vs Δtemperatures
        - difference with IASI, if available, per position in calib. cycle
        - hexbin plots of space counts vs. temperatures
        - hexbin plots of allan deviation vs. temperatures (*)
        - hexbin plots of gain vs. temperatures (*)

        The ones marked with (*) should probably be moved to another
        method.
        """
        C = next(self.counter)
        a = self.fig.add_subplot(self.gridspec[C, 0:int(0.45*k)])
        view_space = M[self.hirs.scantype_fieldname] == self.hirs.typ_space
        view_iwct = M[self.hirs.scantype_fieldname] == self.hirs.typ_iwt
        dsi = self.hirs.dist_space_iwct
        space_and_iwct = (view_space[:-dsi] & view_iwct[dsi:])
        y = numpy.median(M[:-dsi][space_and_iwct]["counts"][:, 8:, ch-1], 1)
        for (i, (tmpfld, xt)) in enumerate(self.loop_through_temps(M[:-dsi][space_and_iwct], temperatures)):
            a.plot(xt, y, '.', label=tmpfld.replace("_", ""),
                   markersize=5, color=colours[i])
        a.set_ylabel("Space counts")
        a.set_xlabel("Temperature [K]")
        a.set_title("Self-emission")
        a.set_xlim(ax2lims)

        a = self.fig.add_subplot(self.gridspec[C, int(0.55*k):])
        # Δself emission in radiance space
        # `y` is on every space view, `med_gain` is only on space
        # views followed by earth views...
        ΔRself = ureg.Quantity(numpy.diff(y), ureg.count) / med_gain[1:]
        if (~ΔRself.mask).sum() < 3:
            logger.error("Found only {:d} valid values for ΔRself, not "
                          "plotting".format((~ΔRself.mask).sum()))
            return []
        # plot ΔR(ΔT) for those temperatures that change considerably
        # (this limitation prevents clutter around ΔT=0)
        maxptp = 0
        for (tmpfld, xt) in self.loop_through_temps(
                M[:-dsi][space_and_iwct], temperatures):
            thisptp = numpy.diff(xt).ptp()
            if thisptp > maxptp:
                maxptp = thisptp
        for (i, (tmpfld, xt)) in enumerate(self.loop_through_temps(
                M[:-dsi][space_and_iwct], temperatures)):
            ΔT = numpy.diff(xt)
            if ΔT.ptp() < .3 * maxptp:
                continue
            #a.plot(ΔT, ΔRself, '.', label=tmpfld)
            typhon.plots.plot_distribution_as_percentiles(a,
                ΔT, ΔRself,
                bins=numpy.linspace(*scipy.stats.scoreatpercentile(ΔT.ravel(), [1, 99]),
                    40),
                color=colours[i],
                label=" " if math.isclose(ΔT.ptp(), maxptp) else None,
                ptiles=self.ptiles,
                linestyles=self.linestyles)
        a.set_xlabel(r"$\Delta$ Temperature"" [K]")
        a.set_ylabel(r"$\Delta$ Rspace btw. calibs"+
            "\n[{:Lx}]".format(ΔRself.u).replace(" ", ""))
        a.set_title("Self-emission evolution")
        a.set_ylim(scipy.stats.scoreatpercentile(
            ΔRself[~ΔRself.mask], [1, 99]))
        a.legend(loc="upper left", bbox_to_anchor=(1, 1.0))

        allcb = []
        if self.Lhiasi is not None:
            C = next(self.counter)

            Lhrs = ureg.Quantity(self.Mhrscmb["radiance_fid_naive"][..., ch-1],
                typhon.physics.units.common.radiance_units["si"])
            Lhrs = Lhrs.to(self.Lhiasi.u, "radiance")
            dL = Lhrs - self.Lhiasi[:, ch-1]
            dL.mask[self.Mhrscmb["lsc"]==0] = True
            ΔRselfint, = self.hirs.interpolate_between_calibs(
                self.Mhrscmb["time"], t_slope[1:], ΔRself,
                kind="nearest")
            D = dict(
                heating = ΔRselfint.m > scipy.stats.scoreatpercentile(ΔRself, 90),
                all = numpy.ones(shape=ΔRselfint.shape, dtype="?"),
                cooling = ΔRselfint.m < scipy.stats.scoreatpercentile(ΔRself, 10))
            if hiasi_mode == "perc":
                # plot ΔL per temperatures
                a = self.fig.add_subplot(self.gridspec[C, 0:int(.45*k)])
#                Lhrs = self.hirs.calculate_radiance(self.M, srf, ch)
                for (i, (tmpfld, xt)) in enumerate(self.loop_through_temps(self.Mhrscmb, temperatures)):
                    typhon.plots.plot_distribution_as_percentiles(a, 
                        xt, dL, nbins=40, color=colours[i],
                        label=tmpfld.replace("_", "") if i==0 else None,
                        ptiles=self.ptiles,
                        linestyles=self.linestyles) 
                    #a.plot(xt, dL[:, ch-1], '.', label=tmpfld)
                a.set_ylabel(r"$\Delta$ R HIRS-HIASI" +
                    "\n[{:Lx}]".format(dL.u).replace(" ", ""))
                a.set_xlabel("Temperature [K]")
                a.set_title("HIASI-anomaly per temperature")

                # plot ΔL per time since last calib
                a = self.fig.add_subplot(self.gridspec[C, int(.55*k):])
                for (i, (lab, selec)) in enumerate(sorted(D.items())):
                    typhon.plots.plot_distribution_as_percentiles(a,
                        #self.Mhrscmb["tsc"][selec], dL[selec],
                        self.Mhrscmb["lsc"][selec], dL[selec],
                        #bins=numpy.linspace(0, 256, 38),
                        bins=numpy.linspace(0, 40, 10, dtype="uint8"),
                        color=colours[i],
                        label=lab, ptiles=self.ptiles[1:-1],
                        linestyles=self.linestyles[1:-1])

                #a.set_xlabel("Time since calibration [s]")
                a.set_xlabel("Scanlines since calibration")
                a.set_ylabel(r"$\Delta$ R HIRS-HIASI"+
                    "\n[{:Lx}]".format(dL.u).replace(" ", ""))
                a.set_title("HIASI-anomaly per calibration position")
                a.legend(loc="upper left", bbox_to_anchor=(1, 1))
            elif hiasi_mode=="hist":
                dLok = numpy.isfinite(dL.m) & (~dL.m.mask)
                xbins = numpy.linspace(0, 40, 10, dtype="uint8")
                ybins = numpy.linspace(
                             *scipy.stats.scoreatpercentile(
                                dL.m[dLok], [1, 99]), 20)
                for (i, (lab, selec)) in enumerate(sorted(D.items())):
                    # do just per calib. position here as for those we
                    # want just three;  for the temperatures, add an
                    # extra layer with the space counts / noise
                    # hist2ds later
                    a = self.fig.add_subplot(self.gridspec[C, (i*(k-2)//3):((i+1)*(k-2)//3)])
                    x = self.Mhrscmb["lsc"][selec]
                    ydL = dL[selec].m
                    ok = (~x.mask) & (~ydL.mask)
                    x = x[ok] # scoreatpercentile dislikes maskedarrays
                    ydL = ydL[ok]
                    im = a.hexbin(x, ydL,
                        extent=[0, 40,
                                *scipy.stats.scoreatpercentile(ydL, [1, 99])],
                        gridsize=20,
                        mincnt=1,
                        cmap="viridis")
#                    (_, _, _, im) = a.hist2d(x, ydL,
#                        bins=[xbins, ybins],
#                        range=[[0, 40],
#                               scipy.stats.scoreatpercentile(ydL, [1, 99])],
#                        cmin=1)
                    typhon.plots.plot_distribution_as_percentiles(
                        a, x, ydL,
                        bins=numpy.linspace(0, 40, 10, dtype="uint8"),
                        color="tan", ptiles=self.ptiles,
                        linestyles=self.linestyles,
                        linewidth=1.5)
                    a.set_title("Self-emis. evol. {:s}".format(lab))
                    if i==0:
                        a.set_ylabel(r"$\Delta$ R HIRS-HIASI" +
                            "\n[{:Lx}]".format(dL.u))
                    else:
                        a.get_yaxis().set_ticklabels([])
                    a.set_xlabel("Scanlines since calibration")
                    for ax in (a.xaxis, a.yaxis):
                        a.xaxis.set_major_locator(
                            matplotlib.ticker.MaxNLocator(nbins=4,
                            prune="both"))
                ac = self.fig.add_subplot(self.gridspec[C, k-2:])
                cb = self.fig.colorbar(im, cax=ac)
                allcb.append(cb)

        C1 = next(self.counter)
        C2 = next(self.counter)
        if self.Lhiasi is not None and hiasi_mode == "hist":
            C3 = next(self.counter)
        # earlier adev was calculated for space counts or iwct counts,
        # but now it must match the temperature axis, so only be for
        # space counts followed by iwct counts, which may differ at
        # the beginning or end
        adv = typhon.math.stats.adev(M[:-dsi][space_and_iwct]["counts"][:, 8:, ch-1], 1)
        Ntemps = len(temperatures)
        ha_ax = {}
        allrng = {}
        for (i, (tmpfld, xt)) in enumerate(self.loop_through_temps(M[:-dsi][space_and_iwct], temperatures)):
            a_h2d = []
            a1 = self.fig.add_subplot(self.gridspec[C1, (i*(k-2)//Ntemps):((i+1)*(k-2)//Ntemps)])
            a_h2d.append(a1)
            # give next one same size, but directly below
            pos = a1.get_position()
            a2 = self.fig.add_axes([pos.x0, pos.y0-pos.height, pos.width, pos.height])
            a_h2d.append(a2)
            if self.Lhiasi is not None and hiasi_mode == "hist":
                pos = a2.get_position()
                a3 = self.fig.add_axes([pos.x0, pos.y0-pos.height, pos.width, pos.height])
                a_h2d.append(a3)

                # add temperature, but needs different temperature loop as
                # it should have all, not only space views!
                ha_ax[tmpfld] = a1

            if include_gain:
                pos = a_h2d[-1].get_position()
                agn = self.fig.add_axes([pos.x0, pos.y0-pos.height, pos.width, pos.height])
                a_h2d.append(agn)


            # assuming outliers are at most 2% of the data, this
            # should be safe even before I filter outliers in my
            # reading routine
            rng = [scipy.stats.scoreatpercentile(xt[~xt.mask], [1, 99]),
                   scipy.stats.scoreatpercentile(y[~y.mask], [1, 99])]
            allrng[tmpfld] = rng[0]

            asc = a2 if (self.Lhiasi is not None and hiasi_mode == "hist") else a1
            aad = a3 if (self.Lhiasi is not None and hiasi_mode == "hist") else a2
            imsc = asc.hexbin(xt, y,
                gridsize=20,
                extent=[rng[0][0], rng[0][1], rng[1][0], rng[1][1]],
                mincnt=1, cmap="viridis")
#            (_, _, _, imsc) = asc.hist2d(xt, y, bins=20, 
#                range=rng, cmap="viridis", cmin=1)
            ptlbins = numpy.linspace(*rng[0], 20)
            typhon.plots.plot_distribution_as_percentiles(
                asc, xt, y, bins=ptlbins,
                color="tan", ptiles=self.ptiles,
                linestyles=self.linestyles,
                linewidth=1.5)
            rng[1] = scipy.stats.scoreatpercentile(adv[~adv.mask], [1, 99])
            imad = aad.hexbin(xt, adv,
                gridsize=20,
                extent=[rng[0][0], rng[0][1], rng[1][0], rng[1][1]],
                mincnt=1, cmap="viridis")
#            (_, _, _, imad) = aad.hist2d(xt, adv, bins=20, 
#                range=rng, cmap="viridis", cmin=1)
            typhon.plots.plot_distribution_as_percentiles(
                aad, xt, adv, bins=ptlbins,
                color="tan", ptiles=self.ptiles,
                linestyles=self.linestyles,
                linewidth=1.5)

            if include_gain and (~med_gain.mask).sum() > 5:
                rng[1] = scipy.stats.scoreatpercentile(med_gain[~med_gain.mask], [1, 99])
                imgn = agn.hexbin(xt, med_gain.m,
                    gridsize=20,
                    extent=[rng[0][0], rng[0][1], rng[1][0], rng[1][1]],
                    cmap="viridis",
                    mincnt=1)
#                (_, _, _, imgn) = agn.hist2d(xt, med_gain.m, bins=20,
#                    range=rng, cmap="viridis", cmin=1)
                typhon.plots.plot_distribution_as_percentiles(
                    agn, xt, med_gain, bins=ptlbins,
                    color="tan", ptiles=self.ptiles,
                    linestyles=self.linestyles,
                    linewidth=1.5)
            else:
                imgn = None

            if i == 0:
                asc.set_ylabel("Space counts")
                aad.set_ylabel("Space Allan dev.\n[counts]")
                if include_gain:
                    agn.set_ylabel("Gain\n[{:~}]".format(med_gain.u))
            else:
                # keep the grid but share ylabel with leftmost
                for a in a_h2d:
                    a.get_yaxis().set_ticklabels([])
            for a in a_h2d[:-1]:
                a.get_xaxis().set_ticklabels([])
            a_h2d[-1].set_xlabel("T [K]")
            a_h2d[0].set_title(tmpfld.replace("_", ""))

            for a in a_h2d:
                a.xaxis.set_major_locator(
                    matplotlib.ticker.MaxNLocator(nbins=4,
                    prune="both"))

        # for the top part
        if self.Lhiasi is not None and hiasi_mode == "hist":
            for (i, (tmpfld, xt)) in enumerate(self.loop_through_temps(self.Mhrscmb, temperatures)):
                ha = ha_ax[tmpfld]
                imha = ha.hexbin(xt.data[dLok],
                    dL.m.data[dLok],
                    gridsize=20,
                    extent=[*allrng[tmpfld],
                            *scipy.stats.scoreatpercentile(dL.m[dLok], [1, 99])],
                    mincnt=1,
                    cmap="viridis")
#                (_, _, _, imha) = ha.hist2d(
#                    xt.data[dLok], dL.m.data[dLok], bins=20,
#                    range=[allrng[tmpfld],
#                           scipy.stats.scoreatpercentile(dL.m[dLok], [1, 99])],
#                    cmin=1)
                typhon.plots.plot_distribution_as_percentiles(ha, 
                    xt, dL, nbins=20, color="tan",
                    ptiles=self.ptiles,
                    linestyles=self.linestyles)
                if i == 0:
                    ha.set_ylabel(r"$\Delta$ R HIRS-HIASI" +
                        "\n[{:Lx}]".format(dL.u))

        # colorbar in remaining half-sized subplot
        if self.Lhiasi is not None and hiasi_mode == "hist":
            acha = self.fig.add_subplot(self.gridspec[C1, k-2:])
            allcb.append(self.fig.colorbar(imha, cax=acha))

            pos = acha.get_position()
            acsc = self.fig.add_axes([pos.x0, pos.y0-pos.height, pos.width, pos.height])
        else:
            acsc = self.fig.add_subplot(self.gridspec[C1, k-2:])
        pos = acsc.get_position()
        acad = self.fig.add_axes([pos.x0, pos.y0-pos.height, pos.width, pos.height])
        if include_gain:
            pos = acad.get_position()
            acgn = self.fig.add_axes([pos.x0, pos.y0-pos.height, pos.width, pos.height])
            ac_all = (acsc, acad, acgn)
            im_all = (imsc, imad, imgn)
        else:
            ac_all = (acsc, acad)
            im_all = (imsc, imad)
        for (a, im) in zip(ac_all, im_all):
            if im is not None:
                cb = self.fig.colorbar(im, cax=a)
                allcb.append(cb)

        return allcb

    @staticmethod
    def loop_through_temps(M, temperatures):
        """Helper to loop through (mean) temperatures contained in M
        """
        for tmpfld in ("temp_{:s}".format(x.lower()) for x in sorted(temperatures)):
            if M[tmpfld].ndim == 3:
                yield (tmpfld[5:], M[tmpfld].mean(-1).mean(-1))
            elif M[tmpfld].ndim == 2:
                yield (tmpfld[5:], M[tmpfld].mean(-1))
            elif M[tmpfld].ndim == 1:
                yield (tmpfld[5:], M[tmpfld])
            else:
                raise RuntimeError("Impossible!")

#    def get_calibcount_range(self, satname="metopa", year=2015):

    def get_accnt(self, typ):
        """Get calibration count anomalies

        Returns an xarray.Dataset.  Due to limitations in pandas and xarray
        (see https://github.com/pydata/xarray/issues/1194), counts are
        converted to floats and masking is done with nans rather than
        masked arrays.  As we only select 2.5% of the data the increase in
        memory is acceptable in this case.
        """
        M = self.Mhrsall

        # Create an xarray Dataset object for easier time series
        # processing
        # need to get rid of masked values first, otherwise xarray changes
        # everything to float
        # (https://github.com/pydata/xarray/issues/1194)
        # …but this OK is too conservative, fails everything if one
        # channel is bad…
        #OK =  ~M["counts"].mask[:, 8:, :19].any(1).any(1)
        # accept conversion to floats and nans as lesser of two bad
        # choices
        D = xarray.Dataset(
            {"counts": (["time", "calibpos", "channel"],
                        M["counts"][:, 8:, :19]),
             "scantype": (["time"],
                            M[self.hirs.scantype_fieldname])},
            coords={"time": M["time"], "calibpos": range(9, 57),
                    "channel": range(1, 20)})
        # select either space or iwt counts
        ccnt = D.isel(time=D["scantype"]==getattr(self.hirs,
            "typ_{:s}".format(typ)))["counts"]
        mccnt = ccnt.mean(dim="calibpos")
        accnt = mccnt - ccnt

        return accnt

    def get_correlations(self, timeres, typ, calibpos):
        """Calculate correlation matrix at time resolution
        """
        accnt = self.get_accnt(typ)

        if accnt["time"].shape[0] < 3:
            raise ValueError("Cannot calculate correlations, only {:d} "
                "time elements found".format(accnt["time"].shape[0]))

        if timeres == "per_cycle":
            times = pandas.DatetimeIndex(accnt["time"].values)
            timeax = times
        else:
            times = pandas.date_range(*accnt["time"][[0, -1]].data, freq=timeres)
            timeax = times[:-1]

        correlations = xarray.DataArray(
            numpy.zeros(shape=(19, 19, timeax.shape[0]), dtype="f4"),
            [("cha", accnt.channel), ("chb", accnt.channel),
             ("time", timeax)])

        for i in range(timeax.shape[0]):
            # accnt is now an array containing nans, because xarray does
            # not support masked arrays
            # see also https://github.com/pydata/xarray/issues/1194 and
            # https://github.com/numpy/numpy/issues/4592 but copying
            # between masked and unmasked does not work very well, need to
            # be careful
            if timeres == "per_cycle":
                cc =  numpy.ma.corrcoef(
                    numpy.ma.masked_invalid(
                        accnt.isel(time=i)))
            else:
                cc =  numpy.ma.corrcoef(
                    numpy.ma.masked_invalid(
                        accnt.sel(time=slice(times[i], times[i+1]),
                        calibpos=calibpos).T))
            cch = cc.data
            cch[cc.mask].fill(numpy.nan)
            correlations[:, :, i] = cch
        return correlations

    
    def plot_noise_correlation_timeseries(self,
            timeres='3H',
            calibpos=20):
        """Plot space/iwt view anomaly correlation timeseries

        Valid time resolution strings are from
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        """
        raise NotImplementedError("Needs updating")
        M = self.Mhrsall

#        try:
#            timeres = timeres.astype(datetime.timedelta)
#        except AttributeError: # hopefully pandas understands as-is
#            pass
# 
#         # Create an xarray Dataset object for easier time series
#         # processing
#         # need to get rid of masked values first, otherwise xarray changes
#         # everything to float
#         # (https://github.com/pydata/xarray/issues/1194)
#         OK =  ~M["counts"].mask[:, 8:, :19].any(1).any(1)
#         D = xarray.Dataset(
#             {"counts": (["time", "scanpos", "channel"],
#                         M["counts"][OK, 8:, :19]),
#              "hrs_scntyp": (["time"],
#                             M["hrs_scntyp"][OK])},
#             coords={"time": M["time"][OK], "scanpos": range(9, 57),
#                     "channel": range(1, 20)})
#         ccnt = D.isel(time=D["hrs_scntyp"]==self.hirs.typ_space)["counts"]
#         mccnt = ccnt.mean(dim="scanpos")
#         accnt = mccnt - ccnt
# 
#         accnt = self.get_accnt()

        correlations = self.get_correlations()

#         times = pandas.date_range(*M["time"][[0, -1]], freq=timeres)
# 
#         correlations = xarray.DataArray(
#             numpy.zeros(shape=(19, 19, times.shape[0]-1), dtype="f4"),
#             [("cha", D.channel), ("chb", D.channel),
#              ("time", times[:-1])])
#         for i in range(times.shape[0]-1):
#             correlations[:, :, i] = numpy.corrcoef(
#                 accnt.sel(time=slice(times[i], times[i+1]),
#                 scanpos=scanpos).T)
# 
        (f, ax_all) = matplotlib.pyplot.subplots(19, 3,
            sharex=True, sharey=True, figsize=(12, 30))

        for ((i, ch_a), (j, ch_b)) in itertools.product(
                enumerate(range(1, 20)), repeat=2):
            if ch_a == ch_b: continue
            correlations.sel(cha=ch_a, chb=ch_b).plot(
                ax=ax_all[i, min(j//6, 2)],
                label="vs. ch. {:d}".format(ch_b))
#            ax_all[i, min(j//3, 2)].plot_date(
#                correlations["time"].astype("M8[s]").astype(datetime.datetime),
#                correlations.sel(cha=ch_a, chb=ch_b).data,
#                label="ch. {:d}+{:d}".format(ch_a, ch_b))
            ax_all[i, min(j//6, 2)].set_title("ch. {:d}".format(ch_a))
        for a in ax_all.ravel():
            if a in ax_all[0, :]:
                a.legend()
            a.set_ylim(-1, 1)
        
        f.suptitle("Inter-channel correlation timeseries, "
                   "{:s} HIRS "
                   "{:%Y-%m-%d}--{:%Y-%m-%d} pos {:d}".format(
                        self.satname, self.start_date, self.end_date,
                        calibpos))
        pyatmlab.graphics.print_or_show(f, False,
            "timeseries_channel_noise_correlation_"
            "HIRS_{:s}{:%Y%m%d%H%M}-{:%Y%m%d%H%M}_p{:d}.".format(
                self.satname, self.start_date, self.end_date, scanpos)
                + "" if self.writefig else "png")


    ##### EXPERIMENTAL NO GO ZONE FOR NOW #####
    def get_noise_with_other(self, 
            ch,
            all_tp=["space", "iwt"], temperatures=["iwt"],
            include_gain=True,
            include_corr=(),
            corr_info={}):
        logger.info("Channel {:d}".format(ch))
        M = self.Mhrsall
#        ch = self.ch
        start_date = self.start_date
        end_date = self.end_date
        #k = int(numpy.ceil(len(temperatures)/2)*2)
        Ntemps = len(temperatures)
        logger.info("Getting calibration counts + noise")
        if len(all_tp) > 0:
            D_ccn = self.get_calib_counts_noise(M=M, ch=ch, all_tp=all_tp)

        if include_gain or include_rself:
            (t_slope, med_gain, u_med_gain) = self.get_gain(M, ch)

        corr_typ = {}
        if len(include_corr)>0:
            for typ in all_tp:
                corr_typ[typ] = self.get_correlations(
                    typ=typ,
                    calibpos=corr_info.get("calibpos", 20),
                    timeres=corr_info.get("timeres", "3H"),
                    N=corr_info.get("count", 2),
                    corr_types=include_corr)
                
        if len(temperatures) > 0:
            tempfields = list(self.loop_through_temps(M, temperatures))
            rv = self.get_temperatures(M=M, temperatures=temperatures)

        # don't do self-emission here, it's not time series anyway
        
        p = pathlib.Path(typhon.config.conf["main"]["myscratchdir"])


    ##### END OF EXPERIMENTAL NO GO ZONE #####
# 1, 2, 8, 19

def main():
    p = parse_cmdline()
    common.set_root_logger(
        logging.DEBUG if p.verbose else logging.INFO
        filename=p.log)
        
    if p.plot_iwt_anomaly:
        write_timeseries_per_day_iwt_anomaly_period(
            "noaa19", datetime.date(2013, 3, 1),
            datetime.date(2014, 3, 1))
        plot_timeseries_temp_iwt_anomaly_all_sats()
    
    if p.plot_noise or p.plot_noise_with_other or p.plot_noise_correlation_timeseries:
        na = NoiseAnalyser(
            datetime.datetime.strptime(p.from_date, p.datefmt),
            datetime.datetime.strptime(p.to_date, p.datefmt),
            p.sat,
            temp_fields=p.temp_fields,
            writefig=p.write_figs)
#            ch=p.channel)

    if p.plot_noise:
        logger.info("Plotting noise")
        na.plot_noise()

    if p.plot_noise_with_other:
        logger.info("Plotting more noise")
        for ch in p.channel:
            if p.store_only:
                na.get_noise_with_other(ch, temperatures=p.temp_fields,
                    all_tp=p.count_fields, include_gain=p.include_gain,
                    include_corr=p.include_corr,
                    corr_info={"pairs": p.corr_pairs,
                               "perc": p.corr_perc,
                               "count": p.corr_count,
                               "timeres": p.corr_timeres,
                               "calibpos": p.corr_calibpos})
            else:
                na.plot_noise_with_other(ch, temperatures=p.temp_fields,
                    all_tp=p.count_fields, include_gain=p.include_gain,
                    include_rself=p.include_rself,
                    hiasi_mode=p.hiasi_mode,
                    include_corr=p.include_corr,
                    corr_info={"pairs": p.corr_pairs,
                               "perc": p.corr_perc,
                               "count": p.corr_count,
                               "timeres": p.corr_timeres,
                               "calibpos": p.corr_calibpos},
                    width_factor=p.width_factor)

    if p.plot_noise_correlation_timeseries:
        logger.info("Plotting noise correlation timeseries")
        na.plot_noise_correlation_timeseries()

    if p.plot_noise_map:
        #make_hirs_orbit_maps(sat, dt, ch, cmap="viridis"):
        make_hirs_orbit_maps("noaa18",
            datetime.datetime(2005, 8, 13, 15, 0),
            5)


#if __name__ == "__main__":
#    main()
