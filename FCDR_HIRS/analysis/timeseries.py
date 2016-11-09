#!/usr/bin/env python3.5

"""Plot various timeseries for HIRS

Anomalies averaged per orbit.
"""

import argparse

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
        choices=["space", "ict", "iwt"],
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

    parser.add_argument("--hiasi_mode", action="store", type=str,
        choices=["perc", "hist"], default="perc",
        help="For HIASI anomalies, show as PERCentiles or as HIST2d")

    parser.add_argument("--sat", action="store", type=str,
        help="Satellite to use.")

    parser.add_argument("--channel", action="store", type=int,
        default=1, help="Channel to consider")

    parser.add_argument("--log", action="store", type=str,
        help="Logfile to write to")

    parser.add_argument("--verbose", action="store_true",
        help="Be verbose", default=False)

    parser.set_defaults(include_gain=True, include_rself=True)
    
    p = parser.parse_args()
    return p
parsed_cmdline = parse_cmdline()

import logging
logging.basicConfig(
    format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
             "%(lineno)s: %(message)s"),
    filename=parsed_cmdline.log,
    level=logging.DEBUG if parsed_cmdline.verbose else logging.INFO)

import datetime
import itertools
import pathlib
import math

import numpy
import matplotlib
matplotlib.use("Agg")
# Source: http://stackoverflow.com/a/20709149/974555
# … but wait until I have LaTeX up and running
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       #r'\usepackage{helvet}',    # set the normal font here
       #r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       #r'\sansmath',              # <- tricky! -- gotta actually tell tex to use!
       r'\DeclareSIUnit\count{count}'  # siunitx doesn't know this one
]
# this too must be before importing matplotlib.pyplot
pathlib.Path("/dev/shm/gerrit/cache").mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot
import matplotlib.gridspec
import matplotlib.dates
import matplotlib.ticker

#from memory_profiler import profile

import typhon.plots
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
import pyatmlab.config
import pyatmlab.graphics
#import pyatmlab.datasets.tovs
#from pyatmlab.units import ureg

from .. import fcdr

srcfile_temp_iwt = pathlib.Path(pyatmlab.config.conf["main"]["myscratchdir"],
                       "hirs_{sat:s}_{year:d}_temp_iwt.npz")

def get_timeseries_temp_iwt_anomaly(sat, year_start=2005, year_end=2017):
    L = []
    for year in range(year_start, year_end):
        try:
            logging.debug("Reading {:d}".format(year))
            D = numpy.load(str(srcfile_temp_iwt).format(sat=sat, year=year))
            L.extend(D["selection"])
        except FileNotFoundError:
            pass
    logging.info("Processing")
    dts = numpy.array([x[0] for x in L], "M8[s]")
    anomalies = numpy.concatenate(
        [(x[1].mean(0).mean(1) - x[1].mean())[:, numpy.newaxis] for x in L], 1).T
    
    return (dts, anomalies)

def plot_timeseries_temp_iwt_anomaly(sat, nrow=4):
    (dts, anomalies) = get_timeseries_temp_iwt_anomaly(sat)
    dts = dts.astype(datetime.datetime)
    (f, ax_all) = matplotlib.pyplot.subplots(nrow, 1)

    (lo, hi) = scipy.stats.scoreatpercentile(anomalies, [1, 99])
    for (i, ax) in enumerate(ax_all):
        ix_st = i * dts.shape[0]//nrow
        ix_end = min((i+1) * dts.shape[0]//nrow, dts.shape[0]-1)
        logging.info("Plotting part {:d}/{:d}".format(i+1, nrow))
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
    dest = pathlib.Path(pyatmlab.config.conf["main"]["myscratchdir"],
        "hirs_iwt_anom_{:s}_{:%Y%m%d}-{:%Y%m%d}".format(sat, start_date, end_date))
    logging.info("Writing {!s}".format(dest))
    with dest.open("wt", encoding="ascii") as fp:
        fp.writelines([("{:%Y-%m-%d}" + 5*" {:.5f}" + "\n").format(
                x["date"].astype(datetime.datetime), *x["anomalies"])
                    for x in X])
            

def plot_timeseries_temp_iwt_anomaly_all_sats():
    for sat in {"noaa18", "noaa19", "metopa", "metopb"}:
        logging.info("Plotting {:s}".format(sat))
        plot_timeseries_temp_iwt_anomaly(sat)
        

class NoiseAnalyser:
    ptiles = (5, 25, 50, 75, 95)
    linestyles = (":", "--", "-", "--", ":")
    fte = 0.67
    fhs = 0.73
#@profile
    def __init__(self, start_date, end_date, satname, ch, temp_fields={"iwt",
                        "fwh", "fwm"}):
        #self.hirs = pyatmlab.datasets.tovs.which_hirs_fcdr(satname)
        self.hirs = fcdr.which_hirs_fcdr(satname)
#        for h in (typhon.datasets.tovs.HIRS4,
#                  typhon.datasets.tovs.HIRS3,
#                  typhon.datasets.tovs.HIRS2):
#            if satname in h.satellites:
#                self.hirs = h()
#                break
#        else:
#            raise ValueError("Unknown satellite: {:s}".format(satname))
        self.satname = satname
        self.ch = ch

#        self.srfs = [typhon.physics.units.em.SRF.fromArtsXML(
#                    satname.upper().replace("NOAA0", "NOAA"), "hirs", i) for i in range(1, 20)]

#        srf1 = pyatmlab.physics.SRF(*self.srfs[1])
        hrsargs=dict(
                fields=["hrs_scnlin", self.hirs.scantype_fieldname, "time",
                        "counts", "calcof_sorted", "radiance"] +
                       ["temp_{:s}".format(f) for f in
                       set(temp_fields) | {"iwt"}],
                locator_args=dict(satname=self.satname),
                reader_args=dict(filter_firstline=True))
        # those need to be read before combining with HIASI, because
        # afterward, I lose the calibration rounds.  But doing it after
        # read a full month (or more) of data takes too much RAM as I will
        # need to copy the entire period; therefore, add it on-the-fly
        Mhrsall = self.hirs.read_period(start_date, end_date,
            pseudo_fields=
                {"tsc": self.hirs.calc_time_since_last_calib,
                 "lsc": self.hirs.count_lines_since_last_calib,
                 "radiance_fid": lambda M:
                    self.hirs.calculate_radiance(M, ch, interp_kind="zero").m},
            NO_CACHE=True, **hrsargs)
#        Mhrsallnew = numpy.ma.zeros(shape=Mhrsall.shape,
#            dtype=Mhrsall.dtype.descr + [("tsc", "u2")])
#        for fld in Mhrsall.dtype.names:
#            Mhrsallnew[fld][...] = Mhrsall[fld][...]
#            Mhrsallnew.mask[fld][...] = Mhrsallnew.mask[fld][...]
#        Mhrsallnew["tsc"][...] = self.hirs.calc_time_since_last_calib(Mhrsall)
#        Mhrsall = Mhrsallnew
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
                    #Miasi = self.hiasi.read_period(start_date, end_date)
                    Miasi = self.hiasi.read_period(dt, dt+step, NO_CACHE=True)
                except typhon.datasets.dataset.DataFileError:
                    logging.info("No IASI found in "
                        "[{:%Y-%m-%d %H:%M}-{:%Y-%m-%d %H:%M}]".format(
                            dt, dt+step))
                    #self.Miasi = None
                    pass
                else:
                    found_iasi = True
                    logging.info("Combining HIASI with HIRS…")
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
            # BUG!  The time-since-calibration calculated here is totally
            # wrong, because calc_time_since_last_calib can only work on
            # the pure HIRS, not on any slicing/subselection!
            #self.tsc = self.hirs.calc_time_since_last_calib(self.Mhrscmb)
            #Lhrs = self.hirs.calculate_radiance(self.M, srf, ch)
            #self.tsc = self.Mhrsall["tsc"]
        else:
            self.Mhrscmb = None
            self.Lhiasi = None
            self.tsc = None

        self.start_date = start_date
        self.end_date = end_date

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
        if Δt < datetime.timedelta(days=(N*2)):
            a.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=
                    math.ceil((24*Δdays)//(N*10)+1)))
            a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
            # FIXME: minor locator
        else:
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
        t = M[views_space]["time"].astype(datetime.datetime)
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
                self.satname, t[0], t[-1]))

    #@profile
    def plot_noise_with_other(self, 
            all_tp=["space", "iwt"], temperatures=["iwt"],
            include_gain=True,
            include_rself=True,
            hiasi_mode="perc"):
        M = self.Mhrsall
        ch = self.ch
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
        N = 2*(len(all_tp)>0) + (len(temperatures)>0) + 2*include_gain + 3*include_rself
        #k = int(numpy.ceil(len(temperatures)/2)*2)
        Ntemps = len(temperatures)
        fact = 16
        k = int((Ntemps+(2/fact))*fact)
        self.ifte = int(self.fte*k)
        self.ifhs = int(self.fhs*k)
        self.gridspec = matplotlib.gridspec.GridSpec(N, k)
        self.fig = matplotlib.pyplot.figure(figsize=(18, 3*N))
        #(f, ax) = matplotlib.pyplot.subplots(N, 1, figsize=(16, 3*N))
        #itax = iter(ax)
        logging.info("Plotting calibration counts + noise")
        self.counter = itertools.count()
        if len(all_tp) > 0:
            self._plot_calib_counts_noise(M=M, ch=ch, all_tp=all_tp, styles=styles)

        if include_gain or include_rself:
            #ax[3].plot_date(t, M[views]["calcof_sorted"][:, ch-1, 1], '.')

            (t_slope, _, slope) = self.hirs.calculate_offset_and_slope(M, ch)
            slope = slope.to(ureg.mW/(ureg.m**2 * ureg.sr *
                1/ureg.cm * ureg.counts), "radiance")
            gain = 1/slope
            med_gain = ureg.Quantity(
                numpy.ma.median(gain.m[:, :], 1),
                gain.u)

        if include_gain:
            self._plot_gain(t_slope=t_slope, med_gain=med_gain)
                
        if len(temperatures) > 0:
            ax2lims = self._plot_temperatures(M=M, temperatures=temperatures)

        if include_rself:
            allcb = self._plot_rself(
                M=M, t_slope=t_slope, med_gain=med_gain, ch=ch,
                temperatures=temperatures, k=k,
                ax2lims=ax2lims, styles=styles,
                hiasi_mode=hiasi_mode,
                include_gain=include_gain)
            # some self-emission characteristics

        logging.info("Finalising")

        for cb in allcb:
            cb.set_label("No.")
            cb.ax.yaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(nbins=5, prune="both"))
            cb.update_ticks()
                    
        t = M["time"].astype(datetime.datetime)
        for a in self.fig.axes:
            a.autoscale_view()
            a.grid("on")
            if isinstance(a.xaxis.major.formatter, matplotlib.dates.AutoDateFormatter):
                self.set_ax_loc(a, t, N=1)
        self.fig.suptitle("Characteristics for {:s} HIRS ch. {:d}, "
                   "{:%Y-%m-%d}--{:%Y-%m-%d}".format(
                        self.satname, ch, start_date, end_date),
                fontsize=26)
        self.fig.subplots_adjust(hspace=0.5, top=0.95)
        logging.info("Writing out")
        # Write only PNG, the rest is too slow / memory-intensive
        # For some reason, sometimes it still fails to use the LaTeX
        # cache.  Make sure we create it /again/ ?!
        pathlib.Path("/dev/shm/gerrit/cache").mkdir(parents=True, exist_ok=True)
        pyatmlab.graphics.print_or_show(self.fig, False,
            "hirs_noise/{self.satname:s}_{tb:%Y}/ch{ch:d}/disect_{self.satname:s}_hrs_ch{ch:d}_{alltyp:s}_{alltemp:s}_{tb:%Y%m%d%H%M}-{te:%Y%m%d%H%M}.png".format(
                self=self, ch=ch, alltyp='_'.join(all_tp),
                alltemp='_'.join(temperatures), tb=t[0], te=t[-1]))

    def _plot_calib_counts_noise(self, M, ch, all_tp, styles):
        C = next(self.counter)
        a0 = self.fig.add_subplot(self.gridspec[C, :self.ifte])
        a0h = self.fig.add_subplot(self.gridspec[C, self.ifhs:])
        C = next(self.counter)
        a1 = self.fig.add_subplot(self.gridspec[C, :self.ifte])
        a1h = self.fig.add_subplot(self.gridspec[C, self.ifhs:])
        L0 = []
        L1 = []
        for (i, tp) in enumerate(all_tp):
            views = M[self.hirs.scantype_fieldname
                        ] == getattr(self.hirs, "typ_"+tp)

            t = M[views]["time"].astype(datetime.datetime)
            x = M[views]["counts"][:, 8:, ch-1]
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
        a0.set_title("Calibration counts over time")
        a0h.set_title("Calib. counts hist.")
        a0h.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=6))
        a0h.legend([x[0] for x in L0], all_tp, loc="upper left",
            bbox_to_anchor=(1.0, 1.0))
        a1.set_title("Calibration noise (Allan deviation) for space "
                        "and IWCT views")
        a1h.set_title("Calib. noise hist.")
        a1.set_xlabel("Date / time")
        a1.set_ylabel("Allan deviation\n[counts]")
        a1h.set_xlabel(a1.get_ylabel().replace("\n", " "))
        a1h.set_ylabel("Number")
        a1h.legend([x[0] for x in L1], all_tp, loc="upper left",
            bbox_to_anchor=(1.0, 1.0))

    def _plot_gain(self, t_slope, med_gain):
        C = next(self.counter)
        a = self.fig.add_subplot(self.gridspec[C, :self.ifte])
        logging.info("Plotting gain")
        a.plot_date(t_slope.astype(datetime.datetime),
                    med_gain, '.',
                        color="black",
                        markersize=5)
        a.set_xlabel("Date / time")
        a.set_ylabel("Gain\n" + "[{:Lx}]".format(med_gain.u))
        valid = (~t_slope.mask) & (~med_gain.mask)
        a.set_ylim(scipy.stats.scoreatpercentile(med_gain[valid], [1, 99]))
        a.set_title("Gain development over time")

        ah = self.fig.add_subplot(self.gridspec[C, self.ifhs:])
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
    
    def _plot_temperatures(self, M, temperatures):
        C = next(self.counter)
        a = self.fig.add_subplot(self.gridspec[C, :self.ifte])
        ah = self.fig.add_subplot(self.gridspec[C, self.ifhs:])
        logging.info("Plotting temperatures")
        ax2lims = None
        t = M["time"].astype(datetime.datetime)
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
        return ax2lims

    def _plot_rself(self, M, t_slope, med_gain, ch, temperatures, k,
                    ax2lims, styles, hiasi_mode, include_gain):
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

            Lhrs = ureg.Quantity(self.Mhrscmb["radiance_fid"],
                typhon.physics.units.common.radiance_units["ir"])
            dL = Lhrs - self.Lhiasi[:, ch-1]
            ΔRselfint, = self.hirs.interpolate_between_calibs(
                self.Mhrscmb, t_slope[1:], ΔRself,
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
                    (_, _, _, im) = a.hist2d(x, ydL,
                        bins=[xbins, ybins],
                        range=[[0, 40],
                               scipy.stats.scoreatpercentile(ydL, [1, 99])],
                        cmin=1)
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
            (_, _, _, imsc) = asc.hist2d(xt, y, bins=20, 
                range=rng, cmap="viridis", cmin=1)
            ptlbins = numpy.linspace(*rng[0], 20)
            typhon.plots.plot_distribution_as_percentiles(
                asc, xt, y, bins=ptlbins,
                color="tan", ptiles=self.ptiles,
                linestyles=self.linestyles,
                linewidth=1.5)
            rng[1] = scipy.stats.scoreatpercentile(adv[~adv.mask], [1, 99])

            (_, _, _, imad) = aad.hist2d(xt, adv, bins=20, 
                range=rng, cmap="viridis", cmin=1)
            typhon.plots.plot_distribution_as_percentiles(
                aad, xt, adv, bins=ptlbins,
                color="tan", ptiles=self.ptiles,
                linestyles=self.linestyles,
                linewidth=1.5)

            rng[1] = scipy.stats.scoreatpercentile(med_gain[~med_gain.mask], [1, 99])
            (_, _, _, imgn) = agn.hist2d(xt, med_gain.m, bins=20,
                range=rng, cmap="viridis", cmin=1)
            typhon.plots.plot_distribution_as_percentiles(
                agn, xt, med_gain, bins=ptlbins,
                color="tan", ptiles=self.ptiles,
                linestyles=self.linestyles,
                linewidth=1.5)

            if i == 0:
                asc.set_ylabel("Space counts")
                aad.set_ylabel("Space Allan dev.\n[counts]")
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
                (_, _, _, imha) = ha.hist2d(
                    xt.data[dLok], dL.m.data[dLok], bins=20,
                    range=[allrng[tmpfld],
                           scipy.stats.scoreatpercentile(dL.m[dLok], [1, 99])],
                    cmin=1)
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
        for (a, im) in zip((acsc, acad, acgn), (imsc, imad, imgn)):
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
        
    

# 1, 2, 8, 19

def main():
    p = parsed_cmdline
    if p.plot_iwt_anomaly:
        write_timeseries_per_day_iwt_anomaly_period(
            "noaa19", datetime.date(2013, 3, 1),
            datetime.date(2014, 3, 1))
        plot_timeseries_temp_iwt_anomaly_all_sats()
    
    if p.plot_noise or p.plot_noise_with_other:
        na = NoiseAnalyser(
            datetime.datetime.strptime(p.from_date, p.datefmt),
            datetime.datetime.strptime(p.to_date, p.datefmt),
            p.sat,
            temp_fields=p.temp_fields,
            ch=p.channel)

    if p.plot_noise:
        logging.info("Plotting noise")
        na.plot_noise()

    if p.plot_noise_with_other:
        logging.info("Plotting more noise")
        na.plot_noise_with_other(temperatures=p.temp_fields,
            all_tp=p.count_fields, include_gain=p.include_gain,
            include_rself=p.include_rself,
            hiasi_mode=p.hiasi_mode)

    if p.plot_noise_map:
        #make_hirs_orbit_maps(sat, dt, ch, cmap="viridis"):
        make_hirs_orbit_maps("noaa18",
            datetime.datetime(2005, 8, 13, 15, 0),
            5)


#if __name__ == "__main__":
#    main()
