"""Plot statistics on calibration counts per "position"

HIRS is supposed to dwell for both the space and IWCT views (NOAA KLM
User's Guide, Table 3.2.1.1-1 and Table 3.2.2.1-1), but only uses 48
elements (NOAA KLM User's Guide, page 3-30, PDF page 125, or Table J-2).
That means that statistically, each of those 48 positions should on
average measure the same.  Let's verify this.
"""

from .. import common
import argparse

import logging
import matplotlib
#matplotlib.use("Agg")
import pathlib
#pathlib.Path("/dev/shm/gerrit/cache").mkdir(parents=True, exist_ok=True)

import datetime
import itertools
import math
import scipy.stats
import random

import numpy
import matplotlib.pyplot
import matplotlib.ticker

import typhon.plots
matplotlib.pyplot.style.use(typhon.plots.styles("typhon"))

from .. import fcdr
from .. import graphics

logger = logging.getLogger(__name__)

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = common.add_to_argparse(parser,
        include_period=True,
        include_sat=True,
        include_channels=True,
        include_temperatures=False)
    
    parser.add_argument("--plot_distributions",
        action="store_true", 
        default=False,
        help="Plot median + quantiles over large number of calibration lines")

    parser.add_argument("--plot_examples",
        action="store",
        type=int,
        default=0,
        help="How many examples to plot")
    
    parser.add_argument("--random_seed",
        action="store",
        type=int,
        default=0,
        help="Randomly select with seed s")

    parser.add_argument("--examples_mode",
        action="store",
        type=str,
        choices=("random", "highcorr", "lowcorr"),
        default="random",
        help="How to select examples to show")

    parser.add_argument("--calibtype",
        action="store",
        type=str,
        choices=("iwt", "ict", "space"),
        default="iwt",
        help="What kind of calibration to show")

    parser.add_argument("--anomaly",
        action="store_true",
        dest="anomalies",
        default=True,
        help="Plot anomalies")

    parser.add_argument("--no_anomaly",
        action="store_false",
        dest="anomalies")

    p = parser.parse_args()
    return p

def plot_calibcount_stats(h, Mall, channels,
        title="", filename=""):
    """Plot statistics on calibration counts

    For all requested channels in the data passed, plot the distribution
    of space counts per "scan position" as a set of percentiles, into a
    figure with one subplot per channel.

    Parameters
    ----------

    h : HIRS
        HIRS object
    Mall : ndarray
        Array from which to extract data to plot
    channels : List[int]
        Channels to plot
    title : str, optional
        Set figure suptitle to this.
    filename : str, optional
        Write output to this filename (inside the plot directory) using
        `graphics.print_or_show`
    """
    N = len(channels)

    (nrow, ncol) = typhon.plots.common.get_subplot_arrangement(N)

    x = numpy.arange(0, 49, 2)
    (f, ax_all) = matplotlib.pyplot.subplots(nrow, ncol,
        sharex=True,
        figsize=(4+3*nrow, 4+2*ncol))
    view_space = Mall[h.scantype_fieldname] == h.typ_space
    view_iwct = Mall[h.scantype_fieldname] == h.typ_iwt
    Msp = Mall[view_space]
    Miwt = Mall[view_iwct]
    for (a1, c) in zip(ax_all.ravel(), channels):
        a2 = a1.twinx()
        for (Mlocal, a, color, lab) in (
                (Msp, a1, "blue", "space"),
                (Miwt, a2, "green", "IWCT")):
            typhon.plots.plot_distribution_as_percentiles(a,
                numpy.arange(1, 49), Mlocal["counts"][:, 8:, c-1].T, bins=x, color=color,
                ptiles=[5, 25, 50, 75, 95],
                linestyles=[":", "--", "-", "--", ":"],
                label=lab,
                linewidth=1.0)
            a.grid(True, which="major")
            for ax in (a.xaxis, a.yaxis):
                ax.set_major_locator(
                    matplotlib.ticker.MaxNLocator(nbins=4, prune=None))
            a.yaxis.set_minor_locator(
                matplotlib.ticker.AutoMinorLocator(5))
            a.set_xlim(0, 48)
        a1.xaxis.set_minor_locator(
            matplotlib.ticker.MultipleLocator(1))
        if c == channels[-1]:
            # http://stackoverflow.com/a/10129461/974555
            lines, labels = a1.get_legend_handles_labels()
            lines2, labels2 = a2.get_legend_handles_labels()
            a1.legend(lines + lines2, labels + labels2,
                loc="upper left", bbox_to_anchor=(1.5, 1.15))
        for tl in a1.get_yticklabels():
            tl.set_color("blue")
        for tl in a2.get_yticklabels():
            tl.set_color("green")
        a1.set_title("Ch. {:d}".format(c))
        if a1 in ax_all[:, -1] or c == channels[-1]:
            a2.set_ylabel("IWCT counts")
        if a1 in ax_all[:, 0]:
            a1.set_ylabel("space counts")
        if a1 in ax_all[-1, :] or c==len(channels)//ncol*ncol:
            a1.set_xlabel("Calib. scanpos")
    # set remaining invisible
    for a in ax_all.ravel()[len(channels):]:
        a.set_visible(False)
    f.suptitle(title, y=1.02)
    f.subplots_adjust(hspace=0.5, wspace=0.5,
        right=0.75 if nrow*ncol==len(channels) else 0.9)
    graphics.print_or_show(f, False, filename)

def plot_calibcount_anomaly_examples(h, M, channels, N,
        mode="random", typ="space", anomaly=True):
    """Plot examples of calibcount anomalies

    Plot calibration count anomalies, i.e. the calibration counts minus
    the mean calibration count for that cycle.

    Parameters
    ----------

    h : HIRS
        HIRS object
    M : ndarray
        structured array such as returned by h.read, must contain at
        least counts and info on scantypes
    channels : List[int]
        channels to plot
    N : int
        How many examples to choose, in as many subplots
    mode : str, optional
        How to select what calibrations counts to choose?  The default is
        ``"random"``, which randomly selects calibration count lines.
        Other alternatives are ``"lowcorr"``, which chooses cases where
        the correlation is lowest (meaning possibly strongly negative),
        and ``"highcorr"``, which selects the cases with the highest
        correlation, 
    typ : str, optional
        Can be ``"space"``, ``"iwt"``, or (HIRS/2 and HIRS/2I only) "ict".
    anomaly : bool, optional
        If True, plot anomalies, the default.  Otherwise plot the real
        count values.
    """
    Mv = M[M[h.scantype_fieldname] == getattr(h, "typ_{:s}".format(typ))]
    ccnt = Mv["counts"][:, h.start_space_calib:, :]
    mccnt = ccnt.mean(1, keepdims=True)
    accnt = ccnt - mccnt if anomaly else ccnt
    aok = ~(accnt[:, :, :].mask.any(2).any(1))
    if not aok.any():
        logger.error("Nothing to plot.  All flagged.")
        return
    accnt = accnt[aok, :, :]
    logger.info("Found {:d} unflagged calibration cycles".format(accnt.shape[0]))
    
    channels = numpy.asarray(channels)
    if mode == "random":
        idx = numpy.random.choice(numpy.arange(accnt.shape[0]),
            size=N, replace=False)
    elif mode in ("lowcorr", "highcorr"): # low/high correlations
        idx = []
        all_corr = numpy.dstack(
            [numpy.corrcoef(accnt[i, :, :][:, channels-1].T)
                for i in range(accnt.shape[0])])
        N_channel_combi = channels.shape[0]*(channels.shape[0]-1)//2
        for (ia, ib) in ((ia, ib) for (ia, ib) in
                itertools.product(range(channels.shape[0]), repeat=2) if ia<ib):
            thiscorr = all_corr[ia, ib, :].copy()
            idx_sorted = numpy.argsort(thiscorr)
            if mode == "highcorr": # sort descending
                idx_sorted = idx_sorted[::-1] 
            idx.extend(idx_sorted[:math.ceil(N/N_channel_combi)])
    else:
        raise ValueError("Expected mode to be 'random', 'lowcorr', or 'highcorr', "
            "got {!s}".format(mode))
    idx.sort() # show examples chronologically
    idx = idx[:N]

    #show = accnt[idx, :, :][:, numpy.asarray(channels)-1]

    (f, ax) = matplotlib.pyplot.subplots(N, 1, sharex=True,
        figsize=(10, 4+2*N), squeeze=False)
    for (a, i) in zip(ax.ravel(), idx):
        for ch in channels:
            a.plot(numpy.arange(h.start_space_calib+1, h.n_perline+1),
                    accnt[i, :,  ch-1],
                    'o-', mfc="none",
                    label="ch. {:d}".format(ch))
        a.set_title(str(Mv["time"][i]))
        a.set_ylabel(("Anomaly to scanline\nmean" if anomaly else
                      "Calibration value") + " [counts]")
        a.grid()
    ax.ravel()[-1].set_xlabel("Scanline position")
    ax.ravel()[0].legend() # FIXME: position

    f.suptitle("{:s} {:s} view calibration {:s}".format(
        h.satname, typ, "anomalies" if anomaly else "values"))

    graphics.print_or_show(f, False,
        "{:s}_{:s}_calib_{:s}_{:s}-{:%Y%m%d%H%M%S}-{:%Y%m%d%H%M%S}_{:d}_{:s}_{:s}.".format(
            typ, mode, "anomalies" if anomaly else "values", h.satname,
            M["time"][idx[0]].astype(datetime.datetime),
            M["time"][idx[-1]].astype(datetime.datetime), N,
            ",".join(str(ch) for ch in channels),
            mode))

def read_and_plot_calibcount_stats(sat, from_date, to_date, channels,
        plot_stats=False,
        plot_examples=0,
        random_seed=0,
        sample_mode="random",
        typ="iwt",
        anomaly=True):
    """Read and plot calibration count statistics.

    Read data for period, then pass it on to `plot_calibcount_stats` and
    `plot_calibcount_anomaly_examples` for plotting statistics and
    examples.

    Note that if you don't pass either ``plot_stats`` or
    ``plot_examples``, nothing will happen.

    Parameters
    ----------

    sat : str
        Name of satellite.
    from_date : datetime.datetime
        Starting datetime.
    to_date : datetime.datetime
        Ending datetime.
    channels : List[int]
        What channel or channels to plot.
    plot_stats : bool, optional
        Whether or not to plot statistics.  Defaults to false.
    plot_examples : int, optional
        How many examples to show.  Defaults to 0.
    random_seed : Number, optional
        Number to seed the random number generator with.  Can be useful if
        you want to recover the same plot again.
    sample_mode : str, optional
        For plotting examples, whether to take ``"random"``, ``"highorr"``,
        or ``"lowcorr"``.  See `plot_calibcount_anomaly_examples`.
    type : str, optional
        For plotting examples, this can be ``"iwt"``, ``"space"``, or
        ``"ict"``.  See `plot_calibcount_anomaly_examples`.
    anomaly : bool, optional
        If true, plot anomalies.  If false, plot actual count values.
        Defaults to True.
    """
    h = fcdr.which_hirs_fcdr(sat)
    M = h.read_period(from_date, to_date,
            fields=["time", "counts", h.scantype_fieldname])
    if plot_stats:
        plot_calibcount_stats(h, M, channels,
            title="HIRS calibration consistency check per scanpos\n"
                  "{sat:s} {from_date:%Y-%m-%d} -- {to_date:%Y-%m-%d}".format(
                    **locals()),
            filename="hirs_calib_per_scanpos_{sat:s}_{from_date:%Y%m%d%H%M}-"
                     "{to_date:%Y%m%d%H%M}_{ch:s}.".format(
                            ch=",".join([str(x) for x in channels]), **locals()))
    if plot_examples > 0:
        numpy.random.seed(random_seed)
        plot_calibcount_anomaly_examples(
            h, M, channels, plot_examples,
            mode=sample_mode,
            typ=typ,
            anomaly=anomaly)

def main():
    """Main function for module

    Expect commandline input.
    """
    p = parse_cmdline()
    from_date = datetime.datetime.strptime(p.from_date, p.datefmt)
    to_date = datetime.datetime.strptime(p.to_date, p.datefmt)
    common.set_logger(
        logging.DEBUG if p.verbose else logging.info,
        p.log,
        loggers={"FCDR_HIRS", "typhon"})
    read_and_plot_calibcount_stats(p.satname, from_date, to_date,
        p.channels, p.plot_distributions, p.plot_examples,
        p.random_seed, 
        p.examples_mode,
        p.calibtype,
        p.anomalies)
