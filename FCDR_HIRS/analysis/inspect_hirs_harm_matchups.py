"""Plotting the health of harmonisation matchups
"""

import argparse
from .. import common
def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = common.add_to_argparse(parser,
        include_period=False,
        include_sat=0,
        include_channels=False,
        include_temperatures=False)

    parser.add_argument("file",
        action="store",
        type=str,
        help="Path to file containing enhanced matchups")

    return parser.parse_args()
p = parse_cmdline()

import sys
import pathlib

import numpy
import matplotlib.pyplot
import xarray
import scipy.stats

import pyatmlab.graphics

def plot_file_summary_stats(path):
    """Plot various summary statistics for harmonisation file

    Assumes it contains the extra fields that I generate for HIRS using
    combine_hirs_hirs_matchups.
    """

    # TODO:
    #   - get extent from data percentiles
    #   - get axes labels from data-array attributes

    (f, ax_all) = matplotlib.pyplot.subplots(2, 3, figsize=(20, 10))

    ds = xarray.open_dataset(path)

    g = ax_all.flat

    cbs = []

    kxrange = scipy.stats.scoreatpercentile(ds["K_forward"], [1, 99])
    kyrange = scipy.stats.scoreatpercentile(ds["K_backward"], [1, 99])
    kΔrange = scipy.stats.scoreatpercentile(ds["K_forward"]+ds["K_backward"], [1, 99])
    Lxrange = scipy.stats.scoreatpercentile(ds["nominal_measurand1"], [1, 99])
    Lyrange = scipy.stats.scoreatpercentile(ds["nominal_measurand2"], [1, 99])
    Lmax = max(Lxrange[1], Lyrange[1])
    Lmin = min(Lxrange[0], Lyrange[0])
    LΔrange = scipy.stats.scoreatpercentile(
        ds["nominal_measurand2"] - ds["nominal_measurand1"],
        [1, 99])

    # K forward vs. K backward
    a = next(g)
    pc = a.hexbin(
        ds["K_forward"],
        ds["K_backward"],
       extent=numpy.concatenate([kxrange, kyrange]),
       mincnt=1)
    a.plot(kxrange, -kxrange, 'k--')
    a.set_xlabel("{description:s}\n[{units:s}]".format(**ds["K_forward"].attrs))  
    a.set_ylabel("{description:s}\n[{units:s}]".format(**ds["K_backward"].attrs))  
    a.set_title("Estimating K forward or backward, comparison")
    a.set_xlim(kxrange)
    a.set_ylim(kyrange)
    cbs.append(f.colorbar(pc, ax=a))

    # histogram of K forward / backward differences
    a = next(g)
    (cnts, bins, patches) = a.hist(
        ds["K_forward"]+ds["K_backward"],
        histtype="step",
        bins=100,
        range=kΔrange)
    a.plot([0, 0], [0, cnts.max()], 'k--')
    a.set_xlabel("Sum of K estimates")
    a.set_ylabel("Count")
    a.set_title("Distribution of sum of K estimates")
    a.set_xlim(kΔrange)

    # radiance comparison
    a = next(g)
    pc = a.hexbin(
        ds["nominal_measurand1"],
        ds["nominal_measurand2"],
        extent=(Lmin, Lmax, Lmin, Lmax),
        mincnt=1)
    a.plot([Lmin, Lmax], [Lmin, Lmax], 'k--')
    a.set_xlabel("Radiance {sensor_1_name:s} [units]".format(**ds.attrs))
    a.set_ylabel("Radiance {sensor_2_name:s} [units]".format(**ds.attrs))
    a.set_title("Radiance comparison")
    a.set_xlim(Lmin, Lmax)
    a.set_ylim(Lmin, Lmax)
    cbs.append(f.colorbar(pc, ax=a))

    # Ks vs. Kforward
    a = next(g)
    pc = a.hexbin(
        ds["K_forward"],
        ds["K_forward"]+ds["K_backward"],
        extent=numpy.concatenate([kxrange, kΔrange]),
        mincnt=1)
    a.plot(kxrange, [0, 0], 'k--')
    a.set_xlabel("{description:s}\n[{units:s}]".format(**ds["K_forward"].attrs))  
    a.set_ylabel("Sum of K estimates")
    a.set_title("K difference vs. K forward")
    a.set_xlim(kxrange)
    a.set_ylim(kΔrange)
    cbs.append(f.colorbar(pc, ax=a))

    # K vs. radiance
    a = next(g)
    pc = a.hexbin(
        ds["nominal_measurand1"],
        ds["K_forward"],
        extent=numpy.concatenate([Lxrange, kxrange]),
        mincnt=1)
    a.set_xlabel("Radiance {sensor_1_name:s} [units]".format(**ds.attrs))
    a.set_ylabel("{description:s}\n[{units:s}]".format(**ds["K_forward"].attrs))  
    a.set_title("K vs. measurement")
    a.set_xlim(Lxrange)
    a.set_ylim(kxrange)
    cbs.append(f.colorbar(pc, ax=a))

    # K vs. measurement difference
    a = next(g)
    extremes = [min([LΔrange[0], kxrange[0]]), max([LΔrange[1], kxrange[1]])]
    pc = a.hexbin(
        ds["nominal_measurand2"] - ds["nominal_measurand1"],
        ds["K_forward"],
        extent=numpy.concatenate([LΔrange, kxrange]),
        mincnt=1)
    a.plot(extremes, extremes, 'k--')
    a.set_xlabel("Radiance {sensor_2_name:s} - {sensor_1_name:s} [units]".format(**ds.attrs))
    a.set_ylabel("{description:s}\n[{units:s}]".format(**ds["K_forward"].attrs))  
    a.set_title("K vs. measurement difference")
    a.set_xlim(LΔrange)
    a.set_ylim(kxrange)
    f.colorbar(pc, ax=a)

    for a in ax_all.flat:
        a.grid(axis="both")

    f.suptitle("K stats for pair {sensor_1_name:s}, {sensor_2_name:s}, {time_coverage:s}".format(**ds.attrs)
        + ", channel " + str(ds["channel"].item()) + "\nchannels used to predict: " +
        ", ".join(str(c) for c in ds["K_forward"].attrs["channels_used"]))
    f.subplots_adjust(hspace=0.35)

    pyatmlab.graphics.print_or_show(f, False,
        "harmonisation_K_stats_{sensor_1_name:s}-{sensor_2_name:s}_ch{channel:d}_{time_coverage:s}.".format(
            channel=ds["channel"].item(), **ds.attrs))

def main():
    plot_file_summary_stats(pathlib.Path(p.file))
