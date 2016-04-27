#!/usr/bin/env python3.5

import logging
import argparse

# logging needs to be handled here, therefore also argument parsing
# see http://stackoverflow.com/q/20240464/974555

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description="Experiment with HIRS SRF estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sat", action="store", default="NOAA18")
    parser.add_argument("--channels", action="store", type=int,
                        default=list(range(1, 13)),
                        choices=list(range(1, 13)), nargs="+")

    parser.add_argument("--makelut", action="store_true", default=False,
        help="Make look up table")
    parser.add_argument("--pca", action="store_true", default=False,
        help="Use Principal Component Analysis.")
    parser.add_argument("--factor", action="store", type=float,
        help="Make LUT denser by factor")
    parser.add_argument("--npca", action="store", type=int,
        help="Number of PCs to retain")

    parser.add_argument("--plot_shifted_srf_in_subplots",
        action="store_true", default=False,
        help="Plot SRF with shifts and fragments of IASI, per channel, "
             "for NOAA-19 and -18")

    parser.add_argument("--plot_spectrum_with_channels",
        action="store_true", default=False, 
        help="Plot a selection of IASI spectra along with HIRS SRFs. ")

    parser.add_argument("--spectrum_count", action="store", type=int,
        default=40,
        help="When plotting IASI spectra, how many to plot?")

    parser.add_argument("--seed", action="store", type=int,
        default=0,
        help="Random seed to use when selecting IASI spectra to plot.  "
             "0 means do not seed.")

    parser.add_argument("--plot_srf_rmse",
        action="store_true", default=False,
        help="Visualise cost function involved in recovering shifted SRF")

    parser.add_argument("--db", action="store",
                    choices=["same", "similar", "different"],
                    default=["prediction"], nargs="+",
        help="Use same, similar, or different DB for training/prediction")

    parser.add_argument("--predict_chan", action="store",
                    choices=["single", "all"],
                    default=["all"],
                    nargs="+",
                    help="Predict single-to-single or all-to-single")

    parser.add_argument("--regression_type", action="store",
        choices=["LR", "PLSR"],
        default=["LR"],
        nargs="+",
        help="What kind of regression to use for prediction: linear "
             "regression of partial least squares regression")

    parser.add_argument("--nplsr", action="store", type=int,
        default=12,
        help="When regressing with PLSR, how many components to use")

    parser.add_argument("--plot_errdist_per_localmin",
        action="store_true", default=False)
    parser.add_argument("--plot_bt_srf_shift",
        action="store_true", default=False)

    parser.add_argument("--latrange",
        action="store", type=float, nargs=2, default=(-90, 90),
        help="Latitude range to use in training database")

    parser.add_argument("--estimate_errorprop", action="store_true")
    parser.add_argument("--shift", action="store", type=float, #type=lambda x: float(x)*ureg.nm,
        help="SRF shift [nm].  Use with estimate_errorprop.",
        default=0.0)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--noise_level_target", action="store", type=float,
                        default=0.0)
    parser.add_argument("--noise_level_master", action="store", type=float,
                        default=0.0)
    parser.add_argument("--threads", action="store", type=int,
                        default=16)

    p = parser.parse_args()
    return p

if __name__ == "__main__":
    parsed_cmdline = parse_cmdline()

logging.basicConfig(
    format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
             "%(lineno)s: %(message)s"),
    level=logging.DEBUG if parsed_cmdline.verbose else logging.INFO)

import sys
import os
import re
import math
import datetime
import itertools
import functools
import pickle
import pathlib

import numpy
import numpy.lib.recfunctions
import scipy.stats

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
    
import matplotlib.pyplot

import progressbar
import numexpr
import mpl_toolkits.basemap
import sklearn.cross_decomposition

import typhon.plots
import typhon.math

import pyatmlab.datasets.tovs
import pyatmlab.io
import pyatmlab.config
import pyatmlab.physics
import pyatmlab.graphics
import pyatmlab.stats
import pyatmlab.db

from pyatmlab.constants import micro, centi, tera, nano
from pyatmlab import ureg

unit_specrad_wn = ureg.W / (ureg.m**2 * ureg.sr * (1/ureg.m))
unit_specrad_freq = ureg.W / (ureg.m**2 * ureg.sr * ureg.Hz)

class LUTAnalysis:
    """Helper class centralising all LUT-related approaches
    """
                
    def _get_next_y_for_lut(self, g, sat, channels=range(1, 13)):
        """Helper for get_lookup_table_*
        """
        self.gran = self.iasi.read(g)
        y = self.get_y("specrad_freq")
        y = y.view(dtype=[("specrad_freq", y.dtype, y.shape[2])])
        tb = self.get_tb_channels(sat, channels)
        tbv = tb.view([("ch{:d}".format(c), tb.dtype)
                 for c in channels]).squeeze()
        return numpy.lib.recfunctions.merge_arrays(
            (tbv, y), flatten=True, usemask=False, asrecarray=False)


    def get_lookup_table_pca(self, sat, npc=4, x=2.0, channels=range(1,
                             13), make_new=True):
        axdata = dict(PCA=dict(
                      npc=npc,
                      scale=x,
                      valid_range=(100, 400),
                      fields=["ch{:d}".format(c) for c in channels]))
        try:
            db = pyatmlab.db.LargeFullLookupTable.fromDir(axdata)
        except FileNotFoundError: # create new
            if not make_new:
                raise
            # First read all, then construct PCA, so that all go into PCA.
            # Hopefully I have enough memory for that, or I need to implement
            # incremental PCA.
            logging.info("Found no LUT, creating new")
            y = None
            for g in itertools.islice(self.graniter, *self.lut_slice_build):
                if y is None:
                    y = self._get_next_y_for_lut(g, sat, channels)
                else:
                    y = numpy.hstack([y, self._get_next_y_for_lut(g, sat, channels)])
            #y = numpy.vstack(y_all)
            logging.info("Constructing PCA-based lookup table")
            db = pyatmlab.db.LargeFullLookupTable.fromData(y,
                dict(PCA=dict(
                    npc=npc,
                    scale=x,
                    valid_range=(100, 400),
                    fields=["ch{:d}".format(c) for c in channels])),
                    use_pca=True)
        return db

    def get_lookup_table_linear(self, sat, x=30, channels={2, 5, 8, 9,
                                11}, make_new=True):
        if not make_new:
            raise NotImplementedError("linear LUT always made new")
        db = None
        for g in itertools.islice(self.graniter, *self.lut_slice_build):
            logging.info("Adding to lookup table: {!s}".format(g))
            y = self._get_next_y_for_lut(g, sat, channels)
            if db is None: # first time
                logging.info("Constructing lookup table")
                db = pyatmlab.db.LargeFullLookupTable.fromData(y,
                    {"ch{:d}".format(i+1):
                     dict(range=(tb[..., i][tb[..., i]>0].min()*0.95,
                                 tb[..., i].max()*1.05),
                          mode="linear",
                          nsteps=x)
                        for i in channels},
                        use_pca=False)
            else:
                logging.info("Extending lookup table")
                db.addData(y)
        return db

    def get_lookup_table(self, sat, pca=False, x=30, npc=2,
                         channels=range(1, 13), make_new=True):
        """
        :param sat: Satellite, i.e. "NOAA18"
        :param bool pca: Use pca or not.
        :param x: If not PCA, this is no. of steps per channel.
            If PCA, this is the scale; see
            pyatmlab.db.SmallLookupTable.fromData.
        :param npc: Only relevant for PCA.  How many PC to use.
            Ignored otherwise.
        """
        # construct single ndarray with both tb and radiances, for binning
        # purposes
        db = None
        if pca:
            self.lut = self.get_lookup_table_pca(sat, x=x, npc=npc,
                                        channels=channels,
                                        make_new=make_new)
        else:
            self.lut = self.get_lookup_table_linear(sat, x=x,
                                        channels=channels,
                                        make_new=make_new)
#        out = "/group_workspaces/cems/fiduceo/Users/gholl/hirs_lookup_table/test/test_{:%Y%m%d-%H%M%S}.dat".format(datetime.datetime.now())
#        logging.info("Storing lookup table to {:s}".format(out))
#        db.toFile(out)

    # Using the lookup table, calculate expected differences: taking a set of
    # 12 IASI-simulated NOAA-18, estimate IASI-simulated NOAA-17, -16, -15,
    # etc., and differences to NOAA-18.  For this, we do:
    #
    # - Load lookup table directory
    # - Calculate PCA
    # - Calculate bin
    # - Read full spectrum
    # - Simulate other satellite/satellites
    # - Calculate differences for each channel
    # (squeeze(mypc.Wt[:18, :].T.dot(atleast_2d(mypc.Y[0, :18]).T))*mypc.sigma+mypc.mu) - tb[0, :]

    # arguments to be passed to itertools.islice.  functools.partial
    # doesn't work because it's the first argument I want to vary, and it
    # doesn't take keyword arguments.
    lut_slice_build = (0, None, 2)
    lut_slice_test = (1, None, 2)
    #
    def lut_load(self, tbl):
        self.lut = pyatmlab.db.LargeFullLookupTable.fromDir(tbl)

    _max = 1000
    def lut_get_spectra_for_channels(self, radiances):
        """For a set of 12 NOAA-18 radiances, lookup LUT spectra
        """

        # FIXME: only use the first N until I'm sure about the method
        for spec in self.lut.lookup_all(radiances[:self._max]):
            yield spec["specrad_freq"]

    def lut_simulate_all_hirs(self, radiances):
        """For a set of 12 NOAA-18 radiances, simulate other NOAAs
        """
        N = 12 # thermal channels only
        if radiances.dtype.fields is None:
            radiances = radiances.view([("ch{:d}".format(i+1),
                    radiances.dtype) for i in range(N)]).squeeze()
        spectra = numpy.vstack(list(self.lut_get_spectra_for_channels(radiances)))
        Tb = numpy.zeros(dtype="f8",
                         shape=(len(self.srfs), N, spectra.shape[0]))
        for (i, sat) in enumerate(self.srfs.keys()):
            for (j, srf) in itertools.islice(enumerate(self.srfs[sat]), N):
                L = srf.integrate_radiances(self.iasi.frequency, spectra)
                Tb[i, j, :] = srf.channel_radiance2bt(L)
        return Tb

    def lut_radiance_delta(self, radiances):
        """How far should radiances per satellite deviate from NOAA-18?

        Takes radiances as input, in the form expected by the LUT, i.e. an
        ndarray.
        As output, tuple with (cont, delta), where cont is just the
        radiances in a more convenient way, and delta is the difference
        between radiances through the LUT and the input, i.e. the
        NOAAx-simulated-from-NOAA18 through the NOAA18 LUT.

        Example plot:
          plot(cont[:, 10], delta[:, 10, :].T, 'o')
                ... is ...
          x-axis: reference BT for channel 11
          y-axis: differences for all channels 11
        """
        N = 12
        logging.info("Simulating radiances using lookup-table...")
        hs = self.lut_simulate_all_hirs(radiances)
        cont = radiances[["ch{:d}".format(i+1) for i in range(N)]].view(
                    radiances["ch1"].dtype).reshape(
                        radiances.shape[0], hs.shape[1])
        # FIXME: I made a workaround in lut_get_spectra_for_channels to
        # only take the first so-many radiances as to speed up
        # development.  Need to propagate that here too.  Remove this
        # later!
        cont = cont[:hs.shape[2], :] # FIXME: remove later
        delta = hs - cont.T[numpy.newaxis, :, :]
        return (cont, delta)

    def plot_lut_radiance_delta(self, radiances):
        (cont, delta) = self.lut_radiance_delta(radiances)
        for i in range(12):
            x = cont[:, i]
            y = delta[:, i, :].T
            ii = numpy.argsort(x)
            (f, a) = matplotlib.pyplot.subplots()
            a.plot(x[ii], y[ii, :], 'x')
            a.legend(list(self.srfs.keys()),
                loc="center right",
                bbox_to_anchor=(1.32, 0.5))
            a.set_title("NOAAx-NOAA18, ch. {:d}".format(i+1))
            a.set_xlabel("IASI-simulated NOAA18-HIRS [K]")
            a.set_ylabel("IASI-simulated other - "
                         "IASI-simulated NOAA18 HIRS [K]")
            a.set_xlim(*scipy.stats.scoreatpercentile(x, [0.3, 99.7]))
            a.set_ylim(*scipy.stats.scoreatpercentile(y, [0.3, 99.7]))
            box = a.get_position()
            a.set_position([box.x0, box.y0, box.width * 0.85, box.height])
            pyatmlab.graphics.print_or_show(f, False,
                    "BT_range_LUT_ch{:d}.".format(i+1))
            matplotlib.pyplot.close(f)

    def plot_lut_radiance_delta_all_iasi(self):
        """With LUT, plot radiance deltas for all IASI

        Although plotting radiance deltas does not depend on IASI that's
        what I quickly happen to have available here.  Will change later
        to use any HIRS data from actual NOAA-18 measurements.
        """
        if self.lut is None:
            self.lut_load("/group_workspaces/cems/fiduceo/Users/gholl/hirs_lookup_table/large_similarity_db_PCA_ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12_4_8.0")

        tb_all = []
#        tb_all = [self.get_tb_channels("NOAA18")[:, :, :12].reshape(-1,
#                    12)]
        #for g in self.graniter:
        self.graniter = self.iasi.find_granules()
        for g in itertools.islice(self.graniter, *self.lut_slice_test):
            self.gran = self.iasi.read(g, CLEAR_CACHE=True)
            tb_all.append(self.get_tb_channels("NOAA18")[:, :,
                        :12].reshape(-1, 12))
        tb_all = numpy.vstack(tb_all).view(
            [("ch{:d}".format(i+1), tb_all[0].dtype) for i in range(12)])
        self.plot_lut_radiance_delta(tb_all)

    def lut_get_stats_unseen_data(self, sat="NOAA18"):
        """Using unseen data, check the density of the lookup table.

        I.e. how frequently do we find 0, 1, 2, ... entries for unseen
        data.
        """

        allrad = []
        chans = [int(x[2:]) for x in self.lut.fields]
        # make sure to reset this iterator
        self.graniter = self.iasi.find_granules()
        for g in itertools.islice(self.graniter, *self.lut_slice_test):
            self.gran = self.iasi.read(g, CLEAR_CACHE=True)
            radiances = self.get_tb_channels(sat, chans).reshape(-1, len(chans))
            radiances = numpy.ascontiguousarray(radiances)
            radiances = radiances.view([(fld, radiances[0].dtype)
                                    for fld in self.lut.fields])
            allrad.append(radiances)
        radiances = numpy.vstack(allrad)
        count = []
        stats = numpy.zeros(radiances.size,
            dtype=[("x", radiances.dtype),
                   ("N", "u4"),
                   ("y_mean", radiances.dtype),
                   ("y_std", radiances.dtype),
                   ("y_ptp", radiances.dtype)])
        logging.info("Using LUT to find IASI for {:,} HIRS spectra".format(radiances.size))
        bar = progressbar.ProgressBar(maxval=radiances.size,
                widgets=pyatmlab.tools.my_pb_widget)
        bar.start()
        bar.update(0)
        for (i, dat) in enumerate(radiances):
            stats[i]["x"] = dat
            try:
                cont = self.lut.lookup(dat)
            except KeyError:
                n = 0
            except EOFError as v:
                logging.error("Could not read from LUT: {!s}".format(v))
                n = 0
            else:
                n = cont.size
                for f in radiances.dtype.names:
                    stats[i]["y_mean"][f] = cont[f].mean()
                    stats[i]["y_std"][f] = cont[f].std()
                    stats[i]["y_ptp"][f] = cont[f].ptp()
            stats[i]["N"] = n
            try:
                count[n] += 1
            except IndexError:
                count.extend([0] * (n-len(count)+1))
                count[n] += 1
            bar.update(i+1)
        bar.finish()
        return (radiances, numpy.array(count), stats)

    def lut_visualise_stats_unseen_data(self, sat="NOAA18"):
        # Supposes that LUT is already loaded, and thus containing most of
        # the required meta-information.

        # FIXME: temporary workaround for development speed
        # reload by removing tmpfile if lut_get_stats_unseen_data changes
        tmp = "/work/scratch/gholl/rad_count_stats_{:s}.dat".format(self.lut.compact_summary())
        try:
            (radiances, counts, stats) = pickle.load(open(tmp, "rb"))
        except FileNotFoundError:
            (radiances, counts, stats) = self.lut_get_stats_unseen_data(sat=sat)
            pickle.dump((radiances, counts, stats), open(tmp, "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)
#        (radiances, counts, stats) = pickle.load(open("rad_count_stats.dat", "rb"))
        (f_tothist, a_tothist) = matplotlib.pyplot.subplots(3)
        (f_errperbin, a_errperbin) = matplotlib.pyplot.subplots(2)
        # in a 2 x 2 grid, show error histograms split by number of bins
        hpb_bnd = [0, 10, 50, 100, stats["N"].max()+1]
        hpb_subsets = [(stats["N"]>hpb_bnd[i]) & (stats["N"]<=hpb_bnd[i+1])
                                for i in range(4)]
        (f_deltahistperbin, a_deltahistperbin) = matplotlib.pyplot.subplots(2, 2)
        (f_bthistperbin, a_bthistperbin) = matplotlib.pyplot.subplots(2, 2)
        (N, b, _) = a_tothist[0].hist(stats["N"], numpy.arange(101.0))
        a_tothist[0].set_ylim(0, N[1:].max())
        a_tothist[0].text(0.8, 0.8, "{:,} ({:.1%}) hit empty bins".format(
                        (stats["N"]==0).sum(), (stats["N"]==0).sum()/stats.size),
                  horizontalalignment="center",
                  verticalalignment="center",
                  transform=a_tothist[0].transAxes)
        a_tothist[0].set_xlabel("No. of IASI spectra in bin")
        a_tothist[0].set_ylabel("No. hits")
        a_tothist[0].set_title("No. IASI spectra per HIRS lookup (total {:,})".format(stats.size))
        hasmean = stats["N"] >= 1
        hasstd = stats["N"] >= 5
        biases = numpy.zeros(dtype="f4",
                             shape=(len(radiances.dtype),
                                    len(hpb_subsets)))
        stds = numpy.zeros_like(biases)

        for (i, field) in enumerate(radiances.dtype.names):
            # field = channel in this case
#            a1[1].plot(stats["x"][field][hasmean], stats["y_mean"][field][hasmean]
#                                        -stats["x"][field][hasmean],
#                      '.', label=field)
            k = 0 if int(field[2:]) < 8 else 1
            a_tothist[k+1].hist(stats["y_mean"][field][hasmean] -
                      stats["x"][field][hasmean], 50,
                      histtype="step", cumulative=False,
                      stacked=False,
                      normed=True,
                      label=field)
            a_errperbin[k].plot(stats["N"][hasmean],
                         stats["y_mean"][field][hasmean] -
                         stats["x"][field][hasmean],
                         linestyle="None",
                         marker=".")
            # Plot error histograms for bins with fewest, few, more, and
            # most spectra contained in them.
            for (k, subset) in enumerate(hpb_subsets):
                if not subset.any():
                    continue
                # besides plotting, write some info to the screet
                delta = stats["y_mean"][field][subset] - stats["x"][field][subset]
#                print("{sat:s} {field:>4s}: [{lo:>3d} – {hi:>4d}] "
#                      "{bins:>12s} "
#                      "{count:>6,} "
#                      "Δ {dm:>5.2f} ({ds:>4.2f}) K".format(
#                      sat=sat, field=field,
#                      lo=stats["N"][subset].min(),
#                      hi=stats["N"][subset].max(),
#                      count=subset.sum(),
#                      bins="-".join([str(b.size) for b in self.lut.bins]),
#                      dm=delta.mean(), ds=delta.std()))
                biases[i, k] = delta.mean()
                stds[i, k] = delta.std()
                a_deltahistperbin.flat[k].hist(delta,
                    numpy.linspace(*scipy.stats.scoreatpercentile(delta, [1, 99]), 50),
                    histtype="step", cumulative=False,
                    stacked=False,
                    normed=True,
                    label=field)
                a_bthistperbin.flat[k].hist(stats["y_mean"][field][subset], 50,
                    histtype="step", cumulative=False,
                    stacked=False, normed=True,
                    label=field)
        for (k, subset) in enumerate(hpb_subsets):
            if not subset.any():
                continue
            a_deltahistperbin.flat[k].set_xlabel(r"$\Delta$ BT [K]")
            a_bthistperbin.flat[k].set_xlabel(r"BT [K]")
            #a_deltahistperbin.flat[k].set_xlim(-10, 10)
            ex = numpy.vstack([scipy.stats.scoreatpercentile(
                    stats["y_mean"][x][subset] - stats["x"][x][subset], [5, 95])
                        for x in stats["y_mean"].dtype.names])
            a_deltahistperbin.flat[k].set_xlim(-abs(ex).max(), abs(ex).max())
            a_bthistperbin.flat[k].set_xlim(200, 300)
            a_deltahistperbin.flat[k].grid(axis="x")
            a_bthistperbin.flat[k].grid(axis="both")
            for a in (a_deltahistperbin, a_bthistperbin):
                a.flat[k].set_title("{:d}-{:,} per bin (tot. {:,})".format(
                    stats["N"][subset].min(),
                    stats["N"][subset].max(),
                    subset.sum()))
                a.flat[k].set_ylabel("PD")
                #a.flat[k].grid()
                if k == 1:
                    a.flat[k].legend(loc="lower right",
                        bbox_to_anchor=(1.6, -1.4),
                        ncol=1, fancybox=True, shadow=True)
        a_tothist[1].legend(loc="upper right", bbox_to_anchor=(1.3, 1.4))
        a_tothist[1].set_xlim(-4, 4)
        a_tothist[2].legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        a_tothist[2].set_xlim(-8, 8)
        for k in {0, 1}:
            a_tothist[k+1].set_xlabel(r"$\Delta$ BT [K]")
            a_tothist[k+1].set_ylabel("Density")
            a_tothist[k+1].set_title("HIRS LUT estimate - reference, "
                                     "normalised (NOAA-18)")
            a_errperbin[k].set_xlabel("No. spectra in bin")
            a_errperbin[k].set_ylabel("$\Delta$ BT [K]")
        #a1[1].set_xlim(200, 300)
        for f in {f_tothist, f_errperbin, f_deltahistperbin, f_bthistperbin}:
            f.suptitle("LUT PCA performance, bins {:s}".format(
                "-".join([str(b.size) for b in self.lut.bins])))
            f.tight_layout(rect=[0, 0, 0.83, 0.97])
        pyatmlab.graphics.print_or_show(f_tothist, False,
            "lut_{:s}_test_hists_{:s}.".format(sat, 
                self.lut.compact_summary().replace(".", "_")))
        pyatmlab.graphics.print_or_show(f_deltahistperbin, False,
            "lut_{:s}_test_deltahistperbin_{:s}.".format(sat,
                self.lut.compact_summary().replace(".", "_")))
        pyatmlab.graphics.print_or_show(f_bthistperbin, False,
            "lut_{:s}_test_bthistperbin_{:s}.".format(sat,
                self.lut.compact_summary().replace(".", "_")))
        for f in {f_tothist, f_errperbin, f_deltahistperbin, f_bthistperbin}:
            matplotlib.pyplot.close(f)
#        pyatmlab.graphics.print_or_show(f_errperbin, False,
#            "lut_{:s}_test_errperbin_{:s}.".format(sat, self.lut.compact_summary()))
        return (biases, stds)
        

    def lut_visualise_multi(self, sat="NOAA18"):
#        basedir = pyatmlab.config.conf["main"]["lookup_table_dir"]
#        subname = ("large_similarity_db_PCA_ch1,ch2,ch3,ch4,ch5,ch6,"
#                   "ch7,ch8,ch9,ch10,ch11,ch12_{npc:d}_{fact:.1f}")
        for channels in {range(1, 13), # all thermal
                         range(1, 8), # all CO₂
                         range(1, 7), # almost all CO₂
                         range(4, 8), # subset of CO₂
                         range(11, 13), # only H₂O
                         range(8, 11), # O₃ and window
                         range(8, 13), # all non-CO₂
                        }:
            D = {}
            for npc in ([i for i in (3, 4) if i < len(channels)] or
                        [len(channels)]):
                for fact in (1.0, 2.0, 4.0, 8.0):
                    logging.info("Considering LUT with npc={:d}, "
                                "fact={:.1f}, ch={!s}".format(npc, fact, list(channels)))
    #                p = pathlib.Path(basedir) / (subname.format(npc=npc, fact=fact))
                    try:
                        self.get_lookup_table(sat="NOAA18", pca=True, x=fact,
                            npc=npc, channels=channels, make_new=False)
                    except FileNotfoundError:
                        logging.error("Does not exist yet, skipping")
                        continue
                    D[tuple(b.size for b in self.lut.bins)] = self.lut_visualise_stats_unseen_data(sat=sat)
            (f1, a1) = matplotlib.pyplot.subplots(2, 2)
            (f2, a2) = matplotlib.pyplot.subplots(2, 2)
            sk = sorted(D.keys())
            stats = numpy.concatenate([numpy.dstack(D[x])[..., numpy.newaxis] for x in sk], 3)
            x = numpy.arange(len(sk))
            subtitlabs = ("<10", "11-50", "51-100", ">100")
            for k in range(a1.size):
                for c in range(stats.shape[0]):
                    a = a1 if channels[c] in range(1, 8) else a2
                    a.flat[k].errorbar(x, stats[c, k, 0, :],
                         yerr=stats[c, k, 1, :],
                         marker="^",
                         linestyle="None",
                         label=channels[c])#"ch{:d}".format(c+1))
                for (a, ymin, ymax) in ((a1, -2, 2), (a2, -5, 5)):
                    a.flat[k].set_xticks(x)
                    a.flat[k].set_xticklabels([str(s) for s in sk],
                                              rotation=25,
                                              size="x-small")
                    a.flat[k].set_title("{:s} per bin".format(subtitlabs[k]))
                    #a.flat[k].set_title("Subset {:d} (FIXME)".format(k+1))
                    a.flat[k].set_xlabel("Bins")
                    a.flat[k].set_ylabel("Mean/std diff. [K]")
                    a.flat[k].set_xlim(x[0]-0.5, x[-1]+0.5)
                    a.flat[k].grid(axis="y")
                    a.flat[k].set_ylim(ymin, ymax)
                    if k == 1:
                        a.flat[k].legend(loc="lower right",
                                bbox_to_anchor=(1.6, -1.4),
                                ncol=1, fancybox=True, shadow=True)
            for (f, lb) in zip((f1, f2), ("ch1-7", "ch8-12")):
                f.suptitle("Mean and s.d. of $\Delta K$ for different binnings")
                f.tight_layout(rect=[0, 0, 0.85, 0.97])
                pyatmlab.graphics.print_or_show(f, False,
                    "lut_{:s}_{!s}_test_perf_all_{:s}.".format(sat, 
                        ",".join(str(x) for x in channels), lb))
                matplotlib.pyplot.close(f)
     

    def estimate_pca_density(self, sat="NOAA18", all_n=[5], bin_scales=[1],
            channels=slice(12), nbusy=4):
        """How efficient is PCA binning?

        Investigates how sparse a PCA-based binning lookup table is.
        This first calculates PCA 

        :param sat: Satellite to use
        :param all_n: Number of PCs to use in lookup table.
            May be an array, will loop through all.
        :param bin_scale: Scaling factor for number of bins per PC.
            Number of bins is proportional to the fraction of variability
            explained by each PC.
        """

        pca = self.get_pca_channels(sat, channels)

        for bin_scale in bin_scales:
            nbins = numpy.ceil(pca.fracs*100*bin_scale)
            bins = [numpy.linspace(pca.Y[:, i].min(), pca.Y[:, i].max(), max(p, 2))
                for (i, p) in enumerate(nbins)]
            for n in all_n:
                logging.info("Binning, scale={:.1f}, n={:d}".format(
                    bin_scale, n))
                bnd = pyatmlab.stats.bin_nd(
                    [pca.Y[:, i] for i in range(n)],
                    bins[:n])
                (no, frac, lowest, med, highest) = self._calc_bin_stats(bnd)
                logging.info("PCA {:d} comp., {:s} bins/comp: {:.3%} {:d}/{:d}/{:d}".format(
                      n, "/".join(["{:d}".format(x) for x in bnd.shape]),
                      frac, lowest, med, highest))
                nos = numpy.argsort(no)
                busiest_bins = bnd.ravel()[nos[-nbusy:]].tolist()
                logging.info("Ranges in {nbusy:d} busiest bins:".format(
                                nbusy=nbusy))
                print("{:>4s} {:>5s}/{:>5s}/{:>5s} {:>5s} {:>5s}".format(
                      "Ch.", "min", "mean", "max", "PTP", "STD"))
                for i in range(bt2d.shape[1]):
                    for b in busiest_bins:
                        print("{:>4d} {:>5.1f}/{:>5.1f}/{:>5.1f} {:>5.2f} "
                              "{:>5.2f}".format(i+1,
                                bt2d[b, i].min(),
                                bt2d[b, i].mean(),
                                bt2d[b, i].max(),
                                bt2d[b, i].ptp(),
                                bt2d[b, i].std()))
                del bnd
                del no
                del nos


#                    for z in zip(range(1, cont.shape[1]+1), cont.min(0),
#                                 cont.mean(0), cont.max(0),
#                                 cont.ptp(0), cont.std(0)):
#                        print("{:>4d} {:>5.1f}/{:>5.1f}/{:>5.1f} {:>5.2f} "
#                              "{:>5.2f}".format(*z))

    def estimate_optimal_channel_binning(self, sat="NOAA18", N=5, p=20):
        """What HIRS channel combi optimises variability?

        :param sat: Satellite to use
        :param N: Number of channels in lookup table
        :param p: Number of bins per channel

        Note that as this method aims to choose an optimal combination of
        channels using rectangular binning (no channel differences), it
        does not use PCA.  For that, see estimate_pca_density.
        """
        bt = self.get_tb_channels(sat)
        btflat = [bt[..., i].ravel() for i in range(bt.shape[-1])]
        bins = [scipy.stats.scoreatpercentile(b[b>0], numpy.linspace(0, 100, p))
                    for b in btflat]
        #bins = numpy.linspace(170, 310, 20)
        chans = range(12) # thermal channels only
        tot = int(scipy.misc.comb(12, N))
        logging.info("Studying {:d} combinations".format(tot))
        for (k, combi) in enumerate(itertools.combinations(chans, N)):
            bnd =  pyatmlab.stats.bin_nd([btflat[i] for i in combi], 
                                         [bins[i] for i in combi])
            (frac, lowest, med, highest) = self._calc_bin_stats(bnd)
            logging.info("{:d}/{:d} channel combination {!s}: {:.3%} {:d}/{:d}/{:d}".format(
                  k, tot, combi, frac, lowest, med, highest))


    @staticmethod
    def _calc_bin_stats(bnd):
        # flattened count
        no = numpy.array([b.size for b in bnd.ravel()])
        #
        frac = (no>0).sum() / no.size
        #
        lowest = no[no>0].min()
        highest = no.max()
        med = int(numpy.median(no[no>0]))
        return (no, frac, lowest, med, highest)

class IASI_HIRS_analyser(LUTAnalysis):
    colors = ("black brown orange magenta burlywood tomato indigo "
              "moccasin cyan teal khaki tan steelblue "
              "olive gold darkorchid pink midnightblue "
              "crimson orchid olive chocolate sienna").split()
    styles = ("solid", "dashed", "dash_dot", "dotted")
    markers = "os^p*hv<>"

    allsats = (pyatmlab.datasets.tovs.HIRS2.satellites |
               pyatmlab.datasets.tovs.HIRS3.satellites |
               pyatmlab.datasets.tovs.HIRS4.satellites)
    allsats = {re.sub(r"0(\d)", r"\1", sat).upper() for sat in allsats}

    x = dict(#converter=dict(
#                wavelength=pyatmlab.physics.frequency2wavelength,
#                wavenumber=pyatmlab.physics.frequency2wavenumber,
#                frequency=lambda x: x),
#             factor=dict(
#                wavelength=micro,
#                wavenumber=centi,
#                frequency=tera),
            unit=dict(
                wavelength=ureg.micrometer,
                wavenumber=1/ureg.centimeter,
                frequency=ureg.terahertz),
#             label=dict(
#                wavelength="Wavelength [µm]",
#                wavenumber="Wave number [cm^-1]",
#                frequency="Frequency [THz]"))
            )
                
    _iasi = None
    @property
    def iasi(self):
        if self._iasi is None:
            self._iasi = pyatmlab.datasets.tovs.IASINC(name="iasinc")
        return self._iasi

    @iasi.setter
    def iasi(self, value):
        self._iasi = value

    _graniter = None
    @property
    def graniter(self):
        if self._graniter is None:
            self._graniter = self.iasi.find_granules()
        return self._graniter

    @graniter.setter
    def graniter(self, value):
        self._graniter = value

    _gran = None
    @property
    def gran(self):
#        if self._gran is None:
#            self._gran = self.iasi.read(next(self.graniter))
        return self._gran

    @gran.setter
    def gran(self, value):
        self._gran = value

    def __init__(self, mode="Viju"):
        #logging.info("Finding and reading IASI")
        if mode == "iasinc":
            self.iasi = pyatmlab.datasets.tovs.IASINC(name="iasinc")
            self.choice = [(38, 47), (37, 29), (100, 51), (52, 11)]
            #self.gran = self.iasi.read(next(self.graniter))
        elif mode == "iasisub":
            self.iasi = pyatmlab.datasets.tovs.IASISub(name="iasisub")
        self.graniter = self.iasi.find_granules()

        hconf = pyatmlab.config.conf["hirs"]
        srfs = {}
        for sat in self.allsats:
            try:
                (hirs_centres, hirs_srf) = pyatmlab.io.read_arts_srf(
                    hconf["srf_backend_f"].format(sat=sat),
                    hconf["srf_backend_response"].format(sat=sat))
            except FileNotFoundError as msg:
                logging.error("Skipping {:s}: {!s}".format(
                              sat, msg))
            else:
                srfs[sat] = [pyatmlab.physics.SRF(f, w) for (f, w) in hirs_srf]
        self.srfs = srfs
#        for coor in self.choice:
#            logging.info("Considering {coor!s}: Latitude {lat:.1f}°, "
#                "Longitude {lon:.1f}°, Time {time!s}, SZA {sza!s})".format(
#                coor=coor, lat=self.gran["lat"][coor[0], coor[1]],
#                lon=self.gran["lon"][coor[0], coor[1]],
#                time=self.gran["time"][coor[0], coor[1]].astype(datetime.datetime),
#                sza=self.gran["solar_zenith_angle"][coor[0], coor[1]]))

    def get_y(self, unit, return_label=False):
        """Get measurement in desired unit
        """
        specrad_wavenum = self.gran["spectral_radiance"]
        if unit.lower() in {"tb", "bt"}:
            y = self.get_tb_spectrum()
            y_label = "Brightness temperature [K]"
        elif unit == "specrad_freq":
            y = pyatmlab.physics.specrad_wavenumber2frequency(specrad_wavenum)
            y_label = "Spectral radiance [W m^-2 sr^-1 Hz^-1]"
        elif unit == "specrad_wavenum":
            y = specrad_wavenum
            y_label = "Spectral radiance [W m^-2 sr^-1 m]"
        else:
            raise ValueError("Unknown unit: {:s}".format(unit))
        return (y[..., :8461], y_label) if return_label else y[..., :8461]

    def get_tb_spectrum(self):
        """Calculate spectrum of brightness temperatures
        """
        specrad_freq = self.get_y(unit="specrad_freq")
#        specrad_wavenum = self.gran["spectral_radiance"]
#        specrad_freq = pyatmlab.physics.specrad_wavenumber2frequency(
#                            specrad_wavenum)

        with numpy.errstate(divide="warn", invalid="warn"):
            logging.debug("...converting radiances to BTs...")
            Tb = pyatmlab.physics.specrad_frequency_to_planck_bt(
                specrad_freq, self.iasi.frequency)
        return Tb

    @typhon.utils.cache.mutable_cache(maxsize=20)
    def get_tb_channels(self, sat, channels=range(1, 13), srfshift=None,
                        specrad_f=None):
        """Get brightness temperature for channels
        """
        #chan_nos = (numpy.arange(19) + 1)[channels]
#        specrad_wn = self.gran["spectral_radiance"]
#        specrad_f = pyatmlab.physics.specrad_wavenumber2frequency(
#                            specrad_wn)
        if srfshift is None:
            srfshift = {}
        if specrad_f is None:
            specrad_f = self.get_y(unit="specrad_freq")
        Tb_chans = numpy.zeros(dtype=numpy.float32,
                               shape=specrad_f.shape[:-1] +
                               (len(channels),)) * ureg.K
        for (i, c) in enumerate(channels):
            srf = self.srfs[sat][c-1]
            if c in srfshift:
                srf = srf.shift(srfshift[c])
                logging.debug("Calculating channel Tb {:s}-{:d}{:+.2~}".format(sat, c, srfshift[c]))
        #for (i, srf) in enumerate(self.srfs[sat]):
            else:
                logging.debug("Calculating channel Tb {:s}-{:d}".format(sat, c))
            #srfobj = pyatmlab.physics.SRF(freq, weight)
            L = srf.integrate_radiances(self.iasi.frequency, specrad_f)

            Tb_chans[..., i] = srf.channel_radiance2bt(L)
        return Tb_chans

    def get_pca_channels(self, sat, channels=slice(12), ret_y=False):
        bt = self.get_tb_channels(sat)
        bt2d = bt.reshape(-1, bt.shape[2])
        bt2d = bt2d[:, channels]
        btok = (bt2d>0).all(1)

        logging.info("Calculating PCA")
        pca = matplotlib.mlab.PCA(bt2d[btok, :])
        Ys = numpy.ma.zeros(shape=bt2d.shape, dtype="f4")
        Ys.mask = numpy.zeros_like(Ys, dtype="bool")
        Ys.mask[~btok] = True
        Ys[btok, :] = pca.Y
        Y = Ys.reshape(bt.shape)

        if ret_y:
            return (pca, Y)
        else:
            return pca

    def plot_full_spectrum_with_all_channels(self, sat,
            y_unit="Tb", x_quantity="frequency",
            selection=None):
#        Tb_chans = self.get_tb_for_channels(hirs_srf)

        (y, y_label) = self.get_y(y_unit, return_label=True)
        y = y.reshape(-1, y.shape[-1])
        spectra = y.reshape(-1, self.iasi.frequency.size)[selection, :]
        logging.info("Visualising")
        (f, a_spectrum) = matplotlib.pyplot.subplots()
        a_srf = a_spectrum.twinx()
        for ch in range(1, 13):
            self._plot_srf_with_spectra(ch, {sat}, x_quantity,
                                   spectra, a_spectrum, a_srf,
                                   shift={})

        # Plot spectrum
        #a.plot(iasi.frequency, specrad_freq[i1, i2, :])
        #x = getattr(self.iasi, x_quantity).to(self.x["unit"][x_quantity])
        #x = self.iasi.frequency.to(x_unit, "sp")
#        for c in self.choice:
#            a.plot(x, y[c[0], c[1], :])
        #a.plot(iasi.wavelength, Tb[i1, i2, :])
        a_spectrum.set_ylabel(y_label)
        a_spectrum.set_xlabel("{:s} [{:~}]".format(x_quantity, self.x["unit"][x_quantity]))
        a_spectrum.set_title("Some IASI spectra with nominal {sat} HIRS"
                        " SRFs".format(sat=sat))

#        # Plot channels
#        a2 = a.twinx()
        for (i, srf) in enumerate(self.srfs[sat]):
            if i>=12:
                break
            #wl = pyatmlab.physics.frequency2wavelength(srf.f)
#            x = getattr(srf, x_quantity).to(self.x["unit"][x_quantity])
#            a2.plot(x, 0.8 * srf.W/srf.W.max(), color="black")
            nomfreq = srf.centroid()
            nomfreq_x = nomfreq.to(self.x["unit"][x_quantity], "sp")
            #nomfreq = pyatmlab.physics.frequency2wavelength(srf.centroid())
            #nomfreq = freq[numpy.argmax(srf.W)]
            #nomfreq = wl[numpy.argmax(weight)]

            # Seems that matplotlib.text.Text.get_unitless_position fails
            # when I keep the unit there
            a_srf.text(nomfreq_x.m, 1.07+(0.06 if (i+1) in {2,4,6} else 0), "{:d}".format(i+1),
                       backgroundcolor="white")

        a_srf.set_ylim(0, 1.2)

#        a.bar(hirs_centres, Tb_chans[self.choice[0], self.choice[1], :], width=2e11, color="red", edgecolor="black",
#              align="center")

        pyatmlab.graphics.print_or_show(f, False,
            "iasi_with_hirs_srf_{:s}_{:s}_{:s}.".format(sat, x_quantity, y_unit))

    @staticmethod
    def _norm_order(x, y):
        """Make sure both are increasing
        """

        ii = numpy.argsort(x)
        return (x[ii], y[ii])

    def _plot_srf_for_sat(self, sat, ch, srf, x_quantity,
            ax, color, linestyle="solid", shift=0.0*ureg.um):
        """Plot SRF into axis.
        """
        x = srf.frequency.to(self.x["unit"][x_quantity], "sp")
        y = srf.W/srf.W.max()
        (x, y) = self._norm_order(x, y)
        ax.plot(x, y, label=(sat if shift.m==0 else "{:s}{:+.2~}".format(sat,shift)),
                color=color, linestyle=linestyle)
        pyatmlab.io.write_data_to_files(
            numpy.vstack(
                (#x/self.x["factor"][x_quantity],
                 x.to(self.x["unit"][x_quantity], "sp"),
                 srf.W/srf.W.max())).T,
            "SRF_{:s}_ch{:d}_{:s}{:+.2~}".format(
                    sat, ch, x_quantity, shift))


    def _plot_srfs(self, ch, sats, x_quantity="wavelength", ax=None,
                    shift={}):
        """For channel `ch` on satellites `sats`, plot SRFs.

        Use axes `ax`.
        """

        for (color, sat) in zip(self.colors, sats):
            srf = self.srfs[sat][ch-1]
            self._plot_srf_for_sat(sat, ch, srf, x_quantity, ax, color)
            if sat in shift:
                self._plot_srf_for_sat(sat, ch, srf.shift(shift[sat]),
                    x_quantity, ax, color, linestyle="dashed",
                    shift=shift[sat])
#            srf = self.srfs[sat][ch-1]
#            x = srf.frequency.to(self.x["unit"][x_quantity], "sp")
#            y = srf.W/srf.W.max()
#            (x, y) = self._norm_order(x, y)
#            ax.plot(x, y, label=sat, color=color)
#            pyatmlab.io.write_data_to_files(
#                numpy.vstack(
#                    (#x/self.x["factor"][x_quantity],
#                     x.to(self.x["unit"][x_quantity], "sp"),
#                     srf.W/srf.W.max())).T,
#                "SRF_{:s}_ch{:d}_{:s}".format(sat, ch, x_quantity))

    def _plot_bts_for_chan(self, ch, sats, x_quantity="wavelength", ax=None):
        """For channel `ch` on satellites `sats`, plot TBs.

        """

        raise NotImplementedError("Pending!")

        # Pending a proper solution: WHAT radiances should I plot?

        for (color, sat) in enumerate(sats):
            srf = self.srfs[sat]
            x = numpy.atleast_1d(srf[i].centroid().to(self.x["unit"][x_quantity], "sp")),
            y = numpy.atleast_2d(
                    [Tb_chans[sat][c[0], c[1], i] for c in self.choice]),
            ax.plot(x, y,
                   markerfacecolor=color,
                   markeredgecolor="black",
                   marker="o", alpha=0.5,
                   markersize=10, linewidth=1.5,
                   zorder=10)

    def _plot_spectra(self, spectra, x_quantity, ax):
        """Small helper plotting spectra into axis
        """
        for spectrum in spectra:
            (x, y) = self._norm_order(
                self.iasi.frequency.to(self.x["unit"][x_quantity], "sp"),
                spectrum)
            ax.plot(x, y,
                    linewidth=0.1, zorder=5)

    def _plot_srf_with_spectra(self, ch, sats, x_quantity,
                               spectra, ax_spectrum, ax_srf,
                               shift={}):
        """For channel `ch`, on satellites `sats`, plot SRFs with spectra.

        Use axes `ax_spectrum` and ax_srf.
        """

        self._plot_srfs(ch, sats, x_quantity, ax=ax_srf, shift=shift)
        #self._plot_bts_for_chan(ch, sats, x_quantity, ax)
        self._plot_spectra(spectra, x_quantity, ax_spectrum)

    
    def _get_freq_range_for_ch(self, ch, sats):
        """In the interest of plotting, get frequency range for channel.

        Considers satellites `sats`
        """

        (freq_lo, freq_hi) = (100*ureg.THz, 0*ureg.Hz)

        for sat in sats:
            srf = self.srfs[sat]
            subset = srf[ch-1].W>srf[ch-1].W.max()/100
            freq_lo = min(freq_lo, srf[ch-1].frequency[subset].min())
            freq_hi = max(freq_hi, srf[ch-1].frequency[subset].max())

        freq_lo = max(freq_lo, self.iasi.frequency.min())
        freq_hi = min(freq_hi, self.iasi.frequency.max())

        return (freq_lo, freq_hi)

    def plot_srf_all_sats(self, x_quantity="wavelength", y_unit="TB",
                          selection=None):
        """Plot part of the spectrum with channel SRF for all sats
        """

        #hirs_srf = {}

        (y, y_label) = self.get_y(y_unit, return_label=True)
#        Tb_chans = {}
#        for (sat, srf) in self.srfs.items():
#            Tb_chans[sat] = self.get_tb_channels(sat)

        #spectra = [y[c[0], c[1], :] for c in self.choice]
        y = y.reshape(-1, y.shape[-1])
        spectra = y.reshape(-1, self.iasi.frequency.size)[selection, :]
        for i in range(12):
            ch = i + 1
            (f, a) = matplotlib.pyplot.subplots()
            a.set_ylabel(y_label)
            a.set_xlabel("{:s} [{:~}]".format(x_quantity, self.x["unit"][x_quantity]))
            a.set_title("Some IASI spectra with different HIRS SRF (ch."
                        "{:d})".format(ch))
            a.grid(axis="y", which="both")
            a2 = a.twinx()

            self._plot_srf_with_spectra(ch, sorted(self.srfs.keys()),
                spectra=spectra,
                x_quantity=x_quantity, ax_spectrum=a, ax_srf=a2)

            box = a.get_position()
            for ax in (a, a2):
                ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

            a2.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
            (freq_lo, freq_hi) = self._get_freq_range_for_ch(ch, self.srfs.keys())
            y_in_view = [spectrum[(self.iasi.frequency>freq_lo) &
                                  (self.iasi.frequency<freq_hi)]
                         for spectrum in spectra]
            a.set_ylim(min([yv.min() for yv in y_in_view]).m,
                       max([yv.max() for yv in y_in_view]).m)
            x_lo = freq_lo.to(self.x["unit"][x_quantity], "sp")
            x_hi = freq_hi.to(self.x["unit"][x_quantity], "sp")
            a.set_xlim(min(x_lo, x_hi).m, max(x_lo, x_hi).m)
            a2.set_xlim(min(x_lo, x_hi).m, max(x_lo, x_hi).m)
            pyatmlab.graphics.print_or_show(f, False,
                    "iasi_with_hirs_srfs_ch{:d}_{:s}_{:s}.".format(
                        ch, x_quantity, y_unit))

    def plot_Te_vs_T(self, sat):
        """Plot T_e as a function of T

        Based on Weinreb (1981), plot T_e as a function of T.  For
        details, see pyatmlab.physics.estimate_effective_temperature.
        """
        hconf = pyatmlab.config.conf["hirs"]
        (hirs_centres, hirs_srf) = pyatmlab.io.read_arts_srf(
            hconf["srf_backend_f"].format(sat=sat),
            hconf["srf_backend_response"].format(sat=sat))

        T = numpy.linspace(150, 300, 1000)
        (fig, a) = matplotlib.pyplot.subplots()
        for (i, (color, f_c, (f, W))) in enumerate(
                zip(self.colors, hirs_centres, hirs_srf)):
            Te = pyatmlab.physics.estimate_effective_temperature(
                    f[numpy.newaxis, :], W, f_c, T[:, numpy.newaxis])
            wl_um = pyatmlab.physics.frequency2wavelength(f_c)/micro
            a.plot(T, (Te-T), color=color,
                   label="ch. {:d} ({:.2f} µm)".format(i+1, wl_um))
            if (Te-T).max() > 0.1 and i!=18:
                print("Max diff ch. {:d} ({:.3f} µm) on {:s}: {:.4f}K".format(
                      i+1, wl_um, sat, (Te-T).max()))
        a.set_xlabel("BT [K]")
        a.set_ylabel("BT deviation Te-T [K]")
        a.set_title("BT deviation without correction, {:s}".format(sat))
        box = a.get_position()
        a.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        a.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
        pyatmlab.graphics.print_or_show(fig, False,
                "BT_Te_corrections_{:s}.".format(sat))

    def plot_channel_BT_deviation(self, sat):
        """Plot BT deviation for mono-/polychromatic Planck
        """

#        hconf = pyatmlab.config.conf["hirs"]
#        (centres, srfs) = pyatmlab.io.read_arts_srf(
#            hconf["srf_backend_f"].format(sat=sat),
#            hconf["srf_backend_response"].format(sat=sat))
#        srfs = [pyatmlab.physics.SRF(f, w) for (f, w) in srfs]

        (fig, a) = matplotlib.pyplot.subplots(2, sharex=True)
        for (i, color, srf) in zip(range(20), self.colors, self.srfs[sat]):
            T = numpy.linspace(srf.T_lookup_table.min(),
                               srf.T_lookup_table.max(),
                               5*srf.T_lookup_table.size)
            L = srf.blackbody_radiance(T)
            freq = srf.centroid()
            wl_um = pyatmlab.physics.frequency2wavelength(freq)/micro
            lab = "ch. {:d} ({:.2f} µm)".format(i+1, wl_um)
            a[0].plot(T[::20], (srf.channel_radiance2bt(L)-T)[::20],
                      color=color, label=lab)
            a[1].plot(T,
                      pyatmlab.physics.specrad_frequency_to_planck_bt(L, freq)-T,
                      color=color, label=lab)
        a[0].set_title("Deviation with lookup table")
        a[1].set_title("Deviation with monochromatic approximation")
        a[1].set_xlabel("Temperature [K]")
        a[1].legend(loc='center left', bbox_to_anchor=(0.8, 1))
        box = a[0].get_position()
        for ax in a:
            ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
            ax.set_ylabel("Delta T [K]")
        fig.subplots_adjust(hspace=0)
        pyatmlab.graphics.print_or_show(fig, False,
                "BT_channel_approximation_{:s}.".format(sat))


    def calc_bt_srf_shift(self, satellite, channel, shift):
        """Calculate BT under SRF shifts

        Per satellite, channel, and shift.

        Expects that self.gran is already set.

        :param str satellite: Satellite, i.e., NOAA18
        :param int channel: Channel, i.e., 11
        :param ndarray shift: Shift in appropriate unit (through ureg).
        """

        y = self.get_y("specrad_freq")

        srf_nom = self.srfs[satellite][channel-1] # index 0 -> channel 1 etc.

        if y.size/y.shape[-1] > 1e5:
            logging.debug("Integrating {:,} spectra to radiances".format(
                y.size//y.shape[-1]))

        L_nom = srf_nom.integrate_radiances(self.iasi.frequency, y)

        if y.size/y.shape[-1] > 1e5:
            logging.debug("Converting {:,} radiances to brightness temperatures".format(
                y.size//y.shape[-1]))

        bt_nom = srf_nom.channel_radiance2bt(L_nom)

        yo = numpy.zeros(shape=bt_nom.shape + shift.shape)

        bar = progressbar.ProgressBar(maxval=len(shift),
                widgets=pyatmlab.tools.my_pb_widget)
        bar.start()

        logging.info("Shifting {:,} spectra by {:d} values between "
            "{:+~} and {:+~}".format(y.size//y.shape[-1], len(shift), shift[0], shift[-1]))
        for (i, sh) in enumerate(shift):
            srf_new = srf_nom.shift(sh)
            L_new = srf_new.integrate_radiances(self.iasi.frequency, y)
            bt_new = srf_new.channel_radiance2bt(L_new)
            yo.reshape(*yo.shape[:2], -1)[:, :, i] = bt_new
            bar.update(i+1)
        bar.finish()

        return yo
       
    def plot_bt_srf_shift(self, satellite, channel):
        """Plot BT changes due to SRF shifts

        :param str satellite: Satellite, i.e., NOAA18
        :param int channel: Channel, i.e., 11
        """

        if self.gran is None:
            self.gran = self.iasi.read(next(self.graniter))
        
        N = 71
        dx = numpy.linspace(-70, 70, N) * ureg.nm
        #dx = numpy.linspace(-30, 30, 7) * ureg.nm
        #dx = dx_many[70:140:10]
        # diff in um for human-readability
#        d_um = (pyatmlab.physics.frequency2wavelength(
#                    self.srfs[satellite][channel-1].centroid()) 
#              - pyatmlab.physics.frequency2wavelength(
#                    self.srfs[satellite][channel-1].centroid()-dx))
        sh = self.calc_bt_srf_shift(satellite, channel, dx)
        nsh = sh[:, :, sh.shape[2]//2]
        dsh = sh - nsh[:, :, numpy.newaxis]
        btbins = numpy.linspace(nsh[nsh>0].min(), nsh.max(), 50)
        ptiles = numpy.array([5, 25, 50, 75, 95])

        scores = numpy.zeros(shape=(btbins.size, ptiles.size, dsh.shape[2]))
        (f, a) = matplotlib.pyplot.subplots()
        for (i, ii) in enumerate(numpy.linspace(1, N-1, 7, dtype=int)):
            q = dx[ii].to(ureg.nm, "sp")
            q = int(q.m)*q.u
            typhon.plots.plot_distribution_as_percentiles(a, nsh.ravel(),
                dsh[:, :, ii].ravel(), bins=btbins, color=self.colors[i],
                label="{:+3~}".format(q),
                ptile_to_legend=True if i==0 else False)

        a.set_title("Radiance change distribution per radiance for shifted SRF, HIRS, {:s} ch. {:d}".format(satellite, channel))
        box = a.get_position()
        a.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        a.legend(loc="center left",ncol=1,bbox_to_anchor=(1,0.5))
        a.set_xlabel("BT [K]")
        a.set_ylabel(r"$\Delta$ BT [K]")
        a.grid(axis="both")
        pyatmlab.graphics.print_or_show(f, False,
            "srf_shifted_dbt_hist_per_radiance_HIRS_{:s}-{:d}.".format(satellite, channel))

        (f, a) = matplotlib.pyplot.subplots()
        typhon.plots.plot_distribution_as_percentiles(a,
            numpy.tile(dx, (dsh.shape[0], dsh.shape[1], 1)).ravel(), dsh.ravel(),
            nbins=50, color="black", label="shift")
        a.set_title("Radiance change distribution per HIRS SRF shift, {:s} ch.  {:d}".format(satellite, channel))
        a.set_xlabel(r"shift [nm]")
        a.set_ylabel(r"$\Delta$ BT [K]")
        a.grid(axis="both")
        pyatmlab.graphics.print_or_show(f, False,
            "srf_shifted_dbt_hist_per_shift_HIRS_{:s}-{:d}.".format(satellite, channel))

    def _prepare_map(self):
        f = matplotlib.pyplot.figure(figsize=(12, 8))
        a = f.add_subplot(1, 1, 1)
        m = mpl_toolkits.basemap.Basemap(projection="cea",
                llcrnrlat=self.gran["lat"].min()-30,
                llcrnrlon=self.gran["lon"].min()-30,
                urcrnrlat=self.gran["lat"].max()+30,
                urcrnrlon=self.gran["lon"].max()+30,
                resolution="c")
#        m = mpl_toolkits.basemap.Basemap(projection="moll", lon_0=0,
#                resolution="c", ax=a)
        m.drawcoastlines()
        m.drawmeridians(numpy.arange(-180, 180, 30))
        m.drawparallels(numpy.arange(-90, 90, 30))
        return (f, a, m)

    def find_matching_hirs_granule(self, h, satname):
        dt1 = self.gran["time"].min().astype(datetime.datetime)
        dt2 = self.gran["time"].max().astype(datetime.datetime)
        return next(h.find_granules(dt1, dt2, satname=satname.lower()))

    def read_matching_hirs_granule(self, h, satname):
        return h.read(self.find_matching_hirs_granule(h, satname.lower()))

    _bt_simul = None
    def map_with_hirs(self, h, satname, c, cmap="viridis"):
        """Plot map with simulated and real HIRS

        Needs hirs object, locates closest granule.
        """

        hg = self.read_matching_hirs_granule(h, satname)
        (f, a, m) = self._prepare_map()

        if self._bt_simul is None:
            bt_simul = self.get_tb_channels(satname)
            self._bt_simul = bt_simul
        else:
            bt_simul = self._bt_simul

        # channel data for iasi-simulated
        pcs = m.pcolor(self.gran["lon"], self.gran["lat"],
                       bt_simul[:, :, c-1], latlon=True, cmap=cmap)

        # channel data for actually measured
        # workaround for http://stackoverflow.com/q/12317155/974555
        safeline = (hg["lon"].data>0).all(1)
        # workaround for numpy/numpy#6723
        X = hg["bt"]
        X._fill_value = X._fill_value.flat[0]
        pcr = m.pcolor(hg["lon"][safeline, :], hg["lat"][safeline, :],
                       X[safeline, :, c-1], latlon=True, cmap=cmap)

        lo = X[safeline, :, c-1].min()
        hi = X[safeline, :, c+1].max()
        pcs.set_clim(lo, hi)
        pcr.set_clim(lo, hi)

        cb = f.colorbar(pcs)
        cb.set_label("BT ch. {:d} [K]".format(c))

        dt1 = self.gran["time"].min().astype(datetime.datetime)
        dt2 = self.gran["time"].max().astype(datetime.datetime)
        a.set_title("{satname:s} hirs-{ch:d}, iasi-simulated or real\n"
                     "{dt1:%y-%m-%d %h:%m} -- {dt2:%H:%M}".format(
                        satname=satname, ch=c, dt1=dt1, dt2=dt2))
        pyatmlab.graphics.print_or_show(f, False,
            "map_BTHIRS_real_simul_ch{:d}_{:%Y%m%d%H%M%S}.".format(c,dt1))

    def map_with_hirs_pca(self, h, satname, cmap="viridis"):

        dt1 = self.gran["time"].min().astype(datetime.datetime)
        dt2 = self.gran["time"].max().astype(datetime.datetime)
        hg = self.read_matching_hirs_granule(h, satname)
        X = hg["bt"][:, :, :12]
        hbt_val = X.reshape(-1, 12)
        allgood = ~hbt_val.mask.any(1)
        hbt_val = hbt_val[allgood]
        hm_pca = matplotlib.mlab.PCA(hbt_val)
        hm_pca_Yf = numpy.ma.zeros(shape=(X.shape[0]*X.shape[1], X.shape[2]),
                        dtype="f4")
        hm_pca_Yf.mask = numpy.zeros_like(hm_pca_Yf, dtype="bool")
        # if at least one channel is masked, mask all PCA
        hm_pca_Yf.mask[~allgood] = True
        hm_pca_Yf[allgood] = hm_pca.Y
        hm_pca_Y = hm_pca_Yf.reshape(X.shape[0], X.shape[1], 12)


        (hs_pca, hs_pca_Y) = self.get_pca_channels(satname, slice(12),
                                                   ret_y=True)

        # Compare weight matrices
        (f, a) = matplotlib.pyplot.subplots(1)
        c = a.pcolor(hm_pca.Wt, cmap="viridis")
        cb = f.colorbar(c)
        cb.set_label("Weight")
        a.set_title("PCA weight matrix HIRS measured {:%Y-%m-%d %H:%M:%S}".format(dt1))
        pyatmlab.graphics.print_or_show(f, False,
            "PCA_weight_HIRS_measured_{:%Y%m%d%H%M%S}.".format(dt1))

        (f, a) = matplotlib.pyplot.subplots(1)
        c = a.pcolor(hs_pca.Wt, cmap="viridis")
        cb = f.colorbar(c)
        cb.set_label("Weight")
        a.set_title("PCA weight matrix IASI-simulated HIRS {:%Y-%m-%d %H:%M:%S}".format(dt1))
        pyatmlab.graphics.print_or_show(f, False,
            "PCA_weight_HIRS_IASI_simul_{:%Y%m%d%H%M%S}.".format(dt1))

        (f, a) = matplotlib.pyplot.subplots(1)
        c = a.pcolor(hm_pca.Wt-hs_pca.Wt, cmap="BrBG")
        cb = f.colorbar(c)
        cb.set_label("Weight")
        a.set_title("PCA weight matrix IASI-meas-simulated HIRS {:%Y-%m-%d %H:%M:%S}".format(dt1))
        pyatmlab.graphics.print_or_show(f, False,
            "PCA_weight_HIRS_IASI_delta_meas_simul_{:%Y%m%d%H%M%S}.".format(dt1))
        
        for i in range(12):
            (f, a, m) = self._prepare_map()
        
        for i in range(12):
            (f, a, m) = self._prepare_map()

            pcs = m.pcolor(self.gran["lon"], self.gran["lat"],
                           hs_pca_Y[:, :, i], latlon=True, cmap=cmap)

            # workaround for http://stackoverflow.com/q/12317155/974555
            safeline = (hg["lon"].data>0).all(1)
            pcs = m.pcolor(hg["lon"][safeline, :], hg["lat"][safeline, :],
                           hm_pca_Y[safeline, :, i], latlon=True, cmap=cmap)

            cb = f.colorbar(pcs)
            cb.set_label("Score PC {:d}".format(i+1))


            a.set_title("{satname:s} HIRS PC {pc:d}, iasi-simulated or real PC scores\n"
                         "{dt1:%Y-%m-%d %H:%M} -- {dt2:%H:%M}".format(
                            satname=satname, pc=i, dt1=dt1, dt2=dt2))
            pyatmlab.graphics.print_or_show(f, False,
                "map_BTHIRS_real_simul_pc{:d}_{:%Y%m%d%H%M%S}.".format(i+1,dt1))

    
    def pls2_prediction_test_pair(self, sat_ref, sat_targ, tb_ref, tb_targ):
        """Use PLS to test prediction between pair of satellites, nominal SRFs

        Employs PLS (Partial Least Squares or Projection to Latent
        Structures) to predict N channels on target satellite from N
        channels on reference satellite, using nominal SRFs.

        :param str sat_ref: Reference satellite
        :param str sat_targ: Target satellite
        :param ndarray tb_ref:
        :param ndarray tb_targ:
        :returns: Differences between test and reference.

        TODO:
            - Wold et al. (2001) propose jack-knifing to estimate
              confidence intervals.  As I understand it, an extension of
              cross-validation by using N subsets of the training data to
              make N predictions.  I'm not sure if I understand it.  Needs
              study and implementation.
            - It appears results are much better when I disable scaling.
              They should not be, I don't understand.  Investigate.
            - It appears I'm not getting any overfitting no matter how
              large I make n_components.  Odd.
            - Cross-validation should be incorporated within the fitting
              procedure, but I don't think sklearn implements this.
              This would set apart some of the training data to
              iteratively independently test convergence and stop
              increasing the no. of components when overfitting occurs.
            - Until that is implemented, need to choose n_components smartly
        """

        # for some reason I don't understand, results are much better with
        # scaling switched off
        pls2 = sklearn.cross_decomposition.PLSRegression(n_components=9, scale=False)
        
        # Divide in training and testing.  Normally there should be
        # validation as well, but I'm not actually using the validation
        # data at the moment to stop overfitting.

        Xtr = tb_ref[::2, :]
        Ytr = tb_targ[::2, :]
        Xv = tb_ref[1::2, :]
        Yv = tb_targ[1::2, :]

        pls2.fit(Xtr, Ytr)
        return pls2.predict(Xv) - Yv


    def plot_srfs_in_subplots(self, sat_ref, sat_targ,
                                   x_quantity="wavelength",
                                   y_unit="Tb",
                                   selection=None,
                                   chans=range(1, 13),
                                   shift={}):
        # Plot SRFs for all
        # FIXME: is there a function to choose a smart subplot arragement
        # automagically?
        nrows = min(len(chans), 3)
        ncols = int(math.ceil(len(chans)/3))
        factor = 0.7
        (f, ax_all) = matplotlib.pyplot.subplots(nrows, ncols,
                            figsize=(8*factor, 10*factor))
        y = self.get_y(y_unit)
        spectra = y.reshape(-1, self.iasi.frequency.size)[selection, :]
        for (i, (ch, a_spec)) in enumerate(zip(chans, ax_all.ravel())):
            a_srf = a_spec.twinx()
            self._plot_srf_with_spectra(ch, {sat_ref, sat_targ},
                x_quantity, spectra, a_spec, a_srf,
                shift=shift)
            (freq_lo, freq_hi) = self._get_freq_range_for_ch(ch, {sat_ref, sat_targ})
            x_lo = freq_lo.to(self.x["unit"][x_quantity], "sp")
            x_hi = freq_hi.to(self.x["unit"][x_quantity], "sp")
            for a in {a_srf, a_spec}:
                a.set_xlim(min((x_lo.m, x_hi.m)),
                           max((x_lo.m, x_hi.m)))
            if i%ncols == 0:
                a_spec.set_ylabel("BT [K]")
            else:
                a_spec.set_yticks([])
            if i%ncols != (ncols-1):
                a_srf.set_yticks([])
            if i == 1:
                a_srf.legend(loc="center",
                             bbox_to_anchor=(-0.14, 1.25),
                             ncol=3,
                             labelspacing=0.25,
                             columnspacing=1.0,
                             frameon=False)

            a_spec.set_xlabel("{:s} [{:~}]".format(x_quantity, self.x["unit"][x_quantity]))
            #a_spec.set_xticks(numpy.linspace(*a.get_xlim(), 3).round(2))
#            a_spec.set_xticks([a.get_xlim()[0].round(2),
#                               ((self.srfs[sat_ref][i].centroid() +
#                                self.srfs[sat_targ][i].centroid())/2).to(
#                                    self.x["unit"][x_quantity], "sp").m.round(2),
#                               a.get_xlim()[1].round(2)])
            a_spec.set_ylim([200, 300])
#            else:
#                a_spec.set_xticks([])
            a_spec.set_title("Ch. {:d}".format(ch))
            a_srf.locator_params(tight=True, nbins=5)
        f.suptitle("Channel positions {:s} and {:s}".format(sat_ref, sat_targ))
        f.tight_layout()
        f.subplots_adjust(top=0.88, right=0.88)

        pyatmlab.graphics.print_or_show(f, False,
                "iasi_with_hirs_srfs_all_Tb_{sat_ref:s}_{sat_targ:s}_{ch:s}_{sh:s}.".format(
                    sat_ref=sat_ref, sat_targ=sat_targ,
                    ch=",".join([str(x) for x in chans]),
                    sh=",".join("{:s}{:+.2~}".format(k,v)
                        for (k,v) in shift.items()).replace(' ','_')))

    @staticmethod
    def _plot_dtb_hist(dtb, ax_all):
        for i in range(12):
            a = ax_all[0 if i<7 else 1]
            a.hist(dtb[:, i], 100, histtype="step",
                   normed=True, label="Ch. {:d}".format(i+1))
        for a in ax_all:
            a.set_xlabel(r"$\Delta$BT [K]")
            a.set_ylabel("Density [1/K]")
            a.legend(ncol=2, loc="best")

    def plot_hist_expected_Tdiff(self, sat_ref, sat_targ, tb_ref, tb_targ):
        (f, ax_all) = matplotlib.pyplot.subplots(2, 1, figsize=(10,5))
        dtb = tb_targ - tb_ref
        self._plot_dtb_hist(dtb, ax_all)
        f.suptitle(r"Expected $\Delta$BT {:s} - {:s}".format(sat_targ, sat_ref))
        f.subplots_adjust(hspace=0.25)
        pyatmlab.graphics.print_or_show(f, False,
            "expected_Tdiff_{:s}-{:s}.".format(sat_targ, sat_ref))

    def plot_hist_pls_perf(self, sat_ref, sat_targ, tb_ref, tb_targ):
        """Plot histograms of PLS performance per channel
        """
        (f, ax_all) = matplotlib.pyplot.subplots(2, 1, figsize=(7,7))
        dtb = self.pls2_prediction_test_pair(sat_ref, sat_targ, tb_ref, tb_targ)
        self._plot_dtb_hist(dtb, ax_all)
        f.suptitle(r"PLS performance $\Delta$BT, predicting {:s} from {:s}".format(sat_targ, sat_ref))
        f.subplots_adjust(hspace=0.25)
        pyatmlab.graphics.print_or_show(f, False,
            "PLS_performance_Tdiff_{:s}-{:s}.".format(sat_targ, sat_ref))

    def plot_fragment_expected_Tdiff(self, sat_ref, sat_targ,
                                     srfshift=0.0*ureg.um, N=40,
                                     col=50):
        """Plot a fragment of TB differences

        Show a 12xN matrix of TB for reference and differences to target.
        Possibly with an SRF-shift.  Same shift applied to all channels.
        """

        tb_ref = self.get_tb_channels(sat_ref)
        tb_targ = self.get_tb_channels(sat_targ,
            srfshift=dict.fromkeys(numpy.arange(1, 13), srfshift))
        dtb = tb_targ - tb_ref
        (f_ref, a_ref) = matplotlib.pyplot.subplots(figsize=(5,5))
        (f_diff, a_diff) = matplotlib.pyplot.subplots(figsize=(5,5))

        im_ref = a_ref.imshow(tb_ref[col, :N, :], cmap="viridis", aspect="auto",
                 interpolation="none", extent=(0.5, 12.5, N+0.5, 0.5))
        im_ref.set_clim(200, 300)
        im_diff = a_diff.imshow(dtb[col, :N, :], cmap="BrBG", aspect="auto",
                 interpolation="none", extent=(0.5, 12.5, N+0.5, 0.5))
        im_diff.set_clim(-2.5, 2.5)

        cb_ref = f_ref.colorbar(im_ref)
        cb_diff = f_diff.colorbar(im_diff)

        cb_ref.set_label("BT {:s} [K]".format(sat_ref))
        cb_diff.set_label(r"$\Delta$ BT {:s}-{:s} [K]".format(sat_targ, sat_ref))

        a_ref.set_title("Slice of IASI-simulated HIRS, {:s}".format(sat_ref))
        a_diff.set_title("Slice of IASI-simulated HIRS differences,\n"
                         "{:s}-{:s} ({:+.2~})".format(sat_targ, sat_ref,
                                                      srfshift))

        for a in (a_ref, a_diff):
            a.set_xlabel("Channel no.")
            a.set_ylabel("Measurement no.")

        pyatmlab.graphics.print_or_show(f_ref, False,
            "slice_TB_{:s}.".format(sat_ref))

        pyatmlab.graphics.print_or_show(f_diff, False,
            "slice_TB_diff_{:s}_{:s}{:+.2f}.".format(sat_targ, sat_ref, 
                srfshift.to(ureg.um, "sp").m))


    def visualise_pls2_diagnostics(self, sat_ref, sat_targ,
                                   x_quantity="wavelength",
                                   y_unit="Tb"):
        """Visualise some PLS2 diagnostics

        Should show:
            - SRFs for channels with underlying spectrum
            - Expected differences between channels for satellite pair
            - (Later) actual differences between channels for satellite
              pair
            - Statistics of PLS2 approximation between channels for
              satellite pair
        """

        # Commented out just to get to the rest more quickly…
        
#        self.plot_srfs_in_subplots(sat_ref, sat_targ,
#            x_quantity, y_unit)

        tb_ref = self.get_tb_channels(sat_ref)
        tb_targ = self.get_tb_channels(sat_targ)
        tb_ref = tb_ref.reshape(-1, 12)
        tb_targ = tb_targ.reshape(-1, 12)
        valid = (tb_ref>0).all(1) & (tb_targ>0).all(1)
        tb_ref = tb_ref[valid, :]
        tb_targ = tb_targ[valid, :]

        self.plot_hist_expected_Tdiff(sat_ref, sat_targ, tb_ref, tb_targ)
        self.plot_hist_pls_perf(sat_ref, sat_targ, tb_ref, tb_targ)
        

    # Can't do cache with random noise addition
    #@typhon.utils.cache.mutable_cache(maxsize=20)
    def _prepare_args_calc_srf_estimate(self, sat, ch, shift, db,
                ref="single", limits={}, noise_level={"master": 0, "target": 0}):
        """Helper for calc_srf_estimate_rmse and others
        
        See documentation for self.calc_srf_estimate_rmse.
        """
        iasi = pyatmlab.datasets.tovs.IASISub(name="iasisub")
        # FIXME: I really should apply limits afterward.  Limits should be
        # applied to training data, which will be from collocations, and
        # to spectral database data, which I will be able to choose to
        # match the characteristics of the collocations.  However, it
        # should NOT be applied to testing data that I need to use to
        # verify how good the prediction is.
        M1 = iasi.read_period(start=datetime.datetime(2011, 1, 1),
            end=datetime.datetime(2011, 6, 30))
        M1_limited = typhon.math.array.limit_ndarray(M1, limits=limits)
        M2 = iasi.read_period(start=datetime.datetime(2012, 1, 1),
            end=datetime.datetime(2012, 6, 30))
        M2_limited = typhon.math.array.limit_ndarray(M2, limits=limits)
        y_master = pyatmlab.physics.specrad_wavenumber2frequency(
            M1["spectral_radiance"][::5, 2, :8461] * unit_specrad_wn)
        if db == "same":
            y_spectral_db = y_master
        elif db == "similar":
            y_spectral_db = pyatmlab.physics.specrad_wavenumber2frequency(
                M1["spectral_radiance"][2::5, 2, :8461] * unit_specrad_wn)
        elif db == "different":
            y_spectral_db = pyatmlab.physics.specrad_wavenumber2frequency(
                M2["spectral_radiance"][::3, 1, :8461] * unit_specrad_wn)
        else:
            raise ValueError("Unrecognised option for db: {:s}".format(db))
        srf_master = self.srfs[sat][ch-1]
        freq = self.iasi.frequency
        
        if ref == "all":
            bt_master = self.get_tb_channels(sat, specrad_f=y_master)
            L_ref = self.get_tb_channels(sat, specrad_f=y_spectral_db)
        elif ref == "single":
            bt_master = srf_master.channel_radiance2bt(
                    srf_master.integrate_radiances(freq, y_master))

            L_ref = srf_master.channel_radiance2bt(
                        srf_master.integrate_radiances(freq, y_spectral_db))
        else:
            raise ValueError("invalid 'ref', expected 'all' or 'single', "
                             "got '{:s}'".format(ref))

        srf_target = srf_master.shift(shift)
        bt_target = srf_target.channel_radiance2bt(
                srf_target.integrate_radiances(freq, y_master))

        # consider numpy.random.multivariate_normal.  And what if the
        # noise level varies with radiance?
        bt_master += noise_level["master"]*numpy.random.randn(*bt_master.shape)*ureg.K
        bt_target += noise_level["target"]*numpy.random.randn(*bt_target.shape)*ureg.K

        return (bt_master, bt_target, srf_master, y_spectral_db,
                    freq, L_ref, y_master)


    _regression_type = {
        "single": (sklearn.linear_model.LinearRegression,
                   {"fit_intercept": True}),
        "all": (sklearn.cross_decomposition.PLSRegression,
                {"n_components": 9, "scale": False})}
    def calc_srf_estimate_rmse(self, sat, ch, shift,
            db="different", ref="single",
            regression_type=None,
            regression_args=None,
            limits={},
            noise_level={"target": 0.0, "master": 0.0}):
        """Calculate cost function for estimating SRF

        Construct artificial HIRS measurements with a prescribed SRF
        shift.  Estimate how well we can recover this SRF shift using
        independent data: as a function of attempted SRF shift, calculate
        the RMSE between estimated and reference radiances.  Hopefully,
        the global minimum of this cost function will coincide with the
        prescribed SRF shift.

        Uses two sets of IASI data:

        - Set 1 is used to calculate two sets of radiances (brightness
          temperatures): the reference brightness temperature
          corresponding to the nominal SRF for satellite `sat`, channel
          `ch`; and a set of radiances when this SRF is shifted by
          `shift`.  In the real world, this set will come from
          collocations/matchups rather than from IASI data.

        - Set 2 is used to simulate many pairs of radiances.  The
          functionality for this is in
          `:func:pyatmlab.math.calc_rmse_for_srf_shift`.  Assuming an SRF
          shift, it simulates radiances for the nominal and a shifted SRF.
          From this shift, it derives a model predicting shifted-SRF radiances
          from nominal-SRF radiances.

        The model derived from set 2 is then applied to predict
        shifted-SRF radiances from the nominal-SRF radiances as calculated
        from set 1.  The set-2-predicted-shifted-SRF-radiances are
        compared to the set-1-directly-calculated-shifted-SRF-radiances,
        the comparison described by the RMSE.  This process is repeated
        for a set of shifts, resulting in a RMSE as a function of SRF
        shift.

        In the real world, set 1 would be from collocations and set 2
        would be still from IASI.  One aspect to investigate is how much
        the correctness of the SRF estimate relies on the similarity of
        the climatology between set2 and set 1.

        Arguments:

            sat [str]: Name of satellite, such as NOAA19
            ch [int]: Channel number
            shift [pint Quantity]: Reference shift, such as 100*ureg.nm.
            db [str]: Indicates whether how similar set2 should be to
                set1.  Valid values are 'same' (identical set), 'similar'
                (different set, but from same region in time and space),
                or 'different' (different set).  Default is 'different'.
            ref [str]: Set to `'single'` if you only want to use a single
                channel to estimate, or `'all'` if you want to use all.
            regression_type [scikit-learn regression class]: Class to use for
                regression.  By default, this is
                sklearn.linear_model.LinearRegression when ref is single,
                and sklearn.cross_decomposition.PLSRegression, when ref is
                all.
            regression_args [tuple]: Default arguments are stored in
                self._regression_type.  See sklearn documentation for
                other possibilities.
            limits [dict]: Limits applied to training/testing data.  See
                `:func:typhon.math.array.limit_ndarray`.
            noise_level [dict]: Noise levels applied to target and master.
                Dictionary {"target": float, "master": float}.

        Returns:

            (dx, dy) [(ndarray, ndarray)]: RMSE [K] as a function of
                attempted SRF shift [nm].  Hopefully, this function will
                have a global minimum corresponding to the actual shift.
        """
        (bt_master, bt_target, srf_master, y_spectral_db, f_spectra,
            L_spectral_db, y_master) = self._prepare_args_calc_srf_estimate(
                    sat, ch, shift, db, ref=ref, limits=limits,
                    noise_level=noise_level)

        regression_type = regression_type or self._regression_type[ref][0]
        regression_args = regression_args or self._regression_type[ref][1]
        dx = numpy.linspace(-100.0, 100.0, 51.0) * ureg.nm
        dy = [pyatmlab.math.calc_rmse_for_srf_shift(q,
                bt_master, bt_target, srf_master, y_spectral_db,
                f_spectra, L_spectral_db, ureg.um,
                regression_type, regression_args) for q in dx]
        dy = numpy.array([d.m for d in dy])*dy[0].u
        return (dx, dy)

    def plot_errdist_per_srf_costfunc_localmin(self, 
            sat, ch, shift_reference, db="different",
            ref="single",
            regression_type=None,
            regression_args=None,
            limits={},
            noise_level={"target": 0.0, "master": 0.0}):
        """Investigate error dist. for SRF cost function local minima

        For all local minima in the SRF shift recovery cost function,
        visualise the error distribution.  Experience has shown that the
        global minimum does not always recover the correct SRF.
        """

        regression_type = regression_type or self._regression_type[ref][0]
        regression_args = regression_args or self._regression_type[ref][1]
        (dx, dy) = self.calc_srf_estimate_rmse(sat, ch, shift_reference,
            db, ref, regression_type, regression_args, limits,
            noise_level)
        localmin = typhon.math.array.localmin(dy)
        (f1, a1) = matplotlib.pyplot.subplots()
        (f2, a2) = matplotlib.pyplot.subplots()
        # although we don't need bt_target to prepare to call
        # calc_bts_for_srf_shift, we still need it to compare its
        # result to what we would like to see
        (bt_master, bt_target, srf_master, y_spectral_db, f_spectra,
            L_spectral_db, y_master) = self._prepare_args_calc_srf_estimate(
                        sat, ch, shift_reference, db=db, ref=ref,
                        limits=limits, noise_level=noise_level)
        for shift_attempt in dx[localmin]:
            bt_estimate = pyatmlab.math.calc_bts_for_srf_shift(shift_attempt,
                bt_master, srf_master, y_spectral_db, f_spectra,
                L_spectral_db, unit=ureg.um,
                regression_type=regression_type,
                regression_args=regression_args)

            # bt_master: BTs according to unshifted SRF
            # bt_target: BTs according to reference shifted SRF
            # bt_estimate: BTs according to regression estimated shifted SRF
            # bt_...: BTs according to attempted shifted SRF (non regression)

            srf_shift_attempt = srf_master.shift(shift_attempt)
            bt_shift_attempt = srf_shift_attempt.channel_radiance2bt(
                srf_shift_attempt.integrate_radiances(self.iasi.frequency, y_master))

            for (a, bt) in ((a1, bt_estimate), (a2, bt_shift_attempt)):
                rmse = numpy.sqrt(((bt_target - bt)**2).mean())
                a.hist((bt_target-bt), 100, histtype="step",
                    label=r"{:+.3~} [RMSE={:.3~}]".format(
                        shift_attempt.to(ureg.nm),
                        rmse.to(ureg.K)))
        
        addendum = ("{sat:s}-{ch:d}, shift {shift_reference:+~}, db "
                    "{db:s}, ref {ref:s}, regr "
                    "{regression_type.__name__:s} args {regression_args!s} "
                    "limits {limits!s} noises {noise_level!s}".format(**vars()))
        a1.set_title("Err. dist at local RMSE minima for shift recovery\n" + addendum)
        a2.set_title("Errors between BTs for estimated and reference SRF\n" + addendum)
        a1.set_xlabel("Residual error for shift [K]")
        a2.set_xlabel("BT error due to inaccurately estimated shift [K]")
        for a in (a1, a2):
            a.set_ylabel("Count")
            a.legend()
            a.grid(axis="both")
        fn_lab = ("{sat:s}_ch{ch:d}_{shift_reference:.0f}_{db:s}_{ref:s}"
                  "_{cls:s}_{args:s}_{lim:s}_noise{noise1:d}mK{noise2:d}mK.").format(
                sat=sat, ch=ch, shift_reference=shift_reference.m, db=db,
                ref=ref, cls=regression_type.__name__,
                args=''.join(str(x) for x in itertools.chain.from_iterable(regression_args.items())),
                lim="global" if limits=={} else "nonglobal",
                noise1=int(noise_level["target"]*1e3),
                noise2=int(noise_level["master"]*1e3))
            
        pyatmlab.graphics.print_or_show(f1, False,
            "srf_estimate_errdist_per_localmin_"+fn_lab)
        pyatmlab.graphics.print_or_show(f2, False,
            "srf_misestimate_bt_propagation_"+fn_lab)
        

    def visualise_srf_estimate_rmse(self, sat, db="different",
                                    ref="single",
                                    regression_type=None,
                                    regression_args=None,
                                    limits={},
                                    noise_level={"target": 0.0, "master": 0.0}):
        regression_type = regression_type or self._regression_type[ref][0]
        regression_args = regression_args or self._regression_type[ref][1]
        (f, ax_all) = matplotlib.pyplot.subplots(4, 3, figsize=(14, 9))
        for i in range(12):
            ch = i + 1
            for (k, shift) in enumerate(numpy.array([-60.0, -30.0, 5.0, 40.0])*ureg.nm):
                logging.info("Estimating {sat:s} ch, {ch:d}, shift "
                             "{shift:+5.3~}, db {db:s}, ref {ref:s}, "
                             "cls {regression_type.__name__:s}, args "
                             "{regression_args!s}, limits "
                             "{limits!s}".format(**vars()))
                (dx, dy) = self.calc_srf_estimate_rmse(sat, ch, shift, db,
                        ref, regression_type, regression_args, limits,
                        noise_level)
                a = ax_all.ravel()[i]
                p = a.plot(dx, dy) #color=self.colors[i%7],
                               #linestyle="solid",
                               #marker=self.markers[k])
#                               label="ch. {:d}, {:+7.3~}".format(ch,shift))
                localmin = typhon.math.array.localmin(dy)
#                localmin = numpy.hstack((False, (dy[1:-1] < dy[0:-2]) & (dy[1:-1] < dy[2:]), False))
                a.plot(dx[localmin], dy[localmin], marker='o', ls="None",
                       color=p[0].get_color(), fillstyle="none",
                       markersize=4)
                a.plot(dx[dy.argmin()], dy[dy.argmin()], marker='o', ls="None",
                       color=p[0].get_color(), fillstyle="full",
                       markersize=6)
                a.vlines(shift.m, 0, a.get_ylim()[1], linestyles="dashed",
                       color=p[0].get_color())
                a.set_title("Ch. {:d}".format(ch))
        for a in ax_all.ravel():
            a.set_xlabel("SRF shift [nm]")
            a.set_ylabel(r"RMSE for estimate [$\Delta$ K]")
            a.grid(axis="both")
#            a.legend(ncol=2, loc="right", bbox_to_anchor=(1, 0.5))
        f.suptitle("Cost function minimisation for recovering shifted {:s} SRF "
                   "({:s} db, channel {:s}, regr {:s})\n"
                   "({!s}, {!s}), noise level {!s}".format(sat, db,
                   ref, regression_type.__name__,
                   regression_args, limits, noise_level))
        f.subplots_adjust(hspace=0.45, wspace=0.32)#, right=0.7)
        pyatmlab.graphics.print_or_show(f, False,
            "SRF_prediction_cost_function_{:s}_{:s}_{:s}_{:s}_{:s}_{:s}_noise{:d}mK{:d}mK.".format(sat,
                db, ref, regression_type.__name__,
                ''.join(str(x) for x in itertools.chain.from_iterable(regression_args.items())),
                "global" if limits=={} else "nonglobal",
                int(1e3*noise_level["target"]),
                int(1e3*noise_level["master"])))

    def estimate_errorprop_srf_recovery(self, sat, ch, shift_reference, db="different",
            ref="all",
            regression_type=sklearn.linear_model.LinearRegression,
            regression_args=(sklearn.cross_decomposition.PLSRegression,
                             {"n_components": 9, "scale": False}),
            optimiser_func=scipy.optimize.minimize_scalar,
            optimiser_args=dict(bracket=[-0.04, 0.04], bounds=[-0.1, 0.1],
                method="brent", args=(ureg.um,)),
            limits={}, noise_level={"target": 1.0, "master": 1.0}):
        """Estimate error propagation under SRF recovery
        """
        N = 100
        estimates = numpy.empty(shape=(N,), dtype="f4")

        bar = progressbar.ProgressBar(maxval=estimates.size,
                widgets=pyatmlab.tools.my_pb_widget)
        bar.start()

        for i in range(estimates.size):
            (bt_master, bt_target, srf_master, y_spectral_db, f_spectra,
                L_spectral_db, _) = self._prepare_args_calc_srf_estimate(
                            sat, ch, shift_reference, db=db, ref=ref,
                            limits=limits, noise_level=noise_level)

            res = pyatmlab.math.estimate_srf_shift(
                bt_master, bt_target, srf_master, y_spectral_db, f_spectra,
                L_spectral_db,
                regression_type=regression_type, regression_args=regression_args,
                optimiser_func=optimiser_func,
                optimiser_args=optimiser_args,
                args=(ureg.um,))
            estimates[i] = res.x
            bar.update(i+1)
        bar.finish()
        stderr = (estimates - (shift_reference.to(ureg.um).m)).std()
        return (stderr*ureg.um).to(ureg.nm)

def main():
    p = parsed_cmdline
    print(p)
    numexpr.set_num_threads(p.threads)


    with numpy.errstate(all="raise"):
        vis = IASI_HIRS_analyser()
        if not isinstance(vis.iasi, pyatmlab.datasets.tovs.IASISub):
            vis.iasi = pyatmlab.datasets.tovs.IASISub(name="iasisub")
        h = pyatmlab.datasets.tovs.HIRS3(name="hirs")

        if p.makelut:
            print("Making LUT only")
            vis.get_lookup_table(sat=p.sat, pca=p.pca, x=p.factor,
                                 npc=p.npca, channels=p.channels)
            return

        if p.plot_bt_srf_shift and vis.gran is None:
            vis.gran = vis.iasi.read_period(start=datetime.datetime(2011, 1, 1), end=datetime.datetime(2011, 6, 30))
        if p.plot_spectrum_with_channels or p.plot_shifted_srf_in_subplots:
            if vis.gran is None:
                vis.gran = vis.iasi.read_period(start=datetime.datetime(2011, 1, 1), end=datetime.datetime(2011, 1, 30))
            if p.seed > 0:
                logging.info("Seeding with {:d}".format(p.seed))
                numpy.random.seed(p.seed)
            selection = numpy.random.choice(range(vis.gran["lat"].size), p.spectrum_count)
#        N = 40
#        col = 50
        if p.plot_shifted_srf_in_subplots:
            for ch in {range(1, 7), range(7, 13)}:
                vis.plot_srfs_in_subplots("NOAA19", "NOAA18",
                    x_quantity="wavelength", y_unit="TB",
                    selection=selection, chans=ch)
                for sh in (0.02*ureg.um, -0.02*ureg.um):
                    vis.plot_srfs_in_subplots("NOAA19", "NOAA18",
                        x_quantity="wavelength", y_unit="TB",
                        selection=selection, chans=ch, shift={"NOAA18": sh})
#
#        vis.plot_fragment_expected_Tdiff("NOAA19", "NOAA18", N=N, col=col)
#        vis.plot_fragment_expected_Tdiff("NOAA19", "NOAA18",
#                -0.02*ureg.um, N=N, col=col)
#        vis.plot_fragment_expected_Tdiff("NOAA19", "NOAA18",
#                +0.02*ureg.um, N=N, col=col)
#        vis.visualise_pls2_diagnostics("NOAA19", "NOAA18")
#        vis.M1 = vis.gran
        regrs = []
        for regr in p.regression_type:
            if regr == "PLSR":
                regrs.append(
                    (sklearn.cross_decomposition.PLSRegression,
                        {"n_components": p.nplsr,
                         "scale": False}))
            elif regr == "LR":
                regrs.append(
                    (sklearn.linear_model.LinearRegression,
                        {"fit_intercept": True}))
            else:
                raise RuntimeError("Impossible")
        dbref = itertools.product(
            p.db,
            p.predict_chan,
            regrs ,
            ({"lat": (*p.latrange, "all")},))
#        dbref = [x for x in dbref if not (x[1]=="single" and
#                    x[2][0] is sklearn.cross_decomposition.PLSRegression)]
#        dbref = list(itertools.product(("different",),
#                                  ("all",)))
#        # Test, why am I getting worse with PLS than with OLS?
#        vis.plot_errdist_per_srf_costfunc_localmin(
#            "NOAA19", 6, 60.0*ureg.nm, db="different", ref="all",
#            limits={"lat": (60, 90, "all")})
        if p.plot_srf_rmse:
            for (db, ref, (cls, args), limits) in dbref:
                vis.visualise_srf_estimate_rmse("NOAA19", db=db, ref=ref,
                    regression_type=cls, regression_args=args,
                    limits=limits)
        if p.plot_errdist_per_localmin:
            for ch in (1, 6, 11, 12):
                for shift in numpy.linspace(-80.0, 80.0, 9)*ureg.nm:
                    for (db, ref, (cls, args), limits) in dbref:
                        vis.plot_errdist_per_srf_costfunc_localmin(
                            "NOAA19", ch, shift, db=db, ref=ref, 
                            regression_type=cls,
                            regression_args=args,
                            limits=limits)
        if p.plot_bt_srf_shift:
            for i in range(1, 13):
                logging.info("Plotting shifts for NOAA19 ch.  {:d}".format(i))
                vis.plot_bt_srf_shift("NOAA19", i)

        shift = p.shift*ureg.nm
        if p.estimate_errorprop:
            for ch in p.channels:
                # I found those channels on HIRS may get local minima:
                if ch in {1, 6, 9, 11, 12}:
                    optimiser_func=scipy.optimize.basinhopping
                    optimiser_args=dict(x0=0, T=0.1, stepsize=0.03,
                                        niter=100, niter_success=20,
                                        interval=10, disp=False,
                                        minimizer_kwargs=dict(
                                            bounds=[(-0.2, +0.2)],
                                            options=dict(
                                                factr=1e12),
                                            args=(ureg.um,)))
                else:
                    optimiser_func=scipy.optimize.minimize_scalar
                    optimiser_args=dict(bracket=[-0.04, 0.04],
                                        bounds=[-0.1, 0.1],
                                        method="brent",
                                        args=(ureg.um,))

                logging.info("Finding variation of minima for "
                    "{p.sat:s} channel {ch:d}, reference {shift:~}. "
                    "Using multiple linear regression. "
                    "Optimising with {optimiser:s}.".format(
                        p=p, ch=ch, shift=shift.to(ureg.nm),
                        optimiser=optimiser_func.__qualname__))
                noise_level = {"target": p.noise_level_target,
                               "master": p.noise_level_master}
                std = vis.estimate_errorprop_srf_recovery(p.sat, ch,
                    shift.to(ureg.um), db="different",
                    ref="all",
                    regression_type=sklearn.linear_model.LinearRegression,
                    regression_args={"fit_intercept": True},
                    optimiser_func=optimiser_func,
                    optimiser_args=optimiser_args,
                    limits={"lat": (60, 90, "all")}, noise_level=noise_level)
                print("Channel {:d} shift of {:~}, noise {!s} has "
                    "stderror {:~}".format(ch, shift, noise_level, std))

#            vis.map_with_hirs(h, "NOAA16", i)
#        vis.lut_load("/group_workspaces/cems/fiduceo/Users/gholl/hirs_lookup_table/large_similarity_db_PCA_ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12_4_8.0")
#        vis.lut_load("/group_workspaces/cems/fiduceo/Users/gholl/hirs_lookup_table/large_similarity_db_PCA_ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12_4_4.0")
#        vis.lut_load("/group_workspaces/cems/fiduceo/Users/gholl/hirs_lookup_table/large_similarity_db_PCA_ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12_4_2.0")
#        vis.lut_load("/group_workspaces/cems/fiduceo/Users/gholl/hirs_lookup_table/large_similarity_db_PCA_ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12_4_1.0")
#        vis.lut_visualise_stats_unseen_data()
#        vis.lut_visualise_multi()
#        (counts, stats) = vis.lut_get_stats_unseen_data()
#        vis.plot_lut_radiance_delta_all_iasi()
#        vis.get_lookup_table(sat="NOAA18", pca=True, x=8)
#        vis.get_lookup_table(sat="NOAA18", pca=True, x=8, npc=3)
#        vis.get_lookup_table(sat="NOAA18", pca=True, x=4.0)
#        vis.get_lookup_table(sat="NOAA18", pca=True, x=4.0, npc=3)
#        vis.get_lookup_table(sat="NOAA18", pca=True, x=2.0)
#        vis.get_lookup_table(sat="NOAA18", pca=True, x=2.0, npc=3)
#        vis.get_lookup_table(sat="NOAA18", pca=True, x=1.0)
#        vis.get_lookup_table(sat="NOAA18", pca=True, x=1.0, npc=3)
#        vis.get_lookup_table("NOAA18", N=40)
#        vis.estimate_optimal_channel_binning("NOAA18", 5, 10)
#        vis.estimate_pca_density("NOAA18", all_n=range(2, 5),
#            bin_scales=[0.5, 2, 4, 6, 8])
        if p.plot_spectrum_with_channels:
            for unit in {"Tb", "specrad_freq"}:
                for x in {"frequency", "wavelength"}:
                    vis.plot_full_spectrum_with_all_channels("NOAA18",
                        y_unit=unit, x_quantity=x,
                        selection=selection)
                    vis.plot_srf_all_sats(y_unit=unit,
                        selection=selection)
#        for h in vis.allsats:
#            try:
#                #vis.plot_Te_vs_T(h)
#                vis.plot_channel_BT_deviation(h)
#            except FileNotFoundError as msg:
#                logging.error("Skipping {:s}: {!s}".format(h, msg))
#        logging.info("Done")

if __name__ == "__main__":
    main()
