import logging
import numpy
import argparse

import sys
import os
import re
import math
import datetime
import itertools
import functools
import pickle
import pathlib
import lzma

import numpy.lib.recfunctions
import scipy.stats
import scipy.odr

import netCDF4

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
    
import matplotlib.pyplot

import progressbar
import numexpr
import mpl_toolkits.basemap
import sklearn.cross_decomposition
#from memory_profiler import profile

import typhon.plots
import typhon.math
import typhon.math.stats

import typhon.datasets.tovs
import pyatmlab.io
import typhon.config
import pyatmlab.physics
import pyatmlab.db

from typhon.constants import (micro, centi, tera, nano)
from typhon.physics.units import ureg, radiance_units as rad_u

from .. import fcdr
from .. import math as fhmath
from .. import common
from .. import graphics

hirs_iasi_matchup = pathlib.Path("/group_workspaces/cems2/fiduceo/Data/Matchup_Data/IASI_HIRS")

unit_specrad_wn = ureg.W / (ureg.m**2 * ureg.sr * (1/ureg.m))
unit_specrad_freq = ureg.W / (ureg.m**2 * ureg.sr * ureg.Hz)
logger = logging.getLogger(__name__)

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description="Experiment with HIRS SRF estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--plot_shifted_srf_in_subplots",
        action="store_true", default=False,
        help="Plot SRF with shifts and fragments of IASI, per channel, "
             "for NOAA-19 and -18")

    parser.add_argument("--plot_spectrum_with_channels",
        action="store_true", default=False, 
        help="Plot a selection of IASI spectra along with HIRS SRFs. ")

    parser.add_argument("--plot_srf_cost",
        action="store_true", default=False,
        help="Visualise cost function involved in recovering shifted SRF")

    parser.add_argument("--plot_errdist_per_localmin",
        action="store_true", default=False,
        help="For each local minimum in the cost function, "
             "visualise the error distribution.")

    parser.add_argument("--plot_bt_srf_shift",
        action="store_true", default=False)

    parser.add_argument("--vis_expected_range", action="store_true",
        help="Visualise expected variability for all channels, as a "
             "function of radiance.")

    parser.add_argument("--compare_hiasi_hirs", action="store_true",
        help="Plot comparison of HIASI and HIRS")

    parser.add_argument("--write_channel_locations",
        action="store", default="", type=str,
        help="Write channel centroids to file as a LaTeX booktabs table.")

    parser.add_argument("--estimate_errorprop", action="store_true",
        help="Estimate uncertainty propagation.  Does not make a figure. "
             "Use with --shift and --channels to choose a single shift "
             "and potentially many channels.")

    parser.add_argument("--makelut", action="store_true", default=False,
        help="Make look up table")

    parser.add_argument("--sat", action="store", default="NOAA19",
        type=str, help="Primary satellite to use")

    parser.add_argument("--sat2", action="store", type=str,
        help="Secondary satellite to use, where applicable.  "
             "For example, when estimating cost function or "
             "error propagation, one may wish to go from one "
             "satellite to another.  By default it will be the same as "
             "the primary satellite")

    parser.add_argument("--channels", action="store", type=int,
                        default=list(range(1, 13)),
                        choices=list(range(1, 20)), nargs="+")

    parser.add_argument("--pca", action="store_true", default=False,
        help="Use Principal Component Analysis for LUT.")

    parser.add_argument("--factor", action="store", type=float,
        help="Make LUT denser by factor")

    parser.add_argument("--npca", action="store", type=int,
        help="Number of PCs to retain in LUT")

    parser.add_argument("--spectrum_count", action="store", type=int,
        default=40,
        help="When plotting IASI spectra, how many to plot?")

    parser.add_argument("--seed", action="store", type=int,
        default=0,
        help="Random seed to use when selecting IASI spectra to plot.  "
             "0 (the default) means do not seed.")

    parser.add_argument("--db", action="store",
                    choices=["same", "similar", "different"],
                    default=["different"], nargs="+",
        help="Use same, similar, or different DB for training/prediction")

    parser.add_argument("--predict_chan", action="store",
                    choices=["single", "all"],
                    default=["all"],
                    nargs="+",
                    help="Predict single-to-single or all-to-single")

    parser.add_argument("--regression_type", action="store",
        choices=["LR", "PLSR", "ODR"],
        default=["LR"],
        nargs="+",
        help="What kind of regression to use for prediction: linear "
             "regression, partial least squares regression, or "
             "orthogonal distance regression")

    parser.add_argument("--nplsr", action="store", type=int,
        default=12,
        help="When regressing with PLSR, how many components to use")

    parser.add_argument("--latrange",
        action="store", type=float, nargs=2, default=(-90, 90),
        help="Latitude range to use in training database")

    parser.add_argument("--shift", action="store", type=float,
        help="SRF shift [nm].  Use with estimate_errorprop.",
        default=0.0)

    parser.add_argument("--ref_shifts", action="store", type=float,
        nargs="+",
        default=[-60.0, -30.0, 5.0, 40.0],
        help="For use with plot_srf_cost, what reference shifts (in nm) to impose.")

    parser.add_argument("--shift_range", action="store", type=float,
        nargs=2, default=[-80.0, 80.0],
        help="For use with plot_srf_cost, what range of shifts to show "
             "cost function for")

    parser.add_argument("--shift_count", action="store", type=int,
        default=41,
        help="How many shifts to try in shift_range")

    parser.add_argument("--verbose", action="store_true", default=False)

    parser.add_argument("--noise_level_target", action="store", type=float,
        default=0.0, nargs="+",
        help="Noise level for the target satellite, that is, the satellite "
             "for which the SRF is to be estimated.  Number of values is "
             "either 1 or equal to number of channels.")

    parser.add_argument("--noise_level_master", action="store", type=float,
        default=0.0, nargs="+",
        help="Noise level for the master satellite, that is, the satellite "
             "where the SRF is assumed known and that is used to predict "
             "radiances.  Number of values is either 1 or equals no. of "
             "channels.")

    parser.add_argument("--noise_quantity", action="store", type=str,
        choices=["radiance", "bt"],
        default="bt",
        help="Quantity to which to apply noise (bt or radiance)")

    parser.add_argument("--noise_units", action="store", type=str,
        default="mW cm/m^2/sr",
        help="Unit for noise.  If noise_quantity is bt, this must be K. "
             "Otherwise, can be anything with dimensions convertible to "
             "radiance units.  If noise_units='relative', the noise is "
             "taken as a factor of a per-channel default which was "
             "estimated based on a single noaa18 orbit 2005-08-18 "
             "14:08-15:54.  In this case, the noise quantity must be "
             "radiance. ")

    parser.add_argument("--n_tries", action="store", type=int,
        default=100,
        help="For use with --estimate_errorprop, how many times to "
             "recover the SRF shift with different noise realisations. "
             "Higher means more accurate estimate of uncertainty but also "
             "longer runtime.")

    parser.add_argument("--predict_quantity", action="store", type=str,
        default="bt", choices=["radiance", "bt"],
        help="For SRF prediction, work in radiances or in BTs. "
             "Use with --estimate_errorprop or --plot_srf_cost")

    parser.add_argument("--threads", action="store", type=int,
                        default=16,
                        help="How many threads to use in "
                             "numexpr-calculations")

    parser.add_argument("--cost_frac_bt", action="store", type=float,
        help="In cost function, relative importance of BT deviation. "
             "Recommend setting this to 1 as minima will be biased low "
             "otherwise.",
        default=1.0)

    parser.add_argument("--cost_frac_dλ", action="store", type=float,
        help="In cost function, relative weight of penalty for larger dλ shift. "
             "Recommend setting this to 0 as minima will be biased low "
             "otherwise.",
        default=0.0)

    parser.add_argument("--cost_mode", action="store", type=str,
        choices=["total", "anomalies"], default="total", 
        help="How to estimate cost function for BT deviation. "
             "If set to 'total', calculate regular cost (y_est - y_ref)**2. "
             "If set to 'anomalies', substract medians for each first.")

    parser.add_argument("--iasi_period", action="store", type=str,
        nargs=2, default=["2011-1-1", "2011-6-30"],
        metavar=("start", "end"),
        help="IASIsub period to read")

    parser.add_argument("--iasi_period2", action="store", type=str,
        nargs=2, default=["2012-1-1", "2012-6-30"],
        metavar=("start", "end"),
        help="IASIsub period to read, alternate")

    parser.add_argument("--hirs_period", action="store", type=str,
        nargs=2,
        metavar=("start", "end"),
        help="HIRS period to read for estimating uncertainty propagation")

    parser.add_argument("--iasi_fraction", action="store", type=float,
        default=0.5,
        help="Fraction of IASIsub data to keep.  Reading 6 months of "
             "IASIsub data needs some 16 GiB RAM.  If you read a year "
             "and set iasi_fraction to 0.1, you should need around 3 GB.")

    parser.add_argument("--datefmt", action="store", type=str,
        default="%Y-%m-%d",
        help="Date format for interpretation of period")
    

    parser.add_argument('--cache', dest='cache', action='store_true',
        help="Use caching.  More memory, higher speed.")

    parser.add_argument('--no-cache', dest='cache', action='store_false',
        help="Suppress caching.  Less memory, slower speed.")

    parser.add_argument("--log", action="store", type=str,
        metavar="file",
        default="",
        help="Log stdout to file")

    parser.set_defaults(cache=True)


    p = parser.parse_args()
    return p

class HIM(typhon.datasets.dataset.MultiFileDataset):
    """For HIRS-IASI-Matchups
    """
    basedir = "/group_workspaces/cems2/fiduceo/Data/Matchup_Data/IASI_HIRS/W_XX-EUMETSAT-Darmstadt,SATCAL+COLLOC+LEOLEOIR,M02+HIRS+M02+IASI_C_EUMS_20130101005203_20130101001158.nc"

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
            logger.info("Found no LUT, creating new")
            y = None
            for g in itertools.islice(self.graniter, *self.lut_slice_build):
                if y is None:
                    y = self._get_next_y_for_lut(g, sat, channels)
                else:
                    y = numpy.hstack([y, self._get_next_y_for_lut(g, sat, channels)])
            #y = numpy.vstack(y_all)
            logger.info("Constructing PCA-based lookup table")
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
            logger.info("Adding to lookup table: {!s}".format(g))
            y = self._get_next_y_for_lut(g, sat, channels)
            if db is None: # first time
                logger.info("Constructing lookup table")
                db = pyatmlab.db.LargeFullLookupTable.fromData(y,
                    {"ch{:d}".format(i+1):
                     dict(range=(tb[..., i][tb[..., i]>0].min()*0.95,
                                 tb[..., i].max()*1.05),
                          mode="linear",
                          nsteps=x)
                        for i in channels},
                        use_pca=False)
            else:
                logger.info("Extending lookup table")
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
#        logger.info("Storing lookup table to {:s}".format(out))
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

        Example plot::

          plot(cont[:, 10], delta[:, 10, :].T, 'o')
                ... is ...
          x-axis: reference BT for channel 11
          y-axis: differences for all channels 11

        """
        N = 12
        logger.info("Simulating radiances using lookup-table...")
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
            graphics.print_or_show(f, False,
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
        logger.info("Using LUT to find IASI for {:,} HIRS spectra".format(radiances.size))
        if self.dobar:
            bar = progressbar.ProgressBar(maxval=radiances.size,
                    widgets=common.my_pb_widget)
            bar.start()
            bar.update(0)
        for (i, dat) in enumerate(radiances):
            stats[i]["x"] = dat
            try:
                cont = self.lut.lookup(dat)
            except KeyError:
                n = 0
            except EOFError as v:
                logger.error("Could not read from LUT: {!s}".format(v))
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
            if self.dobar:
                bar.update(i+1)
        if self.dobar:
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
        graphics.print_or_show(f_tothist, False,
            "lut_{:s}_test_hists_{:s}.".format(sat, 
                self.lut.compact_summary().replace(".", "_")))
        graphics.print_or_show(f_deltahistperbin, False,
            "lut_{:s}_test_deltahistperbin_{:s}.".format(sat,
                self.lut.compact_summary().replace(".", "_")))
        graphics.print_or_show(f_bthistperbin, False,
            "lut_{:s}_test_bthistperbin_{:s}.".format(sat,
                self.lut.compact_summary().replace(".", "_")))
        for f in {f_tothist, f_errperbin, f_deltahistperbin, f_bthistperbin}:
            matplotlib.pyplot.close(f)
#        graphics.print_or_show(f_errperbin, False,
#            "lut_{:s}_test_errperbin_{:s}.".format(sat, self.lut.compact_summary()))
        return (biases, stds)
        

    def lut_visualise_multi(self, sat="NOAA18"):
#        basedir = typhon.config.conf["main"]["lookup_table_dir"]
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
                    logger.info("Considering LUT with npc={:d}, "
                                "fact={:.1f}, ch={!s}".format(npc, fact, list(channels)))
    #                p = pathlib.Path(basedir) / (subname.format(npc=npc, fact=fact))
                    try:
                        self.get_lookup_table(sat="NOAA18", pca=True, x=fact,
                            npc=npc, channels=channels, make_new=False)
                    except FileNotfoundError:
                        logger.error("Does not exist yet, skipping")
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
                graphics.print_or_show(f, False,
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
                logger.info("Binning, scale={:.1f}, n={:d}".format(
                    bin_scale, n))
                bnd = typhon.math.stats.bin_nd(
                    [pca.Y[:, i] for i in range(n)],
                    bins[:n])
                (no, frac, lowest, med, highest) = self._calc_bin_stats(bnd)
                logger.info("PCA {:d} comp., {:s} bins/comp: {:.3%} {:d}/{:d}/{:d}".format(
                      n, "/".join(["{:d}".format(x) for x in bnd.shape]),
                      frac, lowest, med, highest))
                nos = numpy.argsort(no)
                busiest_bins = bnd.ravel()[nos[-nbusy:]].tolist()
                logger.info("Ranges in {nbusy:d} busiest bins:".format(
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
        logger.info("Studying {:d} combinations".format(tot))
        for (k, combi) in enumerate(itertools.combinations(chans, N)):
            bnd =  typhon.math.stats.bin_nd([btflat[i] for i in combi], 
                                         [bins[i] for i in combi])
            (frac, lowest, med, highest) = self._calc_bin_stats(bnd)
            logger.info("{:d}/{:d} channel combination {!s}: {:.3%} {:d}/{:d}/{:d}".format(
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

    allsats = {re.sub(r"0(\d)", "r\1", typhon.datasets.tovs.norm_tovs_name(x)).upper() for x in fcdr.list_all_satellites()}

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

    # Noise order of magnitudes.  Obtained visually from
    # HIRS_radiance_noise_{sat}_ch{ch}... files.

    noise_scale = ureg.Quantity(numpy.array(
        [1.6, 0.8, 0.35, 0.35, 0.3, 0.3, 0.25, 0.1, 0.18, 0.28, 0.20,
        0.18, 1.3e-3, 1.2e-3, 1.0e3, 1.0e-3, 0.9e-3, 0.8e-3, 4.0e-4]),
        rad_u["ir"])

    pred_channels = range(1, 13) # used for prediction
                
    _iasi = None
    @property
    def iasi(self):
        if self._iasi is None:
            self._iasi = typhon.datasets.tovs.IASIEPS(name="iasinc")
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

    def __init__(self, mode="Viju", usecache=True):
        #logger.info("Finding and reading IASI")
        if mode == "iasinc":
            self.iasi = typhon.datasets.tovs.IASIEPS(name="iasinc")
            self.choice = [(38, 47), (37, 29), (100, 51), (52, 11)]
            #self.gran = self.iasi.read(next(self.graniter))
        elif mode == "iasisub":
            self.iasi = typhon.datasets.tovs.IASISub(name="iasisub")
        self.graniter = self.iasi.find_granules()
        self.usecache = usecache

        hconf = typhon.config.conf["hirs"]
        srfs = {}
        for sat in self.allsats:
            try:
                srfs[sat] = [typhon.physics.units.em.SRF.fromArtsXML(sat,
                    "hirs", ch) for ch in range(1, 20)]
            except FileNotFoundError as msg:
                logger.error("Skipping {:s}: {!s}".format(
                              sat, msg))
        self.srfs = srfs
#        for coor in self.choice:
#            logger.info("Considering {coor!s}: Latitude {lat:.1f}°, "
#                "Longitude {lon:.1f}°, Time {time!s}, SZA {sza!s})".format(
#                coor=coor, lat=self.gran["lat"][coor[0], coor[1]],
#                lon=self.gran["lon"][coor[0], coor[1]],
#                time=self.gran["time"][coor[0], coor[1]].astype(datetime.datetime),
#                sza=self.gran["solar_zenith_angle"][coor[0], coor[1]]))
        
        self.dobar = sys.stdout.isatty()

    def get_y(self, unit, return_label=False):
        """Get measurement in desired unit
        """
        specrad_wavenum = self.gran["spectral_radiance"]
        if unit.lower() in {"tb", "bt"}:
            y = self.get_tb_spectrum()
            y_label = "Brightness temperature [K]"
        elif unit == "specrad_freq":
            y = typhon.physics.units.em.specrad_wavenumber2frequency(specrad_wavenum)
            y_label = "Spectral radiance [W m^-2 sr^-1 Hz^-1]"
        elif unit == "specrad_wavenum":
            y = ureg.Quantity(specrad_wavenum,
                ureg.m * ureg.W / (ureg.m**2 * ureg.sr))
            y_label = "Spectral radiance [W m^-2 sr^-1 m]"
        else:
            raise ValueError("Unknown unit: {:s}".format(unit))
        return (y[..., :8461], y_label) if return_label else y[..., :8461]

    def get_tb_spectrum(self):
        """Calculate spectrum of brightness temperatures
        """
        specrad_freq = self.get_y(unit="specrad_freq")

        with numpy.errstate(divide="warn", invalid="warn"):
            logger.info("...converting {:d} spectra to BTs...".format(
                specrad_freq.shape[0]))
            Tb = typhon.physics.units.em.specrad_frequency_to_planck_bt(
                specrad_freq, self.iasi.frequency)
        return Tb

    @typhon.utils.cache.mutable_cache(maxsize=20)
    def get_L_channels(self, sat, channels=range(1, 13), srfshift=None,
                        specrad_f=None):
        """Get radiances for channels
        """
        if srfshift is None:
            srfshift = {}
        if specrad_f is None:
            specrad_f = self.get_y(unit="specrad_freq")
        L_chans = ureg.Quantity(
                numpy.zeros(dtype=numpy.float32,
                        shape=specrad_f.shape[:-1] + (len(channels),)),
                specrad_f.u)
        for (i, c) in enumerate(channels):
            srf = self.srfs[sat][c-1]
            if c in srfshift:
                srf = srf.shift(srfshift[c])
                logger.debug("Calculating channel radiance {:s}-{:d}{:+.2~}".format(sat, c, srfshift[c]))
            else:
                logger.debug("Calculating channel radiance {:s}-{:d}".format(sat, c))
            L = srf.integrate_radiances(self.iasi.frequency, specrad_f)

            L_chans[..., i] = L

        return L_chans

    @typhon.utils.cache.mutable_cache(maxsize=20)
    def get_tb_channels(self, sat, channels=range(1, 13), srfshift=None,
                        specrad_f=None):
        """Get brightness temperature for channels
        """
        L_chans = self.get_L_channels(sat, channels, srfshift, specrad_f)

        Tb_chans = ureg.Quantity(
                numpy.zeros_like(L_chans.m),
                ureg.K)
        for (i, c) in enumerate(channels):
            srf = self.srfs[sat][c-1]
            Tb_chans[..., i] = srf.channel_radiance2bt(L_chans[..., i])
        return Tb_chans

    def get_pca_channels(self, sat, channels=slice(12), ret_y=False):
        bt = self.get_tb_channels(sat)
        bt2d = bt.reshape(-1, bt.shape[2])
        bt2d = bt2d[:, channels]
        btok = (bt2d>0).all(1)

        logger.info("Calculating PCA")
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
            y_unit="Tb", x_quantity="wavelength",
            selection=None, nrows=4):

        (y, y_label) = self.get_y(y_unit, return_label=True)
        if y_unit == "specrad_wavenum":
            y = y.to(rad_u["ir"])
        y = y.reshape(-1, y.shape[-1])
        spectra = y.reshape(-1, self.iasi.frequency.size)[selection, :]
        logger.info("Visualising")
        (f, a_spectrum_all) = matplotlib.pyplot.subplots(nrows, 1, figsize=(12, 8))
        xlims = [(3.6, 4.7), (6.3, 7.5), (9.2, 12.9), (13.0, 15.2)]
        chans = [(13, 14, 15, 16, 17, 18, 19), (11, 12), (8, 9, 10),
                 (1, 2, 3, 4, 5, 6, 7)]
        with numpy.errstate(invalid="ignore"):
            if y_unit == "Tb":
                ylim = (y[y>(0*y.u)].min().m, y.max().m)
            else:
                pass # choose spectra per area
        for (xlim, ch, a_spectrum) in zip(xlims, chans, a_spectrum_all.ravel()):
            a_srf = a_spectrum.twinx()
            logger.info("Plotting spectra")
            self._plot_spectra(spectra, x_quantity, a_spectrum)
#            for ch in range(1, 19):
#                self._plot_srf_with_spectra(ch, {sat}, x_quantity,
#                                       spectra, a_spectrum, a_srf,
#                                       shift={},
#                                       write_srfs=False)
            a_spectrum.set_xlim(xlim)
            if y_unit == "Tb":
                a_spectrum.set_ylim(ylim)
            else:
                inside = ((self.iasi.frequency.to(ureg.um, "sp").m > xlim[0]) &
                          (self.iasi.frequency.to(ureg.um, "sp").m <= xlim[1]))
                y_in_view = spectra[:, inside]
                ylim = (0, y_in_view.max().m)
                a_spectrum.set_ylim(ylim)
            a_spectrum.set_ylabel("Radiance\n[{:~}]".format(spectra.u))
            a_spectrum.set_xlabel("{:s} [{:~}]".format(x_quantity, self.x["unit"][x_quantity]))
            a_spectrum.grid()

            # Plot channels
            logger.info("Plotting channels")
            for c in ch:
                self._plot_srfs(c, {sat}, x_quantity, ax=a_srf)
                srf = self.srfs[sat][c-1]

                nomfreq = srf.centroid()
                nomfreq_x = nomfreq.to(self.x["unit"][x_quantity], "sp")

                # Seems that matplotlib.text.Text.get_unitless_position fails
                # when I keep the unit there
                a_srf.text(nomfreq_x.m, 1.07+(0.06 if c in {13,15} else 0), "{:d}".format(c),
                           backgroundcolor="white",
                           horizontalalignment='center',
                           verticalalignment="center")

            a_srf.set_xlim(xlim)

            a_srf.set_ylim(0, 1.2)


        f.suptitle("Some IASI spectra with measured {sat} HIRS SRFs".format(sat=sat))
        f.subplots_adjust(hspace=0.45)

#        a.bar(hirs_centres, Tb_chans[self.choice[0], self.choice[1], :], width=2e11, color="red", edgecolor="black",
#              align="center")

        graphics.print_or_show(f, False,
            "iasi_with_hirs_srf_{:s}_{:s}_{:s}.".format(sat, x_quantity, y_unit))

    @staticmethod
    def _norm_order(x, y):
        """Make sure both are increasing
        """

        ii = numpy.argsort(x)
        return (x[ii], y[ii])

    def _plot_srf_for_sat(self, sat, ch, srf, x_quantity,
            ax, color, linestyle="solid", shift=0.0*ureg.um,
                linewidth=1.0, writedata=False):
        """Plot SRF into axis.
        """
        x = srf.frequency.to(self.x["unit"][x_quantity], "sp")
        y = srf.W/srf.W.max()
        (x, y) = self._norm_order(x, y)
        ax.plot(x, y, label=(sat if shift.m==0 else "{:s}{:+~g}".format(sat,shift.to(ureg.nm, "sp"))),
                color=color, linestyle=linestyle, linewidth=linewidth)
        ax.plot([srf.centroid().to(self.x["unit"][x_quantity], "sp").m]*2, [0, 0.5], color=color,
            linestyle="dashed", linewidth=linewidth)
        if writedata:
            pyatmlab.io.write_data_to_files(
                numpy.vstack(
                    (#x/self.x["factor"][x_quantity],
                     x.to(self.x["unit"][x_quantity], "sp"),
                     srf.W/srf.W.max())).T,
                "SRF_{:s}_ch{:d}_{:s}{:+.2~}".format(
                        sat, ch, x_quantity, shift))


    def _plot_srfs(self, ch, sats, x_quantity="wavelength", ax=None,
                    shift={}, linewidth=1.0, writedata=False):
        """For channel `ch` on satellites `sats`, plot SRFs.

        Use axes `ax`.
        """

        for (color, sat) in zip(self.colors, sats):
            srf = self.srfs[sat][ch-1]
            self._plot_srf_for_sat(sat, ch, srf, x_quantity, ax, color,
                linewidth=linewidth, writedata=writedata)
            if sat in shift:
                self._plot_srf_for_sat(sat, ch, srf.shift(shift[sat]),
                    x_quantity, ax, color, linestyle="dashed",
                    shift=shift[sat], linewidth=linewidth,
                    writedata=writedata)
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

    def _plot_spectra(self, spectra, x_quantity, ax,
                      lineprops={}):
        """Small helper plotting spectra into axis
        """
        for spectrum in spectra:
            (x, y) = self._norm_order(
                self.iasi.frequency.to(self.x["unit"][x_quantity], "sp"),
                spectrum)
            ax.plot(x, y,
                    linewidth=0.1, zorder=5,
                    **lineprops)

    def _plot_srf_with_spectra(self, ch, sats, x_quantity,
                               spectra, ax_spectrum, ax_srf,
                               shift={}, srf_linewidth=1.0,
                               write_srfs=False,
                               spectra_lineprops={}):
        """For channel `ch`, on satellites `sats`, plot SRFs with spectra.

        Use axes `ax_spectrum` and ax_srf.
        """

        self._plot_srfs(ch, sats, x_quantity, ax=ax_srf, shift=shift,
            linewidth=srf_linewidth, writedata=write_srfs)
        #self._plot_bts_for_chan(ch, sats, x_quantity, ax)
        self._plot_spectra(spectra, x_quantity, ax_spectrum,
            lineprops=spectra_lineprops)

    
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
        if y_unit == "specrad_wavenum":
            y = y.to(rad_u["ir"])
#        Tb_chans = {}
#        for (sat, srf) in self.srfs.items():
#            Tb_chans[sat] = self.get_tb_channels(sat)

        #spectra = [y[c[0], c[1], :] for c in self.choice]
        y = y.reshape(-1, y.shape[-1])
        spectra = y.reshape(-1, self.iasi.frequency.size)[selection, :]
        for i in range(19):
            ch = i + 1
            (f, a) = matplotlib.pyplot.subplots()
            a.set_ylabel("Spectral radiance [{:~}]".format(y.u))
            a.set_xlabel("{:s} [{:~}]".format(x_quantity, self.x["unit"][x_quantity]))
            a.set_title("Some IASI spectra with different HIRS SRF (ch."
                        "{:d})".format(ch))
            a.grid(axis="both", which="both")
            a2 = a.twinx()

            self._plot_srf_with_spectra(ch, sorted(self.srfs.keys()),
                spectra=spectra,
                x_quantity=x_quantity, ax_spectrum=a, ax_srf=a2,
                srf_linewidth=0.5,
                write_srfs=False,
                spectra_lineprops={"color": "gray"})

            # visualise range of centroids and channel radiances
#            centroids = ureg.Quantity(numpy.array(
#                [self.srfs[sat][ch-1].centroid().m
#                    for sat in self.srfs.keys()]), ureg.Hz).to(ureg.um, "sp")
#            radiances = ureg.Quantity(numpy.array([[self.srfs[sat][ch-1].integrate_radiances(
#                                self.iasi.frequency, spectrum).m
#                        for sat in self.srfs.keys()]
#                    for spectrum in spectra]),
#                        rad_u["si"])
            if y_unit == "TB":
                for (i, sat) in self.srfs.keys():
                    radiances[i, :] = srf.channel_radiance2bt(radiances[i, :])
#            cc = centroids.mean()
#            a.errorbar(cc.m, radiances.mean().m,
#                xerr=[[cc.m-centroids.min().m, centroids.max().m-cc.m]],
#                yerr=0, color="black")

            box = a.get_position()
            for ax in (a, a2):
                ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

            a2.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
            (freq_lo, freq_hi) = self._get_freq_range_for_ch(ch, self.srfs.keys())
            y_in_view = [spectrum[(self.iasi.frequency>freq_lo) &
                                  (self.iasi.frequency<freq_hi)]
                         for spectrum in spectra]
            extremes = (min([yv.min() for yv in y_in_view]),
                        max([yv.max() for yv in y_in_view]))
            try:
                a.set_ylim(*[e.m for e in extremes])
            except AttributeError:
                a.set_ylim(*extremes)
            x_lo = freq_lo.to(self.x["unit"][x_quantity], "sp")
            x_hi = freq_hi.to(self.x["unit"][x_quantity], "sp")
            a.set_xlim(min(x_lo, x_hi).m, max(x_lo, x_hi).m)
            a2.set_xlim(min(x_lo, x_hi).m, max(x_lo, x_hi).m)
            graphics.print_or_show(f, False,
                    "iasi_with_hirs_srfs_ch{:d}_{:s}_{:s}.".format(
                        ch, x_quantity, y_unit))

    def plot_Te_vs_T(self, sat):
        """Plot T_e as a function of T

        Based on Weinreb (1981), plot T_e as a function of T.  For
        details, see pyatmlab.physics.estimate_effective_temperature.
        """
        hconf = typhon.config.conf["hirs"]
        (hirs_centres, hirs_srf) = pyatmlab.io.read_arts_srf(
            hconf["srf_backend_f"].format(sat=sat),
            hconf["srf_backend_response"].format(sat=sat))

        T = numpy.linspace(150, 300, 1000)
        (fig, a) = matplotlib.pyplot.subplots()
        for (i, (color, f_c, (f, W))) in enumerate(
                zip(self.colors, hirs_centres, hirs_srf)):
            Te = pyatmlab.physics.estimate_effective_temperature(
                    f[numpy.newaxis, :], W, f_c, T[:, numpy.newaxis])
            wl_um = typhon.physics.units.em.frequency2wavelength(f_c)/micro
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
        graphics.print_or_show(fig, False,
                "BT_Te_corrections_{:s}.".format(sat))

    def plot_channel_BT_deviation(self, sat):
        """Plot BT deviation for mono-/polychromatic Planck
        """

        (fig, a) = matplotlib.pyplot.subplots(2, sharex=True)
        for (i, color, srf) in zip(range(20), self.colors, self.srfs[sat]):
            T = numpy.linspace(srf.T_lookup_table.min(),
                               srf.T_lookup_table.max(),
                               5*srf.T_lookup_table.size)
            L = srf.blackbody_radiance(T)
            freq = srf.centroid()
            wl_um = typhon.physics.units.em.frequency2wavelength(freq)/micro
            lab = "ch. {:d} ({:.2f} µm)".format(i+1, wl_um)
            a[0].plot(T[::20], (srf.channel_radiance2bt(L)-T)[::20],
                      color=color, label=lab)
            a[1].plot(T,
                      typhon.physics.units.em.specrad_frequency_to_planck_bt(L, freq)-T,
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
        graphics.print_or_show(fig, False,
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
            logger.debug("Integrating {:,} spectra to radiances".format(
                y.size//y.shape[-1]))

        L_nom = srf_nom.integrate_radiances(self.iasi.frequency, y)

        if y.size/y.shape[-1] > 1e5:
            logger.debug("Converting {:,} radiances to brightness temperatures".format(
                y.size//y.shape[-1]))

        bt_nom = srf_nom.channel_radiance2bt(L_nom)

        yo = numpy.zeros(shape=bt_nom.shape + shift.shape)

        if self.dobar:
            bar = progressbar.ProgressBar(maxval=len(shift),
                    widgets=common.my_pb_widget)
            bar.start()

        logger.info("Shifting {:,} spectra by {:d} values between "
            "{:+~} and {:+~}".format(y.size//y.shape[-1], len(shift), shift[0], shift[-1]))
        for (i, sh) in enumerate(shift):
            srf_new = srf_nom.shift(sh)
            L_new = srf_new.integrate_radiances(self.iasi.frequency, y)
            bt_new = srf_new.channel_radiance2bt(L_new)
            yo.reshape(*yo.shape[:2], -1)[:, :, i] = bt_new
            if self.dobar:
                bar.update(i+1)
        if self.dobar:
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

        a.set_title("BT change distribution per BT for shifted SRF\n"
            "{:s} HIRS, ch. {:d}".format(satellite, channel))
        box = a.get_position()
        a.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        a.legend(loc="center left",ncol=1,bbox_to_anchor=(1,0.5))
        a.set_xlabel("BT [K]")
        a.set_ylabel(r"$\Delta$ BT [K]")
        a.grid(axis="both")
        graphics.print_or_show(f, False,
            "srf_shifted_dbt_hist_per_radiance_HIRS_{:s}-{:d}.".format(satellite, channel))

        (f, a) = matplotlib.pyplot.subplots()
        typhon.plots.plot_distribution_as_percentiles(a,
            numpy.tile(dx, (dsh.shape[0], dsh.shape[1], 1)).ravel(), dsh.ravel(),
            nbins=50, color="black", label="shift")
        a.set_title("Radiance change distribution per HIRS SRF shift, {:s} ch.  {:d}".format(satellite, channel))
        a.set_xlabel(r"shift [nm]")
        a.set_ylabel(r"$\Delta$ BT [K]")
        a.grid(axis="both")
        graphics.print_or_show(f, False,
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
        graphics.print_or_show(f, False,
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
        graphics.print_or_show(f, False,
            "PCA_weight_HIRS_measured_{:%Y%m%d%H%M%S}.".format(dt1))

        (f, a) = matplotlib.pyplot.subplots(1)
        c = a.pcolor(hs_pca.Wt, cmap="viridis")
        cb = f.colorbar(c)
        cb.set_label("Weight")
        a.set_title("PCA weight matrix IASI-simulated HIRS {:%Y-%m-%d %H:%M:%S}".format(dt1))
        graphics.print_or_show(f, False,
            "PCA_weight_HIRS_IASI_simul_{:%Y%m%d%H%M%S}.".format(dt1))

        (f, a) = matplotlib.pyplot.subplots(1)
        c = a.pcolor(hm_pca.Wt-hs_pca.Wt, cmap="BrBG")
        cb = f.colorbar(c)
        cb.set_label("Weight")
        a.set_title("PCA weight matrix IASI-meas-simulated HIRS {:%Y-%m-%d %H:%M:%S}".format(dt1))
        graphics.print_or_show(f, False,
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
            graphics.print_or_show(f, False,
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
                                   chans=range(1, 20),
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
                shift=shift,
                srf_linewidth=0.5,
                write_srfs=False)
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

        graphics.print_or_show(f, False,
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
        graphics.print_or_show(f, False,
            "expected_Tdiff_{:s}-{:s}.".format(sat_targ, sat_ref))

    def plot_hist_pls_perf(self, sat_ref, sat_targ, tb_ref, tb_targ):
        """Plot histograms of PLS performance per channel
        """
        (f, ax_all) = matplotlib.pyplot.subplots(2, 1, figsize=(7,7))
        dtb = self.pls2_prediction_test_pair(sat_ref, sat_targ, tb_ref, tb_targ)
        self._plot_dtb_hist(dtb, ax_all)
        f.suptitle(r"PLS performance $\Delta$BT, predicting {:s} from {:s}".format(sat_targ, sat_ref))
        f.subplots_adjust(hspace=0.25)
        graphics.print_or_show(f, False,
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

        graphics.print_or_show(f_ref, False,
            "slice_TB_{:s}.".format(sat_ref))

        graphics.print_or_show(f_diff, False,
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
        

    # do caching here, for typhons caching makes a copy
    M1 = None
    M2 = None
    # Can't do cache with random noise addition
    #@typhon.utils.cache.mutable_cache(maxsize=20)
    #@profile
    def _prepare_args_calc_srf_estimate(self, sat, ch, shift, db,
                ref="single", limits={}, noise_level={"master": 0, "target": 0},
                noise_quantity="bt",
                noise_units="K",
                predict_quantity="bt",
                sat2=None,
                iasi_frac=0.5,
                *, start1, end1, start2, end2):
        """Helper for calc_srf_estimate_costfunc and others
        
        See documentation for self.calc_srf_estimate_costfunc.

        Returns:

            y_master (BT or L)

                BTs [K] or radiances [rad units] for master: unshifted SRF
                from same DB as y_target

            y_target (BT or L)

                BTs [K] or radiances [rad units] for target: shifted SRF
                from same DB as y_master

            srf0

            L_spectral_db

                Full radiance spectra [W/m^2/Hz/sr] for reference
                database.

            L_full_testing

                Full radiance spectra for test dataset

            freq

            y_ref (BT or L)

                BTs [K] or radiances [rad units] for reference.  Unshifted
                SRF from different DB.  Corresponds to radiances in
                L_spectral_db.

            L_master
        """
        sat2 = sat2 or sat
        iasi = typhon.datasets.tovs.IASISub(name="iasisub")
        if not noise_quantity in {"bt", "radiance"}:
            raise ValueError("Unrecognised noise_quantity: "
                "{:s}".format(noise_quantity))
        if predict_quantity == "radiance" and noise_quantity != "radiance":
            raise ValueError("when predicting in radiance, noise_quantity "
                "MUST be radiance, found {:s} instead!".format(noise_quantity))

        # I really should apply limits afterward.  Limits should be
        # applied to training data, which will be from collocations, and
        # to spectral database data, which I will be able to choose to
        # match the characteristics of the collocations.  However, it
        # should NOT be applied to testing data that I need to use to
        # verify how good the prediction is.
        #
        # For memory reduction, immediately throw away ⅔ of the data.
        M1 = self.M1 if self.M1 is not None else iasi.read_period(start=start1, end=end1,
                NO_CACHE=True,
                filters={lambda M: M[numpy.random.random(M.shape[0])<iasi_frac]})
        self.M1 = M1
        #M1_limited = typhon.math.array.limit_ndarray(M1, limits=limits)
        L_full_testing = typhon.physics.units.em.specrad_wavenumber2frequency(
            M1["spectral_radiance"][::5, 2, :8461] * unit_specrad_wn)
        if db == "same":
            L_spectral_db = L_full_testing
        elif db == "similar":
            L_spectral_db = typhon.physics.units.em.specrad_wavenumber2frequency(
                M1["spectral_radiance"][2::5, 2, :8461] * unit_specrad_wn)
        elif db == "different":
            M2 = self.M2 if self.M2 is not None else iasi.read_period(start=start2, end=end2,
                NO_CACHE=True,
                limits=limits)
            self.M2 = M2
            #M2_limited = typhon.math.array.limit_ndarray(M2, limits=limits)
            L_spectral_db = typhon.physics.units.em.specrad_wavenumber2frequency(
                M2["spectral_radiance"][::3, 1, :8461] * unit_specrad_wn)
        else:
            raise ValueError("Unrecognised option for db: {:s}".format(db))
        srf0 = self.srfs[sat2][ch-1]
        srf_target = srf0.shift(shift)
        freq = self.iasi.frequency

        if ref == "all": # means 1–12 for now
            L_master = self.get_L_channels(sat, channels=self.pred_channels,
                                           specrad_f=L_full_testing).to(
                                                rad_u["ir"], "radiance")
            L_ref = self.get_L_channels(sat, channels=self.pred_channels,
                                           specrad_f=L_spectral_db).to(
                                                rad_u["ir"], "radiance")
            if predict_quantity == "bt":
                bt_ref = self.get_tb_channels(sat, channels=self.pred_channels,
                                               specrad_f=L_spectral_db)
        elif ref == "single":
            L_master = self.srfs[sat][ch-1].integrate_radiances(freq,
                L_full_testing).to(rad_u["ir"], "radiance")
            L_ref = self.srfs[sat][ch-1].integrate_radiances(freq,
                L_spectral_db).to(rad_u["ir"], "radiance")
            if predict_quantity == "bt":
                bt_ref = self.srfs[sat][ch-1].channel_radiance2bt(L_ref)
        else:
            raise ValueError("invalid 'ref', expected 'all' or 'single', "
                             "got '{:s}'".format(ref))


        noise_level = noise_level.copy()
        L_target = srf_target.integrate_radiances(freq, L_full_testing).to(
                                                rad_u["ir"], "radiance")
        if noise_units == "relative":
            if noise_quantity != "radiance":
                raise ValueError("When noise_units=relative, "
                    "you must use noise_quantity=radiance")

            noise_level["master"] *= self.noise_scale[
                numpy.array(self.pred_channels)-1]
            noise_level["target"] *= self.noise_scale[ch-1]
            noise_units = self.noise_scale.u

        if noise_quantity == "radiance":
            L_target += ureg.Quantity(
                noise_level["target"] * numpy.random.randn(*L_target.shape),
                noise_units).to(L_target.u, "radiance")
            if numpy.ndim(noise_level["master"]) == 0:
                L_master += ureg.Quantity(
                    noise_level["master"]*numpy.random.randn(*L_master.shape),
                    noise_units).to(L_master.u, "radiance")
            else: # noise level per channel
                L_master += (noise_level["master"]
                                        [numpy.asarray(self.pred_channels)-1]
                                       [numpy.newaxis, :] *
                    numpy.random.randn(*L_master.shape)).to(
                        L_master.u, "radiance")

        if predict_quantity == "bt":
            bt_master = self.srfs[sat][ch-1].channel_radiance2bt(L_master)
            bt_target = srf_target.channel_radiance2bt(L_target)

        if noise_quantity == "bt":
            bt_master += ureg.Quantity(
                noise_level["master"]*numpy.random.randn(*bt_master.shape),
                noise_units)
            bt_target += ureg.Quantity(
                noise_level["target"]*numpy.random.randn(*bt_target.shape),
                noise_units)

        #L_master, L_target, L_spectral_db
        L_spectral_db = L_spectral_db.to(rad_u["ir"], "radiance")
        if predict_quantity == "bt":
            return (bt_master, bt_target, srf0, L_spectral_db,
                        L_full_testing,
                        freq, bt_ref, L_master, noise_level["master"],
                        noise_level["target"])
        elif predict_quantity == "radiance":
            return (L_master, L_target, srf0, L_spectral_db,
                        L_full_testing,
                        freq, L_ref, L_master, 
                        noise_level["master"], noise_level["target"])
        else:
            raise ValueError("Unknown prediction quantity: "
                             "{:s}.  Expecting bt or radiance.".format(
                                predict_quantity))


    _regression_type = {
        "single": (sklearn.linear_model.LinearRegression,
                   {"fit_intercept": True}),
        "all": (sklearn.cross_decomposition.PLSRegression,
                {"n_components": 9, "scale": False})}
    def calc_srf_estimate_costfunc(self, sat, ch, shift,
            db="different", ref="single",
            regression_type=None,
            regression_args=None,
            limits={},
            noise_level={"target": 0.0, "master": 0.0},
            noise_quantity="bt",
            noise_units="K",
            predict_quantity="bt",
            A=1.,
            B=0.,
            cost_mode="total",
            dλ=ureg.Quantity(numpy.linspace(-50.0, 50.0, 41), ureg.nm),
            sat2=None, 
            iasi_frac=0.5, *,
            start1, end1, start2, end2):
        """Calculate cost function for estimating SRF

        Construct artificial HIRS measurements with a prescribed SRF
        shift.  Estimate how well we can recover this SRF shift using
        independent data: as a function of attempted SRF shift, calculate
        the cost function.  That is either
        
            C₁ = A/µ_y**2 * \sum_i^N (y_est,i - y_ref,i)**2 +
               + N B/λ**2 + dλ**2

        or

            C₂ = \sum_i^N (y_est,i - y_ref,i - <y_est,i - y_ref,i>)**2

        where A and B are weights, µ_y is the mean brightness temperature,
        y_est is the BT estimated with attempted SRF shift dλ, y_ref is
        the reference brightness temperature calculated with reference
        shift, λ is the centroid wavelength for the reference SRF, and dλ
        is the attempted shift.  Returned will be (dλ, C).

        I recommend setting A=1 and B=0, because anything else just
        minimises to the wrong shift.

        Hopefully, the global minimum of this cost function will coincide
        with the prescribed SRF shift.

        Uses two sets of IASI data:

        - Set 1 is used to calculate two sets of radiances (brightness
          temperatures): the reference brightness temperature
          corresponding to the nominal SRF for satellite `sat`, channel
          `ch`; and a set of radiances when this SRF is shifted by
          `shift`.  In the real world, this set will come from
          collocations/matchups rather than from IASI data.

        - Set 2 is used to simulate many pairs of radiances.  The
          functionality for this is in
          `:func:fhmath.calc_rmse_for_srf_shift`.  Assuming an SRF
          shift, it simulates radiances for the nominal and a shifted SRF.
          From this shift, it derives a model predicting shifted-SRF radiances
          from nominal-SRF radiances.

        The model derived from set 2 is then applied to predict
        shifted-SRF radiances from the nominal-SRF radiances as calculated
        from set 1.  The set-2-predicted-shifted-SRF-radiances are
        compared to the set-1-directly-calculated-shifted-SRF-radiances,
        the comparison described by the cost function.  This process is repeated
        for a set of shifts, resulting in a cost as a function of SRF
        shift according to the cost functions above.

        In the real world, set 1 would be from collocations and set 2
        would be still from IASI.  One aspect to investigate is how much
        the correctness of the SRF estimate relies on the similarity of
        the climatology between set2 and set 1.

        Arguments:

            sat [str]: Name of satellite, such as NOAA19
            ch [int]: Channel number
            shift [pint Quantity]: Reference shift, such as 10*ureg.nm.
            db [str]: Indicates whether how similar set2 should be to
                set1.  Valid values rare 'same' (identical set), 'similar'
                (different set, but from same region in time and space),
                or 'different' (different set).  Default is 'different'.
            ref [str]: Set to `'single'` if you only want to use a single
                channel to estimate, or `'all'` if you want to use all.
            regression_type [scikit-learn regression class]: Class to use for
                regression.  By defarult, this is
                sklearn.linear_model.LinearRegression when ref is single,
                and sklearn.cross_decomposition.PLSRegression, when ref is
                all.
            regression_args [tuple]: Default arguments are stored in
                self._regression_typre.  See sklearn documentation for
                other possibilities.
            limits [dict]: Limits applied to training/testing data.  See
                `:func:typhon.math.array.limit_ndarray`.
            noise_level [dict]: Noise levels applied to target and master.
                Dictionary {"target": float, "master": float}.
            noise_quantity [str]: Add noise in "bt" or "radiance"
                quantity.
            noise_units [str]: Add noise in "K" (for bt), in radiance
                units, or in "relative" units.
            predict_quantity [str]: Do prediction in "bt" or "radiance"
                space.
            A [float]: Weight for cost function component due to radiance
                differences between reference and attempted SRF.  Defaults
                to 1 for backward compatibility.
            B [float]: Weight for cost function component.  Defaults to 0.
            cost_mode (str): "total" or "anomalies"
            dλ [ndarray]: values of dλ to try
            sat2 [str]: Secondary satellite.  This is the satellite on
                which the channel is for which we wish to estimate the
                shift.  By default, sat2 equals sat.
            iasi_frac [float]: Fraction of IASIsub to keep

        Returns:

            (dλ, dy) [(ndarray, ndarray)]: Cost function [no unit] as a
                function of attempted SRF shift [nm].  Hopefully, this
                function will have a global minimum corresponding to the
                actual shift.
        """
        sat2 = sat2 or sat
        (y_master, y_target, srf0, L_spectral_db, L_full_testing, f_spectra,
            y_ref, L_master, u_y_ref, u_y_target) = self._prepare_args_calc_srf_estimate(
                    sat, ch, shift, db, ref=ref, limits=limits,
                    noise_level=noise_level,
                    noise_quantity=noise_quantity,
                    noise_units=noise_units,
                    predict_quantity=predict_quantity,
                    sat2=sat2,
                    iasi_frac=iasi_frac,
                    start1=start1, start2=start2,
                    end1=end1, end2=end2)

        regression_type = regression_type or self._regression_type[ref][0]
        regression_args = regression_args or self._regression_type[ref][1]
        #dλ = numpy.linspace(-100.0, 100.0, 51.0) * ureg.nm
        C1 = [fhmath.calc_cost_for_srf_shift(q,
                y_master, y_target, srf0, L_spectral_db,
                f_spectra, y_ref, ureg.um,
                regression_type, regression_args,
                cost_mode, predict_quantity=predict_quantity,
                u_y_ref=u_y_ref,
                u_y_target=u_y_target) for q in dλ]
        C1 = ureg.Quantity(numpy.array([d.m for d in C1]), C1[0].u)

        # add penalty for larger shifts.  This ensures that we prefer a
        # local minima with a small shift than with a large shift.

        (A, B) = (A/(A+B), B/(A+B)) # ensure they add up to 1
        N = y_master.shape[0]

        C = (A * C1 + 
             N * B / (srf0.centroid().to("nm", "sp")**2) * dλ**2)

        return (dλ, C)

    def plot_errdist_per_srf_costfunc_localmin(self, 
            sat, ch, shift_reference, db="different",
            ref="single",
            regression_type=None,
            regression_args=None,
            limits={},
            noise_level={"target": 0.0, "master": 0.0},
            noise_quantity="bt",
            noise_units="K",
            predict_quantity="bt",
            dλ=ureg.Quantity(numpy.linspace(-80, 80, 50), ureg.nm),
            A=1.,
            B=0.,
            cost_mode="total",
            sat2=None,
            iasi_frac=0.5,
            *,
            start1, start2, end1, end2):
        """Investigate error dist. for SRF cost function local minima

        For all local minima in the SRF shift recovery cost function,
        visualise the error distribution.  Experience has shown that the
        global minimum does not always recover the correct SRF.

        Arguments are identical as for self.calc_srf_estimate_costfunc.
        """

        sat2 = sat2 or sat
        regression_type = regression_type or self._regression_type[ref][0]
        regression_args = regression_args or self._regression_type[ref][1]
        (dx, dy) = self.calc_srf_estimate_costfunc(sat, ch, shift_reference,
            db, ref, regression_type, regression_args, limits,
            noise_level, noise_quantity=noise_quantity,
            noise_units=noise_units,
            predict_quantity=predict_quantity,
            sat2=sat2,
            iasi_frac=iasi_frac,
            dλ=dλ,
            A=A, B=B, cost_mode=cost_mode,
            start1=start1, end1=end1, start2=start2, end2=end2)
        localmin = typhon.math.array.localmin(dy)
        (f1, a1) = matplotlib.pyplot.subplots(figsize=(12,9))
        (f2, a2) = matplotlib.pyplot.subplots(figsize=(12,9))
        # although we don't need y_target to prepare to call
        # calc_y_for_srf_shift, we still need it to compare its
        # result to what we would like to see
        (y_master, y_target, srf0, L_spectral_db, L_full_testing, f_spectra,
            y_ref, L_master, u_y_ref, u_y_target) = self._prepare_args_calc_srf_estimate(
                        sat, ch, shift_reference, db=db, ref=ref,
                        limits=limits, noise_level=noise_level,
                        noise_quantity=noise_quantity,
                        noise_units=noise_units,
                        predict_quantity=predict_quantity,
                        sat2=sat2,
                        iasi_frac=iasi_frac,
                        start1=start1, start2=start2,
                        end1=end1, end2=end2)
        shift_attempts = dx[localmin]
        if not shift_reference in shift_attempts:
            shift_attempts = ureg.Quantity(numpy.concatenate(
                (numpy.atleast_1d(shift_reference.m),
                 dx[localmin].m)), shift_reference.u)
        for shift_attempt in shift_attempts:
            y_estimate = fhmath.calc_y_for_srf_shift(shift_attempt,
                y_master, srf0, L_spectral_db, f_spectra,
                y_ref, unit=ureg.um,
                regression_type=regression_type,
                regression_args=regression_args,
                predict_quantity=predict_quantity)

            # y_master: BTs or radiances according to unshifted SRF srf0
            # y_target: BTs according to reference shifted SRF srf_target
            # y_estimate: BTs according to regression estimated shifted SRF
            # bt_...: BTs according to attempted shifted SRF (non regression)

            srf_shift_attempt = srf0.shift(shift_attempt)
            L_shift_attempt = srf_shift_attempt.integrate_radiances(
                self.iasi.frequency, L_full_testing)
#            L_shift_attempt = srf_shift_attempt.integrate_radiances(self.iasi.frequency, L_master)
            if predict_quantity == "bt":
                y_shift_attempt = srf_shift_attempt.channel_radiance2bt(L_shift_attempt)
            elif predict_quantity == "radiance":
                y_shift_attempt = L_shift_attempt.to(rad_u["ir"], "radiance")
                y_target = y_target.to(rad_u["ir"], "radiance")
                y_estimate = y_estimate.to(rad_u["ir"], "radiance")
            else:
                raise ValueError("Unknown prediction quantity: "
                    "{:s}".format(predict_quantity))

            for (a, y) in ((a1, y_estimate), (a2, y_shift_attempt)):
                rmse = numpy.sqrt(((y_target - y)**2).mean())
                a.hist((y_target-y), 100, histtype="step",
                    label=r"{:+.3~} [RMSE={:.5~}; {:s}]".format(
                        shift_attempt.to(ureg.nm), rmse,
                        ", ".join(
                            (["truth"] if numpy.isclose(
                                shift_attempt.m, shift_reference.m)
                                       else []) +
                            (["localmin"] if shift_attempt in dx[localmin]
                                       else []))))
            
        addendum = ("{sat:s} to {sat2:s}-{ch:d}, shift {shift_reference:+~}, db "
                    "{db:s}, ref {ref:s}, regr "
                    "{regression_type.__name__:s} args {regression_args!s} "
                    "limits {limits!s}\nnoises {noise_level!s} "
                    "prediction in {predict_quantity:s}, "
                    "noise added in {noise_quantity:s} in "
                    "{noise_units:s}".format(**vars()))
        a1.set_title("Err. dist at local RMSE minima for shift recovery\n" + addendum)
        a2.set_title("Errors between ys for estimated and reference SRF\n" + addendum)
        a1.set_xlabel("Residual error for shift [{:~}]".format(y.u))
        a2.set_xlabel("y error due to inaccurately estimated shift [{:~}]".format(y.u))
        for a in (a1, a2):
            a.set_ylabel("Count")
            a.legend()
            a.grid(axis="both")
        fn_lab = ("{sat:s}_{sat2:s}_ch{ch:d}_{shift_reference:.0f}_{db:s}_{ref:s}"
                  "_{cls:s}_{args:s}_{lim:s}_noise{noise1:d}_{noise2:d}_"
                  "pq{predict_quantity:s}_nq{noise_quantity:s}_"
                  "nu{noise_units:s}.").format(
                sat=sat, sat2=sat2, ch=ch,
                shift_reference=shift_reference.m, db=db,
                ref=ref, cls=regression_type.__name__,
                args=''.join(str(x) for x in itertools.chain.from_iterable(
                    sorted(regression_args.items()))),
                lim="global" if limits=={} else "nonglobal",
                noise1=int(numpy.array(noise_level["target"]).squeeze()*1e3),
                noise2=int(numpy.array(noise_level["master"]).squeeze()*1e3),
                predict_quantity=predict_quantity,
                noise_quantity=noise_quantity,
                noise_units=noise_units)
            
        graphics.print_or_show(f1, False,
            "srf_estimate_errdist_per_localmin_"+fn_lab)
        graphics.print_or_show(f2, False,
            "srf_misestimate_bt_propagation_"+fn_lab)
            

    def visualise_srf_estimate_costfunc(self, sat, db="different",
                    ref="single",
                    regression_type=None,
                    regression_args=None,
                    limits={},
                    noise_level={"target": 0.0, "master": 0.0},
                    noise_quantity="bt",
                    noise_units="K",
                    predict_quantity="bt",
                    A=1.,
                    B=0.,
                    cost_mode="total",
                    sat2=None,
                    iasi_frac=0.5,
                    ref_dλ=ureg.Quantity(numpy.array([-10.0, -2.0, +5.0,
                            +15.0]), ureg.nm),
                    dλ=ureg.Quantity(numpy.linspace(-80, 80, 50), ureg.nm),
                    channels=range(1, 13),
                    *,
                    start1, end1, start2, end2):
        """Visualise cost function for SRF minimisation
        """
        sat2 = sat2 or sat
        if regression_type is None:
            regression_type = self._regression_type[ref][0]
        if regression_args is None:
            regression_args = self._regression_type[ref][1]
        (f, ax_all) = matplotlib.pyplot.subplots(
                *typhon.plots.get_subplot_arrangement(len(channels)),
                figsize=(14, 9))
        for (i, ch) in enumerate(channels):
            p_all = []
            for shift in ref_dλ:
                logger.info("Estimating {sat2:s} ch {ch:d} from {sat:s} "
                             "channels {self.pred_channels!s}, shift "
                             "{shift:+5.3~}, db {db:s}, ref {ref:s}, "
                             "cls {regression_type.__name__:s}, args "
                             "{regression_args!s}, limits "
                             "{limits!s}".format(**vars()))
                (dx, dy) = self.calc_srf_estimate_costfunc(sat, ch, shift, db,
                        ref, regression_type, regression_args, limits,
                        noise_level, noise_quantity=noise_quantity,
                        noise_units=noise_units,
                        predict_quantity=predict_quantity,
                        A=A, B=B, cost_mode=cost_mode,
                        iasi_frac=iasi_frac,
                        dλ=dλ,
                        start1=start1, end1=end1, start2=start2, end2=end2)
                a = ax_all.ravel()[i]
                p = a.plot(dx, dy)
                localmin = typhon.math.array.localmin(dy)
                a.plot(dx[localmin], dy[localmin], marker='o', ls="None",
                       color=p[0].get_color(), fillstyle="none",
                       markersize=4)
                a.plot(dx[dy.argmin()], dy[dy.argmin()], marker='o', ls="None",
                       color=p[0].get_color(), fillstyle="full",
                       markersize=6)
                p_all.append(p[0])
            for (p, shift) in zip(p_all, ref_dλ):
                a.vlines(shift.m, 0, a.get_ylim()[1], linestyles="dashed",
                       color=p.get_color())
                a.set_title("Ch. {:d}".format(ch))
        for a in ax_all.ravel():
            a.set_xlabel("SRF shift [nm]")
            a.set_ylabel(r"Cost function")
            a.grid(axis="both")
#            a.legend(ncol=2, loc="right", bbox_to_anchor=(1, 0.5))
        f.suptitle("Cost function evaluation for recovering shifted {:s} SRF "
                   "({:s} db, channel {:s}, regr {:s})\n"
                   "from {:s} "
                   "({!s}, {!s}), noise level {!s},\nA={:.2f}, B={:.2f} "
                   "cm={:s}, pred in {:s}, noise at {:s} in {:s}".format(
                   sat2, db, ref, regression_type.__name__,
                   sat,
                   regression_args, limits, noise_level, A, B,
                   cost_mode, predict_quantity, noise_quantity,
                   noise_units))
        f.subplots_adjust(hspace=0.47, wspace=0.35)#, right=0.7)
        graphics.print_or_show(f, False,
            "SRF_prediction_cost_function_{sat:s}_{sat2:s}_{db:s}_{ref:s}_"
            "{regression_type.__name__:s}_{regrargs:s}_lim{limstr:s}_A{A:d}"
            "_B{B:d}_noise{noise_targ:d},{noise_lev:d}_cm{cm:s}_{cst:s}_"
            "pq{pq:s}_nq{nq:s}_nu{nu:s}.".format(sat=sat, sat2=sat2,
                db=db, ref=ref, regression_type=regression_type,
                regrargs=''.join(str(x) for x in itertools.chain.from_iterable(regression_args.items())),
                limstr="".join("{:s}{:.0f}-{:.0f}".format(k, *v) for (k,v) in limits.items()),
                A=int(100*A), B=int(100*B),
                noise_targ=int(1e3*numpy.array(noise_level["target"]).squeeze()),
                noise_lev=int(1e3*numpy.array(noise_level["master"]).squeeze()),
                cm=cost_mode,
                cst=",".join("{:d}".format(int(x.m)) for x in ref_dλ),
                pq=predict_quantity, nq=noise_quantity, nu=noise_units))

    #@profile
    def estimate_errorprop_srf_recovery(self, sat, ch, shift_reference, db="different",
            ref="all",
            regression_type=sklearn.linear_model.LinearRegression,
            regression_args=(sklearn.cross_decomposition.PLSRegression,
                             {"n_components": 9, "scale": False}),
            optimiser_func=scipy.optimize.minimize_scalar,
            optimiser_args=dict(bracket=[-0.04, 0.04], bounds=[-0.1, 0.1],
                method="bounded", args=(ureg.um,)),
            limits={"lat": (-90, 90, "all")}, noise_level={"target": 1.0, "master": 1.0},
            noise_quantity="bt", noise_units="K", 
            cost_mode="total", N=100,
            predict_quantity="BT", 
            sat2=None,
            iasi_frac=0.5,
            *,
            hirs_start, hirs_end, start1, start2, end1, end2):
        """Estimate error propagation under SRF recovery.

        Mandatory arguments:

            sat
            ch
            shift_reference

        Optional arguments:

            db
            ref
            regression_type
            regression_args
            optimiser_func
            optimiser_args
            limits
            noise_level
            noise_quantity
            noise_units
            cost_mode   "total" or "anomalies"
            N
            predict_quantity
            sat2
            iasi_frac

        Mandatory keyword arguments:
            start1
            start2
            end1
            end2
            hirs_start
            hirs_end
        """
        sat2 = sat2 or sat
        estimates = numpy.empty(shape=(N,), dtype="f4")

        # to estimate propagation through calibration, use real HIRS
        # measurements.  Memory intensive; rad_wn_all will be
        # N * m * 56 * 4 bytes, or around 3 GB/day when N=100.
        h = fcdr.which_hirs_fcdr(sat.lower())
        M = h.read_period(hirs_start, hirs_end,
            locator_args={"satname": sat.lower()},
            reader_args={"filter_firstline": False}, # FIXME: update to new format
            fields=["time", "hrs_scntyp", "counts", "temp_iwt", "lat",
                    "lon", "hrs_scnlin"])
        rad_wn_all = ureg.Quantity(
            numpy.empty(shape=(N, M.shape[0], 56), dtype="f4"),
            rad_u["ir"])

        rad_wn_ref = h.Mtorad(M, self.srfs[sat2][ch-1], ch)

        if self.dobar:
            bar = progressbar.ProgressBar(maxval=estimates.size,
                    widgets=common.my_pb_widget)
            bar.start()

        for i in range(estimates.size):
            (y_master, y_target, srf0, L_spectral_db, L_full_testing, f_spectra,
                y_ref, _, u_y_ref, u_y_target) = self._prepare_args_calc_srf_estimate(
                            sat, ch, shift_reference, db=db, ref=ref,
                            limits=limits, noise_level=noise_level,
                            noise_quantity=noise_quantity,
                            noise_units=noise_units,
                            predict_quantity=predict_quantity,
                            sat2=sat2,
                            iasi_frac=iasi_frac,
                            start1=start1, start2=start2,
                            end1=end1, end2=end2)

            res = fhmath.estimate_srf_shift(
                y_master, y_target, srf0, L_spectral_db, f_spectra,
                y_ref,
                regression_type=regression_type, regression_args=regression_args,
                optimiser_func=optimiser_func,
                optimiser_args=optimiser_args,
                cost_mode=cost_mode,
                args=(ureg.um,),
                predict_quantity=predict_quantity,
                u_y_ref=u_y_ref,
                u_y_target=u_y_target)
            estimates[i] = res.x

            rad_wn_all[i, :, :] = h.Mtorad(M,
                srf0.shift(ureg.Quantity(res.x, ureg.um)), ch)

            logger.debug("Estimate {:d}/{:d}: {:.5f} nm (“truth”: {:.3f} nm), "
                          "ΔL = {:.3~}".format(
                i+1, estimates.size, estimates[i]*1e3,
                shift_reference.to(ureg.nm).m,
                (rad_wn_all[i, :, :] - rad_wn_ref).mean()))

            if self.dobar:
                bar.update(i+1)
        if self.dobar:
            bar.finish()
        bias = estimates.mean() - shift_reference.to(ureg.um).m
        stderr = (estimates - shift_reference.to(ureg.um).m).std()

        pdd = pathlib.Path(pyatmlab.io.plotdatadir())

        regrargs=''.join(str(x) for x in itertools.chain.from_iterable(sorted(regression_args.items())))
        basename = ("srf_errorprop_{sat:s}_{sat2:s}_ch{ch:d}_"
                    "db{db:s}_ref{ref:s}_"
                    "rt{regression_type.__name__:s}_"
                    "ra{regrargs:s}_"
                    "lats{lowlat}-{highlat}_"
                    "nq{noise_quantity:s}_nu{noise_units:s}_"
                    "cm{cost_mode:s}_pq{predict_quantity:s}".format(
                        lowlat=int(limits["lat"][0]),
                        highlat=int(limits["lat"][1]), **vars()))
        dmpdir = pathlib.Path(typhon.config.conf["main"]["mydatadir"])
        dmpfile = dmpdir / "srf_errorprop" / (basename + 
            "_sr{:d}_N{:d}_nl{:d},{:d}_estimates".format(
                int(round(shift_reference.to(ureg.nm).m)),
                N,
                int(noise_level["target"]*1e3),
                int(noise_level["master"]*1e3)))
                
        tofile = pdd / basename

        tofile.parent.mkdir(parents=True, exist_ok=True)
        # Too slow and I really am not using it.
#        dmpfile.parent.mkdir(parents=True, exist_ok=True)
#        with lzma.open(str(dmpfile.with_suffix(".pkl.lzma")),
#                mode="wb", preset=lzma.PRESET_DEFAULT) as fp:
#            logger.info("Dumping to {!s}".format(
#                    dmpfile.with_suffix(".pkl.lzma")))
#            pickle.dump((estimates, rad_wn_ref, rad_wn_all), fp)

        logger.info("Writing to {!s}".format(tofile) + ".{dat,info}")
        Δrad = rad_wn_all - rad_wn_ref[numpy.newaxis, ...]

        with tofile.with_suffix(".info").open("a", encoding="utf-8") as fp:
            fp.write(" ".join(sys.argv) + "\n")
            fp.write(common.get_verbose_stack_description())

        with tofile.with_suffix(".dat").open("a", encoding="ascii") as fp:
            fp.write(
                "{ch:<6d} "
                "{shift_reference.m:<8.2f} "
                "{N:<6d} "
                "{noise_level[target]:<9.3f} "
                "{noise_level[master]:<9.3f} "
                "{bias:<15.9f} "
                "{stderr:<15.9f} "
                "{radbias:<15.9f} "
                "{radstderr:<15.9f} "
                "\n".format(
                    radbias=Δrad.mean().m,
                    # not interested in std across channels / radiances
                    radstderr=Δrad.m.data.std(0).mean(),
                    **vars()))
        

        return (ureg.Quantity(bias, ureg.um).to(ureg.nm),
                ureg.Quantity(stderr, ureg.um).to(ureg.nm))

    def write_channel_locations(self, outfile, channels):
        """Write a LaTeX table with locations for all channels for all satellites
        """
        with open(outfile, "wt", encoding="ascii") as s:
            s.write(r"\begin{tabular}{r" + "l"*len(channels) + "}\n")
            s.write(r"\toprule" + "\n")
            for ch in channels:
                s.write("& {:d} ".format(ch))
            s.write(r"\\" + "\n")
            s.write(r"\midrule" + "\n")
            centroids = numpy.empty(shape=(len(channels), len(self.srfs)))*ureg.um
            for (i, sat) in enumerate(sorted(self.srfs.keys())):
                s.write(sat + " ")
                k = 0
                for (ch, srf) in enumerate(self.srfs[sat], start=1):
                    if ch in channels:
                        c = srf.centroid().to(ureg.um, "sp")
                        centroids[k, i] = c
                        s.write("& {:.3f} ".format(c.m))
                        k += 1
                s.write(r"\\" + "\n")
            s.write(r"\bottomrule" + "\n")
            s.write("Mean ")
            for i in range(len(channels)):
                s.write("& {:.4f} ".format(centroids[i, :].mean().m))
            s.write(r"\\" + "\n")
            s.write("Std [nm] ")
            for i in range(len(channels)):
                s.write("& {:.4f} ".format(centroids[i, :].std().to(ureg.nm).m))
            s.write(r"\\" + "\n")
            s.write(r"\bottomrule" + "\n")
            s.write(r"\end{tabular}" + "\n")

    def compare_hiasi_hirs(self, ch=1, start=0):
        i = ch - 1
        g = hirs_iasi_matchup.glob("*.nc")
        for k in range(start+1):
            himfile = next(g)
        ds = netCDF4.Dataset(himfile)
        # This uses the counts included in the NetCDF files, but those
        # appear to be wrong for all but channel 1...
        iasi_specrad = (ds["ref_radiance"][...] * rad_u["ir"]).to(
                rad_u["si"], "radiance")
        freq = numpy.loadtxt(self.iasi.freqfile) * ureg.Hz
        iasi_rad = self.srfs["METOPA"][i].integrate_radiances(freq,
                    iasi_specrad).to(rad_u["ir"], "radiance")
        hirs_rad = ((ds["mon_c0"][:, i] + ds["mon_c1"][:, i] * ds["mon_counts"][:, i])
                    * rad_u["ir"])
        (f, a) = matplotlib.pyplot.subplots()
        a.plot(iasi_rad, hirs_rad-iasi_rad, 'o')
        a.set_xlabel("IASI-simulated HIRS [{:~}]".format(iasi_rad.u))
        a.set_ylabel("HIRS - IASI-simulated HIRS [{:~}]".format(iasi_rad.u))
        a.set_title("IASI-HIRS comparison {:s}".format(ds.RefStartTimeSec))
        graphics.print_or_show(f, False,
            "HIRS_IASI_comparison_{:s}.".format(ds.RefStartTimeSec))


    def plot_expected_range(self, nbins=80, lab="", channels=range(1, 20)):
        # spectral_radiance
        logger.info("Obtaining spectral radiance")
        L_specrad = self.get_y("specrad_freq")
        sats = self.srfs.keys()
        tbs = {}
        for sat in sats:
            logger.info("Calculating brightness temperatures for {:s}".format(sat))
            tbs[sat] = self.get_tb_channels(sat, specrad_f=L_specrad,
                channels=range(1, 20))

        for ch in channels:
            logger.info("Visualising channel {:d}".format(ch))
            tb_ch = numpy.array([tbs[sat][:, 0, ch-1].m for sat in sats])
            x = tb_ch.mean(0)
            y = tb_ch.ptp(0)
#            bins = numpy.linspace(x.min(), x.max(), nbins)
#            binned = typhon.math.stats.bin(x, y, bins)
            (f, a) = matplotlib.pyplot.subplots()
#            pcs = numpy.array([scipy.stats.scoreatpercentile(b, [5, 25, 50, 75, 95])
#                   for b in binned]).T
            (_, _, _, c) = a.hist2d(x, y, bins=80, cmap="gray_r")
#            for (pc, ls, lb) in zip(pcs, (":", "--", "-", "--", ":"),
#                ("p5/95", "p25/75", "median", None, None)):
#                a.plot(bins, pc, linestyle=ls, color="blue", label=lb)
            typhon.plots.plot_distribution_as_percentiles(a, x, y,
                nbins=nbins, label="BT range", color="blue")
            a.set_xlabel("Brightness temperature [K]")
            a.set_ylabel("Brightness temperature PTP across sats [K]")
            a.set_title("HIRS channel {:d} expected BT range between "
                        "sats".format(ch))
            a.grid()
            a.legend(loc="upper left")
            cb = f.colorbar(c)
            cb.set_label("No. spectra")
            graphics.print_or_show(f, False,
                "HIRS_{:d}_exp_radrange_{:s}.".format(ch, lab))

def main():
    p = parse_cmdline()
    common.set_logger(
        logging.DEBUG if p.verbose else logging.INFO,
        p.log,
        loggers={"FCDR_HIRS", "typhon"})
        
    print(p)
    numexpr.set_num_threads(p.threads)

    with numpy.errstate(divide="raise", over="raise", under="warn", invalid="raise"):
        vis = IASI_HIRS_analyser(usecache=p.cache)
        if not isinstance(vis.iasi, typhon.datasets.tovs.IASISub):
            vis.iasi = typhon.datasets.tovs.IASISub(name="iasisub")

        if p.makelut:
            print("Making LUT only")
            vis.get_lookup_table(sat=p.sat, pca=p.pca, x=p.factor,
                                 npc=p.npca, channels=p.channels)
            return

        start = datetime.datetime.strptime(p.iasi_period[0], p.datefmt)
        end = datetime.datetime.strptime(p.iasi_period[1], p.datefmt)
        start_alt = datetime.datetime.strptime(p.iasi_period2[0], p.datefmt)
        end_alt = datetime.datetime.strptime(p.iasi_period2[1], p.datefmt)
        if (p.plot_bt_srf_shift or p.vis_expected_range) and vis.gran is None:
            vis.gran = vis.iasi.read_period(start=start, end=end,
                NO_CACHE=not p.cache)
        if p.plot_spectrum_with_channels or p.plot_shifted_srf_in_subplots:
            if vis.gran is None:
                vis.gran = vis.iasi.read_period(start=start, end=end,
                    NO_CACHE=not p.cache)
            if p.seed > 0:
                logger.info("Seeding with {:d}".format(p.seed))
                numpy.random.seed(p.seed)
            selection = numpy.random.choice(range(vis.gran["lat"].size), p.spectrum_count)
#        N = 40
#        col = 50
        if p.plot_shifted_srf_in_subplots:
            for ch in {range(1, 7), range(7, 13), range(13, 19)}:
                logger.info("Plotting SRFS in subplots, {!s}".format(ch))
                vis.plot_srfs_in_subplots("NOAA19", "NOAA18",
                    x_quantity="wavelength", y_unit="TB",
                    selection=selection, chans=ch)
                for sh in (0.02*ureg.um, -0.02*ureg.um):
                    logger.info("Plotting SRFS in subplots, {!s}, "
                        "shifted {:~}".format(ch, sh))
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
            elif regr == "ODR":
                regrs.append((scipy.odr.odrpack.ODR, {}))
            else:
                raise RuntimeError("Impossible")
        dbref = itertools.product(
            p.db,
            p.predict_chan,
            regrs ,
            ({"lat": (*p.latrange, "all")},))
        dbref = list(dbref) # may want multiple iterations
#        dbref = [x for x in dbref if not (x[1]=="single" and
#                    x[2][0] is sklearn.cross_decomposition.PLSRegression)]
#        dbref = list(itertools.product(("different",),
#                                  ("all",)))
#        # Test, why am I getting worse with PLS than with OLS?
#        vis.plot_errdist_per_srf_costfunc_localmin(
#            "NOAA19", 6, 60.0*ureg.nm, db="different", ref="all",
#            limits={"lat": (60, 90, "all")})

        if p.plot_srf_cost:
            for (db, ref, (cls, args), limits) in dbref:
                vis.visualise_srf_estimate_costfunc(p.sat.upper(), db=db, ref=ref,
                    regression_type=cls, regression_args=args,
                    limits=limits, A=p.cost_frac_bt, B=p.cost_frac_dλ,
                    cost_mode=p.cost_mode, 
                    ref_dλ=ureg.Quantity(p.ref_shifts, ureg.nm),
                    dλ=ureg.Quantity(numpy.linspace(*p.shift_range, p.shift_count), ureg.nm),
                    sat2=p.sat2.upper(),
                    start1=start, start2=start_alt,
                    end1=end, end2=end_alt,
                    noise_level={"target": p.noise_level_target,
                                 "master": p.noise_level_master},
                    predict_quantity=p.predict_quantity,
                    noise_quantity=p.noise_quantity,
                    noise_units=p.noise_units,
                    channels=p.channels)

        if p.plot_errdist_per_localmin:
            for ch in p.channels:
                for shift in ureg.Quantity(numpy.array(p.ref_shifts),
                                  ureg.nm):
                    for (db, ref, (cls, args), limits) in dbref:
                        vis.plot_errdist_per_srf_costfunc_localmin(
                            p.sat.upper(), ch, shift, db=db, ref=ref, 
                            regression_type=cls,
                            regression_args=args,
                            limits=limits,
                            predict_quantity=p.predict_quantity,
                            noise_level={"target": p.noise_level_target,
                                         "master": p.noise_level_master},
                            noise_quantity=p.noise_quantity,
                            noise_units=p.noise_units,
                            dλ=ureg.Quantity(
                                numpy.linspace(*p.shift_range, p.shift_count),
                                               ureg.nm),
                            A=p.cost_frac_bt,
                            B=p.cost_frac_dλ,
                            cost_mode=p.cost_mode,
                            sat2=p.sat2,
                            iasi_frac=p.iasi_fraction,
                            start1=start, start2=start_alt,
                            end1=end, end2=end_alt)

        if p.plot_bt_srf_shift:
            for ch in p.channels:
                logger.info("Plotting shifts for {:s} ch. {:d}".format(
                    p.sat, ch))
                vis.plot_bt_srf_shift(p.sat, ch)

        shift = ureg.Quantity(p.shift, ureg.nm)
        if p.estimate_errorprop:
            hirs_start = datetime.datetime.strptime(p.hirs_period[0], p.datefmt)
            hirs_end = datetime.datetime.strptime(p.hirs_period[1], p.datefmt)
            for (ch_i, ch) in enumerate(p.channels):
                # I found those channels on HIRS may get local minima:
                if ch in {1, 6, 9, 11, 12}:
                    optimiser_func=scipy.optimize.basinhopping
                    optimiser_args=dict(x0=0, T=0.1, stepsize=0.03,
                                        niter=100, niter_success=20,
                                        interval=10, disp=False,
                                        minimizer_kwargs=dict(
                                            bounds=[(-0.1, +0.1)],
                                            options=dict(
                                                factr=1e12),
                                            args=(ureg.um,)))
                else:
                    optimiser_func=scipy.optimize.minimize_scalar
                    optimiser_args=dict(bracket=[-0.04, 0.04],
                                        bounds=[-0.1, 0.1],
                                        method="bounded",
                                        args=(ureg.um,))

                logger.info("Finding variation of minima for "
                    "{p.sat:s} channel {ch:d}, reference {shift:~}. "
                    "Using multiple linear regression. "
                    "Optimising with {optimiser:s}.".format(
                        p=p, ch=ch, shift=shift.to(ureg.nm),
                        optimiser=optimiser_func.__qualname__))
                if len(p.noise_level_target) > 1:
                    noise_level_target = p.noise_level_target[ch_i]
                    noise_level_master = p.noise_level_master
                else:
                    noise_level_target = p.noise_level_target[0]
                    noise_level_master = p.noise_level_master[0]
                noise_level = {"target": noise_level_target,
                               "master": noise_level_master}

                N = p.n_tries
                    
                for (db, ref, (cls, args), limits) in dbref:
                    (bias, std) = vis.estimate_errorprop_srf_recovery(p.sat, ch,
                        shift.to(ureg.um), db=db,
                        ref=ref,
                        regression_type=cls,
                        regression_args=args,
                        optimiser_func=optimiser_func,
                        optimiser_args=optimiser_args,
                        limits=limits,
                        noise_level=noise_level,
                        noise_quantity=p.noise_quantity,
                        noise_units=p.noise_units,
                        cost_mode=p.cost_mode,
                        sat2=p.sat2,
                        iasi_frac=p.iasi_fraction,
                        hirs_start=hirs_start,
                        hirs_end=hirs_end,
                        start1=start, start2=start_alt,
                        end1=end, end2=end_alt,
                        N=N,
                        predict_quantity=p.predict_quantity)
                    print("Channel {:d} shift of {:~}, noise {!s} ({:s} in ({:s})) has "
                        "bias {:~}, stderror {:~}, based on {:d} attempts".format(
                            ch, shift, noise_level, p.noise_quantity, 
                            p.noise_units, bias, std, N))

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
            for unit in ("specrad_wavenum", "Tb", "specrad_freq"):
                vis.plot_full_spectrum_with_all_channels(p.sat,
                    y_unit=unit, x_quantity="wavelength",
                    selection=selection)
                vis.plot_srf_all_sats(y_unit=unit,
                    selection=selection)
#        for h in vis.allsats:
#            try:
#                #vis.plot_Te_vs_T(h)
#                vis.plot_channel_BT_deviation(h)
#            except FileNotFoundError as msg:
#                logger.error("Skipping {:s}: {!s}".format(h, msg))
        if p.write_channel_locations != "":
            vis.write_channel_locations(p.write_channel_locations,
                channels=p.channels)
        if p.compare_hiasi_hirs:
            for i in range(25):
                vis.compare_hiasi_hirs(ch=1, start=i)

        if p.vis_expected_range:
            vis.plot_expected_range(lab="{start:%Y%m%d}-{end:%Y%m%d}".format(
                start=start, end=end), channels=p.channels)
        logger.info("Done")
