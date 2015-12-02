#!/usr/bin/env python3.5

import sys
import os
import re
import datetime
import itertools
import functools
import pickle
import logging
if __name__ == "__main__":
    logging.basicConfig(
        format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
                 "%(lineno)s: %(message)s"),
        level=logging.DEBUG)
import pathlib
import argparse

import numpy
import numpy.lib.recfunctions
import scipy.stats

import matplotlib
if not os.getenv("DISPLAY"): # None or empty string
    matplotlib.use("Agg")
    
import matplotlib.pyplot

import progressbar
import numexpr
import mpl_toolkits.basemap

import pyatmlab.datasets.tovs
import pyatmlab.io
import pyatmlab.config
import pyatmlab.physics
import pyatmlab.graphics
import pyatmlab.stats
import pyatmlab.db

from pyatmlab.constants import micro, centi, tera, nano

class IASI_HIRS_analyser:
    colors = ("brown orange magenta burlywood tomato indigo "
              "moccasin cyan teal khaki tan steelblue "
              "olive gold darkorchid pink midnightblue "
              "crimson orchid olive chocolate sienna").split()
    allsats = (pyatmlab.datasets.tovs.HIRS2.satellites |
               pyatmlab.datasets.tovs.HIRS3.satellites |
               pyatmlab.datasets.tovs.HIRS4.satellites)
    allsats = {re.sub(r"0(\d)", r"\1", sat).upper() for sat in allsats}

    x = dict(converter=dict(
                wavelength=pyatmlab.physics.frequency2wavelength,
                wavenumber=pyatmlab.physics.frequency2wavenumber,
                frequency=lambda x: x),
             factor=dict(
                wavelength=micro,
                wavenumber=centi,
                frequency=tera),
             label=dict(
                wavelength="Wavelength [µm]",
                wavenumber="Wave number [cm^-1]",
                frequency="Frequency [THz]"))
                
    _iasi = None
    @property
    def iasi(self):
        if self._iasi is None:
            self._iasi = pyatmlab.datasets.tovs.IASI(name="iasi")
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

    def __init__(self):
        logging.info("Finding and reading IASI")
        #self.iasi = pyatmlab.datasets.tovs.IASI(name="iasi")
        #self.graniter = self.iasi.find_granules()
        #self.gran = self.iasi.read(next(self.graniter))
        self.choice = [(38, 47), (37, 29), (100, 51), (52, 11)]

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
        return (y, y_label) if return_label else y

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

    def get_tb_channels(self, sat, channels=range(1, 13)):
        """Get brightness temperature for channels
        """
        #chan_nos = (numpy.arange(19) + 1)[channels]
#        specrad_wn = self.gran["spectral_radiance"]
#        specrad_f = pyatmlab.physics.specrad_wavenumber2frequency(
#                            specrad_wn)
        specrad_f = self.get_y(unit="specrad_freq")
        Tb_chans = numpy.zeros(dtype=numpy.float32,
                               shape=specrad_f.shape[0:2] + (len(channels),))
        for (i, c) in enumerate(channels):
            srf = self.srfs[sat][c-1]
        #for (i, srf) in enumerate(self.srfs[sat]):
            logging.debug("Calculating channel Tb {:s}-{:d}".format(sat, c))
            #srfobj = pyatmlab.physics.SRF(freq, weight)
            L = srf.integrate_radiances(self.iasi.frequency, specrad_f)

            Tb_chans[:, :, i] = srf.channel_radiance2bt(L)
        return Tb_chans


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

    def plot_full_spectrum_with_all_channels(self, sat,
            y_unit="Tb"):
#        Tb_chans = self.get_tb_for_channels(hirs_srf)

        (y, y_label) = self.get_y(y_unit, return_label=True)
        logging.info("Visualising")
        (f, a) = matplotlib.pyplot.subplots()

        # Plot spectrum
        #a.plot(iasi.frequency, specrad_freq[i1, i2, :])
        for c in self.choice:
            a.plot(self.iasi.frequency, y[c[0], c[1], :])
        #a.plot(iasi.wavelength, Tb[i1, i2, :])
        a.set_ylabel(y_label)
        a.set_xlabel("Frequency [Hz]")
        a.set_title("Some arbitrary IASI spectra with nominal {sat} HIRS"
                        " SRFs".format(sat=sat))

        # Plot channels
        a2 = a.twinx()
        for (i, srf) in enumerate(self.srfs[sat]):
            #wl = pyatmlab.physics.frequency2wavelength(srf.f)
            a2.plot(srf.f, 0.8 * srf.W/srf.W.max(), color="black")
            nomfreq = srf.centroid()
            #nomfreq = pyatmlab.physics.frequency2wavelength(srf.centroid())
            #nomfreq = freq[numpy.argmax(srf.W)]
            #nomfreq = wl[numpy.argmax(weight)]
            a2.text(nomfreq, 0.9, "{:d}".format(i+1))

        a2.set_ylim(0, 1)

#        a.bar(hirs_centres, Tb_chans[self.choice[0], self.choice[1], :], width=2e11, color="red", edgecolor="black",
#              align="center")

        pyatmlab.graphics.print_or_show(f, False,
            "iasi_with_hirs_srf_{:s}_{:s}.".format(sat, y_unit))

    def plot_srf_all_sats(self, x_quantity="wavelength", y_unit="TB"):
        """Plot part of the spectrum with channel SRF for all sats
        """

        #hirs_srf = {}

        (y, y_label) = self.get_y(y_unit, return_label=True)
        Tb_chans = {}
        for (sat, srf) in self.srfs.items():
            #sat = re.sub(r"0(\d)", r"\1", sat)
            #sat = sat.upper()
#            try:
#                (_, hirs_srf[sat]) = pyatmlab.io.read_arts_srf(
#                    pyatmlab.config.conf["hirs"]["srf_backend_f"].format(sat=sat),
#                    pyatmlab.config.conf["hirs"]["srf_backend_response"].format(sat=sat))
#            except FileNotFoundError as err:
#                logging.error("Skipped {!s}: {!s}".format(sat, err))
#            else:
#                logging.info("Calculating channel radiances for {:s}".format(sat))
            Tb_chans[sat] = self.get_tb_channels(sat)

        for i in range(19):
            ch = i + 1
            (f, a) = matplotlib.pyplot.subplots()
            #spectrum = y[self.choice[0], self.choice[1], :]
            spectra = [y[c[0], c[1], :] for c in self.choice]
            a.set_ylabel(y_label)
            a.set_xlabel(self.x["label"][x_quantity])
            a.set_title("A IASI spectrum with different HIRS SRF (ch."
                        "{:d})".format(ch))
            a.grid(axis="y", which="both")
            #a.set_ylim(200, 300)
            a2 = a.twinx()
            (freq_lo, freq_hi) = (1e14, 0)
            # Plot SRFs for all channels
            for (color, (sat, srf)) in zip(self.colors, self.srfs.items()):
#                (freq, weight) = srf[i]
                x = self.x["converter"][x_quantity](srf[i].f)
                a2.plot(x/self.x["factor"][x_quantity],
                        srf[i].W/srf[i].W.max(),
                        label=sat[0] + "-" + sat[-2:].lstrip("SA"),
                        color=color)
                freq_lo = min(freq_lo, srf[i].f.min())
                freq_hi = max(freq_hi, srf[i].f.max())
                a.plot(
                    self.x["converter"][x_quantity](numpy.atleast_1d(srf[i].centroid()))/self.x["factor"][x_quantity],
                    numpy.atleast_2d(
                        [Tb_chans[sat][c[0], c[1], i] for c in self.choice]),
                       markerfacecolor=color,
                       markeredgecolor="black",
                       marker="o", alpha=0.5,
                       markersize=10, linewidth=1.5,
                       zorder=10)
                pyatmlab.io.write_data_to_files(
                    numpy.vstack(
                        (x/self.x["factor"][x_quantity],
                         srf[i].W/srf[i].W.max())).T,
                    "SRF_{:s}_ch{:d}_{:s}".format(sat, ch, x_quantity))
            # Plot IASI spectra
            for spectrum in spectra:
                a.plot(self.x["converter"][x_quantity](self.iasi.frequency)/self.x["factor"][x_quantity], spectrum,
                       linewidth=1.0, zorder=5)
            freq_lo = max(freq_lo, self.iasi.frequency.min())
            freq_hi = min(freq_hi, self.iasi.frequency.max())
            box = a.get_position()
            for ax in (a, a2):
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#            a.set_ylim(0, 1)
            a2.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
#            y_in_view = spectrum[(self.iasi.frequency>freq_lo) &
#                                 (self.iasi.frequency<freq_hi)]
            y_in_view = [spectrum[(self.iasi.frequency>freq_lo) &
                                  (self.iasi.frequency<freq_hi)]
                         for spectrum in spectra]
#            a.set_ylim(y_in_view.min(), y_in_view.max())
            a.set_ylim(min([yv.min() for yv in y_in_view]),
                       max([yv.max() for yv in y_in_view]))
            x_lo = self.x["converter"][x_quantity](freq_lo)/self.x["factor"][x_quantity]
            x_hi = self.x["converter"][x_quantity](freq_hi)/self.x["factor"][x_quantity]
            #wl_lo = pyatmlab.physics.frequency2wavelength(freq_lo)
            #wl_hi = pyatmlab.physics.frequency2wavelength(freq_hi)
            #a.set_xlim(wl_lo/micro, wl_hi/micro)
            #a2.set_xlim(wl_lo/micro, wl_hi/micro)
            a.set_xlim(min(x_lo, x_hi), max(x_lo, x_hi))
            a2.set_xlim(min(x_lo, x_hi), max(x_lo, x_hi))
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

        T = numpy.linspace(150, 330, 1000)
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
                widgets=[progressbar.Bar("=", "[", "]"), " ",
                         progressbar.Percentage()])
        bar.start()
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
     

    def calc_bt_srf_shift(self, satellite, channel, shift):
        """Calculate BT under SRF shifts

        Per satellite, channel, and shift.

        Expects that self.gran is already set.

        :param str satellite: Satellite, i.e., NOAA18
        :param int channel: Channel, i.e., 11
        :param ndarray shift: Shift in Hz.
        """

        y = self.get_y("specrad_freq")

        srf_nom = self.srfs[satellite][channel-1] # index 0 -> channel 1 etc.
        L_nom = srf_nom.integrate_radiances(self.iasi.frequency, y)
        bt_nom = srf_nom.channel_radiance2bt(L_nom)

        yo = numpy.zeros(shape=bt_nom.shape + shift.shape)

        for (i, sh) in enumerate(numpy.atleast_1d(shift).flat):
            srf_new = srf_nom.shift(sh)
            L_new = srf_new.integrate_radiances(self.iasi.frequency, y)
            bt_new = srf_new.channel_radiance2bt(L_new)
            yo.reshape(*yo.shape[:2], -1)[:, :, i] = bt_new

        return yo
       
    def plot_bt_srf_shift(self, satellite, channel):
        """Plot BT changes due to SRF shifts

        :param str satellite: Satellite, i.e., NOAA18
        :param int channel: Channel, i.e., 11
        """

        if self.gran is None:
            self.gran = self.iasi.read(next(self.graniter))
        
        dx = numpy.linspace(-1.5e11, 1.5e11, 7)
        # diff in um for human-readability
        d_um = (pyatmlab.physics.frequency2wavelength(
                    self.srfs[satellite][channel-1].centroid()) 
              - pyatmlab.physics.frequency2wavelength(
                    self.srfs[satellite][channel-1].centroid()-dx))
        sh = self.calc_bt_srf_shift(satellite, channel, dx)
        nsh = sh[:, :, sh.shape[2]//2]
        dsh = sh - nsh[:, :, numpy.newaxis]
        btbins = numpy.linspace(nsh[nsh>0].min(), nsh.max(), 50)
        ptiles = numpy.array([5, 25, 50, 75, 90])

        scores = numpy.zeros(shape=(btbins.size, ptiles.size, dsh.shape[2]))
        (f, a) = matplotlib.pyplot.subplots()
        for i in range(dsh.shape[2]):
            btbinned = pyatmlab.stats.bin(nsh.ravel(), dsh[:, :, i].ravel(), btbins)
            scores[:, :, i] = numpy.vstack([scipy.stats.scoreatpercentile(b, ptiles) for b in btbinned])

            for k in range(ptiles.size):
                a.plot(btbins, scores[:, k, i], color=self.colors[i],
                       ls=(":", "--", "-", "--", ":")[k],
                       label="shift {:3d} nm".format(int(d_um[i]//nano)) if k==2 else None)
        a.set_title("Effect of HIRS SRF shift, {:s} ch. {:d}".format(satellite, channel))
        a.legend(loc="upper left")
        a.set_xlabel("BT [K]")
        a.set_ylabel(r"\Delta BT [K]")
        pyatmlab.graphics.print_or_show(f, False,
            "srf_shift_direct_estimate_HIRS_{:s}-{:d}.".format(satellite, channel))


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

def main():
    print(numexpr.set_num_threads(8))
    parser = argparse.ArgumentParser("Experiment with HIRS SRF estimation")
    parser.add_argument("--makelut", action="store_true", default=False)
    parser.add_argument("--pca", action="store_true", default=False)
    parser.add_argument("--sat", action="store", default="NOAA18")
    parser.add_argument("--factor", action="store", type=float)
    parser.add_argument("--npc", action="store", type=int)
    parser.add_argument("--channels", action="store", type=int,
                        default=list(range(1, 13)), nargs="+")
    p = parser.parse_args()
    print(p)

    with numpy.errstate(all="raise"):
        vis = IASI_HIRS_analyser()
        h = pyatmlab.datasets.tovs.HIRS3()

        if p.makelut:
            print("Making LUT only")
            vis.get_lookup_table(sat=p.sat, pca=p.pca, x=p.factor,
                                 npc=p.npc, channels=p.channels)
            return

        if vis.gran is None:
            vis.gran = vis.iasi.read(next(vis.graniter))
            
        vis.map_with_hirs_pca(h, "NOAA16")
        for i in range(1, 13):
            vis.plot_bt_srf_shift("NOAA18", i)
            vis.map_with_hirs(h, "NOAA16", i)
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
#        for unit in {"Tb", "specrad_freq"}:
#            vis.plot_full_spectrum_with_all_channels("NOAA18",
#                y_unit=unit)
#            vis.plot_srf_all_sats(y_unit=unit)
#        for h in vis.allsats:
#            try:
#                #vis.plot_Te_vs_T(h)
#                vis.plot_channel_BT_deviation(h)
#            except FileNotFoundError as msg:
#                logging.error("Skipping {:s}: {!s}".format(h, msg))
#        logging.info("Done")

if __name__ == "__main__":
    main()
