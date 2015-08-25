#!/usr/bin/env python3.4

import os
import re
import datetime
import itertools
import logging
if __name__ == "__main__":
    logging.basicConfig(
        format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
                 "%(lineno)s: %(message)s"),
        level=logging.DEBUG)

import numpy
import numpy.lib.recfunctions
import scipy.stats

import matplotlib
if not os.getenv("DISPLAY"): # None or empty string
    matplotlib.use("Agg")
    
import matplotlib.pyplot

import numexpr

import pyatmlab.datasets.tovs
import pyatmlab.io
import pyatmlab.config
import pyatmlab.physics
import pyatmlab.graphics
import pyatmlab.stats
import pyatmlab.db

from pyatmlab.constants import micro, centi, tera

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
        if self._gran is None:
            self._gran = self.iasi.read(next(self.graniter))
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
        for coor in self.choice:
            logging.info("Considering {coor!s}: Latitude {lat:.1f}°, "
                "Longitude {lon:.1f}°, Time {time!s}, SZA {sza!s})".format(
                coor=coor, lat=self.gran["lat"][coor[0], coor[1]],
                lon=self.gran["lon"][coor[0], coor[1]],
                time=self.gran["time"][coor[0], coor[1]].astype(datetime.datetime),
                sza=self.gran["solar_zenith_angle"][coor[0], coor[1]]))

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

    def get_tb_channels(self, sat, channels=slice(None)):
        """Get brightness temperature for channels
        """
        chan_nos = (numpy.arange(19) + 1)[channels]
#        specrad_wn = self.gran["spectral_radiance"]
#        specrad_f = pyatmlab.physics.specrad_wavenumber2frequency(
#                            specrad_wn)
        specrad_f = self.get_y(unit="specrad_freq")
        Tb_chans = numpy.zeros(dtype=numpy.float32,
                               shape=specrad_f.shape[0:2] + (chan_nos.size,))
        for (i, srf) in enumerate(self.srfs[sat]):
            logging.debug("Calculating channel Tb {:s}-{:d}".format(sat, i+1))
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

        bt = self.get_tb_channels(sat)
        bt2d = bt.reshape(-1, bt.shape[2])
        bt2d = bt2d[:, channels]
        bt2d = bt2d[(bt2d>0).all(1), :]

        logging.info("Calculating PCA")
        pca = matplotlib.mlab.PCA(bt2d)
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


    def build_lookup_table(self, sat, pca=False, x=30):
        """
        :param sat: Satellite, i.e. "NOAA18"
        :param bool pca: Use pca or not.
        :param x: If not PCA, this is no. of steps per channel.
            If PCA, this is the scale; see
            pyatmlab.db.SmallLookupTable.fromData.
        """
        # construct single ndarray with both tb and radiances, for binning
        # purposes
        logging.info("Constructing data")
        db = None
        for g in self.graniter:
            
            self.gran = self.iasi.read(g)
            y = self.get_y("specrad_freq")
            y = y.view(dtype=[("specrad_freq", y.dtype, y.shape[2])])
            tb = self.get_tb_channels(sat)
            tbv = tb.view([("ch{:d}".format(i+1), tb.dtype)
                     for i in range(tb.shape[-1])]).squeeze()
            y = numpy.lib.recfunctions.merge_arrays(
                (tbv, y), flatten=True, usemask=False, asrecarray=False)
    #            [self.get_y("specrad_freq").view(dtype=[("specrad_freq", 8641],
    #            usemask=False,
    #            asrecarray=False)
            if db is None: # first time
                logging.info("Constructing lookup table")
                if pca:
                    logging.info("Constructing PCA")
                    db = pyatmlab.db.LargeFullLookupTable.fromData(y,
                        dict(PCA=dict(
                            npc=4,
                            scale=x,
                            valid_range=(100, 400),
                            fields=["ch{:d}".format(i+1) for i in
                                        range(12)])),
                            use_pca=True)
                else:
                    db = pyatmlab.db.LargeFullLookupTable.fromData(y,
                        {"ch{:d}".format(i+1):
                         dict(range=(tb[..., i][tb[..., i]>0].min()*0.95,
                                     tb[..., i].max()*1.05),
                              mode="linear",
                              nsteps=x)
                            for i in {2, 5, 8, 9, 11}},
                            use_pca=False)
            else:
                logging.info("Extending lookup table")
                db.addData(y)
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
    #
    # FIXME: there are problems with the lookup table.  It is too sparse.
    # Unseen data rarely hits a bin.  Need a more intelligent method.
    lut = None
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

        # I think that when I built the lookup table, I inadvertently
        # skipped the first granule.  Until I am surer about optimising
        # the lookup table, I want to use the exact same entries, so I
        # skip the first one now again.
        # FIXME: Remove this workaround
        tb_all = []
#        tb_all = [self.get_tb_channels("NOAA18")[:, :, :12].reshape(-1,
#                    12)]
        #for g in self.graniter:
        for g in itertools.islice(self.graniter, 2):
            self.gran = self.iasi.read(g)
            tb_all.append(self.get_tb_channels("NOAA18")[:, :,
            :12].reshape(-1, 12))
        tb_all = numpy.vstack(tb_all).view(
            [("ch{:d}".format(i+1), tb_all[0].dtype) for i in range(12)])
        self.plot_lut_radiance_delta(tb_all)

def main():
    print(numexpr.set_num_threads(8))
    with numpy.errstate(all="raise"):
        vis = IASI_HIRS_analyser()
        vis.plot_lut_radiance_delta_all_iasi()
#        vis.build_lookup_table(sat="NOAA18", pca=True, x=8)
#        vis.build_lookup_table("NOAA18", N=40)
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
