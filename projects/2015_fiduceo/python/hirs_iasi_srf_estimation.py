#!/usr/bin/env python3.4

import os
import itertools
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
                

    def __init__(self):
        logging.info("Finding and reading IASI")
        self.iasi = pyatmlab.datasets.tovs.IASI(name="iasi")
        self.graniter = self.iasi.find_granules()
        self.gran = self.iasi.read(next(self.graniter))
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

    def estimate_optimal_channel_binning(self, sat="NOAA18", N=5, p=20):
        """What HIRS channel combi optimises variability?

        :param sat: Satellite to use
        :param N: Number of channels in lookup table
        :param p: Number of bins per channel
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
            # flattened count
            no = numpy.array([b.size for b in bnd.ravel()])
            #
            frac = (no>0).sum() / no.size
            #
            lowest = no[no>0].min()
            highest = no.max()
            med = int(numpy.median(no[no>0]))
            logging.info("{:d}/{:d} channel combination {!s}: {:.3%} {:d}/{:d}/{:d}".format(
                  k, tot, combi, frac, lowest, med, highest))

    def build_lookup_table(self, sat, N=30):
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
                db = pyatmlab.db.LargeFullLookupTable.fromData(y,
                    {"ch{:d}".format(i+1):
                     dict(range=(tb[..., i][tb[..., i]>0].min()*0.95,
                                 tb[..., i].max()*1.05),
                          mode="linear",
                          nsteps=N)
                        for i in {2, 5, 8, 9, 11}})
            else:
                logging.info("Extending lookup table")
                db.addData(y)
#        out = "/group_workspaces/cems/fiduceo/Users/gholl/hirs_lookup_table/test/test_{:%Y%m%d-%H%M%S}.dat".format(datetime.datetime.now())
#        logging.info("Storing lookup table to {:s}".format(out))
#        db.toFile(out)


def main():
    print(numexpr.set_num_threads(8))
    with numpy.errstate(all="raise"):
        vis = IASI_HIRS_analyser()
#        vis.build_lookup_table("NOAA18", N=40)
#        vis.estimate_optimal_channel_binning("NOAA18", 5, 10)
        for unit in {"Tb", "specrad_freq"}:
#            vis.plot_full_spectrum_with_all_channels("NOAA18",
#                y_unit=unit)
            vis.plot_srf_all_sats(y_unit=unit)
#        for h in vis.allsats:
#            try:
#                #vis.plot_Te_vs_T(h)
#                vis.plot_channel_BT_deviation(h)
#            except FileNotFoundError as msg:
#                logging.error("Skipping {:s}: {!s}".format(h, msg))
#        logging.info("Done")

if __name__ == "__main__":
    main()
