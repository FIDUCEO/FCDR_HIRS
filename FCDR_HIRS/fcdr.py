"""Datasets for TOVS/ATOVS
"""

import io
import tempfile
import subprocess
import datetime
import logging
import gzip
import shutil
import abc
import pathlib
import dbm

import numpy
import scipy.interpolate

import netCDF4
import dateutil
import progressbar

try:
    import coda
except ImportError:
    logging.warn("Unable to import coda, won't read IASI EPS L1C")
    
import typhon.datasets.dataset
import typhon.utils.metaclass
import typhon.physics.units
from typhon.physics.units.common import ureg
from typhon.datasets.tovs import (Radiometer, HIRS, HIRSPOD, HIRS2,
    HIRSKLM, HIRS3, HIRS4)

from pyatmlab import tools

class HIRSFCDR:
    """Produce, write, study, and read HIRS FCDR.

    Mixin for kiddies HIRS?FCDR
    """

    realisations = 100
    srfs = None
    satname = None

    # Read in some HIRS data, including nominal calibration
    # Estimate noise levels from space and IWCT views
    # Use noise levels to propagate through calibration and BT conversion

    def __init__(self, *args, satname, **kwargs):
        self.srfs = [typhon.physics.units.em.SRF.fromArtsXML(
                     satname.upper(), "hirs", i) for i in range(1, 20)]
        super().__init__(*args, satname=satname, **kwargs)
        self.my_pseudo_fields.update(radiance_fid=self.calculate_radiance_all)
        #self.hirs = hirs
        #self.srfs = srfs

    def interpolate_between_calibs(self, M, calib_time, *args, kind="nearest"):
        """Interpolate calibration parameters between calibration cycles

        This method is just beginning and likely to improve considerably
        in the upcoming time.

        Doesn't even have to be between calibs, can be any times.

        FIXME: Currently implementing linear interpolation.

        Arguments:
        
            M [ndarray]
            
                ndarray with dtype such as returned by self.read.  Must
                contain enough fields.

            calib_time [ndarray, dtype time]

                times corresponding to offset and slope, such as returned
                by HIRS.calculate_offset_and_slope.

            *args
                
                anything defined only at calib_time, such as slope,
                offset, or noise_level
        
        Returns:

            list, corresponding to args, interpolated to all times in M
        """

        x = numpy.asarray(calib_time.astype("u8"))
        xx = numpy.asarray(M["time"].astype("u8"))
        out = []
        for y in args:
            try:
                u = y.u
            except AttributeError:
                u = None
            y = numpy.ma.asarray(y)
            # explicitly set masked data to nan, for scipy.interpolate
            # doesn't understand this
            y.data[y.mask] = numpy.nan
            fnc = scipy.interpolate.interp1d(
                x, y,
                kind=kind,
                #fill_value="extrapolate",
                fill_value="extrapolate" if kind=="nearest" else numpy.nan,
                bounds_error=False,
                axis=0)

            yy = numpy.ma.masked_invalid(fnc(xx))
            if u is None:
                out.append(yy)
            else:
                out.append(ureg.Quantity(yy, u))

        return out

    def custom_calibrate(self, counts, slope, offset):
        """Calibrate with my own slope and offset

        Currently linear.  Uncertainties currently considered upstream in
        MC sense, to be amended.
        """
        return offset[:, numpy.newaxis] + slope[:, numpy.newaxis] * counts

    def extract_calibcounts_and_temp(self, M, ch, srf=None):
        """Calculate calibration counts and IWCT temperature

        In the IR, space view temperature can be safely estimated as 0
        (radiance at 3K is around 10^200 times less than at 300K)

        Arguments:

            M

                ndarray such as returned by self.read, corresponding to
                scanlines

            ch

                Channel for which counts shall be returned and IWCT
                temperature shall be calculated.

            srf [typhon.physics.em.SRF]

                SRF object used to estimate IWCT.  Optional; if not given
                or None, use the NOAA-reported SRF for channel.

        Returns:

            time

                time corresponding to remaining arrays

            L_iwct

                radiance corresponding to IWCT views.  Calculated by
                assuming ε=1 (blackbody), an arithmetic mean of all
                temperature sensors on the IWCT, and the SRF passed to the
                method.
                        counts_iwct

                counts corresponding to IWCT views

            counts_space

                counts corresponding to space views
        """

        srf = srf or self.srfs[ch-1]

        views_space = M[self.scantype_fieldname] == self.typ_space
        views_iwct = M[self.scantype_fieldname] == self.typ_iwt

        # select instances where I have both in succession.  Should be
        # always, unless one of the two is missing or the start or end of
        # series is in the middle of a calibration.  Take this from
        # self.dist_space_iwct because for HIRS/2 and HIRS/2I, there is a
        # views_icct in-between.
        dsi = self.dist_space_iwct
        space_followed_by_iwct = (views_space[:-dsi] & views_iwct[dsi:])
        #M15[1:][views_space[:-1]]["hrs_scntyp"]

        M_space = M[:-dsi][space_followed_by_iwct]
        M_iwct = M[dsi:][space_followed_by_iwct]

        counts_space = ureg.Quantity(M_space["counts"][:, 8:, ch-1],
                                     ureg.count)
        counts_iwct = ureg.Quantity(M_iwct["counts"][:, 8:, ch-1],
                                    ureg.count)

        T_iwct = ureg.Quantity(
            M_space["temp_iwt"].mean(-1).mean(-1).astype("f4"), ureg.K)

        L_iwct = srf.blackbody_radiance(T_iwct)
        L_iwct = ureg.Quantity(L_iwct.astype("f4"), L_iwct.u)

        return (M_space["time"], L_iwct, counts_iwct, counts_space)


    def calculate_offset_and_slope(self, M, ch, srf=None):
        """Calculate offset and slope.

        Arguments:

            M [ndarray]

                ndarray with dtype such as returned by self.read.  Must
                contain enough fields.

            ch [int]

                Channel that the SRF relates to.

            srf [typhon.physics.em.SRF]

                SRF used to estimate slope.  Needs to implement the
                `blackbody_radiance` method such as `typhon.physics.em.SRF`
                does.  Optional: if not provided, use standard one.

        Returns:

            tuple (time, offset, slope) where:

            time [ndarray] corresponding to offset and slope

            offset [ndarray] offset calculated at each calibration cycle

            slope [ndarray] slope calculated at each calibration cycle

        """

        srf = srf or self.srfs[ch-1]
        (time, L_iwct, counts_iwct, counts_space) = self.extract_calibcounts_and_temp(M, ch, srf)
        L_space = ureg.Quantity(numpy.zeros_like(L_iwct), L_iwct.u)

        slope = (
            (L_iwct - L_space)[:, numpy.newaxis] /
            (counts_iwct - counts_space))

        offset = -slope * counts_space

        return (time,
                offset,
                slope)

    def calculate_radiance(self, M, ch, interp_kind="nearest", srf=None):
        """Calculate radiance

        Wants ndarray as returned by read, SRF, and channel.

        Returns pint quantity with masked array underneath.
        """
        srf = srf or self.srfs[ch-1]
        (time, offset, slope) = self.calculate_offset_and_slope(
            M, ch, srf)
        if offset.shape[0] > 1:
            (interp_offset, interp_slope) = self.interpolate_between_calibs(M, time,
                ureg.Quantity(numpy.ma.median(offset.m, 1), offset.u),
                ureg.Quantity(numpy.ma.median(slope.m, 1), slope.u),
                kind=interp_kind)
        elif offset.shape[0] == 1:
            interp_offset = numpy.ma.zeros(dtype=offset.dtype, shape=M.shape)
            interp_offset[:] = numpy.ma.median(offset.m, 1)
            interp_offset = ureg.Quantity(interp_offset, offset.u)
            interp_slope = numpy.ma.zeros(dtype=offset.dtype, shape=M.shape)
            interp_slope[:] = numpy.ma.median(slope.m, 1)
            interp_slope = ureg.Quantity(interp_slope, offset.u)
        elif M.shape[0] > 0:
            raise ValueError("Found {:d} calibration cycles, too few!".format(offset.shape[0]))
        else:
            return ureg.Quantity(
                numpy.zeros(shape=M["radiance"][:, :, ch-1].shape, dtype="f4"),
                typhon.physics.units.common.radiance_units["ir"])
        rad_wn = self.custom_calibrate(
            ureg.Quantity(M["counts"][:, :, ch-1].astype("f4"), ureg.count),
            interp_slope, interp_offset).to(typhon.physics.units.common.radiance_units["ir"], "radiance")
        rad_wn = ureg.Quantity(numpy.ma.array(rad_wn), rad_wn.u)
        rad_wn.m.mask = M["counts"][:, :, ch-1].mask
        rad_wn.m.mask |= M["radiance"][:, :, ch-1].mask
        rad_wn.m.mask |= numpy.isnan(rad_wn)
        return rad_wn
    Mtorad = calculate_radiance

    def calculate_radiance_all(self, M, interp_kind="zero", srf=None):
        """Calculate radiances for all channels

        """

        all_rad = [self.calculate_radiance(M, i, interp_kind=interp_kind)
            for i in range(1, 20)]
        return ureg.Quantity(numpy.ma.concatenate([rad.m[...,
            numpy.newaxis] for rad in all_rad], 2), all_rad[0].u)

        
    
    def estimate_noise(self, M, ch, typ="both"):
        """Calculate noise level at each calibration line.

        Currently implemented to return noise level for IWCT and space
        views.
        """
        if typ == "both":
            calib = M[self.scantype_fieldname] != self.typ_Earth
        else:
            calib = M[self.scantype_fieldname] == getattr(self, "typ_{:s}".format(typ))

        calibcounts = ureg.Quantity(M["counts"][calib, 8:, ch-1],
                                    ureg.counts)
        return (M["time"][calib], typhon.math.stats.adev(calibcounts, 1))

    def recalibrate(self, M, ch, srf=None, realisations=None):
        """Recalibrate counts to radiances with uncertainties

        Arguments:

            M [ndarray]

                Structured array such as returned by self.read.  Should
                have at least fields "hrs_scntyp", "counts", "time", and
                "temp_iwt".

            ch [int]

                Channel to calibrate.

            srf [pyatmlab.physics.SRF]

                SRF to use for calibrating the channel and converting
                radiances to units of BT.  Optional; if None, use
                “default" SRF for channel.

        TODO: incorporate SRF-induced uncertainties --- how?
        """
        srf = self.srfs[ch-1]
        if realisations is None:
            realisations = self.realisations
        logging.info("Estimating noise")
        (t_noise_level, noise_level) = self.estimate_noise(M, ch)
        # note, this can't be vectorised easily anyway because of the SRF
        # integration bit
        logging.info("Calibrating")
        (time, offset, slope) = self.calculate_offset_and_slope(M, ch, srf)
        # NOTE:
        # See https://github.com/numpy/numpy/issues/7787 on numpy.median
        # losing the unit
        logging.info("Interpolating") 
        (interp_offset, interp_slope) = self.interpolate_between_calibs(M,
            time, 
            ureg.Quantity(numpy.median(offset, 1), offset.u),
            ureg.Quantity(numpy.median(slope, 1), slope.u))
        interp_noise_level = numpy.interp(M["time"].view("u8"),
                    t_noise_level.view("u8")[~noise_level.mask],
                    noise_level[~noise_level.mask])
        logging.info("Allocating")
        rad_wn = ureg.Quantity(numpy.empty(
            shape=M["counts"].shape[:2] + (realisations,),
            dtype="f4"), typhon.physics.units.common.radiance_units["ir"])
        bt = ureg.Quantity(numpy.empty_like(rad_wn), ureg.K)
        logging.info("Estimating {:d} realisations for "
            "{:,} radiances".format(realisations,
               rad_wn.size))
        bar = progressbar.ProgressBar(maxval=realisations,
                widgets = tools.my_pb_widget)
        bar.start()
        for i in range(realisations):
            with ureg.context("radiance"):
                # need to explicitly convert .to(rad_wn.u),
                # see https://github.com/hgrecco/pint/issues/394
                rad_wn[:, :, i] = self.custom_calibrate(
                    ureg.Quantity(M["counts"][:, :, ch-1].astype("f4")
                        + numpy.random.randn(*M["counts"].shape[:-1]).astype("f4")
                            * interp_noise_level[:, numpy.newaxis],
                                 ureg.count).astype("f4"),
                    interp_slope, interp_offset).to(rad_wn.u)
                    
    
            bt[:, :, i] = ureg.Quantity(
                srf.channel_radiance2bt(rad_wn[:, :, i]).astype("f4"),
                ureg.K)
            bar.update(i)
        bar.finish()
        logging.info("Done")

        return (rad_wn, bt)

    def read_and_recalibrate_period(self, start_date, end_date):
        M = self.read(start_date, end_date,
                fields=["time", "counts", "bt", "calcof_sorted"])
        return self.recalibrate(M)

    def extract_and_interp_calibcounts_and_temp(self, M, ch, srf=None):
        srf = srf or self.srfs[ch-1]
        (time, L_iwct, C_iwct, C_space) = self.extract_calibcounts_and_temp(M, ch, srf)
        views_Earth = M[self.scantype_fieldname] == self.typ_Earth
        C_Earth = M["counts"][views_Earth, :, ch-1]
        # interpolate all of those to cover entire time period
        (L_iwct, C_iwct, C_space) = self.interpolate_between_calibs(
            M, time, L_iwct, C_iwct, C_space)
        (C_Earth,) = self.interpolate_between_calibs(
            M, M["time"][views_Earth], C_Earth)
        C_space = ureg.Quantity(numpy.median(C_space, 1), C_space.u)
        C_iwct = ureg.Quantity(numpy.median(C_iwct, 1), C_iwct.u)
        C_Earth = ureg.Quantity(C_Earth, ureg.counts)

        return (L_iwct, C_iwct, C_space, C_Earth)

    def calc_sens_coef(self, typ, M, ch, srf): 
        """Calculate sensitivity coefficient.

        Actual work is delegated to calc_sens_coef_{name}

        Arguments:

            typ
            M
            SRF
            ch
        """

        f = getattr(self, "calc_sens_coef_{:s}".format(typ))

        (L_iwct, C_iwct, C_space, C_Earth) = (
            self.extract_and_interp_calibcounts_and_temp(M, ch, srf))

        return f(L_iwct[:, numpy.newaxis], C_iwct[:, numpy.newaxis],
                 C_space[:, numpy.newaxis], C_Earth)
    
    def calc_sens_coef_C_Earth(self, L_iwct, C_iwct, C_space, C_Earth):
        return L_iwct / (C_iwct - C_space)

    def calc_sens_coef_C_iwct(self, L_iwct, C_iwct, C_space, C_Earth):
        return - L_iwct * (C_Earth - C_space) / (C_iwct - C_space)**2

    def calc_sens_coef_C_space(self, L_iwct, C_iwct, C_space, C_Earth):
        return L_iwct * (C_Earth - C_iwct) / (C_iwct - C_space)**2

    def calc_urad(self, typ, M, ch, *args, srf=None):
        """Calculate uncertainty

        Arguments:

            typ [str]
            
                Sort of uncertainty.  Currently implemented: "noise" and
                "calib".

            M
            ch

            *args

                Depends on the sort of uncertainty, but should pass all
                the "base" uncertainties needed for propagation.  For
                example, for calib, must be u_C_iwct and u_C_space.

            srf
                
                Only if different from the nominal
        """

        srf = srf or self.srfs[ch-1]
        f = getattr(self, "calc_urad_{:s}".format(typ))
        (L_iwct, C_iwct, C_space, C_Earth) = (
            self.extract_and_interp_calibcounts_and_temp(M, ch, srf))
        return f(L_iwct[:, numpy.newaxis],
                 C_iwct[:, numpy.newaxis],
                 C_space[:, numpy.newaxis], C_Earth, *args)

    def calc_urad_noise(self, L_iwct, C_iwct, C_space, C_Earth, u_C_Earth):
        """Calculate uncertainty due to random noise
        """

        s = self.calc_sens_coef_C_Earth(L_iwct, C_iwct, C_space, C_Earth)
        return abs(s) * u_C_Earth

    def calc_urad_calib(self, L_iwct, C_iwct, C_space, C_Earth,
                              u_C_iwct, u_C_space):
        s_iwct = self.calc_sens_coef_C_iwct(
                    L_iwct, C_iwct, C_space, C_Earth)
        s_space = self.calc_sens_coef_C_space(
                    L_iwct, C_iwct, C_space, C_Earth)
        return numpy.sqrt((s_iwct * u_C_iwct)**2 +
                    (s_space * u_C_space)**2)

    def calc_S_noise(self, u):
        """Calculate covariance matrix between two uncertainty vectors

        Random noise component, so result is a diagonal
        """

        if u.ndim == 1:
            return ureg.Quantity(numpy.diag(u**2), u.u**2)
        elif u.ndim == 2:
            # FIXME: if this is slow, I will need to vectorise it
            return ureg.Quantity(
                numpy.rollaxis(numpy.dstack(
                    [numpy.diag(u[i, :]**2) for i in range(u.shape[0])]),
                    2, 0),
                u.u**2)
        else:
            raise ValueError("u must have 1 or 2 dims, found {:d}".format(u.ndim))

    def calc_S_calib(self, u, c_id):
        """Calculate covariance matrix between two uncertainty vectors

        Calibration (structured random) component.

        For initial version of my own calibration implementation, where
        only one calibartion propagates into each uncertainty.

        FIXME: make this vectorisable

        Arguments:
            
            u [ndarray]

                Vector of uncertainties.  Last dimension must be the
                dimension to estimate covariance matrix for.

            c_id [ndarray]

                Vector with identifier for what calibration cycle was used
                in each.  Most commonly, the time.  Shape must match u.
        """

        u = ureg.Quantity(numpy.atleast_2d(u), u.u)
        u_cross = u[..., numpy.newaxis] * u[..., numpy.newaxis].swapaxes(-1, -2)

        # r = 1 when using same calib, 0 otherwise...
        c_id = numpy.atleast_2d(c_id)
        r = (c_id[..., numpy.newaxis] == c_id[..., numpy.newaxis].swapaxes(-1, -2)).astype("f4")

        S = u_cross * r

        #S.mask |= (u[:, numpy.newaxis].mask | u[numpy.newaxis, :].mask) # redundant

        return S.squeeze()

    def calc_S_srf(self, u):
        """Calculate covariance matrix between two uncertainty vectors

        Component due to uncertainty due to SRF
        """
        
        raise NotImplementedError("Not implemented yet!")

class HIRS2FCDR(HIRSFCDR, HIRS2):
    pass

class HIRS3FCDR(HIRSFCDR, HIRS3):
    pass

class HIRS4FCDR(HIRSFCDR, HIRS4):
    pass

def which_hirs_fcdr(satname):
    """Given a satellite, return right HIRS object
    """
    for h in {HIRS2FCDR, HIRS3FCDR, HIRS4FCDR}:
        for (k, v) in h.satellites.items():
            if satname in {k}|v:
                return h(satname=k)
    else:
        raise ValueError("Unknown HIRS satellite: {:s}".format(satname))
