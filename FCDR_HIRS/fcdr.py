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

from . import models
from . import effects
from . import measurement_equation as me

class HIRSFCDR:
    """Produce, write, study, and read HIRS FCDR.

    Some of the methods need context-information.  A class that helps in
    passing in the requirement information is at
    FCDR_HIRS.processing.generate_fcdr.FCDRGenerator.

    Mixin for kiddies HIRS?FCDR

    Relevant papers:
    - NOAA: cao07_improved_jaot.pdf
    - PDF_TEN_990007-EPS-HIRS4-PGS.pdf
    """

    realisations = 100
    srfs = None
    satname = None

    # NB: first 8 views of space counts deemed always unusable, see
    # NOAA or EUMETSAT calibration papers/documents.  I've personaly
    # witnessed (on NOAA-18) that later positions are sometimes also
    # systematically offset
    start_space_calib = 8
    start_iwct_calib = 8

    # Read in some HIRS data, including nominal calibration
    # Estimate noise levels from space and IWCT views
    # Use noise levels to propagate through calibration and BT conversion

    def __init__(self, *args, satname, **kwargs):
        for nm in {satname}|self.satellites[satname]:
            try:
                self.srfs = [typhon.physics.units.em.SRF.fromArtsXML(
                             nm, "hirs", i) for i in range(1, 20)]
            except FileNotFoundError:
                pass # try the next one
            else:
                break
        else:
            raise ValueError("Could not find SRF for any of: {:s}".format(
                ','.join({satname}|self.satellites[satname])))
        super().__init__(*args, satname=satname, **kwargs)
        # if the user has asked for headers to be returned, M is a tuple
        # (head, lines) so we need to extract the lines.  Otherwise M is
        # just lines.
        # the following line means the pseudo field is only calculated if
        # the value of the keyword "calibrate" (passed to
        # read/read_period/…) is equal to any of the values in the tuple
        cond = {"calibrate": (None, True)}
        self.my_pseudo_fields["radiance_fid"] = (
            ["radiance", self.scantype_fieldname, "temp_iwt", "time"],
            lambda M, D:
            self.calculate_radiance_all(
                M[1] if isinstance(M, tuple) else M, interp_kind="zero"),
            cond)
        self.my_pseudo_fields["bt_fid"] = (["radiance_fid"],
            self.calculate_bt_all,
            cond)

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
            if not numpy.isscalar(y.mask):
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

        Currently linear.
        """
        return offset[:, numpy.newaxis] + slope[:, numpy.newaxis] * counts

    def extract_calibcounts_and_temp(self, M, ch, srf=None,
            return_u=False, return_ix=False):
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

            return_u [bool]

                Also return uncertainty estimates.  Defaults to False.

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

            u_counts_iwct

                (if return_u is True)

            u_counts_space

                (if return_u is True)

            ix

                (if return_ix is True)
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

        counts_space = ureg.Quantity(M_space["counts"][:,
            self.start_space_calib:, ch-1], ureg.count)
        self._quantities[me.symbols["C_space"]] = self._quantity_to_xarray(
            counts_space, name="C_space")
        # For IWCT, at least EUMETSAT uses all 56…
        counts_iwct = ureg.Quantity(M_iwct["counts"][:,
            self.start_iwct_calib:, ch-1], ureg.count)
        self._quantities[me.symbols["C_IWCT"]] = self._quantity_to_xarray(
            counts_iwct, name="C_IWCT")

        # FIXME wart: I should use the IWCT observation line, not the
        # space observation line, for the IWCT temperature measurement…
        T_iwct = ureg.Quantity(
            M_space["temp_iwt"].mean(-1).mean(-1).astype("f4"), ureg.K)
        self._quantities[me.symbols["T_IWCT"]] = self._quantity_to_xarray(
            T_iwct, name="T_IWCT_calib_mean")

        L_iwct = srf.blackbody_radiance(T_iwct)
        self._quantities[me.symbols["R_IWCT"]] = self._quantity_to_xarray(
            L_iwct, name="R_IWCT")
        L_iwct = ureg.Quantity(L_iwct.astype("f4"), L_iwct.u)

        extra = []
        if return_u:
            u_counts_iwct = (typhon.math.stats.adev(counts_iwct, 1) /
                numpy.sqrt(counts_iwct.shape[1]))
            u_counts_space = (typhon.math.stats.adev(counts_space, 1) /
                numpy.sqrt(counts_space.shape[1]))
            extra.extend([u_counts_iwct, u_counts_space])
        if return_ix:
            extra.append(space_followed_by_iwct.nonzero()[0])
            
        return (M_space["time"], L_iwct, counts_iwct, counts_space) + tuple(extra)

    def _quantity_to_xarray(self, quantity, *args, **kwargs):
        """Convert quantity to xarray

        Quantity can be masked and with unit, which will be converted.
        """
        
        da = xarray.DataArray(numpy.asarray(quantity), name,
            dims=self.data_vars_props[name][1],
            attrs=self.data_vars_props[name][2],
            encoding=self.data_vars_props[name][3],
            dtype="f4") # DataArray only supports masking for floats
        da.attrs.setdefault("units", str(getattr(da, "u", "UNDEFINED")))
        da[quantity.mask] = numpy.nan
        return da


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

                If true, additionally return uncertainties on offset and
                slope.

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

        # sometimes IWCT or space counts seem to drift over a “scan line”
        # of calibration.  Identify this by comparing the IQR to the
        # counts.  For truly random normally distributed data:
        # (25, 75) … > 2: false positive 0.2%
        # (10, 90) … > 3.3: false positive 0.5%
        # …based on a simple simulated # experiment.
        bad_iwct = (scipy.stats.iqr(counts_iwct, 1, (10, 90)) > 3.3 *
            typhon.math.stats.adev(counts_iwct, 1))
        bad_space = (scipy.stats.iqr(counts_space, 1, (10, 90)) > 3.3 *
            typhon.math.stats.adev(counts_space, 1))

        bad = bad_iwct | bad_space
        slope.mask |= bad[:, numpy.newaxis]
        offset.mask |= bad[:, numpy.newaxis]
        self._quantities[me.symbols["a_0"]] = self._quantity_to_xarray(
            offset, name="offset")
        self._quantities[me.symbols["a_1"]] = self._quantity_to_xarray(
            slope, name="slope")
        return (time,
                offset,
                slope)

    _quantities = {}
    def calculate_radiance(self, M, ch, interp_kind="nearest", srf=None):
        """Calculate radiance

        Wants ndarray as returned by read, SRF, and channel.

        Returns pint quantity with masked array underneath.
        """
        # When calculating uncertainties I depend on the same quantities
        # as when calculating radiances, so I really should keep track of
        # the quantities I calculate so I can use them for the
        # uncertainties after.
        self._quantities.clear() # don't accidentally use old quantities…

        srf = srf or self.srfs[ch-1]
        (time, offset, slope) = self.calculate_offset_and_slope(
            M, ch, srf)
        # NOTE: taking the median may not be an optimal solution.  See,
        # for example, plots produced by the script
        # plot_hirs_calibcounts_per_scanpos in the FCDR_HIRS package
        # within FIDUCEO, in particular for noaa18 channels 1--12, where
        # the lowest scan positions are systematically offset compared to
        # the higher ones.  See also the note at
        # calculate_offset_and_slope. 
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
            raise typhon.datasets.dataset.InvalidFileError("Found {:d} calibration cycles, too few!".format(offset.shape[0]))
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

    def calculate_bt_all(self, M, D): 
        bt_all = ureg.Quantity(
            numpy.ma.concatenate(
                [self.srfs[ch-1].channel_radiance2bt(
                    D["radiance_fid"][:, :, ch-1])[..., numpy.newaxis]#.astype("f4")
                        for ch in range(1, 20)], 2),
            ureg.K)
        if numpy.isscalar(bt_all.m.mask):
            bt_all.m.mask = D["radiance_fid"].mask
        else:
            bt_all.m.mask |= D["radiance_fid"].mask
        return bt_all
   
    def estimate_noise(self, M, ch, typ="both"):
        """Calculate noise level at each calibration line.

        Currently implemented to return noise level for IWCT and space
        views.

        Warning: this does not ensure that only space views followed by
        IWCT views are taken into account.  If you need such an assurance,
        use extract_calibcounts_and_temp instead.
        """
        if typ == "both":
            calib = M[self.scantype_fieldname] != self.typ_Earth
        else:
            calib = M[self.scantype_fieldname] == getattr(self, "typ_{:s}".format(typ))

        calibcounts = ureg.Quantity(M["counts"][calib, 8:, ch-1],
                                    ureg.counts)
        return (M["time"][calib],
                typhon.math.stats.adev(calibcounts, 1) /
                    numpy.sqrt(calibcounts.shape[1]))

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

    rself_model = None
    def estimate_Rself(self, ds_core, ds_context):
        """Estimate self-emission and associated uncertainty

        Arguments:
            
            ds_core [xarray.Dataset]

                Data for which to estimate self-emission.  Should be an
                xarray Dataset covering the period for which the
                self-emission shall be evaluated.

            ds_context [xarray.Dataset]

                Context data.  Should be an xarray Dataset containing a
                longer period than ds_core, will be used to estimate the
                self-emission model parameters and uncertainty.  Must
                contain for calibration lines (space and IWCT views)
                including temperatures and space/IWCT counts.

        Returns:

            xarray.DataArray?
        """
        
        if self.rself_model is None:
            self.rself_model = models.RSelf(self)

        raise NotImplementedError("Not implemented yet")


    def calc_u_for_variable(self, var, quantities, all_effects,
                            cached_uncertainties):
        """Calculate total uncertainty

        This just gathers previously calculated quantities; this should be
        called after all effects have been populated and the measurement
        equation has been evaluated.

        Arguments:

            var [str] or [symbol]

                Variable for which to calculate uncertainty

            quantities

                Dictionary with numerical values for quantities

            all_effects

                Dictionary with sets of effects (effect.Effect objects)
                with magnitudes filled in.

            cached_uncertainties

                Dictionary with cached uncertainties for quantities that
                for which we do not directly estimate uncertainties, but
                that are expressions of other quantities including
                uncertainties and effects (i.e. R_IWCT uncertainty results
                from uncertainties in T_IWCT, ε, φ, etc.).  Note that this
                dictionary will be changed by this function!
        """

        # Traversing down the uncertainty expression for the measurement
        # equation.  Example expression for u²(R_e):
        #
#    4  2          2  2                      2  2         2          2             2    
# C_E ⋅u (a₂) + C_E ⋅u (a₁) + (2⋅C_E⋅a₂ + a₁) ⋅u (C_E) + u (O_Re) + u (R_selfE) + u (a₀)
        #
        # Here:
        #
        #   1. C_E, a₂ are values that gets directly substituted,
        #
        #   2. u(a₂), u(C_E), u(O_Re), u(R_selfE) are uncertainties that
        #      get directly substituted,
        #
        #   3. u(a₁) and u(a₀) are uncertainties on a sub-expression, for which we
        #      will recursively call ourselves,
        #
        #   4. a₁ has an expression that needs to be evaluated, but should
        #      already have been prior to calling this method, so should be
        #      provided in some way.
        #
        # We need to build a dictionary where for each expression/symbol
        # we need the values, so we can substitute them and get a value
        # for the uncertainty.
        # 
        # We should make use of a cache (dictionary).  For cases (1) and
        # (4), those get built when evaluating the measurement equation.
        # For case (2), the cache should be built into the effects tables,
        # when those are being populated.  For case (3), recursively call
        # myself but also fill in cached_uncertainties dictionary.

        s = me.symbols.get(var, var)

        if s not in me.expressions.keys():
            # If there is no expression for this value, the uncertainty is
            # simply what should have already been calculated
            #all_effects = effects.effects()

            if s in all_effects.keys():
                # Responsibility to put name and attributes onto effect
                # belong to effects.Effect.magnitude setter property.
                return functools.reduce(
                    operator.add,
                    (eff.magnitude for eff in all_effects[s]))
            else:
                return xarray.DataArray(0, name="u_{!s}".format(var),
                    attrs={"quantity": str(var), "note":
                        "No documented effect associated with this quantity"})

        # evaluate expression for this quantity
        e = me.expressions[me.symbols.get(var, var)]
        u_e = typhon.physics.metrology.express_uncertainty(e)
        fu = sympy.Function("u")
        args = typhon.physics.metrology.recursive_args(u_e,
            stop_at=(sympy.Symbol, sympy.Indexed, fu))

        # NB: adict is the dictionary of everything (uncertainties and
        # quantities) that needs to be
        # substituted to evaluate the magnitude of the uncertainty.
        # cached_uncertainties is a dictionary persistent between function
        # calls (until cleared) to avoid recalculating identical
        # expressions
        adict = {}
        for v in args:
            # check which one of the four aforementioned applies
            if isinstance(v, fu):
                # it's an uncertainty function
                assert len(v.args) == 1
                # this covers both cases (2) and (3); if there is no
                # expression for the uncertainty, it will be read from the
                # effects tables (see above)
                if v.args[0] in cached_uncertainties.keys():
                    adict[v] = cached_uncertainties[v.args[0]]
                else:
                    adict[v] = self.calc_u_for_variable(v.args[0])
                    cached_uncertainties[v.args[0]] = adict[v]
            else:
                # it's a quantity
                if v not in quantities:
                    if v not in me.expressions.keys():
                        raise ValueError(
                            "Calculation of {:s} needs defined value for "
                            "quantity {:s} but this is not set.  I have values "
                            "for: {:s}.".format(str(list(me.expressions.keys()))))
                    quantities[v] = me.evaluate_quantity(v, quantities)

                adict[v] = quantities[v]
                    
        # now I have adict with values for uncertainties and other
        # quantities, that I need to substitute into the expression
        raise NotImplementedError("Tot hier heeft de heer ons geholpen")


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
    
    @typhon.math.common.calculate_precisely
    def calc_sens_coef_C_Earth(self, L_iwct, C_iwct, C_space, C_Earth):
        return L_iwct / (C_iwct - C_space)

    @typhon.math.common.calculate_precisely
    def calc_sens_coef_C_iwct(self, L_iwct, C_iwct, C_space, C_Earth):
        return - L_iwct * (C_Earth - C_space) / (C_iwct - C_space)**2

    @typhon.math.common.calculate_precisely
    def calc_sens_coef_C_iwct_slope(self, L_iwct, C_iwct, C_space):
        """Sensitivity coefficient for C_IWCT for slope (a₁) calculation

        Arguments:

            L_iwct [ndarray]

                Radiance for IWCT.  Can be obtained with
                self.extract_calibcounts_and_temp.  Should
                be 1-D [N].

            C_iwct [ndarray]

                Counts for IWCTs.  Should be 2-D [N × 48]

            C_space [ndarray]

                Counts for space views.  Same shape as C_iwct.

        Returns:

            Sensitivity coefficient.
        """
        return L_iwct[:, numpy.newaxis] / (C_iwct - C_space)**2

    @typhon.math.common.calculate_precisely
    def calc_sens_coef_C_space(self, L_iwct, C_iwct, C_space, C_Earth):
        return L_iwct * (C_Earth - C_iwct) / (C_iwct - C_space)**2

    @typhon.math.common.calculate_precisely
    def calc_sens_coef_C_space_slope(self, L_iwct, C_iwct, C_space):
        """Sensitivity coefficient for C_space for slope (a₁) calculation

        Input as for calc_sens_coef_C_iwct_slope
        """
        return -L_iwct[:, numpy.newaxis] / (C_iwct - C_space)**2


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
        """Calculate radiance uncertainty due to calibration
        """
        s_iwct = self.calc_sens_coef_C_iwct(
                    L_iwct, C_iwct, C_space, C_Earth)
        s_space = self.calc_sens_coef_C_space(
                    L_iwct, C_iwct, C_space, C_Earth)
        return numpy.sqrt((s_iwct * u_C_iwct)**2 +
                    (s_space * u_C_space)**2)

    def calc_uslope(self, M, ch, srf=None):
        """Direct calculation of slope uncertainty

        Such as for purposes of visualising uncertainties in slope/gain
        """
        srf = srf or self.srfs[ch-1]
        (time, L_iwct, C_iwct, C_space, u_C_iwct,
            u_C_space) = self.extract_calibcounts_and_temp(
                M, ch, srf, return_u=True)
        s_iwct = self.calc_sens_coef_C_iwct_slope(L_iwct, C_iwct, C_space)
        s_space = self.calc_sens_coef_C_space_slope(L_iwct, C_iwct, C_space)
#        (t_iwt_noise_level, u_C_iwct) = self.estimate_noise(M, ch, typ="iwt")
#        (t_space_noise_level, u_C_space) = self.estimate_noise(M, ch, typ="space")
#        (u_C_iwct,) = h.interpolate_between_calibs(M,
#            t_iwt_noise_level, u_C_iwct)
#        (u_C_space,) = h.interpolate_between_calibs(M,
#            t_space_noise_level, u_C_space)

        return numpy.sqrt((s_iwct * u_C_iwct[:, numpy.newaxis])**2 +
                          (s_space * u_C_space[:, numpy.newaxis])**2)

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

def list_all_satellites():
    """Return a set with all possible satellite names of any kind
    """
    S = set()
    for h in {HIRS2FCDR, HIRS3FCDR, HIRS4FCDR}:
        for sats in h.satellites.values():
            S |= sats
    return S
