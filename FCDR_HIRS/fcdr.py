"""Datasets for TOVS/ATOVS
"""

import logging
import itertools
import warnings
import functools
import operator
import datetime
import numbers

import numpy
import scipy.interpolate
import progressbar
import pandas
import xarray
import sympy
from typhon.physics.units.tools import UnitsAwareDataArray as UADA
    
import typhon.datasets.dataset
import typhon.physics.units
from typhon.physics.units.common import ureg, radiance_units as rad_u
from typhon.datasets.tovs import (Radiometer, HIRS, HIRSPOD, HIRS2,
    HIRSKLM, HIRS3, HIRS4)

from pyatmlab import tools

from . import models
from . import effects
from . import measurement_equation as me
from . import _fcdr_defs

class FCDRError(typhon.datasets.dataset.InvalidDataError):
    pass

class FCDRWarning(UserWarning):
    pass

class HIRSFCDR(typhon.datasets.dataset.HomemadeDataset):
    """Produce, write, study, and read HIRS FCDR.

    Some of the methods need context-information.  A class that helps in
    passing in the requirement information is at
    FCDR_HIRS.processing.generate_fcdr.FCDRGenerator.

    Mixin for kiddies HIRS?FCDR.  Not to be constructed directly.

    Relevant papers:
    - NOAA: cao07_improved_jaot.pdf
    - PDF_TEN_990007-EPS-HIRS4-PGS.pdf

    Construct with 'read' which can be "L1B" or "L1C".
    """

    name = section = "fcdr_hirs"
    # See spreadsheet attached to e-mail from Tom Block to fiduceo-project
    # mailing list, 2017-03-31
    stored_name = ("FIDUCEO_FCDR_L1C_HIRS{version:d}_{satname:s}_"
                   "{year:04d}{month:02d}{day:02d}{hour:02d}{minute:02d}{second:02d}_"
                   "{year_end:04d}{month_end:02d}{day_end:02d}{hour_end:02d}{minute_end:02d}{second_end:02d}_"
                   "{fcdr_type:s}_v{data_version:s}_fv{format_version:s}.nc")
    write_subdir = "{fcdr_type:s}/{satname:s}/{year:04d}/{month:02d}/{day:02d}"
    stored_re = (r"FIDUCEO_FCDR_L1C_HIRS(?P<version>[2-4])_"
                 r"(?P<satname>.{6})_"
                 r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})"
                 r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})_"
                 r"(?P<year_end>\d{4})(?P<month_end>\d{2})(?P<day_end>\d{2})"
                 r"(?P<hour_end>\d{2})(?P<minute_end>\d{2})(?P<second_end>\d{2})_"
                 r"(?P<fcdr_type>[a-zA-Z]*)_"
                 r"(?P<data_version>v.+)_"
                 r"(?P<format_version>fv.+)\.nc")
    
    # before data_version v0.5
    old_stored_re = (
                 r'FCDR_HIRS_(?P<satname>.{6})_(?P<fcdr_version>.+)_'
                 r'(?P<fcdr_type>[a-zA-Z]*)_'
                 r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})'
                 r'(?P<hour>\d{2})(?P<minute>\d{2})_'
                 r'(?P<year_end>\d{4})(?P<month_end>\d{2})(?P<day_end>\d{2})'
                 r'(?P<hour_end>\d{2})(?P<minute_end>\d{2})\.nc')

    # format changelog:
    #
    # v0.3: removed copied flags, added own flag fields
    format_version="0.3"

    realisations = 100
    srfs = None
    satname = None

    band_dir = None
    band_file = None
    read_mode = "L1B"

    read_returns = "xarray" # for NetCDFDataset in my inheritance tree

    l1b_base = None # set in child

    # NB: first 8 views of space counts deemed always unusable, see
    # NOAA or EUMETSAT calibration papers/documents.  I've personaly
    # witnessed (on NOAA-18) that later positions are sometimes also
    # systematically offset
    start_space_calib = 8
    start_iwct_calib = 8

    ε = 0.98 # from Wang, Cao, and Ciren (2007, JAOT), who give so further
             # source for this number

    # Read in some HIRS data, including nominal calibration
    # Estimate noise levels from space and IWCT views
    # Use noise levels to propagate through calibration and BT conversion

    def __init__(self, read="L1B", *args, satname, **kwargs):
        for nm in {satname}|self.satellites[satname]:
            try:
                self.srfs = [typhon.physics.units.em.SRF.fromArtsXML(
                             nm, self.section, i) for i in range(1, 20)]
            except FileNotFoundError:
                pass # try the next one
            else:
                break
        else:
            raise ValueError("Could not find SRF for any of: {:s}".format(
                ','.join({satname}|self.satellites[satname])))
        self.read_mode = read
        if read == "L1C":
            self.re = self.stored_re # before super()
        super().__init__(*args, satname=satname, **kwargs)
        if read == "L1B":
            self.read_returns = "ndarray"
        elif read == "L1C":
            #self.name = self.section = self.write_name
            #self.stored_name = self.write_stored_name
            self.basedir = self.write_basedir
            self.subdir = self.write_subdir
            self.mandatory_fields.clear()
            self.mandatory_fields.add("time")
            self.read_returns = "xarray"
        else:
            raise ValueError("'read' must be 'L1B' or 'L1C', "
                             "got {!s}".format(read))
        # if the user has asked for headers to be returned, M is a tuple
        # (head, lines) so we need to extract the lines.  Otherwise M is
        # just lines.
        # the following line means the pseudo field is only calculated if
        # the value of the keyword "calibrate" (passed to
        # read/read_period/…) is equal to any of the values in the tuple
        if read == "L1B":
            cond = {"calibrate": (None, True)}
            self.my_pseudo_fields["radiance_fid_naive"] = (
                ["radiance", self.scantype_fieldname, "temp_iwt", "time"],
                lambda M, D:
                self.calculate_radiance_all(
                    M[1] if isinstance(M, tuple) else M, interp_kind="zero",
                    return_ndarray=True),
                cond)
            self.my_pseudo_fields["bt_fid_naive"] = (["radiance_fid_naive"],
                self.calculate_bt_all,
                cond)

        self._data_vars_props.update(_fcdr_defs.FCDR_data_vars_props)

        #self.hirs = hirs
        #self.srfs = srfs

    def _read(self, *args, **kwargs):
        if self.read_mode == "L1C":
            return super()._read(*args, **kwargs)
        elif self.read_mode == "L1B":
            return self.l1b_base._read(self, *args, **kwargs)
#            return super(typhon.datasets.dataset.HomemadeDataset, self)._read(
#                *args, **kwargs)
        else:
            raise RuntimeError("Messed up!  Totally bad!")

    def interpolate_between_calibs(self, target_time, calib_time, *args, kind="nearest"):
        """Interpolate calibration parameters between calibration cycles

        This method is just beginning and likely to improve considerably
        in the upcoming time.

        Doesn't even have to be between calibs, can be any times.

        Arguments:
        
            target_time [ndarray, dtype time]
            
                Dataset with enough fields

            calib_time [ndarray, dtype time]

                times corresponding to offset and slope, such as returned
                by HIRS.calculate_offset_and_slope.  Will only be used for
                arguments not carrying their own time.

            *args
                
                anything defined only at calib_time, such as slope,
                offset, or noise_level.  Can be ndarrays or
                xarray.DataArrays.  If the latter come with their own
                time, this is used instead of calib_time.
        
        Returns:

            list, corresponding to args, interpolated to all times in ds
        """

        if not numpy.issubdtype(target_time.dtype, numpy.datetime64):
            raise TypeError("As of 2017-02-22, interpolate_between_calibs "
                "takes time directly")

        x = numpy.asarray(calib_time.astype("M8[ms]").astype("u8"))
        xx = numpy.asarray(target_time.astype("M8[ms]").astype("u8"))
        out = []
        for y in args:
            try:
                u = y.u
            except AttributeError:
                try:
                    u = y.attrs["units"]
                except (AttributeError, KeyError):
                    u = None
            try:
                xh = y["time"]
            except (ValueError, IndexError):
                xh = x
            if not isinstance(y, (numpy.ndarray, xarray.DataArray)):
                y = numpy.ma.asarray(y)
            # explicitly set masked data to nan, for scipy.interpolate
            # doesn't understand this
            try:
                if not numpy.isscalar(y.mask):
                    y.data[y.mask] = numpy.nan
            except AttributeError:
                pass # not a masked array
            fnc = scipy.interpolate.interp1d(
                x, y,
                kind=kind,
                #fill_value="extrapolate",
                fill_value="extrapolate" if kind=="nearest" else numpy.nan,
                bounds_error=False,
                axis=0)

            yy = numpy.ma.masked_invalid(fnc(xx))
            if isinstance(y, xarray.DataArray):
                out.append(
                    UADA(
                        fnc(xx),
                        dims=("time",),
                        coords={"time": target_time},
                        attrs={"units": y.attrs["units"]} if u else {}))
            elif u is None:
                out.append(yy)
            else:
                out.append(ureg.Quantity(yy, u))

        return out

    def custom_calibrate(self, counts, slope, offset, a2, Rself):
        """Calibrate with my own slope and offset

        All arguments should be xarray.DataArray or preferably
        UnitsAwareDataArrays.
        """
        return offset + slope * counts + a2 * counts**2 - Rself
#        return (offset[:, numpy.newaxis]
#              + slope[:, numpy.newaxis] * counts
#              + a2 * counts**2
#              - Rself)

    def extract_calibcounts_and_temp(self, ds, ch, srf=None,
            return_u=False, return_ix=False, tuck=False):
        """Calculate calibration counts and IWCT temperature

        In the IR, space view temperature can be safely estimated as 0
        (radiance at 3K is around 10^200 times less than at 300K)

        Arguments:

            ds

                ndarray such as returned by self.as_karray_dataset, corresponding to
                scanlines.  Must have at least variables 'time',
                'scantype', 'counts', and 'temperature_iwct'.

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
                assuming ε from self.ε, an arithmetic mean of all
                temperature sensors on the IWCT, and the SRF passed to the
                method.  Earthshine / reflection through the blackbody is
                not yet implemented (see #18)

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

        # 2017-02-22 backward compatibility
        if not isinstance(ds, xarray.Dataset):
            warnings.warn("Passing ndarray to extract_calibcounts_and_temp "
                "is deprecated since 2017-02-22, should pass "
                "xarray.Dataset ", DeprecationWarning)
            ds = self.as_xarray_dataset(ds)

        # xarray.core.array.nputils.array_eq (and array_neq) use the
        # context manager 'warnings.catch_warnings'.  This causes the
        # warnings registry to be reset such that warnings that should be
        # printed once get printed every time (see
        # http://bugs.python.org/issue29672 and
        # https://github.com/pydata/xarray/blob/master/xarray/core/nputils.py#L73)
        # Therefore, avoid xarray array_eq for now. 
#        views_space = ds["scantype"] == self.typ_space
        views_space = xarray.DataArray(ds["scantype"].values == self.typ_space, coords=ds["scantype"].coords)
#        views_iwct = ds["scantype"] == self.typ_iwt
        views_iwct = xarray.DataArray(ds["scantype"].values == self.typ_iwt, coords=ds["scantype"].coords)

        # select instances where I have both in succession.  Should be
        # always, unless one of the two is missing or the start or end of
        # series is in the middle of a calibration.  Take this from
        # self.dist_space_iwct because for HIRS/2 and HIRS/2I, there is a
        # views_icct in-between.
        dsi = self.dist_space_iwct
        space_followed_by_iwct = (views_space[:-dsi].variable &
                                   views_iwct[dsi:].variable)
#        space_followed_by_iwct = (views_space[:-dsi].variable &
#                                   views_iwct[dsi:].variable)

        ds_space = ds.isel(time=slice(None, -dsi)).isel(
                    time=space_followed_by_iwct)

        #M_space = M[:-dsi][space_followed_by_iwct]

        ds_iwct = ds.isel(time=slice(dsi, None)).isel(
                    time=space_followed_by_iwct)
        #M_iwct = M[dsi:][space_followed_by_iwct]

        counts_space = UADA(ds_space["counts"].sel(
                scanpos=slice(self.start_space_calib, None),
                channel=ch).rename({"channel": "calibrated_channel"}),
            name="counts_space")
        counts_space.attrs.update(units="counts")
#        counts_space = ureg.Quantity(M_space["counts"][:,
#            self.start_space_calib:, ch-1], ureg.count)
        # For IWCT, at least EUMETSAT uses all 56…
        counts_iwct = UADA(ds_iwct["counts"].sel(
                scanpos=slice(self.start_iwct_calib, None),
                channel=ch).rename({"channel": "calibrated_channel"}),
            name="counts_iwct")
        counts_iwct.attrs.update(units="counts")
#        counts_iwct = ureg.Quantity(M_iwct["counts"][:,
#            self.start_iwct_calib:, ch-1], ureg.count)

        T_iwct = ds_iwct["temperature_iwct"].mean(
                dim="prt_reading").mean(dim="prt_number_iwt")
        T_iwct.attrs["units"] = ds_iwct["temperature_iwct"].attrs["units"]
#        T_iwct = ureg.Quantity(ds_iwct["temperature_iwct"].mean(
#                dim="prt_reading").mean(dim="prt_number_iwt"),
#                ureg.K)
#        T_iwct = ureg.Quantity(
#            M_space["temp_iwt"].mean(-1).mean(-1).astype("f4"), ureg.K)
        # store directly, not using R_IWCT, as this is the same across
        # channels
        # FIXME wart: I'm storing the same information 19 times (for each
        # channel), could assert they are identical or move this out of a
        # higher loop, or I could simply leave it
        if tuck:
            self._quantities[me.symbols["T_IWCT"]] = self._quantity_to_xarray(
                T_iwct, name=me.names[me.symbols["T_IWCT"]])
            self._quantities[me.symbols["N"]] = self._quantity_to_xarray(
                    numpy.array(ds.dims["prt_number_iwt"], "u1"),
                    name=me.names[me.symbols["N"]])

        # FIXME: for consistency, should replace this one also with
        # band-corrections — at least temporarily.  Perhaps this wants to
        # be implemented inside the blackbody_radiance method…
        # NB: SRF does not understand DataArrays yet
        # NB: pint seems to silently drop xarray.DataArray information,
        # see https://github.com/hgrecco/pint/issues/479
        # instead use UADA
        L_iwct = self.ε * srf.blackbody_radiance(
            ureg.Quantity(T_iwct.values, ureg.K))
        #L_iwct = ureg.Quantity(L_iwct.astype("f4"), L_iwct.u)
        L_iwct = UADA(L_iwct,
            dims=T_iwct.dims,
            coords={**T_iwct.coords, "calibrated_channel": ch},
            attrs={"units": str(L_iwct.u)})

        extra = []
        # this implementation is slightly more sophisticated than in
        # self.estimate_noise although there is some code duplication.
        # Here, we only use real calibration lines, where both space and
        # earth views were successful.
        u_counts_space = (typhon.math.stats.adev(counts_space, "scanpos") /
            numpy.sqrt(counts_space.shape[1]))
        if tuck:
            self._tuck_effect_channel("C_space", u_counts_space, ch)

        u_counts_iwct = (typhon.math.stats.adev(counts_iwct, "scanpos") /
            numpy.sqrt(counts_iwct.shape[1]))
        if tuck:
            # before 'tucking', I want to make sure the time coordinate
            # corresponds to the calibration cycle, i.e. the same as for
            # space.  This is ultimately used by passing a dictionary to
            # a lambda generated by sympy.lambdify so we need to reassign
            # coordinates in advance
            self._tuck_effect_channel("C_IWCT", 
                u_counts_iwct.assign_coords(time=u_counts_space["time"]),
                ch)
            self._tuck_effect_channel("C_Earth",
                UADA((u_counts_space.variable + u_counts_iwct.variable)/2,
                      coords=u_counts_space.coords,
                      attrs=u_counts_space.attrs), ch)

        if return_u:
            extra.extend([u_counts_iwct, u_counts_space])
        if return_ix:
            extra.append(space_followed_by_iwct.values.nonzero()[0])

        if tuck:
            coords = {"calibration_cycle": ds_space["time"].values}
            #coords = {"calibration_cycle": M_space["time"]}
            self._tuck_quantity_channel("C_s", counts_space,
                calibrated_channel=ch, **coords)
            self._tuck_quantity_channel("C_IWCT", counts_iwct,
                calibrated_channel=ch, **coords)
            self._tuck_quantity_channel("R_IWCT", L_iwct,
                calibrated_channel=ch, **coords)
            # store 'N', C_PRT[n], d_PRT[n, k], O_TPRT, O_TIWCT…
        return (UADA(ds_space["time"]),
                UADA(L_iwct),
                UADA(counts_iwct),
                UADA(counts_space)) + tuple(extra)
        #return (M_space["time"], L_iwct, counts_iwct, counts_space) + tuple(extra)

    def _quantity_to_xarray(self, quantity, name, dropdims=(), dims=None):
        """Convert quantity to xarray

        Quantity can be masked and with unit, which will be converted.
        Can also pass either dropdims (dims subtracted from ones defined
        for the quantity) or dims (hard list of dims to include).
        """
        
        da = UADA(
            numpy.asarray(quantity,
                dtype=("f4" if hasattr(quantity, "mask") else
                    quantity.dtype)), # masking only for floats
            dims=dims if dims is not None else [d for d in self._data_vars_props[name][1] if d not in dropdims],
            attrs=self._data_vars_props[name][2],
            encoding=self._data_vars_props[name][3])
        # 1st choice: quantity.u (pint unit)
        # 2nd choice: defined in quantity.attrs (UADA)
        # 3rd choice: defined in da.attrs
        # fallback: "UNDEFINED"
        try:
            u = quantity.u
        except AttributeError:
            try:
                u = quantity.attrs["units"]
            except (AttributeError, KeyError):
                u = da.attrs.get("units", "UNDEFINED")

        da.attrs["units"] = str(u)
        try:
            da.values[quantity.mask] = numpy.nan
        except AttributeError:
            pass # not a masked array
        return da

    def _tuck_quantity_channel(self, symbol_name, quantity, **coords):
        """Convert quantity to xarray and put into self._quantities

        TODO: need to assign time coordinates so that I can later
        extrapolate calibration_cycle dimension to scanline dimension.

        Returns quantity as stored.
        """

        s = me.symbols[symbol_name]
        name = me.names[s]
        q = self._quantity_to_xarray(quantity, name,
                dropdims=["channel", "calibrated_channel"]).assign_coords(
                    **coords)
        if s in self._quantities:
            da = self._quantities[s]
            in_coords = [x for x in ("channel", "calibrated_channel")
                            if x in da.coords]
            if len(in_coords) != 1:
                raise ValueError("{:s} does not contain exactly one "
                                 "channel coordinate, found {:d}.".format(
                                    symbol_name, len(in_coords)))
            # FIXME: need to check if we're tucking the same channel twice
            da = xarray.concat([da, q], dim=in_coords[0])
            # NB: https://github.com/pydata/xarray/issues/1297
            da.encoding = q.encoding
            self._quantities[s] = da
            return da
        else:
            self._quantities[s] = q
            return q

    def _tuck_effect_channel(self, name, quantity, channel):
        """Convert quantity to xarray and put into self._effects
        """

        # NB: uncertainty does NOT always have the same dimensions as quantity
        # it belongs to!  For example, C_IWCT has dimensions
        # ('calibration_cycle', 'calibration_position', 'channel'), but
        # u_C_IWCT has dimensions  ('calibration_cycle', 'channel').
        # Therefore, effects have 'dims' attribute in case there is a
        # difference.

        # NEE!  Dit gaat helemaal mis want ik heb dan verschillende
        # coordinates voor verschillende variables en als ik dan de
        # expression evaluate met xarray krĳg ik results met 0 dimension
        # want xarray tries to realign the coordinates...  earlier I
        # accidentally removed the coordinates and that is why it worked
        # at all!
        if isinstance(quantity, xarray.DataArray):
            q = quantity.rename(
                dict(zip(quantity.dims,
                self._effects_by_name[name].dimensions)))
        else:
            q = self._quantity_to_xarray(quantity, name,
                dropdims=["channel", "calibrated_channel"],
                dims=self._effects_by_name[name].dimensions)
        q = q.assign_coords(calibrated_channel=channel)
        if self._effects_by_name[name].magnitude is None:
            self._effects_by_name[name].magnitude = q
        else:
            da = self._effects_by_name[name].magnitude
            # check if we're tucking it for the same channel twice...
            if channel in da.calibrated_channel.values:
                if da.calibrated_channel.size == 1:
                    if numpy.array_equal(da.values, q.values):
                        return # nothing to do
                    else:
                        raise ValueError("Inconsistent values for same channel!")
                else:
                    raise NotImplementedError("TBD")

            da = xarray.concat([da, q], dim="calibrated_channel")
            # NB: https://github.com/pydata/xarray/issues/1297
            da.encoding = q.encoding
            self._effects_by_name[name].magnitude = da

    def calculate_offset_and_slope(self, ds, ch, srf=None, tuck=False):
        """Calculate offset and slope.

        Arguments:

            ds [xarray.Dataset]

                xarray dataset with fields such as returned by
                self.as_xarray_dataset.  Must
                contain at least variables 'time', 'scantype', 'counts',
                and 'temperature_iwct'.

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
        
        # 2017-02-22 backward compatibility
        if not isinstance(ds, xarray.Dataset):
            warnings.warn("Passing ndarray to calculate_offset_and_slope "
                "is deprecated since 2017-02-22, should pass "
                "xarray.Dataset ", DeprecationWarning)
            ds = self.as_xarray_dataset(ds)
        (time, L_iwct, counts_iwct, counts_space) = self.extract_calibcounts_and_temp(ds, ch, srf, tuck=tuck)
        #L_space = ureg.Quantity(numpy.zeros_like(L_iwct), L_iwct.u)
        L_space = UADA(xarray.zeros_like(L_iwct),
            coords={k:v 
                for (k, v) in counts_space.isel(scanpos=0).coords.items()
                if k in L_iwct.coords.keys()})

        ΔL = UADA(L_iwct.variable - L_space.variable,
                  coords=L_space.coords,
                  attrs=L_space.attrs)
        Δcounts = UADA(
            counts_iwct.variable - counts_space.variable,
            coords=counts_space.coords, name="Δcounts",
            attrs=counts_space.attrs)
        slope = ΔL/Δcounts

        # non-linearity is set to 0 for now
        a2 = UADA(0, name="a2", coords={"calibrated_channel": ch}, attrs={"units":
            str(rad_u["si"]/(ureg.count**2))})

        offset = -counts_space**2 * a2 -slope * counts_space

        # sometimes IWCT or space counts seem to drift over a “scan line”
        # of calibration.  Identify this by comparing the IQR to the
        # counts.  For truly random normally distributed data:
        # (25, 75) … > 2: false positive 0.2%
        # (10, 90) … > 3.3: false positive 0.5%
        # …based on a simple simulated # experiment.
        if counts_iwct.coords["time"].size > 0:
            # NB: need to encapsulate this or it will fail with
            # ValueError, see https://github.com/scipy/scipy/issues/7178
            bad_iwct = (counts_iwct.reduce(
                scipy.stats.iqr, dim="scanpos", rng=(10, 90)) > 3.3 *
                typhon.math.stats.adev(counts_iwct, dim="scanpos"))
            bad_space = (counts_space.reduce(
                scipy.stats.iqr, dim="scanpos", rng=(10, 90)) > 3.3 *
                typhon.math.stats.adev(counts_space, dim="scanpos"))

            bad_calib = xarray.DataArray(bad_iwct.variable | bad_space.variable,
                    coords=bad_space.coords, name="bad_calib")
            slope.sel(time=bad_calib)[...] = numpy.nan
            offset.sel(time=bad_calib)[...] = numpy.nan
        coords = {"calibration_cycle": time.values}
        if tuck:
            self._tuck_quantity_channel("a_0", offset,
                calibrated_channel=ch, **coords)
            self._tuck_quantity_channel("a_1", slope,
                calibrated_channel=ch, **coords)
            self._tuck_quantity_channel("a_2", a2,
                calibrated_channel=ch)
        return (time,
                offset,
                slope,
                a2)

    _quantities = {}
    _effects = None
    _effects_by_name = None
    _flags = {"scanline": {}, "channel": {}}
    def calculate_radiance(self, ds, ch, interp_kind="zero", srf=None,
                context=None,
                Rself_model=None,
                Rrefl_model=None, tuck=False, return_bt=False):
        """Calculate radiance

        Wants ndarray as returned by read, SRF, and channel.

        Returns pint quantity with masked array underneath.

        Arguments:
            
            ds [xarray.Dataset]

                xarray Dataset with at least variables 'time', 'scantype',
                'temperature_iwct', and 'counts'.  Such is returned by
                self.as_xarray_dataset.  Those are values for which
                radiances will be calculated.

            ch [int]

                Channel to calculate radiance for.  If you want to
                calculate the radiance for all channels, use
                calculate_radiance_all.

            interp_kind [str]

                Legacy, must be equal to "zero".  Instead of
                interpolation, pass a self-emission model.

            srf [SRF]

                SRF to use.  If not passed, use default (as measured
                before launch).

            context [xarray.dataset]

                Like ds, but used for context.  For example, calibration
                information may have to be found outside the range of `ds`.
                It is also needed for developing the Rself and Rrefl
                models when not provided to the function already.

            Rself_model [RSelf]

                Model to use for self-emission.  See models.RSelf.

            Rrefl_model [RRefl]

                Model to use for Earthshine.  See models.RRefl.

        """
        if interp_kind != "zero":
            raise NotImplementedError("You asked for {:s} interpolation, "
                "but I refuse to do so.  Interpolating calibration "
                "must be done with a zero-order spline. Corrections on "
                "top of that must be physics-based, such as through "
                "the self-emission model. ".format(interp_kind))

        if not isinstance(ds, xarray.Dataset):
            warnings.warn("Passing ndarray to calculate_radiance "
                "is deprecated since 2017-02-22, should pass "
                "xarray.Dataset ", DeprecationWarning)
            ds = self.as_xarray_dataset(ds)
        srf = srf or self.srfs[ch-1]
        has_context = context is not None
        context = context if has_context else ds
        # NB: passing `context` here means that quantities and effects are
        # stored with dimension for entire context period, rather than
        # just the ds period.  Should take this into account with later
        # processing.
        (time, offset, slope, a2) = self.calculate_offset_and_slope(
            context, ch, srf, tuck=tuck)
        # NOTE: taking the median may not be an optimal solution.  See,
        # for example, plots produced by the script
        # plot_hirs_calibcounts_per_scanpos in the FCDR_HIRS package
        # within FIDUCEO, in particular for noaa18 channels 1--12, where
        # the lowest scan positions are systematically offset compared to
        # the higher ones.  See also the note at
        # calculate_offset_and_slope. 
        if offset.shape[0] > 1 or has_context:
            (interp_offset, interp_slope) = self.interpolate_between_calibs(
                ds["time"], time,
                offset.median(dim="scanpos", keep_attrs=True),
                #ureg.Quantity(numpy.ma.median(offset.m, 1), offset.u),
                slope.median(dim="scanpos", keep_attrs=True),
                #ureg.Quantity(numpy.ma.median(slope.m, 1), slope.u),
                kind=interp_kind) # kind is always "zero" (see above)
        elif offset.shape[0] == 1:
            interp_offset = xarray.zeros_like(ds["counts"].sel(
                scanpos=1, channel=ch))
            interp_offset.values[:] = UADA(offset.median("scanpos"))
            interp_offset.attrs["units"] = offset.units
            interp_slope = xarray.zeros_like(ds["counts"].sel(
                scanpos=1, channel=ch))
            interp_slope.values[:] = UADA(slope.median("scanpos"))
            interp_slope.attrs["units"] = slope.units
        elif ds["counts"].coords["time"].size > 0:
            raise typhon.datasets.dataset.InvalidFileError("Found {:d} calibration cycles, too few!".format(offset.shape[0]))
        else: # zero scanlines…
            raise FCDRError("Dataset has zero scanlines, nothing to do")

        # see note near line 270
        views_Earth = xarray.DataArray(ds["scantype"].values == self.typ_Earth, coords=ds["scantype"].coords)
        #views_Earth = ds["scantype"] == self.typ_Earth
        # NB: C_Earth has counts for all channels but only
        # calibrated_channel will be used
        C_Earth = UADA(ds["counts"].isel(time=views_Earth).sel(
            channel=ch)).rename({"channel": "calibrated_channel"})

        if Rself_model is None:
            warnings.warn("No self-emission defined, assuming 0!",
                FCDRWarning)
            Rself = UADA(numpy.zeros(shape=C_Earth["time"].shape),
                         coords=C_Earth["time"].coords,
                         name="Rself", attrs={"units":   
                str(rad_u["si"])})
            RselfIWCT = Rselfspace = UADA(numpy.zeros(shape=offset["time"].shape),
                    coords=offset["time"].coords, attrs=Rself.attrs)
            u_Rself = UADA([0], dims=["rself_update_time"],
                           coords={"rself_update_time": [ds["time"].values[0]]})
        else:
            try:
                Rself_model.fit(context, ch)
            except ValueError as e:
                if e.args[0].startswith("Found array with 0 sample(s)"):
                    raise FCDRError(
                        "Unable to train self-emission model, no valid "
                        "training data in {:%Y-%m-%d %H:%M}--{:%Y-%m-%d %H:%M}".format(
                            context["time"].values[0].astype("M8[s]").astype(datetime.datetime),
                            context["time"].values[-1].astype("M8[s]").astype(datetime.datetime)))
                else:
                    raise
            (Xt, Y_reft, Y_predt) = Rself_model.test(context, ch)
            (X, Y_pred) = Rself_model.evaluate(ds, ch)
            # Y_pred is rather the offset than the self-emission
            Rself = interp_offset.isel(time=views_Earth) - Y_pred#.sel(time=views_Earth["time"].isel(time=views_Earth))
            #Rself = (interp_offset - Y_pred).isel(time=views_Earth)
            Rself.attrs["note"] = ("Implemented as ΔRself in pre-β. "
                "RselfIWCT = Rselfspace = 0.")
            RselfIWCT = Rselfspace = UADA(numpy.zeros(shape=offset["time"].shape),
                    coords=offset["time"].coords,
                    attrs={**Rself.attrs,
                           "note": "Rself is implemented as ΔRself in pre-β"})
            u_Rself = numpy.sqrt(((Y_reft - Y_predt)**2).mean(
                keep_attrs=True, dim="time"))
            # trick to make sure there is a time dimension…
            # http://stackoverflow.com/a/42999562/974555
            u_Rself = xarray.concat((u_Rself,), dim="rself_update_time")
            # and, for now, establish fake time based u_Rself to ensure
            # there's always u_Rself info in every orbit file even though
            # it's updated more rarely than that.  Take a range that is
            # too large (context), another lie, so we can later
            # interpolate to the right segment (lie will not propagate)
            # interpolation will happen in _make_dims_consistent
            times = pandas.date_range(
                context["time"].values[0],
                context["time"].values[-1],
                freq='10min')
            u_Rself = u_Rself[numpy.tile([0], times.size)]
            u_Rself = u_Rself.assign_coords(
                rself_update_time=times.values)# [ds["time"].values[0]])
            u_Rself.attrs["U_RSELF_WARNING"] = (
                "Self-emission uncertainty repeated "
                "every ten minutes as a stop-gap measure to ensure even "
                "short slices contain this info, but not a sustainable "
                "solution.  Coordinates are lying!")
        # according to Wang, Cao, and Ciren (2007), ε=0.98, no further
        # source or justification given
        ε = UADA(1, name="emissivity")
        a_3 = UADA(self.ε-1, name="correction to emissivity")
        if Rrefl_model is None:
            warnings.warn("No Earthshine model defined, assuming 0!",
                FCDRWarning)
            Rrefl = UADA(numpy.zeros(shape=offset.shape),
                    coords=offset.coords,
                    attrs={"units": str(rad_u["si"])})
        else:
            raise NotImplementedError("Evalutation of Earthshine "
                "model not implemented yet")
        a_4 = UADA(0, name="harmonisation bias",
            attrs={"units": rad_u["si"]})
        rad_wn = self.custom_calibrate(C_Earth, interp_slope,
            interp_offset, a2, Rself)

        (α, β, λ_eff, Δα, Δβ, Δλ_eff) = srf.estimate_band_coefficients(
            self.satname, self.section, ch)
#        (α, β, f_eff, Δα, Δβ, Δf_eff) = (numpy.float32(0),
#            numpy.float32(1), srf.centroid().to(ureg.THz, "sp"), 0, 0, 0)

        coords = {"calibration_cycle": time.values}
        if tuck:
            self._tuck_quantity_channel("R_selfE", Rself, 
                calibrated_channel=ch)
            self._tuck_effect_channel("Rself", u_Rself, ch)
            self._tuck_quantity_channel("R_selfIWCT", RselfIWCT, 
                calibrated_channel=ch, **coords)
            self._tuck_quantity_channel("R_selfs", Rselfspace,
                calibrated_channel=ch, **coords)
            self._tuck_quantity_channel("C_E", C_Earth, 
                calibrated_channel=ch, scanline_earth=C_Earth["time"].values)
    #        self._tuck_quantity_channel("R_e", rad_wn[views_Earth, :], ch)
            # keep result but copy over only encoding, because
            # _tuck_quantity_channel also renames dimensions and I don't want
            # that yet
            R_e = self._tuck_quantity_channel("R_e", rad_wn,
                calibrated_channel=ch)
            rad_wn.encoding = R_e.encoding
            if return_bt:
                self._tuck_quantity_channel("T_b",
                    rad_wn.to("K", "radiance", srf=srf),
                    calibrated_channel=ch)
            self._tuck_quantity_channel("R_refl", Rrefl,
                calibrated_channel=ch, **coords)

            self._tuck_quantity_channel("α", α, 
                calibrated_channel=ch)
            self._tuck_effect_channel("α", Δα, ch)
            self._tuck_quantity_channel("β", β,
                calibrated_channel=ch)
            self._tuck_effect_channel("β", Δβ, ch)
            self._tuck_quantity_channel("fstar", λ_eff.to(ureg.THz, "sp"),
                calibrated_channel=ch)
            self._tuck_effect_channel("f_eff", Δλ_eff.to(ureg.GHz, "sp"), ch)
            self._quantities[me.symbols["ε"]] = self._quantity_to_xarray(
                ε, name=me.names[me.symbols["ε"]])
            self._quantities[me.symbols["a_3"]] = self._quantity_to_xarray(
                a_3, name=me.names[me.symbols["a_3"]])
            self._quantities[me.symbols["a_4"]] = self._quantity_to_xarray(
                a_4, name=me.names[me.symbols["a_4"]])
        rad_wn = rad_wn.rename({"time": "scanline_earth"})
        if return_bt:
            return (rad_wn, rad_wn.to("K", "radiance", srf=srf))
        else:
            return rad_wn
    Mtorad = calculate_radiance

    def calculate_radiance_all(self, ds, interp_kind="zero", srf=None,
                context=None,
                Rself_model=None,
                Rrefl_model=None,
                return_ndarray=False):
        """Calculate radiances for all channels

        See calculate_radiance for documentation on inputs.
        """
        if not isinstance(ds, xarray.Dataset):
            warnings.warn("Passing ndarray to calculate_radiance_all "
                "is deprecated since 2017-02-22, should pass "
                "xarray.Dataset ", DeprecationWarning)
            ds = self.as_xarray_dataset(ds)

        # When calculating uncertainties I depend on the same quantities
        # as when calculating radiances, so I really should keep track of
        # the quantities I calculate so I can use them for the
        # uncertainties after.
        self._quantities.clear() # don't accidentally use old quantities…
        self._flags["scanline"].clear()
        self._flags["channel"].clear()

        # the two following dictionary should and do point to the same
        # effect objects!  I want both because when I'm evaluating the
        # effects it's easier to get them by name, but when I'm
        # substituting them into the uncertainty-expression it's easier to
        # get them by symbol.
        self._effects = effects.effects()
        self._effects_by_name = {e.name: e for e in
                itertools.chain.from_iterable(self._effects.values())}

        (all_rad, all_bt) = zip(*[self.calculate_radiance(ds, ch, interp_kind=interp_kind,
                context=context, Rself_model=Rself_model,
                Rrefl_model=Rrefl_model, tuck=True, return_bt=True)
            for ch in range(1, 20)])
        da = xarray.concat(all_rad, dim="calibrated_channel")
        da.encoding = all_rad[0].encoding
        # NB: https://github.com/pydata/xarray/issues/1297
        da = da.transpose("scanline_earth", "scanpos", "calibrated_channel") 
        # until all of typhon can handle xarrays (see
        # https://arts.mi.uni-hamburg.de/trac/rt/ticket/145) I will
        # unfortunately sometimes need to move back to regular arrays
        if return_ndarray:
            dam = xarray.DataArray(numpy.zeros_like(ds["toa_brightness_temperature"].values), coords=ds["toa_brightness_temperature"].coords)
            dam.loc[dict(time=da.scanline_earth)] = da
            return ureg.Quantity(
                numpy.ma.masked_invalid(dam.values),
                da.attrs["units"])
        else:
            return da
#        return ureg.Quantity(numpy.ma.concatenate([rad.m[...,
#            numpy.newaxis] for rad in all_rad], 2), all_rad[0].u)

    def calculate_bt_all(self, M, D): 
        if isinstance(D["radiance_fid_naive"], xarray.DataArray):
            bt_all = xarray.concat(
                [D["radiance_fid_naive"].sel(channel=i).to(
                    "K", "radiance", srf=self.srfs[i-1])
                    for i in range(1, 20)],
                "channel")
            # NB: https://github.com/pydata/xarray/issues/1297
            # but encoding set later
        else:
            bt_all = ureg.Quantity(
                numpy.ma.concatenate(
                    [self.srfs[ch-1].channel_radiance2bt(
                        D["radiance_fid_naive"][:, :, ch-1])[..., numpy.newaxis]#.astype("f4")
                            for ch in range(1, 20)], 2),
                ureg.K)
            if numpy.isscalar(bt_all.m.mask):
                bt_all.m.mask = D["radiance_fid_naive"].mask
            else:
                bt_all.m.mask |= D["radiance_fid_naive"].mask
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
        (time, offset, slope, a2) = self.calculate_offset_and_slope(M, ch, srf)
        # NOTE:
        # See https://github.com/numpy/numpy/issues/7787 on numpy.median
        # losing the unit
        logging.info("Interpolating") 
        (interp_offset, interp_slope) = self.interpolate_between_calibs(M["time"],
            time, 
            ureg.Quantity(numpy.median(offset, 1), offset.u),
            ureg.Quantity(numpy.median(slope, 1), slope.u))
        interp_noise_level = numpy.interp(M["time"].view("u8"),
                    t_noise_level.view("u8")[~noise_level.mask],
                    noise_level[~noise_level.mask])
        logging.info("Allocating")
        rad_wn = ureg.Quantity(numpy.empty(
            shape=M["counts"].shape[:2] + (realisations,),
            dtype="f4"), rad_u["ir"])
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
            M["time"], time, L_iwct, C_iwct, C_space)
        raise RuntimeError("What!  I'm interpolating Earth Counts?  NO!!!")
        (C_Earth,) = self.interpolate_between_calibs(
            M["time"], M["time"][views_Earth], C_Earth)
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
                            cached_uncertainties, return_more=False):
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

            return_components [bool]

                Optional.  If true, also return a xarray.Dataset with all
                uncertainty components.
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
                baddies = [eff for eff in all_effects[s]
                    if eff.magnitude is None]
                goodies = [eff for eff in all_effects[s]
                    if eff.magnitude is not None]
                if baddies:
                    warnings.warn("Effects with unquantified "
                        "uncertainty: {!s}".format(
                            '; '.join(eff.name for eff in baddies)))
                # Responsibility to put name and attributes onto effect
                # belong to effects.Effect.magnitude setter property.
                if goodies:
                    u = functools.reduce(
                        operator.add,
                        (eff.magnitude for eff in goodies))
                else:
                    u = UADA(0, name="u_{!s}".format(s),
                        attrs={
                            "quantity": str(s),
                            "note": "No uncertainty quantified for: {:s}".format(
                                ';'.join(eff.name for eff in baddies)),
                            "units": self._data_vars_props[
                                        me.names[s]][2]["units"],
                            "encoding": self._data_vars_props[
                                        me.names[s]][3]})
                cached_uncertainties[s] = u
                return (u, {}, {}) if return_more else u
            else:
                u = UADA(0, name="u_{!s}".format(s),
                    attrs={
                        "quantity": str(s),
                        "note": "No documented effect associated with this "
                                "quantity",
                        "units": self._data_vars_props[
                                    me.names[s]][2]["units"],
                        "encoding": self._data_vars_props[
                                    me.names[s]][3]})
                cached_uncertainties[s] = u
                return (u, {}, {}) if return_more else u

        # evaluate expression for this quantity
        e = me.expressions[s]
        # BUG: expressing u_e /before/ substitution in the presence of
        # integrals can cause expressions to be taken out of the Planck
        # function where they should remain inside — express_uncertainty
        # has no knowledge of quantities / expressions / constants /
        # functions; but replacing all by functions puts far too many
        # dependencies into each, which gives confusing uncertainty
        # estimates.  Will need to be something in-between, but keep the
        # version I know to be incorrect under an integral as long as we
        # apply implementation issue#55 rather than issue#56.
#        u_e = typhon.physics.metrology.express_uncertainty(
#            e.subs({sm: me.functions.get(sm, sm)
#                    for sm in typhon.physics.metrology.recursive_args(e)}))
        failures = set()
        (u_e, sensitivities, components) = typhon.physics.metrology.express_uncertainty(
            e, on_failure="warn", collect_failures=failures,
            return_sensitivities=True,
            return_components=True)

        if u_e == 0: # constant
            # FIXME: bookkeep where I have zero uncertainty
            warnings.warn("Assigning u=0 to {!s}".format(s))
            u = UADA(0, name="u_{!s}".format(s),
                attrs={
                    "quantity": str(s),
                    "note": "This appears to be a constant value with "
                            "neglected uncertainty",
                    "units": str(me.units[s])
                })
            cached_uncertainties[s] = u
            return (u, {}, {}) if return_more else u

        fu = sympy.Function("u")
        args = typhon.physics.metrology.recursive_args(u_e,
            stop_at=(sympy.Symbol, sympy.Indexed, fu))

        # Before I proceed, I want to check for zero arguments; this
        # might mean that I need to evaluate less.  Hence two runs through
        # args: first to check the zeroes, then to see what's left.
        for v in args:
            if isinstance(v, fu):
                # comparing .values to avoid entering
                # xarray.core.nputils.array_eq which has a catch_warnings
                # context manager destroying the context registry, see
                # http://bugs.python.org/issue29672
                if (v.args[0] in cached_uncertainties.keys() and
                        numpy.all(cached_uncertainties.get(v.args[0]).values == 0)):
                    u_e = u_e.subs(v, 0)
                    del sensitivities[v.args[0]]
                    del components[v.args[0]]
                elif ((v.args[0] not in me.expressions.keys() or
                       isinstance(me.expressions[v.args[0]], sympy.Number)) and
                      v.args[0] not in all_effects.keys()):
                    # make sure it gets into cached_uncertainties so that
                    # it is "documented".  That is a side-effect for
                    # calc_u_for_variable so I don't otherwise need the
                    # result.
                    ua = self.calc_u_for_variable(v.args[0],
                        quantities, all_effects, cached_uncertainties)
                    # ua==0 triggers __eq__ which triggers catch_warnings
                    # which causes warnings to be printed over and over
                    # again…
                    #assert ua==0, "Impossible"
                    assert ua.values==0, "Impossible"
                    u_e = u_e.subs(v, 0)
                    del sensitivities[v.args[0]]
                    del components[v.args[0]]
                    
        # This may be different from before because I have substituted out
        # arguments that evaluate to zero, so there are less args now
        oldargs = args # not used in code but may want to inspect in
                       # debugging
        args = typhon.physics.metrology.recursive_args(u_e,
            stop_at=(sympy.Symbol, sympy.Indexed, fu))

        # NB: adict is the dictionary of everything (uncertainties and
        # quantities) that needs to be
        # substituted to evaluate the magnitude of the uncertainty.
        # cached_uncertainties is a dictionary persistent between function
        # calls (until cleared) to avoid recalculating identical
        # expressions.  The keys are expressions (such as Symbol or
        # u(Symbol)), the values are numbers or numpy arrays of numbers
        adict = {}
        sub_sensitivities = {}
        sub_components = {}
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
                    # I must have already calculated subsens and subcomp,
                    # but I don't have access to the value.  Either it is
                    # already in sub_sensitivities/sub_components, or the
                    # caller has access to it.  In the latter case, the
                    # caller will assign it (see just below) but first I
                    # will do a search.
                    subsens = _recursively_search_for(
                        sub_sensitivities,
                        v.args[0])
                    subcomp = _recursively_search_for(
                        sub_components,
                        v.args[0])
                else:
                    (adict[v], subsens, subcomp) = self.calc_u_for_variable(
                        v.args[0], quantities, all_effects,
                        cached_uncertainties, return_more=True)
                    # Callee may have taken some uncertainties from cache;
                    # the associated subsens/subcomp are probably the ones
                    # /I/ calculated… see just above!
                    for (ddto, ddfrom) in (
                            (subsens, sub_sensitivities),
                            (subcomp, sub_components)):
                        for (kk, vv) in ddto.items():
                            if vv[1] is None:
                                ddto[kk] = (ddto[kk][0],
                                    _recursively_search_for(ddfrom, kk))
                                assert ddto[kk] is not None, \
                                       "Should not be None now :("
                    cached_uncertainties[v.args[0]] = adict[v]
                # NB: We should have sensitivities.keys() ==
                # components.keys() == our current loop
                sub_sensitivities[v.args[0]] = (
                    sensitivities[v.args[0]], subsens)
                sub_components[v.args[0]] = (
                    components[v.args[0]], subcomp)
            else:
                # it's a quantity
                if v not in quantities:
                    if v not in me.expressions.keys():
                        raise ValueError(
                            "Calculation of u({!s})={!s} needs defined value or "
                            "expression for "
                            "quantity {!s} but this is not set.  I have values "
                            "or expressions for: {:s}.".format(
                                s, e, v, str(list(quantities.keys()))))
                    quantities[v] = me.evaluate_quantity(v, quantities)
                    vname = me.names.get(v, str(v))
                    quantities[v].name = vname
                    quantities[v].attrs.update(
                        self._data_vars_props[me.names[v]][2])
                    quantities[v].encoding.update(
                        self._data_vars_props[me.names[v]][3])

                adict[v] = quantities[v]

        # NB: syntax, cannot use parentheses as it will test an always-True
        # tuple!
        if len(failures) == 0:
            assert sensitivities.keys() == components.keys() == \
                    sub_sensitivities.keys() == sub_components.keys(), \
                    "Found inconsistencies between bookkeeping dicts!"
                    
        # now I have adict with values for uncertainties and other
        # quantities, that I need to substitute into the expression
        # I expect I'll have to do some trick to substitute u(x)? no?
        ta = tuple(args)
        # dummify=True because some args are not valid identifiers
        f = sympy.lambdify(ta, u_e, numpy, dummify=True)
        # multiple dimensions with time coordinates:
        # - calibration_cycle
        # - scanline_earth
        # Any dimension other than calibration_cycle needs to be
        # interpolated to have dimension scanline_earth before further
        # processing.
        src_dims = set().union(itertools.chain.from_iterable(
            x.dims for x in adict.values() if hasattr(x, 'dims')))
        dest_dims = set(self._data_vars_props[me.names[s]][1])
        if not dest_dims <= src_dims: # problem!
            warnings.warn("Cannot correctly estimate uncertainty u({!s}). "
                "Destination has dimensions {!s}, arguments (between them) "
                "have {!s}!".format(s, dest_dims, src_dims or "none"),
                FCDRWarning)
        if not src_dims <= dest_dims: # needs reducing
            adict = self._make_dims_consistent(adict)
        # verify/convert dimensions
        u = f(*[typhon.math.common.promote_maximally(
                    adict[x]).to_root_units() for x in ta])
        u = u.to(self._data_vars_props[me.names[s]][2]["units"])
        u = u.rename("u_"+me.names[s])
        cached_uncertainties[s] = u
        if return_more:
            var_unit = self._data_vars_props[me.names[s]][2]["units"]
            # turn expressions into data for the dictionairies
            # sub_sensitivities and sub_components
            for k in sub_sensitivities.keys():
                # I already verified that sub_sensitivities and
                # sub_components have the same keys
                args = typhon.physics.metrology.recursive_args(u_e,
                    stop_at=(sympy.Symbol, sympy.Indexed, fu))
                ta = tuple(args)
                for dd in (sub_sensitivities, sub_components):
                    # dummify=True because some args are not valid
                    # identifiers
                    f = sympy.lambdify(ta, dd[k][0], numpy, dummify=True)
                    dd[k] = (f(
                        *[typhon.math.common.promote_maximally(adict[x]).to_root_units()
                            for x in ta]),
                            dd[k][1])
            # make units nicer.  This may prevent loss of precision
            # problems when values become impractically large or small
            for (k, v) in sub_sensitivities.items():
                if isinstance(sub_sensitivities[k][0], numbers.Number):
                    # sensitivity is a scalar / constant
                    k_unit = self._data_vars_props[me.names[k]][2]["units"]
                    if not k_unit == var_unit:
                        raise ValueError("∂{s!s}/∂{k!s} = {sens:g} should be "
                            "dimensionless, but {s!s} is in "
                            "{var_unit!s} and {k!s} is in "
                            "{k_unit!s}".format(sens=sub_sensitivities[k][0],
                                **vars()))
                    # nothing else to do
                else: # must be xarray.DataArray
                    sub_sensitivities[k] = (
                        sub_sensitivities[k][0].to(
                            ureg.Unit(self._data_vars_props[me.names[s]][2]["units"])/
                            ureg.Unit(self._data_vars_props[me.names[k]][2]["units"])),
                        sub_sensitivities[k][1])
            for (k, v) in sub_components.items():
                sub_components[k] = (
                    sub_components[k][0].to(
                        self._data_vars_props[me.names[s]][2]["units"]),
                   sub_components[k][1])
            # FIXME: perhaps I need to add prefixes such that the
            # magnitude becomes close to 1?
            return (u, sub_sensitivities, sub_components)
        else:
            return u

    @staticmethod
    def _make_dims_consistent(adict):
        """Ensure dims are consistent

        The components of adict are the contents to calculate var.  Make
        sure dimensions are consistent, through interpolation, averaging,
        etc.  Requires that the desired dimensions of var occur in at
        least one of the values for adict so that coordinates can be read
        from it.

        Currently hardcoded for:
            
            - calibration_cycle → interpolate → scanline_earth
            - calibration_position → average → ()
        """
        new_adict = {}

        dest_time = adict[me.symbols["C_E"]].scanline_earth.astype("u8")

        for (k, v) in adict.items():
            if "calibration_position" in v.dims:
                v = v.mean(dim="calibration_position", keep_attrs=True)
            if "calibration_cycle" in v.dims or "rself_update_time" in v.dims:
                d = ("calibration_cycle"
                     if "calibration_cycle" in v.dims
                     else "rself_update_time")
                src_time = v[d].astype("u8")#adict[me.symbols["a_1"]][d].astype("u8")
                fnc = scipy.interpolate.interp1d(
                    src_time, v,
                    kind="zero",
                    bounds_error=True,
                    axis=v.dims.index(d))
                new_adict[k] = UADA(fnc(dest_time),
                    dims=[x.replace(d, "scanline_earth") for
                            x in v.dims],
                    coords=adict[me.symbols["C_E"]].coords,
                    attrs=v.attrs,
                    encoding=v.encoding)
            else:
                new_adict[k] = v

        return new_adict


    def numerically_propagate_ΔL(self, L, ΔL):
        """Temporary method to numerically propagate L to Tb

        Until I find a proper solution for the exploding Tb uncertainties
        (see https://github.com/FIDUCEO/FCDR_HIRS/issues/78 ) approximate
        these numerically
        """
        ΔTb = xarray.zeros_like(L).drop(("scanline", "lat", "lon"))
        ΔTb.attrs["units"] = "K"
        for ch in range(1, 20):
            srf = self.srfs[ch-1]
            Lch = L.sel(calibrated_channel=ch)
            ΔLch = ΔL.sel(calibrated_channel=ch)
            low = (Lch-ΔLch).to("K", "radiance", srf=srf)
            high = (Lch+ΔLch).to("K", "radiance", srf=srf)
            ΔTb.loc[{"calibrated_channel": ch}] = (high-low)/2
        return ΔTb

    def estimate_channel_correlation_matrix(self, ds_context, calpos=20):
        """Estimate channel correlation matrix

        Calculates correlation coefficients between space view anomalies
        accross channels.  

        As per #87.  
        """
        Cs = ds_context["counts"].isel(time=ds_context["scantype"].values == self.typ_space)

        ΔCs = (Cs - Cs.mean("scanpos"))
        S = numpy.corrcoef(ΔCs.sel(scanpos=calpos).T)
        da = xarray.DataArray(S,
            coords={"channel": ds_context.coords["channel"]},
            dims=("channel", "channel"))
        da.name = "channel_correlation_matrix"
        da.attrs = self._data_vars_props[da.name][2]
        da.encoding = self._data_vars_props[da.name][3]
        return da

    def get_BT_to_L_LUT(self):
        """Returns LUT to translate BT to LUT
        """

        n = 100
        LUT_BT = xarray.DataArray(
            numpy.tile(numpy.linspace(200, 300, 101)[:, numpy.newaxis],
                       19),
            coords={"calibrated_channel": range(1, 20)},
            dims=("LUT_index", "calibrated_channel"),
            name="LUT_BT")
        LUT_radiance = xarray.DataArray(
            numpy.zeros(shape=(101, 19), dtype="f4"),
            coords=LUT_BT.coords,
            dims=LUT_BT.dims,
            name="LUT_radiance")
        for ch in LUT_BT.calibrated_channel.values:
            srf = typhon.physics.units.em.SRF.fromArtsXML(
                self.satname.upper(), "hirs", ch)
            LUT_radiance.loc[{"calibrated_channel": ch}] = (
                srf.blackbody_radiance(LUT_BT.sel(calibrated_channel=ch)).to(
                    rad_u["ir"], "radiance"))
        return (LUT_BT, LUT_radiance)

    def get_flags(self, ds, context, R_E):
        """Get flags for FCDR

        Only those for which I have the information I need are set, in
        practice those that have been copied.
        """

        flags_scanline = xarray.DataArray(
            numpy.zeros(
                shape=R_E["scanline_earth"].size,
                dtype=self._data_vars_props["quality_scanline_bitmask"][3]["dtype"]),
            dims=("scanline_earth",),
            coords={"scanline_earth": R_E.coords["scanline_earth"]},
            name="quality_scanline_bitmask",
            attrs=self._data_vars_props["quality_scanline_bitmask"][2]
            )

        flags_channel = xarray.DataArray(
            numpy.zeros(
                shape=(R_E["scanline_earth"].size, R_E["calibrated_channel"].size),
                dtype=self._data_vars_props["quality_scanline_bitmask"][3]["dtype"]),
            dims=("scanline_earth", "calibrated_channel"),
            coords={"scanline_earth": R_E.coords["scanline_earth"],
                    "calibrated_channel": R_E.coords["calibrated_channel"]},
            name="quality_channel_bitmask",
            attrs=self._data_vars_props["quality_channel_bitmask"][2]
            )
        
        da_qfb = ds["quality_flags_bitfield"].sel(
            time=R_E.coords["scanline_earth"])
#        da_lqfb = ds["line_quality_flags_bitfield"].sel(
#            time=R_E.coords["scanline_earth"])
        fd_qif = typhon.datasets._tovs_defs.QualIndFlagsHIRS[self.version]
#        fd_qfb = typhon.datasets._tovs_defs.LinQualFlagsHIRS[self.version]
        fs = _fcdr_defs.FlagsScanline
#        fc = _fcdr_defs.FlagsChannel

#        probs = {st:
#            functools.reduce(operator.or_,
#                (v for (k, v) in fd_qfb.__members__.items() if k.startswith(st)))
#                    for st in {"tm", "el", "ca"}}
#        probs["tm"] |= (da_qfb & fd_qif.qitimeseqerr)

        flags_scanline[{"scanline_earth":((da_qfb & fd_qif.qidonotuse)!=0)}] |= fs.DO_NOT_USE
        flags_scanline[{"scanline_earth":((da_qfb & fd_qif.qitimeseqerr)!=0)}] |= fs.SUSPECT_TIME
        flags_scanline[{"scanline_earth":((da_qfb & fd_qif.qinofullcalib)!=0)}] |= fs.SUSPECT_CALIB
        flags_scanline[{"scanline_earth":((da_qfb & fd_qif.qinoearthloc)!=0)}] |= fs.SUSPECT_GEO
#        flags_scanline[{"scanline_earth":((da_lqfb & probs["el"])!=0)}] |= fs.SUSPECT_GEO
#        flags_scanline[{"scanline_earth":((da_lqfb & probs["tm"])!=0)}] |= fs.SUSPECT_TIME
#        flags_scanline[{"scanline_earth":((da_lqfb & probs["ca"])!=0)}] |= fs.SUSPECT_CALIB

#        flags_channel[{"scanline_earth":((da_qfb & fd_qif.qidonotuse)!=0)}] |= fc.DO_NOT_USE

        return (flags_scanline, flags_channel)


    # The remaining methods are no longer used.

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
#        (u_C_iwct,) = h.interpolate_between_calibs(M["time"],
#            t_iwt_noise_level, u_C_iwct)
#        (u_C_space,) = h.interpolate_between_calibs(M["time"],
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

class HIRSPODFCDR:
    """Mixin for HIRS POD FCDRs
    """
    def get_flags(self, ds, context, R_E):
        (flags_scanline, flags_channel) = super().get_flags(ds, context, R_E)

        # might add stuff here later

        return (flags_scanline, flags_channel)

class HIRSKLMFCDR:
    """Mixin for HIRS KLM FCDRs
    """
    def get_flags(self, ds, context, R_E):
        """Get flags for FCDR

        Only those for which I have the information I need are set, in
        practice those that have been copied.
        """

        (flags_scanline, flags_channel) = super().get_flags(ds, context, R_E)

        da_qfb = ds["quality_flags_bitfield"].sel(
            time=R_E.coords["scanline_earth"])
        da_lqfb = ds["line_quality_flags_bitfield"].sel(
            time=R_E.coords["scanline_earth"])
        fd_qif = typhon.datasets._tovs_defs.QualIndFlagsHIRS[self.version]
        fd_qfb = typhon.datasets._tovs_defs.LinQualFlagsHIRS[self.version]
        fs = _fcdr_defs.FlagsScanline
        fc = _fcdr_defs.FlagsChannel

        probs = {st:
            functools.reduce(operator.or_,
                (v for (k, v) in fd_qfb.__members__.items() if k.startswith(st)))
                    for st in {"tm", "el", "ca"}}

        flags_scanline[{"scanline_earth":((da_lqfb & probs["el"])!=0)}] |= fs.SUSPECT_GEO
        flags_scanline[{"scanline_earth":((da_lqfb & probs["tm"])!=0)}] |= fs.SUSPECT_TIME
        flags_scanline[{"scanline_earth":((da_lqfb & probs["ca"])!=0)}] |= fs.SUSPECT_CALIB

        flags_channel[{"scanline_earth":((da_qfb & fd_qif.qidonotuse)!=0)}] |= fc.DO_NOT_USE

        return (flags_scanline, flags_channel)

class HIRS2FCDR(HIRSPODFCDR, HIRSFCDR, HIRS2):
    l1b_base = HIRS2

class HIRS3FCDR(HIRSKLMFCDR, HIRSFCDR, HIRS3):
    l1b_base = HIRS3

class HIRS4FCDR(HIRSKLMFCDR, HIRSFCDR, HIRS4):
    l1b_base = HIRS4

def which_hirs_fcdr(satname, *args, **kwargs):
    """Given a satellite, return right HIRS object
    """
    for h in {HIRS2FCDR, HIRS3FCDR, HIRS4FCDR}:
        for (k, v) in h.satellites.items():
            if satname in {k}|v:
                return h(*args, satname=k, **kwargs)
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

def _recursively_search_for(sub, var):
    """Search if 'var' already exists in the tree for
    sub_sensitivities or sub_components
    """

    for (k, v) in sub.items():
        if k is var:
            return sub[k][1]
        elif sub[k][1] is not None:
            res = _recursively_search_for(sub[k][1], var)
            if res is not None:
                return res


# Patch xarray.core._ignore_warnings_if to avoid repeatedly hearing the
# same warnings.  This function contains the catch_warnings contextmanager
# which is buggy, see http://bugs.python.org/issue29672
import contextlib
import xarray.core.ops

@contextlib.contextmanager
def do_nothing(*args, **kwargs):
    yield
xarray.core.ops._ignore_warnings_if = do_nothing
xarray.core.duck_array_ops._ignore_warnings_if = do_nothing
