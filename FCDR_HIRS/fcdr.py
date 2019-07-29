"""Datasets for TOVS/ATOVS

This is the main workhorse of the FCDR, in particular the `HIRSFCDR`
class, which is nearly 3 KLOC by itself.  Although the implementation is
essentially all in the `HIRSFCDR` class, you should not instantiate that
one directly.  Rather, you should instantiate either `HIRS2FCDR`,
`HIRS3FCDR`, or `HIRS4FCDR`, which contain both the functionality for
`HIRSFCDR` (which is shared between different types of HIRS), and the
functionality from the classes `typhon.datasets.tovs.HIRS2`,
`typhon.datasets.tovs.HIRS3`, and `typhon.datasets.tovs.HIRS4`,
which in turn inherit from `typhon.datasets.tovs.HIRS`.  To get an object
from one of the classes in this module, you will want to call
`which_hirs_fcdr`.

Classes defined here:

`HIRSFCDR`
    Contains the core functionality for calculating calibrated HIRS
    radiances and their uncertainties.
`HIRS2FCDR`, `HIRS3FCDR`, `HIRS4FCDR`
    Specific classes to be instantiated.
`HIRSPODFCDR`, `HIRSKLMFCDR`
    Some limited specific functionality and definitions that is only
    relevant for particular types of HIRS.

Important functionality inherited from typhon:

`typhon.datasets.tovs.HIRS`
    Contains the bulk of the HIRS-specific L1B functionality, in
    particular a reading routine, but also filtering that is part of a
    reading routine even when not producing an FCDR (see also the module
    :mod:`typhon.datasets.filters`).
`typhon.datasets.tovs.HIRS2`, `typhon.datasets.tovs.HIRS3`, typhon.datasets.tovs.HIRS4`
    Specific classes to be instantiated; the :class:`HIRS2FCDR`, :class:`HIRS3FCDR` and
    :class:`HIRS4FCDR` have those in their inheritance tree.
`typhon.datasets.dataset.Dataset` and subclasses:
    Generic functionality for reading and processing satellite data, such
    as an abstraction on how to read longer periods of data in
    `typhon.datasets.dataset.MultiFileDataset.read_period`.

This module also defines a number of utility functions.  The most useful
may be `which_hirs_fcdr`, which accepts a satellite name in any spelling
variation and returns the instance of the correct :class:`HIRS2FDCR`,
:class:`HIRS3FCDR`, or :class:`HIRS4FCDR` class.
"""

import logging
import itertools
import warnings
import functools
import operator
import datetime
import numbers
import math

import numpy
import scipy.interpolate
import progressbar
import pandas
import xarray
import sympy
import pyorbital.orbital
import pyorbital.astronomy
import fiduceo.fcdr.writer.fcdr_writer

import typhon.physics.units.tools
import typhon.physics.units.em
from typhon.physics.units.tools import UnitsAwareDataArray, UnitsAwareDataArray as UADA
from typhon.utils import get_time_dimensions
    
import typhon.datasets.dataset
import typhon.physics.units
from typhon.physics.units.common import ureg, radiance_units as rad_u
from typhon.datasets.tovs import (Radiometer, HIRS, HIRSPOD, HIRS2,
    HIRSKLM, HIRS3, HIRS4)

from . import models
from . import effects
from . import measurement_equation as me
from . import filters
from . import _fcdr_defs
from . import _harm_defs
from . import common
from .common import list_all_satellites
from .exceptions import (FCDRError, FCDRWarning) # used to be here

logger = logging.getLogger(__name__)

class HIRSFCDR(typhon.datasets.dataset.HomemadeDataset):
    """Generic methods related to HIRS FCDR
    
    This class contains the core functionality to produce, read, and write
    the HIRS FCDR.  A higher level class building on top of the
    functionality is in :class:`~FCDR_HIRS.processing.generate_fcdr.FCDRGenerator`.
    There are three HIRS FCDR classes for the three generations of HIRS
    covered by FIDUCEO, and the user should produce one instance per
    specific instance of HIRS (per satellite).  Each instance covers all
    channels.  To adapt how the actual radiances and uncertainties are
    calculated, adapt methods in this class.  To adapt the more overall
    processing and storage, and to actually process the FCDR, use methods
    in :class:`~FCDR_HIRS.processing.generate_fcdr.FCDRGenerator`.

    The production of the HIRS FCDR works on segments that can be as short
    as one scanline and as long as memory permits.  Along with the segment
    for which the HIRS FCDR is being produced, context must be provided.
    The context must contain at least the calibration cycle before and
    after the beginning and ending of the segment to be processed, but 
    may contain considerable more; for example, the default self-emission
    model (see `models.RSelf`) requires a 24-hour context.

    Read in some HIRS data, including nominal calibration.
    Estimate noise levels from space and IWCT views.
    Use noise levels to propagate through calibration and BT conversion.

    Objects of `HIRSFCDR` should not be constructed directly.  Users
    should instead produce objects from `HIRS2FCDR`, `HIRS3FCDR`, and
    `HIRS4FCDR`.  The utility function `which_hirs_fcdr` will return an
    object of the correct class, depending on the satellite.

    Attributes can either be set as keyword arguments when constructing an
    object, or as part of the ``.typhonrc`` configuration file.

    This class always produces L1C, but can be used to read either L1B or
    L1C.  Pass the ``read`` keyword argument as ``"L1B"`` or ``"L1C"``.

    Note that variable names in this class may not match the meaning in
    the measurement equation in D2.2.

    Relevant papers:

        - Cao, Jarva, and Ciren, An Improved Algorithm for the Operational
          Calibration of the High-Resolution Infrared Radiation Sounder,
          JOURNAL OF ATMOSPHERIC AND OCEANIC TECHNOLOGY, 24, 2007, 
          DOI: 10.1175/JTECH2037.1
        - HIRS 4 Level 1 Product Generation Specification,
          EUM.EPS.SYS.SPE.990007, v6, 17 September 2013

    
    Some methods and attributes in this class are very old and not used in
    actual FCDR production, but retained because some old analysis scripts
    rely on them to work.  **DO NOT USE** those methods:

    .. deprecated:

        - `recalibrate` **DO NOT USE**
        - `estimate_noise` **DO NOT USE**
        - `read_and_recalibrate_period` **DO NOT USE**
        - `extract_and_interp_calibcounts_and_temp` **DO NOT USE**
        - `estimate_Rself` **DO NOT USE**
        - `calc_sens_coef` **DO NOT USE**
        - `calc_sens_coef_C_Earth` **DO NOT USE**
        - `calc_sens_coef_C_iwct` **DO NOT USE**
        - `calc_sens_coef_C_iwct_slope` **DO NOT USE**
        - `calc_sens_coef_C_space` **DO NOT USE**
        - `calc_sens_coef_C_space_slope` **DO NOT USE**
        - `calc_urad` **DO NOT USE**
        - `calc_urad_noise` **DO NOT USE**
        - `calc_urad_calib` **DO NOT USE**
        - `calc_uslope` **DO NOT USE**
        - `calc_S_noise` **DO NOT USE**
        - `calc_S_calib` **DO NOT USE**

    """

    #: name and configuration section for FCDR HIRS

    name = section = "fcdr_hirs"
    # See spreadsheet attached to e-mail from Tom Block to fiduceo-project
    # mailing list, 2017-03-31

    #: stored name according to TB format
    stored_name = ("FIDUCEO_FCDR_L1C_HIRS{version:d}_{satname:s}_"
                   "{year:04d}{month:02d}{day:02d}{hour:02d}{minute:02d}{second:02d}_"
                   "{year_end:04d}{month_end:02d}{day_end:02d}{hour_end:02d}{minute_end:02d}{second_end:02d}_"
                   "{fcdr_type:s}_v{data_version:s}_fv{format_version:s}.nc")

    #: structure of the directory tree where FCDR files are stored
    write_subdir = "{fcdr_type:s}/{satname:s}/{year:04d}/{month:02d}/{day:02d}"
    
    #: regular expression to read stored L1C files
    stored_re = (r"FIDUCEO_FCDR_L1C_HIRS(?P<version>[2-4])_"
                 r"(?P<satname>.{6})_"
                 r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})"
                 r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})_"
                 r"(?P<year_end>\d{4})(?P<month_end>\d{2})(?P<day_end>\d{2})"
                 r"(?P<hour_end>\d{2})(?P<minute_end>\d{2})(?P<second_end>\d{2})_"
                 r"(?P<fcdr_type>[a-zA-Z]*)_"
                 r"v(?P<data_version>.+)_"
                 r"fv(?P<format_version>.+)\.nc")
    
    #: regular expression to read stored L1C files before data_version v0.5
    old_stored_re = (
                 r'FCDR_HIRS_(?P<satname>.{6})_(?P<fcdr_version>.+)_'
                 r'(?P<fcdr_type>[a-zA-Z]*)_'
                 r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})'
                 r'(?P<hour>\d{2})(?P<minute>\d{2})_'
                 r'(?P<year_end>\d{4})(?P<month_end>\d{2})(?P<day_end>\d{2})'
                 r'(?P<hour_end>\d{2})(?P<minute_end>\d{2})\.nc')

    format_version=fiduceo.fcdr.writer.fcdr_writer.__version__
    """Format version for stored FCDR data

    Format changelog:
    
    v0.3
        removed copied flags, added own flag fields
    v0.4
        attribute with flag masks is now a numeric one
    v0.5
        renamed random -> independent, non-random -> structured,
        changed e-mail address to fiduceo-coordinator
    v0.6
        changed y-coordinate which now simply continues numerically
    v0.7
        various changes, adapted to Tom Blocks format, now with SRFs,
        LUTs, correlation lengths and scales, new encodings, correct
        filenames (uppercase satellite names)
    v2.0.0
        jumped to 2.0.0 because I now follow the FCDRTools convention

    """

    #: Not used.
    realisations = 100

    srfs = None
    """List of :class:`~typhon.physics.units.SRF` objects.

    Set by `__init__`, after instantiation this attribute will contain a
    list of SRFs, specifically of :class:`typhon.physics.units.SRF` objects.
    """

    #: name of satellite to which this edition of the HIRS FCDR belongs
    satname = None

    #: location of directory with band coefficient parameters
    band_dir = None
    #: location of files with band coefficient parameters, within `band_dir`
    band_file = None
    #: should reading methods read ``'L1B'`` or ``'L1C'``?
    read_mode = "L1B"

    #: must always be set to ``"xarray"``
    read_returns = "xarray" # for NetCDFDataset in my inheritance tree

    l1b_base = None
    """:class:`~typhon.datasets.Dataset` : Which dataset is used for reading.

    Value is set by :meth:`__init__` depending an :attr:`read_mode`.
    """

    start_space_calib = 8
    """Index at which to start using space calibration

    The first 8 views of space counts deemed always unusable, see
    NOAA or EUMETSAT calibration papers/documents.  I've personaly
    witnessed (on NOAA-18) that later positions are sometimes also
    systematically offset; see :issue:`12` and the class
    `filters.CalibrationMirrorFilter`.
    """   

    #: Index at which to start using IWCT calibration
    start_iwct_calib = 8

    #: Filter to use for calibration counts based on IQR
    calibfilter = filters.IQRCalibFilter()

    #: Filter to use for Earth counts
    filter_earthcounts = typhon.datasets.filters.MEDMAD(5,hirs=True,\
                                                            calibration_data=False,\
                                                            prt=False)

    #: Filter to weed out unlikely cold Earth scenes (colder than space)
    filter_coldearth = filters.ImSoColdFilter()

    #: Filter to use for PRT counts
    filter_prtcounts = typhon.datasets.filters.MEDMAD(5,prt=True,hirs=True,\
                                                          calibration_data=False)

    #: Filter to use for calibration counts, based on MEDMAD
    filter_calibcounts = typhon.datasets.filters.MEDMAD(5,hirs=True,\
                                                            calibration_data=\
                                                            True,prt=False)

    ε = 0.98
    """Assumed reference emissivity for the IWCT
    
    Wang, Cao, and Ciren (2007, JAOT) use ε=0.98 who give no further
    source for this number
    """

    no_harm = False #: suppress harmonisation

    def __new__(cls, name=None, **kwargs):
        if name is None and "satname" in kwargs:
            name = "fcdr_hirs_" + kwargs["satname"]
        return super().__new__(cls, name, **kwargs)

    def __init__(self, read="L1B", *args, satname, **kwargs):
        for nm in {satname}|self.satellites[satname]:
            try:
                self.srfs = [typhon.physics.units.em.SRF.fromRTTOV(
                              typhon.datasets.tovs.norm_tovs_name(nm, "RTTOV"),
                              self.section, i) for i in range(1, 20)]
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
            self.default_orbit_filters = []
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
                lambda M, D, H, fn:
                self.calculate_radiance_all(
                    M[1] if isinstance(M, tuple) else M, 
                    return_ndarray=True,
                    naive=True),
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

    def within_enough_context(self, ds, context, ch, n=1):
        """Get an indexer for ``ds`` that ensures enough context

        In some cases, there is not enough context around a segment for
        which we are calculating radiances.  This methods takes a segment
        of L1B data in ``ds``, as well as a context segment ``context``.
        It returns a dictionary that can be used to index the
        `xarray.Dataset` that is ``ds``, such that the ``context``
        contains at least ``n`` calibration cycles on both ends of ``ds``.
        When all is working well, i.e. there is enough context already,
        the indexing of ``ds`` with this dictionary will return exactly
        ``ds`` again.

        Parameters
        ----------

        ds : `xarray.Dataset`
            `xarray.Dataset` object containing HIRS L1B data from which
            the FCDR is to be calculated.
        context : `xarray.Dataset` containing the context of HIRS L1B data
        ch : int
            Channel for which the index will be calculated.  The indexing
            may be channel-dependent if flagging or filtering filters out
            different calibration lines for different channels.
        n : int, optional
            How many calibration cycles before and after the core segment
            are needed.  Defaults to 1.

        Returns
        -------

        dict
            Dictionary to be used to index ``ds``.
        """

        if n==0:
            return ds

        (counts_space, counts_iwct) = self.extract_calibcounts(context,
            ch, fail_if_none=True)
        
        # let's also trigger a failure if the context does but the core
        # does not have calibration counts
        self.extract_calibcounts(ds, ch, fail_if_none=True)

        if counts_space["time"].size == 0:
            # set slices that will make things empty
            ii = dict.fromkeys(
                    get_time_dimensions(ds),
                    slice(ds["time"][-1].values.astype("M8[ms]"),
                          ds["time"][0].values.astype("M8[ms]")))
        else:
            ii = dict.fromkeys(
                    get_time_dimensions(ds),
                    slice(counts_space["time"][n-1].values.astype("M8[ms]").astype(datetime.datetime),
                          counts_space["time"][-n].values.astype("M8[ms]").astype(datetime.datetime)))
        return ii
        

    def interpolate_between_calibs(self, target_time, calib_time, *args, kind="nearest"):
        """Interpolate calibration parameters between calibration cycles

        This method is designed to provide basic interpolation of the calibration
        parameters slope and offset between calibration parameters.  If
        you are using a self-emission model, you probably want to use
        ``kind="zero"``, as to not double-count self-emission. Note that this
        interpolates at a constant value as we need

        Although designed between calibration parameters, it is really
        just a wrapper around `scipy.interpolate.interp1d` so it can be
        used in other contexts as well.

        Parameters
        ----------
        
        target_time : ndarray, dtype time
            
            Times for which the parameters should be returned.  This may
            be an array with the time for every scanline.

        calib_time : ndarray, dtype time

            Times for which a "measurement" is available.  This may be
            offset and slope, such as returned by
            :meth:`calculate_offset_and_slope`.

        *args : List[ndarray]
                
            List of arrays of anything defined only at ``calib_time``.
            For example, that may be ``slope`` and ``offset``.

        kind : str, optional

            Type of interpolation to use.  Options are as for
            `scipy.interpolate.interp1d`, which is where the actual
            interpolation is being done.  Defaults to ``"nearest"``.
        
        Returns
        -------

        List[typhon.physics.units.utils.UnitsAwareDataArray]
            
            Each of the arrays in ``args``, interpolated to all times in
            ``target_time``.
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

    def custom_calibrate(self, counts, slope, offset, a2, Rself, a_4=0):
        """Apply core of measurement equation: calibrate with own slope and offset

        Apply the very core of the measurement equation, calculating Earth
        radiance as a function of offset, slope, counts, non-linearity,
        self-emission, and harmonisation offset.  This is the lowest level
        method of FCDR radiance calculations.  The dimensions of the
        parameters must be compatible, to avoid unwanted broadcasting.
        That means that the slope and offset must already have been
        interpolated to have the same dimensions as the counts.  Of
        course, dimensions can be omitted, for example, ``a2`` and ``a_4``
        can be scalar.

        The arguments should be either `xarray.DataArray` or, preferably,
        `typhon.physics.units.tools.UnitsAwareDataArray`.  I recommend the
        latter so that unit compatibility is ensured.

        Parameters
        ----------

        counts : xarray.DataArray
            Earth counts.
        slope : xarray.DataArray
            Interpolated slope as obtained from `interpolate_between_calibs`.
        a2 : xarray.DataArray
            Harmonisation non-linearity coefficient.
        Rself: xarray.DataArray
            Self-emission estimate.
        a_4: xarray.DataArray
            Harmonisation offset.

        Returns
        -------

        `xarray.DataArray` or `typhon.physics.units.tools.UnitsAwareDataArray`
            Calibrated Earth radiances
        """
        return offset + slope * counts + a2 * counts**2 - Rself + a_4
#        return (offset[:, numpy.newaxis]
#              + slope[:, numpy.newaxis] * counts
#              + a2 * counts**2
#              - Rself)

    def extract_calibcounts(self, ds, ch, fail_if_none=False):
        """Extract calibration counts from data

        Extract space counts and IWCT counts as `xarray.DataArray`.  This
        is done by searching for lines that are space views, followed by
        lines that are IWCT views, possibly with an ICCT view in-between,
        in case of HIRS/2 (see the attribute :attr:`~HIRS2FCDR.dist_space_iwct`).  The
        space and IWCT views are then each assigned the time coordinates
        corresponding to the space views, thus pretending they were
        calculated at the same time.  The latter ensures coordinates are
        aligned so we can do ``offset / slope`` or such, without xarray
        inadvertently broadcasting based on different coordinates, or
        returning and empty array because no coordinates for the same
        dimension match.

        Use the higher level method
        `HIRSFCDR.extract_calibcounts_and_temp` if you additionally want
        to obtain IWCT radiances and IWCT and space count uncertainty
        estimates.

        Parameters
        ----------

        ds : `xarray.Dataset`
            L1B HIRS data from which to extract the calibration counts.
        ch : int
            Channel for which to extract the calibration counts.
        fail_if_none : bool, optional
            If there are none at all, raise an exception.  Defaults to
            False, which simply returns an empty array.

        Returns
        -------

        :class:`~typhon.physics.units.tools.UnitsAwareDataArray`
            Space counts
        :class:`~typhon.physics.units.tools.UnitsAwareDataArray`
            IWCT counts, but with space counts coordinates
        """
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

        if fail_if_none and not views_space.any():
            raise FCDRError("No space views found, giving up")
        elif fail_if_none and not views_iwct.any():
            raise FCDRError("No IWCT views found, giving up")
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
        if fail_if_none and not space_followed_by_iwct.any():
            raise FCDRError(
                "I have space and IWCT views, but not in same cycle?!")

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
        counts_space = counts_space.drop(counts_space.coords.keys()&{"scanline", "lat", "lon"})
        counts_space.attrs.update(units="counts")
#        counts_space = ureg.Quantity(M_space["counts"][:,
#            self.start_space_calib:, ch-1], ureg.count)
        # For IWCT, at least EUMETSAT uses all 56…
        counts_iwct = UADA(ds_iwct["counts"].sel(
                scanpos=slice(self.start_iwct_calib, None),
                channel=ch).rename({"channel": "calibrated_channel"}),
            name="counts_iwct")
        counts_iwct = counts_iwct.drop(counts_iwct.coords.keys()&{"scanline", "lat", "lon"})
        counts_iwct.attrs.update(units="counts")
#        counts_iwct = ureg.Quantity(M_iwct["counts"][:,
#            self.start_iwct_calib:, ch-1], ureg.count)

        return (counts_space, counts_iwct)

    def extract_calibcounts_and_temp(self, ds, ch, srf=None,
            return_u=False, return_ix=False, tuck=False,
            include_emissivity_correction=True):
        """Get IWCT radiance, calibration counts, and their uncertainties

        From L1B data in the form of an xarray dataset as returned by
        `typhon.datasets.tovs.HIRS.as_xarray_dataset`, extract calibration
        counts (space counts and IWCT counts), calculate the IWCT
        radiance, and calculate uncertainties for IWCT and space counts.
        Use the lower level method `HIRSFCDR.extract_calibcounts` if you
        only want to obtain the IWCT and space counts, and see the
        documentation for that method on more details on how the space and
        IWCT counts are extracted.

        Parameters
        ----------

        ds : xarray.Dataset

            L1B data such as returned by `as_xarray_dataset`, for which
            the radiances and counts are calculated and extracted.  Must
            have at least the fields ``time``, ``scantype``, ``counts``,
            and ``temperature_internal_warm_calibration_target``.

        ch : int

            Channel for which counts shall be returned and IWCT
            temperature shall be calculated.

        srf : :class:`typhon.physics.units.SRF`, optional

            SRF object used to estimate IWCT.  Optional; if not given
            or None, use the standard SRF for the channel as given by
            RTTOV.

        return_u : bool, optional

            Also return uncertainty estimates.  Defaults to False.

        return_ix : bool, optional
            
            Also return indices to ds corresponding to the calibration
            lines.  Defaults to False.

        tuck : bool, optional

            Cache values and uncertainties.  This is needed if we
            subsequently want to calculate the overall radiance
            uncertainty.  Defaults to False.

        include_emissivity_correction : bool, optional

            Whether to consider tho harmonisation parameter correcting the
            emissivity or not.  Normally this should be True, which is the
            default, but in some cases the user may want to switch it off
            for debugging purposes.

        Returns
        -------

        time : `xarray.DataArray`

            Time corresponding to the other returned arrays.

        L_iwct : `xarray.DataArray`

            radiance corresponding to IWCT views.  Calculated by
            assuming ε from self.ε, an arithmetic mean of all
            temperature sensors on the IWCT, and the SRF passed to the
            method.  Earthshine / reflection through the blackbody is
            not yet implemented (see #18)

        counts_iwct : `xarray.DataArray`

            counts corresponding to IWCT views

        counts_space : `xarray.DataArray`

            counts corresponding to space views

        u_counts_iwct : `xarray.DataArray`

            uncertainty on counts_iwct, only returned if input argument
            ``return_u`` is True

        u_counts_space : `xarray.DataArray`

            uncertainties on counts_space, only returned if input argument
            ``return_u`` is True

        ix : `xarray.DataArray`

            indices to lines containing space views, only returned if input
            argument ``return_ix`` is True
        """

        srf = srf or self.srfs[ch-1]

        # 2017-02-22 backward compatibility
        if not isinstance(ds, xarray.Dataset):
            warnings.warn("Passing ndarray to extract_calibcounts_and_temp "
                "is deprecated since 2017-02-22, should pass "
                "xarray.Dataset ", DeprecationWarning)
            ds = self.as_xarray_dataset(ds)

        (counts_space, counts_iwct) = self.extract_calibcounts(ds, ch)

        # starting with xarray 0.10, “Alignment between coordinates on
        # indexed and indexing objects is also now enforced.”.  That means
        # that if ds.coords["calibrated_channel"] == [1, 2, ..., 19], but
        # counts_space["time"].coords["calibrated_channel] == [1], then
        # xarray will raise IndexError.  Therefore, I need to state
        # explicitly that I care only about one channel (even though it
        # doesn't matter for ds_iwct).
        ds_iwct = ds.sel(calibrated_channel=ch, time=counts_space["time"])
        T_iwct = UADA(ds_iwct["temperature_internal_warm_calibration_target"].mean(
                dim="prt_reading").mean(dim="prt_number_iwt")).drop(
                    "scanline")
        T_iwct.attrs["units"] = ds_iwct["temperature_internal_warm_calibration_target"].attrs["units"]
#        T_iwct = ureg.Quantity(ds_iwct["temperature_internal_warm_calibration_target"].mean(
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
            calibcycle_coords = {"calibration_cycle": counts_space["time"].values}
            self._quantities[me.symbols["T_IWCT"]] = self._quantity_to_xarray(
                T_iwct, name=me.names[me.symbols["T_IWCT"]],
                **calibcycle_coords)
            self._quantities[me.symbols["N"]] = self._quantity_to_xarray(
                    numpy.array(ds.dims["prt_number_iwt"], "u1"),
                    name=me.names[me.symbols["N"]])
            self._quantities[me.symbols["M"]] = self._quantity_to_xarray(
                    numpy.array(ds.dims["prt_reading"], "u1"),
                    name=me.names[me.symbols["M"]])
            self._quantities[me.symbols["A"]] = self._quantity_to_xarray(
                    6, name=me.names[me.symbols["A"]])

        # emissivity correction
        # order: see e-mail RQ 2018-04-06
        if self.no_harm or not include_emissivity_correction:
            a_3 = UADA(0,
                name="correction to emissivity")
        else:
            a_3 = UADA(_harm_defs.harmonisation_parameters[self.satname].get(ch, [0,0,0])[1],
                name="correction to emissivity")
        if not -0.98 < a_3 < 0.02:
            warnings.warn(f"Channel {ch:d}: ε + a₃ = {(self.ε+a_3).item():.4f}.  Problem?", FCDRWarning)

        # FIXME: for consistency, should replace this one also with
        # band-corrections — at least temporarily.  Perhaps this wants to
        # be implemented inside the blackbody_radiance method…
        # NB: SRF does not understand DataArrays yet
        # NB: pint seems to silently drop xarray.DataArray information,
        # see https://github.com/hgrecco/pint/issues/479
        # instead use UADA
        B = srf.blackbody_radiance(ureg.Quantity(T_iwct.values, ureg.K))
        L_iwct = (self.ε + a_3).item() * B
        #L_iwct = ureg.Quantity(L_iwct.astype("f4"), L_iwct.u)
        B = UADA(B,
            dims=T_iwct.dims,
            coords={**T_iwct.coords, "calibrated_channel": ch},
            attrs={"units": str(B.u)})
        L_iwct = UADA(L_iwct,
            dims=T_iwct.dims,
            coords={**T_iwct.coords, "calibrated_channel": ch},
            attrs={"units": str(L_iwct.u)})

        extra = []
        # this implementation is slightly more sophisticated than in
        # self.estimate_noise although there is some code duplication.
        # Here, we only use real calibration lines, where both space and
        # earth views were successful.
        counts_space_adev, Ndata = \
            typhon.math.stats.adev(counts_space, "scanpos",outN=True)
        
        #
        # Divide by SQRT of number of elements on the scanline as for space 
        # (and IWCT etc.) the data are averaged
        #
        # Account for missing data
        # Get names of other dimenstions to loop round
        # Note make sure missing data are dealt with correctly via Ndata
        u_counts_space = (counts_space_adev /
            numpy.sqrt(Ndata.values))
        if tuck:
            self._tuck_effect_channel("C_space", u_counts_space, ch)

        counts_iwct_adev, Ndata = \
            typhon.math.stats.adev(counts_iwct, "scanpos",outN=True)
        u_counts_iwct = (counts_iwct_adev / numpy.sqrt(Ndata.values))
        if tuck:
            # before 'tucking', I want to make sure the time coordinate
            # corresponds to the calibration cycle, i.e. the same as for
            # space.  This is ultimately used by passing a dictionary to
            # a lambda generated by sympy.lambdify so we need to reassign
            # coordinates in advance
            self._tuck_effect_channel("C_IWCT", 
                u_counts_iwct.assign_coords(time=u_counts_space["time"]),
                ch)
            # NB: for the estimate on space and IWCT counts, it is
            # correct to divide by sqrt(N), because we use N repeated
            # measurements for the variable estimate; but in case of Earth
            # views, this is NOT correct because we use only a single
            # measurement.  See #125.
            self._tuck_effect_channel("C_Earth",
                UADA((counts_space_adev.variable + counts_iwct_adev.variable)/2,
                      coords=u_counts_space.coords,
                      attrs=u_counts_space.attrs), ch)

        if return_u:
            extra.extend([u_counts_iwct, u_counts_space])
        if return_ix:
            #extra.append(space_followed_by_iwct.values.nonzero()[0])
            extra.append(xarray.DataArray(
                numpy.arange(ds["time"].size),
                dims=["time"],
                coords={"time": ds["time"]}).sel(time=counts_space["time"].values))

        if tuck:
            #coords = {"calibration_cycle": M_space["time"]}
            self._tuck_quantity_channel("C_s", counts_space,
                calibrated_channel=ch, **calibcycle_coords)
            self._tuck_quantity_channel("C_IWCT", counts_iwct,
                calibrated_channel=ch, **calibcycle_coords)
            self._tuck_quantity_channel("R_IWCT", L_iwct,
                calibrated_channel=ch, **calibcycle_coords)
            # store 'N', C_PRT[n], d_PRT[n, k], O_TPRT, O_TIWCT…
            self._tuck_quantity_channel("B",
                B.assign_coords(time=counts_space["time"].values),
                calibrated_channel=ch)
        return (UADA(counts_space["time"]),
                UADA(L_iwct),
                UADA(counts_iwct),
                UADA(counts_space)) + tuple(extra)
        #return (M_space["time"], L_iwct, counts_iwct, counts_space) + tuple(extra)

    def _quantity_to_xarray(self, quantity, name, dropdims=(), dims=None,
            **coords):
        """Convert quantity to UnitsAwareDataArray with correct attributes

        Given a quantity representing the values for any 
        parameter in the measurement equation, convert this to a
        `typhon.physics.units.tools.UnitsAwareDataArray` (itself a
        subclass of `xarray.DataArray`) and make sure the attributes,
        encoding, and name are all set according to the format
        specification for the debug FCDR.  In particular, the units will
        be set such that the resulting `UnitsAwareDataArray` can be used
        in calculations.

        Quantity can be masked and with unit, which will be converted.
        Can also pass either dropdims (dims subtracted from ones defined
        for the quantity) or dims (hard list of dims to include).

        Parameters
        ----------

        quantity : array_like
            Quantity that shall be converted to `UnitsAwareDataArray`.  It
            must have the dimensions that are expected according to the
            debug format specifications.
        name : str
            Name under which this quantity is stored in the debug format
            specification in `_fcdr_defs`.
        dropdims : tuple, optional
            Dimensions to be removed from the `DataArray`.  Defaults to
            ``()``.
        dims : tuple, optional
            Names of dimensions, can be used to override the ones defined
            in `_data_vars_props_`.
        **coords : Mapping
            Coordinates to assign to the resulting `DataArray`. 

        Returns
        -------

        da : `UnitsAwareDataArray`
            A new `UnitsAwareDataArray` following the specifications of
            the debug FCDR.
        """
        
        if isinstance(quantity, UADA):
            da = quantity.rename(
                dict(
                    zip(quantity.dims,
                        [d for d in self._data_vars_props[name][1]
                            if d not in dropdims])))
            da.attrs.update(self._data_vars_props[name][2])
            da.encoding.update(self._data_vars_props[name][3])
        else:
            da = UADA(
                numpy.asarray(quantity,
                    dtype=("f4" if hasattr(quantity, "mask") else
                        quantity.dtype if hasattr(quantity, "dtype") else
                        type(quantity))), # masking only for floats
                dims=dims if dims is not None else [d for d in self._data_vars_props[name][1] if d not in dropdims],
                attrs=self._data_vars_props[name][2],
                encoding=self._data_vars_props[name][3])
        if not da.name:
            da.name = self._data_vars_props[name][0]
        # also drop dimensions when they are now dimensionless coordinates
        for d in dropdims:
            if d in da.coords:
                da = da.drop(d)
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
        return da.assign_coords(**coords)

    def _tuck_quantity_channel(self, symbol_name, quantity, 
            concat_coords=(), **coords):
        """Convert quantity to xarray and put into self._quantities

        Convert a quantity to the correct `UnitsAwareDataArray` format
        using `HIRSFCDR._quantity_to_xarray`, then cache it either in
        ``self._quantities`` if it is in the measurement equation, or in
        ``self._other_quantities`` if it is not.

        .. todo::

            Need to assign time coordinates so that I can later
            extrapolate calibration_cycle dimension to scanline dimension.
        
        Parameters
        ----------

        symbol_name : str
            Name under which the symbol is stored in
            `measurement_equation.symbols`.
        quantity : array_like
            Quantity to be converted and stored.
        concat_coords : tuple, optional
            I don't remember exactly what this is for, but it is certainly
            used.  Defaults to ``()``.
        **coords : Mapping
            Remaining coordinates.

        Returns
        -------

        Returns quantity as stored.
        """

        s = me.symbols.get(symbol_name)
        name = me.names.get(s, symbol_name) # FIXME: fails for Indexed?
        q = self._quantity_to_xarray(quantity, name,
                dropdims=["channel", "calibrated_channel"],
                **coords)
        if s is None: 
            s = name
            dest = self._other_quantities
        else:
            dest = self._quantities
        if s in dest:
            da = dest[s]
            in_coords = [x for x in ("channel", "calibrated_channel")
                            if x in da.coords]
            if len(in_coords) != 1:
                raise ValueError("{:s} does not contain exactly one "
                                 "channel coordinate, found {:d}.".format(
                                    symbol_name, len(in_coords)))
            # FIXME: need to check if we're tucking the same channel twice
            da = xarray.concat([da, q], dim=in_coords[0],
                coords=[in_coords[0]]+list(concat_coords))
            # NB: https://github.com/pydata/xarray/issues/1297
            da.encoding = q.encoding
            dest[s] = da
            return da
        else:
            dest[s] = q
            return q

    def _tuck_effect_channel(self, name, quantity, channel,
            covariances=None):
        """Convert uncertainty quantity to xarray and put into self._effects

        Convert an uncertainty to the correct format for internal storage,
        then cache it in the ``_effects`` attribute.

        Parameters
        ----------

        name : str
            Name of parameter in `measurement_equation`.
        quantity : array_like
        `   Magnitude of uncertainty.
        channel : int
            Channel for which this uncertainty applies.  This still needs
            to be passed even if the uncertainty is not channel dependent.
        covariances : Mapping[Effect, array_like], optional
            If applicable, covariances with other effects.  Currently,
            these are only used for covariances between fundamental
            calibration parameters / harmonisation parameters.
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

        if covariances is None:
            covariances = {}
        eff = self._effects_by_name[name]
        if isinstance(quantity, xarray.DataArray):
            q = quantity.rename(
                dict(zip(quantity.dims,
                eff.dimensions)))
        else:
            q = self._quantity_to_xarray(quantity, name,
                dropdims=["channel", "calibrated_channel"],
                dims=eff.dimensions)
        if q.name is None:
            q.name = f"u_{name:s}"
        q = q.assign_coords(calibrated_channel=channel)
        if eff.magnitude is None:
            eff.magnitude = q
        else:
            da = eff.magnitude
            # check if we're tucking it for the same channel twice...
            if channel in da.calibrated_channel.values:
                if da.calibrated_channel.size == 1:
                    if numpy.array_equal(da.values, q.values):
                        return # nothing to do
                    else:
                        raise ValueError("Inconsistent values for same channel!")
                else:
                    raise NotImplementedError("TBD")

            # make sure this fails if the other dimension coordinates do
            # not match
            da = xarray.concat([da, q], dim="calibrated_channel",
                compat="identical")
            # NB: https://github.com/pydata/xarray/issues/1297
            da.encoding = q.encoding
            eff.magnitude = da
        for (other, val) in covariances.items():
            eff.set_covariance(self._effects_by_name[other], channel, val)

    def calculate_offset_and_slope(self, ds, ch, srf=None, tuck=False,
            naive=False,
            include_emissivity_correction=True, 
            include_nonlinearity=True,
            accept_nan_for_nan=True):
        """Calculate offset and slope.

        From a segment of L1B data, calculate the offset and the slope for
        a particular channel, for all calibration lines within this
        segment.

        To get offset and slope for all scanlines, pass the result into
        `HIRSFCDR.interpolate_between_calibs`.

        Parameters
        ----------

        ds : xarray.Dataset

            xarray dataset with fields such as returned by
            `as_xarray_dataset`.  Must
            contain at least variables ``time``, ``scantype``, ``counts``,
            and ``temperature_internal_warm_calibration_target``.

        ch : int

            Channel that the SRF relates to.

        srf : :class:`typhon.physics.units.SRF`, optional

            SRF used to estimate slope.  Needs to implement the
            :meth:`typhon.physics.units.SRF.blackbody_radiance` method
            such as :class:`typhon.physics.units.SRF`
            does.  Optional: if not provided, use standard one.

        tuck : bool, optional

            Cache any calculated values where appropriate.  This is needed
            if subsequently calculating uncertainties, which need the same
            measurement equation parameters as calculating the
            measurements themselves.  Defaults to False.

        naive : bool, optional

            Naive FCDR calculation: no self-emission or harmonisation, and
            reducing checks.  Defaults to False.

        include_emissivity_correction : bool, optional

            Include the harmonisation parameter correcting the emissivity.
            This should normally be True, but the user may set this to
            False for debugging purposes.  Defaults to True.

        include_nonlinearity : bool, optional

            Include the harmonisation parameter for nonlinearity.
            This should normally be True, but the user may set this to
            False for debugging purposes.  Defaults to True.

        accept_nan_for_nan : bool, optional

            Accept nans in offset as long as same nans occur in slope and
            vice versa.  Defaults to False, which means a ValueError will
            be raised.

        Returns
        -------

        time : ndarray

            corresponding to offset and slope

        offset : ndarrayo

            offset calculated at each calibration cycle

        slope : ndarray
        
            slope calculated at each calibration cycle
        """

        srf = srf or self.srfs[ch-1]
        
        # 2017-02-22 backward compatibility
        if not isinstance(ds, xarray.Dataset):
            warnings.warn("Passing ndarray to calculate_offset_and_slope "
                "is deprecated since 2017-02-22, should pass "
                "xarray.Dataset ", DeprecationWarning)
            ds = self.as_xarray_dataset(ds)

        # nonlinearity
        # order: see e-mail RQ 2018-04-06 and 2018-10-11
        if self.no_harm or not include_nonlinearity:
            a2 = UADA(0, 
                name="a2", coords={"calibrated_channel": ch},
                attrs = {"units": str(rad_u["si"]/(ureg.count**2))})
        else:
            a2 = UADA(_harm_defs.harmonisation_parameters[self.satname].get(ch, [0,0,0])[2],
                name="a2", coords={"calibrated_channel": ch},
                attrs = {"units": str(rad_u["si"]/(ureg.count**2))})

        (time, L_iwct, counts_iwct,
            counts_space) = self.extract_calibcounts_and_temp(
                ds, ch, srf, tuck=tuck,
                include_emissivity_correction=include_emissivity_correction)
        #L_space = ureg.Quantity(numpy.zeros_like(L_iwct), L_iwct.u)
        L_space = UADA(xarray.zeros_like(L_iwct),
            coords={k:v 
                for (k, v) in counts_space.isel(scanpos=0).coords.items()
                if k in L_iwct.coords})

        ΔL = UADA(L_iwct.variable - L_space.variable,
                  coords=L_space.coords,
                  attrs=L_space.attrs)
        Δcounts = UADA(
            counts_iwct.variable - counts_space.variable,
            coords=counts_space.coords, name="Δcounts",
            attrs=counts_space.attrs)
        ΔLΛ2 = UADA(
            counts_iwct.variable**2 - counts_space.variable**2,
            coords=counts_space.coords, name="Δcounts",
            attrs=counts_space.attrs)
        ΔLΛ2.attrs["units"] = ureg.count**2
        slope = (ΔL - a2*ΔLΛ2)/Δcounts

        offset = -counts_space**2 * a2 -slope * counts_space

        # in some (rare) cases, counts_space and counts_iwct are all zero.
        # This will cause slope to be inf, and slope * counts_space to be
        # nan.  There should be no other possible way for getting nans in
        # offset.
        # what if we have already filtered counts_iwct or counts_space
        # to set outliers to nan, and the result is nan simply because
        # of that?  In this case we want to accept nans where counts are
        # nan too, hence the flag accept_nan_for_nan
        offsetnan = numpy.isnan(offset)
        countsnan = numpy.isnan(counts_space.values) | numpy.isnan(counts_iwct.values)
        counts0both = (counts_space.values == 0) & (counts_iwct.values == 0)
        nansok = counts0both
        if accept_nan_for_nan:
            nansok |= countsnan
        if not naive and not numpy.array_equal(offsetnan, nansok):
            raise ValueError("Problematic data propagating unexpectedly. "
                "I can except offset nans to correspond to cases where "
                "counts_space == counts_iwct == 0, such as "
                "NOAA-12 1997-05-31T16:02:42.528000, or when "
                "you have approved counts_space or counts_iwct to be "
                "nan (such as due to pre-filtering), but there "
                "appears to be something else going on here. "
                "I cannot proceed like this, please investigate what's "
                "going on and handle it properly.")
        elif not naive and numpy.isnan(offset).any():
            logger.warn("Found cases where counts_space == counts_iwct == 0.  "
                "Setting both slope and offset to inf (instead of nan).")
            offset.values[numpy.isnan(offset)] = numpy.inf

        if counts_iwct.coords["time"].size > 0:
            # sometimes IWCT or space counts seem to drift over a “scan line”
            # of calibration.  Identify this by comparing the IQR to the
            # counts.  For truly random normally distributed data:
            # (25, 75) … > 2: false positive 0.2%
            # (10, 90) … > 3.3: false positive 0.5%
            # …based on a simple simulated # experiment.
            bad_iwct = self.calibfilter.filter_calibcounts(counts_iwct)
            bad_space = self.calibfilter.filter_calibcounts(counts_space)

            # filter IWCT and space outliers
            bad_iwct |= self.filter_calibcounts.filter_outliers(
                counts_iwct.median("scanpos").values)
            bad_space |= self.filter_calibcounts.filter_outliers(
                counts_space.median("scanpos").values)

            # filter PRT outliers 
            bad_iwct |= self.filter_prttemps.filter_outliers(
                ds["temperature_internal_warm_calibration_target"].sel(time=counts_iwct["time"]).mean(dim="prt_number_iwt").mean(dim="prt_reading").values)

            bad_calib = xarray.DataArray(bad_iwct.variable | bad_space.variable,
                    coords=bad_space.coords, name="bad_calib")
            # TODO/FIXME: some of this might be salvageable!  Just set to DONOTUSE
#            slope.sel(time=bad_calib)[...] = numpy.nan
#            offset.sel(time=bad_calib)[...] = numpy.nan
        coords = {"calibration_cycle": time.values}
        if tuck:
            self._tuck_quantity_channel("a_0", offset,
                calibrated_channel=ch, **coords)
            self._tuck_quantity_channel("a_1", slope,
                calibrated_channel=ch, **coords)
            self._tuck_quantity_channel("a_2", a2,
                calibrated_channel=ch)
            if self.no_harm or not include_nonlinearity:
                self._tuck_effect_channel("a_2", 0, ch,
                    covariances={"a_3": 0, "a_4": 0})
            else:
                self._tuck_effect_channel("a_2",
                    _harm_defs.harmonisation_parameter_uncertainties[self.satname].get(ch, [0,0,0])[2],
                    ch,
                    covariances={
                        "a_3": _harm_defs.harmonisation_parameter_covariances[self.satname].get(ch, numpy.zeros((3,3)))[2, 1],
                        "a_4": _harm_defs.harmonisation_parameter_covariances[self.satname].get(ch, numpy.zeros((3,3)))[2, 0]})
        return (time,
                offset,
                slope,
                a2)

    _quantities = {}
    _other_quantities = {}
    _effects = None
    _effects_by_name = None
    _flags = {"scanline": {}, "channel": {}}
    def calculate_radiance(self, ds, ch, srf=None,
                context=None,
                Rself_model=None,
                Rrefl_model=None, tuck=False, return_bt=False,
                naive=False):
        """Calculate FIDUCEO HIRS FCDR radiance for channel

        Apply the measurement equation to calculate the calibrated FIDUCEO
        radiance for a particular channel.  To calculate all channels, use
        `calculate_radiance_all`.

        When ``tuck`` is set to True, this method and the methods it calls
        store the values for the following quantities in cache::

            ``Rself``, ``u_Rself``, ``R_selfIWCT``, ``R_selfs``,
            ``R_self_start``, ``R_self_end``, ``C_E``, ``R_E``, ``T_b``,
            ``R_refl``, ``α``, ``Δα``, ``β``, ``Δβ``, ``fstar``,
            ``Δλ_eff``, ``a_3``, ``a_4``

        These cached values are required to calculate uncertainties using
        `HIRSFCDR.calc_u_for_variable`.

        In principle, it is possible to swap out the self-emission model
        by passing a different ``Rself_model``.  In practice, there is
        currently exactly one self-emission model implemented, and the
        interface is designed for this single purpose.  Some recoding
        would be needed to make the interface to the self-emission model
        more generic.

        Parameters
        ----------
            
        ds : `xarray.Dataset`

            xarray Dataset with at least variables 'time', 'scantype',
            'temperature_internal_warm_calibration_target', and
            'counts'.  Such is returned by self.as_xarray_dataset.
            Those are values for which radiances will be calculated.

        ch : int

            Channel to calculate radiance for.  If you want to
            calculate the radiance for all channels, use
            `HIRSFCDR.calculate_radiance_all`.

        srf : :class:`~typhon.physics.units.SRF`, optional

            SRF to use.  If not passed, use default, as obtained from
            RTTOV, which is a shift developed by Paul Menzel compared to
            the SRFs measured before launch.

        context : `xarray.Dataset`, optional

            Like ``ds``, but used for context.  For example, calibration
            information may have to be found outside the range of ``ds``.
            The self-emission model may also need context.

        Rself_model : RSelf, optional

            Model to use for self-emission.  See `models.RSelf` for a
            suitable class from which to create an object to pass here.

        Rrefl_model : RRefl, optional

            Model to use for Earthshine.  See `models.RRefl`.  Not
            currently used.

        tuck : bool, optional

            If true, store/cache intermediate values in
            self._quantities and self._effects.  This is needed for
            `HIRSFCDR.calc_u_for_variable` to work.  Defaults to False.

        return_bt : bool, optional

            If true, return (radiance, bt).  If false, return only
            radiance.

        Returns
        -------

        radiance : `typhon.physics.units.tools.UnitsAwareDataArray`
            Calculated radiances for all scanlines and pixels in ``ds``.
        bt : `typhon.physics.units.tools.UnitsAwareDataArray`
            Only returned if ``return_bt`` is True, calculated brightness
            temperatures.
        """

        if not isinstance(ds, xarray.Dataset):
            warnings.warn("Passing ndarray to calculate_radiance "
                "is deprecated since 2017-02-22, should pass "
                "xarray.Dataset ", DeprecationWarning)
            ds = self.as_xarray_dataset(ds)
        srf = srf or self.srfs[ch-1]
        has_context = context is not None
        context = context if has_context else ds

        # emissivity correction
        # order: see e-mail RQ 2018-04-06 and 2018-10-11
        if naive or self.no_harm:
            a_3 = UADA(0,
                name="correction to emissivity",
                attrs={"units": "dimensionless"})
        else:
            a_3 = UADA(_harm_defs.harmonisation_parameters[self.satname].get(ch, [0,0,0])[1],
                name="correction to emissivity",
                attrs={"units": "dimensionless"})
        # self-emission bias
        # order: see e-mail RQ 2018-04-06 and 2018-10-11
        if naive or self.no_harm:
            a_4 = UADA(0,
                name="harmonisation bias",
                attrs={"units": rad_u["si"]})
        else:
            a_4 = UADA(_harm_defs.harmonisation_parameters[self.satname].get(ch, [0,0,0])[0],
                name="harmonisation bias",
                attrs={"units": rad_u["si"]})

        dsix = self.within_enough_context(ds, context, ch, 1)
        n_within_context = ds.loc[dsix]["time"].size
        if 0 < n_within_context < ds["time"].size and not naive:
            logger.warning("It appears that, despite best efforts, "
                "the context does not sufficiently cover the period "
                "for which the FCDR is to be calculated.  I want to "
                "calculate FCDR for channel {:d} for "
                "{:%Y-%m-%d %H:%M:%S} – {:%Y-%m-%d %H:%M:%S}.  Context is "
                "needed for interpolation of calibration and self-emission "
                "model (among others), but context only available "
                "between {:%Y-%m-%d %H:%M:%S} – {:%Y-%m-%d %H:%M:%S}.  I "
                "will limit the FCDR calculation to the period of "
                "{:%Y-%m-%d %H:%M:%S} – {:%Y-%m-%d %H:%M:%S}. "
                "Remaining data will be MISSING pending #137.  Sorry!".format(
                ch,
                *ds["time"][[0,-1]].values.astype("M8[ms]").astype(datetime.datetime),
                *context["time"][[0,-1]].values.astype("M8[ms]").astype(datetime.datetime),
                *ds.loc[dsix]["time"][[0,-1]].values.astype("M8[ms]").astype(datetime.datetime),
                ))
            ds = ds.loc[dsix]
            # this means I need to fix the flags as well
            for k in self._flags.keys():
                self._flags[k] = self._flags[k].sel(scanline_earth=dsix["time"])

        # some stuff I can do whether I have enough context or not

        views_Earth = xarray.DataArray(ds["scantype"].values == self.typ_Earth, coords=ds["scantype"].coords)
        # NB: C_Earth has counts for all channels but only
        # calibrated_channel will be used
        C_Earth = UADA(ds["counts"].isel(time=views_Earth).sel(
            channel=ch)).rename({"channel": "calibrated_channel"})

        if n_within_context < 2:
            logging.error("Less than two calibration lines in period "
                "{:%Y-%m-%d %H:%M}--{:%Y-%m-%d %H:%M} for channel {:d}!  Data unusable.".format(
                *context["time"][[0,-1]].values.astype("M8[ms]").astype(datetime.datetime),
                ch))
            self._flags["channel"].loc[{"calibrated_channel": ch}] |= (
                _fcdr_defs.FlagsChannel.DO_NOT_USE|
                _fcdr_defs.FlagsChannel.CALIBRATION_IMPOSSIBLE)
            # no point in even trying to calibrate or do self-emission


            # need to set to dummy:
            #
            # Rself, u_Rself, R_selfIWCT, R_selfs, R_self_start, R_self_end,
            # C_E, R_E, T_b, R_refl, α, Δα, β, Δβ, fstar, Δλ_eff, N, M, A
            
            # full Earth count dimensions
            par = functools.partial(UADA,
                numpy.zeros(shape=C_Earth.shape),
                coords=C_Earth.coords)

            rad_wn = par(attrs={"units": str(rad_u["si"])})
            T_b = par(attrs={"units": "K"})
            
            # once per scanline
            par = functools.partial(UADA,
                numpy.zeros(shape=C_Earth["time"].shape),
                coords=C_Earth["time"].coords)

            Rself = par(attrs={"units": str(rad_u["si"])})

            # once per time_rself
            u_Rself = UADA([0], dims="time_rself", attrs={"units": str(rad_u["si"])})

            # once per calibration cycle (a.k.a. never)
            # NB: dtype must be ns, https://github.com/pydata/xarray/issues/1494
            par = functools.partial(UADA,
                numpy.zeros(shape=0),
                dims=("time",),
                coords={"time": numpy.zeros(shape=0, dtype="M8[ns]")})

            B = offset = a_0 = L_iwct = u_Rself = RselfIWCT = Rselfspace = \
                     R_refl =  par(attrs=Rself.attrs)

            slope = a_1 = par(attrs={"units":
                              str(self._data_vars_props["slope"][2]["units"])})

            T_IWCT = par(attrs={"units": "K"})

            u_counts_earth = counts_space = counts_iwct = u_counts_space = u_counts_iwct = par(
                attrs={"units": "counts"})

            if naive or self.no_harm:
                a2 = UADA(0,
                    name="a2", coords={"calibrated_channel": ch},
                    attrs = {"units": str(rad_u["si"]/(ureg.count**2))})
            else:
                a2 = UADA(_harm_defs.harmonisation_parameters[self.satname].get(ch, [0,0,0])[2],
                    name="a2", coords={"calibrated_channel": ch},
                    attrs = {"units": str(rad_u["si"]/(ureg.count**2))})

            Rself_start = Rself_end = xarray.DataArray(
                [numpy.datetime64(0, 's')], dims=["time_rself"])
            has_Rself = False

            coords = {"calibration_cycle": numpy.array([], dtype="M8[ns]")}

            # those are normally done in calculate_offset_and_slope and
            # extract_calibcounts_and_temp
            self._tuck_effect_channel("C_space", u_counts_space, ch)
            self._tuck_effect_channel("C_IWCT", u_counts_iwct, ch)
            self._tuck_effect_channel("C_Earth", u_counts_earth, ch)
            calibcycle_coords = {"calibration_cycle": counts_space["time"].values}
            self._tuck_quantity_channel("C_s", counts_space,
                calibrated_channel=ch, **calibcycle_coords)
            self._tuck_quantity_channel("C_IWCT", counts_iwct,
                calibrated_channel=ch, **calibcycle_coords)
            self._tuck_quantity_channel("R_IWCT", L_iwct,
                calibrated_channel=ch, **calibcycle_coords)
            self._tuck_quantity_channel("a_0", offset,
                calibrated_channel=ch, **calibcycle_coords)
            self._tuck_quantity_channel("a_1", slope,
                calibrated_channel=ch, **calibcycle_coords)
            self._tuck_quantity_channel("a_2", a2,
                calibrated_channel=ch)
            self._tuck_quantity_channel("B", B,
                calibrated_channel=ch)
            self._quantities[me.symbols["T_IWCT"]] = self._quantity_to_xarray(
                T_IWCT, name=me.names[me.symbols["T_IWCT"]],
                **calibcycle_coords)
            self._quantities[me.symbols["N"]] = self._quantity_to_xarray(
                    numpy.array(ds.dims["prt_number_iwt"], "u1"),
                    name=me.names[me.symbols["N"]])

#            interp_slope = UADA(xarray.zeros_like(
#                ds["counts"].sel(scanpos=1, channel=ch)),
#                attrs={"units": str(self._data_vars_props["slope"][2]["units"])})*numpy.nan
#            interp_offset = UADA(xarray.zeros_like(
#                ds["counts"].sel(scanpos=1, channel=ch)),
#                attrs={"units": str(self._data_vars_props["offset"][2]["units"])})*numpy.nan
#            interp_slope_modes = {"zero": interp_slope, "linear": interp_slope, "cubic": interp_slope}
#            interp_offset_modes = {"zero": interp_offset, "linear": interp_offset, "cubic": interp_offset}
        else:
            # this means I have at least two calibration cycles with data
            # in-between, so in theory I can do everything I want to.
            
        
            # NB: passing `context` here means that quantities and effects are
            # stored with dimension for entire context period, rather than
            # just the ds period.  Should take this into account with later
            # processing.
            (time, offset, slope, a2) = self.calculate_offset_and_slope(
                context, ch, srf, tuck=tuck, naive=naive)
            
            if not numpy.array_equal(numpy.isfinite(offset),
                                     numpy.isfinite(slope)):
                raise ValueError("Expecting offset and slope to have same "
                    "finite values, but I got disappointed.")

            # median treats nan and inf differently, so where I have inf
            # in slope but not in offset it goes wrongly.  In any case,
            # for the purpose of the median, I really want to ignore infs
            # so I'd rather set them to nans
            for x in (slope, offset):
                x.values[~numpy.isfinite(x.values)] = numpy.nan

            if Rself_model is None:
                warnings.warn("No self-emission defined, assuming 0!",
                    FCDRWarning)
                has_Rself = False
            else:
                try:
                    Rself_model.fit(context, ch)
                except ValueError as e:
                    if (e.args[0].startswith("Space views fail normality") or
                        e.args[0].startswith("All space views in fitting") or
                        e.args[0].startswith("Some offsets are infinite")):
                        errmsg = (
                            "Unable to train self-emission model for channel {:d} with "
                            "training data in {:%Y-%m-%d %H:%M}--{:%Y-%m-%d %H:%M}: "
                            "{:s} ".format(
                                ch,
                                context["time"].values[0].astype("M8[s]").astype(datetime.datetime),
                                context["time"].values[-1].astype("M8[s]").astype(datetime.datetime),
                                e.args[0]))
                         # This should not happen anymore now that I have the
                         # context check at the top
    #                    if offset.shape[0] == 1:
    #                        raise FCDRError(errmsg + " Moreover, I have only a "
    #                            "single valid calibration cycle in the period "
    #                            "{:%Y-%m-%d %H:%M}--{:%Y-%m-%d %H:%M}, which "
    #                            "means I cannot fall back to interpolation "
    #                            "either!  Sorry, I give up!")
                        logger.error(errmsg +
                            "Will flag data and reduce to basic interpolation!")
    #                    if not ch in self._flags["channel"]:
    #                        self._flags["channel"][ch] = _fcdr_defs.FlagsChannel.SELF_EMISSION_FAILS
    #                    else:
                        self._flags["channel"].loc[{"calibrated_channel": ch}] |= _fcdr_defs.FlagsChannel.SELF_EMISSION_FAILS
                        self._flags["channel"].loc[{"calibrated_channel": ch}] |= _fcdr_defs.FlagsChannel.DO_NOT_USE
                        has_Rself = False
                    else:
                        raise
                else:
                    has_Rself = True

            # NOTE: taking the median may not be an optimal solution.  See,
            # for example, plots produced by the script
            # plot_hirs_calibcounts_per_scanpos in the FCDR_HIRS package
            # within FIDUCEO, in particular for noaa18 channels 1--12, where
            # the lowest scan positions are systematically offset compared to
            # the higher ones.  See also the note at
            # calculate_offset_and_slope. 
            interp_offset_modes = {}
            interp_slope_modes = {}
            interp_bad_modes = {}
            if offset.shape[0] > 1 or ((naive or has_context) and time.shape[0]>0):
                for mode in ("zero",) if naive else ("zero", "linear", "cubic"):
                    moff = offset.median(dim="scanpos", keep_attrs=True)
                    mslp = slope.median(dim="scanpos", keep_attrs=True)
                    bad = (
                        self.filter_calibcounts.filter_outliers(moff.values) |
                        self.filter_calibcounts.filter_outliers(mslp.values))
                    (interp_offset, interp_slope, interp_bad) = self.interpolate_between_calibs(
                        ds["time"], time,
                        moff, mslp, bad,
                        kind=mode)
                    naive
                    interp_offset_modes[mode] = interp_offset
                    interp_slope_modes[mode] = interp_slope
                    interp_bad_modes[mode] = interp_bad
            else:
                raise RuntimeError("This should never happen again")
                # check version history to see what was here before
            if not naive:
                # in naive mode, interp_bad_modes["zero"] will contain
                # nans, then .astype(numpy.bool_) will turn it into an
                # object array, either way boolean indexing will fail
                # but in naive mode we don't need flags so let's just skip
                # it.
                # NB: this may also happen if there's insufficient
                # context?
                self._flags["channel"].sel(
                    calibrated_channel=ch)[
                    {"scanline_earth":
                     xarray.DataArray(
                        interp_bad_modes["zero"].astype(numpy.bool_),
                        dims=("time",),
                        coords={
                            "time":
                             ds.coords["time"]}
                    ).sel(time=C_Earth.coords["time"]).values
                    }] |= (
                    _fcdr_defs.FlagsChannel.DO_NOT_USE |
                    _fcdr_defs.FlagsChannel.CALIBRATION_IMPOSSIBLE)
            # might be attractive to do "linear" here, but no: that would
            # complicate the uncertainty propagation, for which zero-order
            # interpolation occurs in _make_dims_consistent.
            # And in naive mode I certainly want to have 'zero'.
            interp_slope = interp_slope_modes["zero"]
            # may be overwritten but this is so I carry out the right check;
            # should not be set to cubic here...
            interp_offset = interp_offset_modes["zero"]
            if not naive and (not numpy.isfinite(interp_offset).all() or
                              not numpy.isfinite(interp_slope).all()):
                if not numpy.array_equal(numpy.isfinite(interp_offset),
                                         numpy.isfinite(interp_slope)):
                    raise ValueError("There's nans in slope or offset but "
                        "they're not the same, this is a bug.")
                if ((offset.size>0 and numpy.isfinite(offset).all()) or
                    (slope.size>0 and numpy.isfinite(slope).all())):
                    raise ValueError("There's nans in slope or offset when "
                        "interpolated but not in original, this is a bug.")
                logger.error("Looks like some or all slopes/offsets are "
                     "impossible to calculate "
                    f"for channel {ch:d}.  That's not good.  Do not touch.")
                self._flags["channel"].loc[{"calibrated_channel": ch}]  |= (
                    _fcdr_defs.FlagsChannel.DO_NOT_USE|
                    _fcdr_defs.FlagsChannel.CALIBRATION_IMPOSSIBLE)
            # to be used if I have none, but also for debugging otherwise
            Rself_0 = UADA(numpy.zeros(shape=C_Earth["time"].shape),
                         coords=C_Earth["time"].coords,
                         name="Rself", attrs={"units":   
                str(rad_u["si"])})

            Rself_0_start = Rself_0_end = xarray.DataArray(
                [numpy.datetime64(0, 's')], dims=["time_rself"])


            if has_Rself:
                if Rself_model.name == "Gerrit_Self_Emission":
                    # we do have a working self-emission model, probably
                    # This is Gerrit's temperature dependent self 
                    # emission model
                    interp_offset = interp_offset_modes["zero"]
                    (Xt, Y_reft, Y_predt) = Rself_model.test(context, ch)
                    (X, Y_pred) = Rself_model.evaluate(ds, ch)
                    # Y_pred is rather the offset than the self-emission
                    Rself = interp_offset.isel(time=views_Earth) - Y_pred#.sel(time=views_Earth["time"].isel(time=views_Earth))
                #Rself = (interp_offset - Y_pred).isel(time=views_Earth)
                    Rself.attrs["note"] = ("Implemented as ΔRself in pre-β. "
                                           "RselfIWCT = Rselfspace = 0.")
                    Rself.attrs["model_info"] = str(Rself_model).replace("\n", " ")
                    RselfIWCT = Rselfspace = UADA(numpy.zeros(shape=offset["time"].shape),
                                                  coords=offset["time"].coords,
                                                  attrs={**Rself.attrs,
                                                           "note": "Rself is implemented as ΔRself in pre-β",
                                                           })
                    Rself_start = xarray.DataArray(
                        [Rself_model.fit_time[0]],
                        dims=["time_rself"])
                    Rself_end = xarray.DataArray(
                        [Rself_model.fit_time[1]],
                        dims=["time_rself"])
                    Rself = Rself.assign_coords(
                        Rself_start=xarray.DataArray(
                            numpy.tile(Rself_start, Rself.shape[0]),
                            dims=("time",),
                            coords={"time": Rself.time}),
                        Rself_end=xarray.DataArray(
                            numpy.tile(Rself_end, Rself.shape[0]),
                            dims=("time",),
                            coords={"time": Rself.time}))
                    u_Rself = numpy.sqrt(((Y_reft - Y_predt)**2).mean(
                            keep_attrs=True, dim="time"))
                    
                    T_outliers = functools.reduce(
                        operator.or_,
                        [self.filter_prttemps.filter_outliers(
                                ds[k].values.mean(-1).mean(-1)
                                if ds[k].values.ndim==3
                                else ds[k].values)
                         for k in X.data_vars.keys()])
                    
                    T_outliers = xarray.DataArray(
                        T_outliers,
                        dims=("time",),
                        coords={"time": ds["time"]})
                    
                    self._flags["scanline"][{"scanline_earth":
                                                 T_outliers.sel(time=C_Earth["time"]).values}] |= (
                        _fcdr_defs.FlagsScanline.DO_NOT_USE |
                        _fcdr_defs.FlagsScanline.BAD_TEMP_NO_RSELF
                        )
                elif Rself_model.name == "Linear_Interpolation_by_time":
                    # JMittaz
                    # Note that we have to give interplated slope and offset
                    # to self emission model as this is what the measurement
                    # equation uses. This is not actually a correct model
                    # of the instrument and self emission and seems 
                    # inconsistent with the available FIDUCEO HIRS documentation
                    # One other difference - u_Rself is now calculated 
                    # scanline by scanline when within calibration cycles
                    # (not necessarily constant). Note sure how this works
                    # with code below
                    (rself_time, X, Rself, uRself) = \
                        Rself_model.evaluate(ds, ch)

                    u_Rself = UADA(numpy.array([uRself]),\
                                       dims=["time_rself"],\
                                       attrs={**Rself.attrs})
                    Rself.attrs["note"] = ("Implemented as ΔRself in pre-β. "
                                           "RselfIWCT = Rselfspace = 0.")
                    Rself.attrs["model_info"] = str(Rself_model).replace("\n", " ")
                    RselfIWCT = Rselfspace = UADA(numpy.zeros(shape=offset["time"].shape),
                                                  coords=offset["time"].coords,
                                                  attrs={**Rself.attrs,
                                                           "note": "Rself is implemented as ΔRself in pre-β",
                                                           })
                    Rself_start = xarray.DataArray(
                        [rself_time[0]],
                        dims=["time_rself"])
                    Rself_end = xarray.DataArray(
                        [rself_time[1]],
                        dims=["time_rself"])
                    Rself = Rself.assign_coords(
                        Rself_start=xarray.DataArray(
                            numpy.tile(Rself_start, Rself.shape[0]),
                            dims=("time",),
                            coords={"time": Rself.time}),
                        Rself_end=xarray.DataArray(
                            numpy.tile(Rself_end, Rself.shape[0]),
                            dims=("time",),
                            coords={"time": Rself.time}))
                    #
                    # Bad uRself when some/all of Ts are bad
                    # denote these are bad Ts 
                    #
                    T_outliers = functools.reduce(
                        operator.or_,
                        [self.filter_prttemps.filter_outliers(
                                ds[k].values.mean(-1).mean(-1)
                                if ds[k].values.ndim==3
                                else ds[k].values)
                         for k in X.data_vars.keys()])
                    
                    T_outliers = xarray.DataArray(
                        T_outliers,
                        dims=("time",),
                        coords={"time": ds["time"]})
                    
                    self._flags["scanline"][{"scanline_earth":
                                                 T_outliers.sel(time=C_Earth["time"]).values}] |= (
                        _fcdr_defs.FlagsScanline.DO_NOT_USE |
                        _fcdr_defs.FlagsScanline.BAD_TEMP_NO_RSELF
                        )
            else:
                # we don't have a working self-emission model.  Take
                # self-emission as an interpolation between adjecent
                # calibration cycles, and include an uncertainty corresponding
                # to the differ18ence between linear and zero-order
                # interpolation.
                interp_offset = interp_offset_modes["zero" if naive else "cubic"]
                Rself = Rself_0
                RselfIWCT = Rselfspace = UADA(numpy.zeros(shape=offset["time"].shape),
                        coords=offset["time"].coords, attrs=Rself.attrs)
    #            u_Rself = UADA([0], dims=["rself_update_time"],
    #                           coords={"rself_update_time": [ds["time"].values[0]]})
                # Although I could potentially omit the RMSE here and just
                # take the difference between the modes, that yields one value
                # for each Earth view.  Under normal circumstances, the
                # self-emission model gives me one value for each calibration
                # cycle.  If some channels work but others don't, I'll have
                # different time coordinates for different channels, which
                # means I can't concatenate them into a single DataArray.
                # Since this is an adhoc bad format anyway I don't care that
                # I'm losing information.
                u_Rself = UADA(
                    [abs(interp_offset_modes["zero"]).mean()
                        if naive
                        else numpy.sqrt((abs(interp_offset_modes["linear"] - interp_offset_modes["zero"])**2).mean())],
                    dims=["time_rself"],
                    attrs={"units": offset.attrs["units"]})

    #            # make sure dimensions etc. are the same as when we do have a
    #            # working model
    #            try:
    #                u_Rself = u_Rself.rename(dict(channel="calibrated_channel", time="time_rself")).drop(("scanpos", "scanline", "lon", "lat"))
    #            except ValueError:
    #                pass # not sure why this would happen

            # trick to make sure there is a time dimension…
            # http://stackoverflow.com/a/42999562/974555
            u_Rself = xarray.concat((u_Rself,), dim="time_rself")

            # want to do this differently… see #128
            # and, for now, establish fake time based u_Rself to ensure
            # there's always u_Rself info in every orbit file even though
            # it's updated more rarely than that.  Take a range that is
            # too large (context), another lie, so we can later
            # interpolate to the right segment (lie will not propagate)
            # interpolation will happen in _make_dims_consistent
            # JMittaz: Note sure how this works with linear interpolation
            # model which potentially has variable u_Rself
            # Gerrit seems to have hardwired an assumption that the 
            # uncertainty is a single number into susequent code so giving
            # a variable number doesn't work. Too difficult to correct
            # at the moment (lots of iterables etc.) so fix uncertainty
            # to mean value of uRself for the moment
            # Don't know if this will ever be fixed given the difficulty 
            # of Gerrit's code...
            times = pandas.date_range(
                context["time"].values[0],
                context["time"].values[-1],
                freq='10min')
            # This repeats a single number
            u_Rself = u_Rself[numpy.tile([0], times.size)]
            u_Rself = u_Rself.assign_coords(
                time_rself=times.values)# [ds["time"].values[0]])
            u_Rself.attrs["U_RSELF_WARNING"] = (
                "Self-emission uncertainty repeated "
                "every ten minutes as a stop-gap measure to ensure even "
                "short slices contain this info, this does not imply "
                "an actual update of the information.  Check coordinates "
                "Rself_start and Rself_end.  See #128."
                "For Linear time interpolated model time variable uncertainty "
                "is implemented but not possible with this version of the "
                "code (JMittaz comment)")
            
            # according to Wang, Cao, and Ciren (2007), ε=0.98, no further
            # source or justification given
            if Rrefl_model is None:
                warnings.warn("No Earthshine model defined, assuming 0!",
                    FCDRWarning)
                R_refl = UADA(numpy.zeros(shape=offset["time"].shape),
                        coords=offset["time"].coords,
                        name="R_refl",
                        attrs={"units": str(rad_u["si"])})
            else:
                raise NotImplementedError("Evalutation of Earthshine "
                    "model not implemented yet")
            rad_wn = self.custom_calibrate(C_Earth, interp_slope,
                interp_offset, a2, Rself, a_4)

            if not has_Rself: # need to set manually
                newcoor = dict(
                    Rself_start=xarray.DataArray(
                        numpy.tile(Rself_0_start, Rself.shape[0]),
                        dims=("time",),
                        coords={"time": Rself.time}),
                    Rself_end=xarray.DataArray(
                        numpy.tile(Rself_0_end, Rself.shape[0]),
                        dims=("time",),
                        coords={"time": Rself.time}))
                Rself = Rself_0.assign_coords(**newcoor)

            # for debugging purposes, calculate various other radiances
            a2_0 = UADA(0,
                    name="a2", coords={"calibrated_channel": ch},
                    attrs = {"units": str(rad_u["si"]/(ureg.count**2))})

            # Rself_0 already set, but not with right coords
            Rself_0 = Rself_0.assign_coords(**Rself.coords)
            a4_0 = UADA(0,
                    name="harmonisation bias",
                    attrs={"units": rad_u["si"]})


            rad_wn_dbg = {}

            for (skipa2, skiprself, skipa4, skipa3) in itertools.product((0,1),repeat=4):
                lab = ("linear"*skipa2 +
                       "norself"*skiprself+
                       "nooffset"*skipa4+
                       "noεcorr"*skipa3)
                if not lab:
                    continue

                (time, offset_dbg, slope_dbg,
                    a2_dbg) = self.calculate_offset_and_slope(
                        context, ch, srf, tuck=False,
                        include_nonlinearity=not skipa2,
                        include_emissivity_correction=not skipa3)
                (interp_offset_dbg, interp_slope_dbg,
                    interp_bad_dbg) = self.interpolate_between_calibs(
                        ds["time"], time,
                        offset_dbg.median(dim="scanpos", keep_attrs=True),
                        slope_dbg.median(dim="scanpos", keep_attrs=True),
                        bad, kind="zero")

                rad_wn_dbg[lab] = self.custom_calibrate(
                    C_Earth,
                    interp_slope_dbg,
                    interp_offset_dbg,
                    a2_dbg,
                    Rself_0 if skiprself else Rself,
                    a4_0 if skipa4 else a_4)

            bad = self.filter_earthcounts.filter_outliers(C_Earth.values)
            # I need to compare to space counts.
            C_space = self._quantities[me.symbols["C_s"]]
            # 2017-12-16, for the first channel there is a coordinate but
            # not a dimension for the channel.  Should be properly fixed
            # up where it's tucked away but I tried two fixed that both
            # had side-effects I couldn't quite understand right away, so
            # workaround here instead. -- 2017-12-16, GH
            if "calibrated_channel" in C_space.dims:
                C_space = C_space.sel(calibrated_channel=ch)
            bad |= self.filter_coldearth.filter(
                C_Earth,
                self.interpolate_between_calibs(
                    C_Earth["time"],
                    time,
                    C_space.median("calibration_position").values,
                    kind="zero")[0])
            if has_Rself: # otherwise I have no use of T and no T_ouliers
                bad |= T_outliers.isel(time=views_Earth).values[:, numpy.newaxis]
            # NB, need to test if this takes care of most remaining
            # bad outliers!  May need an entire rerun of the archive for
            # that...
            self._flags["pixel"].loc[{"calibrated_channel": ch}].values[bad.values] |= (
                _fcdr_defs.FlagsPixel.DO_NOT_USE|_fcdr_defs.FlagsPixel.OUTLIER_NOS)
            # 
            coords = {"calibration_cycle": time.values}

            T_b = rad_wn.to("K", "radiance", srf=srf)
        # end if n_within_context == 0.  From here only code that can be
        # executed whether I have good data or not.

        if not has_Rself: # need to set manually
            newcoor = dict(
                Rself_start=xarray.DataArray(
                    numpy.tile(Rself_0_start, Rself.shape[0]),
                    dims=("time",),
                    coords={"time": Rself.time}),
                Rself_end=xarray.DataArray(
                    numpy.tile(Rself_0_end, Rself.shape[0]),
                    dims=("time",),
                    coords={"time": Rself.time}))
            Rself = Rself_0.assign_coords(**newcoor)
            rad_wn = rad_wn.assign_coords(**newcoor)
            T_b = T_b.assign_coords(**newcoor)


        (α, β, λ_eff, Δα, Δβ, Δλ_eff) = srf.estimate_band_coefficients(
            self.satname, self.section, ch, include_shift=False)
#        (α, β, f_eff, Δα, Δβ, Δf_eff) = (numpy.float32(0),
#            numpy.float32(1), srf.centroid().to(ureg.THz, "sp"), 0, 0, 0)

        if tuck:
            # I want to keep the channel dimension on the time coordinate
            # for the 'last update' self-emission.  Sometimes some
            # channels have a more recently updated self-emission model
            # than others (imagine if one channel or set of channels is
            # not functioning for a while, we wouldn't want to abandon
            # updating all channels)
            self._tuck_quantity_channel("R_selfE", Rself, 
                calibrated_channel=ch,
                concat_coords=["Rself_start", "Rself_end"])
            self._tuck_effect_channel("Rself", u_Rself, ch)
            self._tuck_quantity_channel("R_selfIWCT", RselfIWCT, 
                calibrated_channel=ch, **coords)
            self._tuck_quantity_channel("R_selfs", Rselfspace,
                calibrated_channel=ch, **coords)
#            self._tuck_quantity_channel("R_self_start", Rself_start)
#            self._tuck_quantity_channel("R_self_end", Rself_end)
            self._tuck_quantity_channel("C_E", C_Earth, 
                calibrated_channel=ch, scanline_earth=C_Earth["time"].values)
    #        self._tuck_quantity_channel("R_e", rad_wn[views_Earth, :], ch)
            # keep result but copy over only encoding, because
            # _tuck_quantity_channel also renames dimensions and I don't want
            # that yet
            R_e = self._tuck_quantity_channel("R_e", rad_wn,
                calibrated_channel=ch,
                concat_coords=["Rself_start", "Rself_end"])
            rad_wn.encoding = R_e.encoding
            if return_bt:
                self._tuck_quantity_channel("T_b",
                    T_b,
                    calibrated_channel=ch,
                    concat_coords=["Rself_start", "Rself_end"])
            self._tuck_quantity_channel("R_refl", R_refl,
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
                self.ε, name=me.names[me.symbols["ε"]])
            self._tuck_quantity_channel("a_3", a_3, calibrated_channel=ch)
            self._tuck_quantity_channel("a_4", a_4, calibrated_channel=ch)
            if naive or self.no_harm:
                self._tuck_effect_channel("a_3", 0, ch,
                    covariances={"a_2": 0, "a_4": 0})
                self._tuck_effect_channel("a_4", 0, ch,
                    covariances={"a_2": 0, "a_3": 0})
            else:
                self._tuck_effect_channel("a_3",
                    _harm_defs.harmonisation_parameter_uncertainties[self.satname].get(ch, [0,0,0])[1],
                    ch,
                    covariances={
                        "a_2": _harm_defs.harmonisation_parameter_covariances[self.satname].get(ch, numpy.zeros((3,3)))[1, 2],
                        "a_4": _harm_defs.harmonisation_parameter_covariances[self.satname].get(ch, numpy.zeros((3,3)))[1, 0]})
                self._tuck_effect_channel("a_4",
                    _harm_defs.harmonisation_parameter_uncertainties[self.satname].get(ch, [0,0,0])[0],
                    ch,
                    covariances={
                        "a_2": _harm_defs.harmonisation_parameter_covariances[self.satname].get(ch, numpy.zeros((3,3)))[0, 2],
                        "a_3": _harm_defs.harmonisation_parameter_covariances[self.satname].get(ch, numpy.zeros((3,3)))[0, 1]})

            # the zero-terms
            for s in (s for s in me.symbols.keys() if s.startswith("O_")):
                self._tuck_quantity_channel(s, 0, calibrated_channel=ch)

            # debug radiances
            for (k, v) in rad_wn_dbg.items():
                self._tuck_quantity_channel(
                    f"rad_wn_{k:s}", v, calibrated_channel=ch,
                    concat_coords=["Rself_start", "Rself_end"])
        rad_wn = rad_wn.rename({"time": "scanline_earth"})


#
        # prevent http://bugs.python.org/issue29672
        # Tb0 = (T_b.variable == 0)
        Tb0 = xarray.DataArray((T_b.values==0), dims=T_b.dims, coords=T_b.coords)
        # skip this check for SW channels when R_e is often really so small
        # that we can't define a meaningful T_b
        seix = Tb0.any("scanpos").values
        if ch<13 and not (
            (self._flags["pixel"].sel(calibrated_channel=ch).any("scanpos").isel(scanline_earth=seix)) |
            (self._flags["scanline"].isel(scanline_earth=seix)) |
            (self._flags["channel"].sel(calibrated_channel=ch).isel(scanline_earth=seix))).all():
            idx0 = {"scanline_earth": seix}
            flag_p = self._flags["pixel"].sel(calibrated_channel=ch).any("scanpos")[idx0]
            flag_s = self._flags["scanline"][idx0]
            flag_c = self._flags["channel"].sel(calibrated_channel=ch)[idx0]
            flag_any = (flag_c.values!=0)|(flag_p.values!=0)|(flag_s.values!=0) 
            logger.warning(
                "Despite best efforts, after filtering outliers and "
                "accounting for flagged data due to various problems and all "
                "that, I still find {:d} scanlines for channel {:d} "
                "where my T_b estimate is 0 but unflagged.  Perhaps outlier detection is "
                "failing.  See #146.".format(int((~flag_any).sum()), ch))
            # bad Earth:
#            C_Earth.isel(time=Tb0.any("scanpos")).isel(time=~flag_any)
#            raise ValueError("I ended up with unflagged T_b==0!")
#

        if return_bt:
            return (rad_wn, T_b)
        else:
            return rad_wn
    Mtorad = calculate_radiance

    def calculate_radiance_all(self, ds, srf=None,
                context=None,
                Rself_model=None,
                Rrefl_model=None,
                return_ndarray=False,
                naive=False):
        """Calculate radiances for all channels

        Parameters
        ----------
            
        ds : `xarray.Dataset`

            xarray Dataset with at least variables 'time', 'scantype',
            'temperature_internal_warm_calibration_target', and
            'counts'.  Such is returned by self.as_xarray_dataset.
            Those are values for which radiances will be calculated.

        context : `xarray.Dataset`, optional

            Like ``ds``, but used for context.  For example, calibration
            information may have to be found outside the range of ``ds``.
            The self-emission model may also need context.

        Rself_model : RSelf, optional

            Model to use for self-emission.  See `models.RSelf` for a
            suitable class from which to create an object to pass here.

        Rrefl_model : RRefl, optional

            Model to use for Earthshine.  See `models.RRefl`.  Not
            currently used.

        return_ndarray : bool, optional

            If true, return an `ndarray`.  Otherwise, return an
            `xarray.DataArray`.

        naive : bool, optional

            Defaults to False

        Returns
        -------

        ndarray or `xarray.DataArray`
            Radiances for all channels.
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
        self._other_quantities.clear()
        self._reset_flags(ds)
#        self._flags["scanline"].clear()
#        self._flags["channel"].clear()

        # the two following dictionary should and do point to the same
        # effect objects!  I want both because when I'm evaluating the
        # effects it's easier to get them by name, but when I'm
        # substituting them into the uncertainty-expression it's easier to
        # get them by symbol.
        self._effects = effects.effects()
        self._effects_by_name = {e.name: e for e in
                itertools.chain.from_iterable(self._effects.values())}

        (all_rad, all_bt) = zip(*[self.calculate_radiance(ds, ch, 
                context=context, Rself_model=Rself_model,
                Rrefl_model=Rrefl_model, tuck=True, return_bt=True,
                naive=naive)
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

    def calculate_bt_all(self, M, D, H=None, fn=None): 
        """Naive BT for all channels --- DO NOT USE!

        Perform a **naive** calculation of BT for all channels.  This is
        also known as an incorrect calculation.  You should not use this
        method.

        This method has the signature of a :class:`typhon.datasets.Dataset`
        ``pseudo_field`` and is not intended to be used directly.

        Parameters
        ----------

        M : not used
        D : Mapping
            contains earlier calculated pseudofields, must contain
            ``radiance_fid_naive``
        H : not used, optional
        fn : not used, optional
            
        Returns
        -------

        masked_array
        """
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

    def get_L_cached_meq(self):
        """Calculate radiance from cached quantities using measurement equation

        After we have calculated L using regular calculate_radiance, we
        can make another estimate using the measurement equation, mostly
        to check consistency.  The regular L calculation is not currently
        directly through the measurement equation as defined in the
        measurement_equation module, but is a rather directly in code
        implementation of the same.

        Parameters
        ----------

        none

        Returns
        -------

        `typhon.physics.units.tools.UnitsAwareDataArray`
            Calculated radiance for all channels.
        """

        L_meq = []
        for e in (me.expressions[me.symbols["R_e"]],
                  me.expression_Re_simplified):
            fargs = typhon.physics.metrology.recursive_args(
                e, stop_at=(sympy.Symbol, sympy.Indexed))
            ta = tuple(fargs)
            fe = sympy.lambdify(ta, e, numpy, dummify=True)
            adict = {k:v for (k,v) in self._quantities.items() if k in fargs}
            adict = self._make_adict_dims_consistent_if_needed(adict, me.symbols["R_e"])
            L_meq.append(fe(*[typhon.math.common.promote_maximally(adict[x]).to_root_units()
                              for x in ta]))
        return (self._quantity_to_xarray(
                    L_meq[0].to(rad_u["si"]),
                    "R_e_alt_meq_full"),
                self._quantity_to_xarray(
                    L_meq[1].to(rad_u["si"]),
                    "R_e_alt_meq_simple"))

    def estimate_noise(self, M, ch, typ="both"):
        """Calculate noise level at each calibration line. **DO NOT USE**

        Old implementation to return noise level for IWCT and space
        views.  **DO NOT USE**

        Use `HIRSFCDR.extract_calibcounts_and_temp` with ``tuck=True``,
        then obtain uncertainty in Earth counts in
        ``self._effects_by_name["C_Earth"].magnitude``.

        Warning: this does not ensure that only space views followed by
        IWCT views are taken into account.  If you need such an assurance,
        use ``HIRSFCDR.extract_calibcounts_and_temp`` instead.

        Parameters
        ----------

        M : structured ndarray
        ch : int
        typ : str

        Returns
        -------

        structured ndarray
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
        """**DO NOT USE** Recalibrate counts to radiances with uncertainties

        **DO NOT USE**

        This implementation is old, incomplete, and incorrect.

        Parameters
        ----------

        M : ndarray

            Structured array such as returned by self.read.  Should
            have at least fields "hrs_scntyp", "counts", "time", and
            "temp_iwt".

        ch : int

            Channel to calibrate.

        srf : :class:`typhon.physics.units.SRF`

            SRF to use for calibrating the channel and converting
            radiances to units of BT.  Optional; if None, use
            “default" SRF for channel.

        Returns
        -------

        radiance : ndarray
        bt : ndarray
        """
        warnings.warn("Deprecated, use calculate_radiance", DeprecationWarning)
        srf = self.srfs[ch-1]
        if realisations is None:
            realisations = self.realisations
        logger.info("Estimating noise")
        (t_noise_level, noise_level) = self.estimate_noise(M, ch)
        # note, this can't be vectorised easily anyway because of the SRF
        # integration bit
        logger.info("Calibrating")
        (time, offset, slope, a2) = self.calculate_offset_and_slope(M, ch, srf)
        # NOTE:
        # See https://github.com/numpy/numpy/issues/7787 on numpy.median
        # losing the unit
        logger.info("Interpolating") 
        (interp_offset, interp_slope) = self.interpolate_between_calibs(M["time"],
            time, 
            ureg.Quantity(numpy.median(offset, 1), offset.u),
            ureg.Quantity(numpy.median(slope, 1), slope.u))
        interp_noise_level = numpy.interp(M["time"].view("u8"),
                    t_noise_level.view("u8")[~noise_level.mask],
                    noise_level[~noise_level.mask])
        logger.info("Allocating")
        rad_wn = ureg.Quantity(numpy.empty(
            shape=M["counts"].shape[:2] + (realisations,),
            dtype="f4"), rad_u["ir"])
        bt = ureg.Quantity(numpy.empty_like(rad_wn), ureg.K)
        logger.info("Estimating {:d} realisations for "
            "{:,} radiances".format(realisations,
               rad_wn.size))
        bar = progressbar.ProgressBar(maxval=realisations,
                widgets = common.my_pb_widget)
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
        logger.info("Done")

        return (rad_wn, bt)

    def read_and_recalibrate_period(self, start_date, end_date):
        """**DO NOT USE**"""
        M = self.read(start_date, end_date,
                fields=["time", "counts", "bt", "calcof_sorted"])
        return self.recalibrate(M)

    def extract_and_interp_calibcounts_and_temp(self, M, ch, srf=None):
        """**DO NOT USE**"""
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
        """Estimate self-emission and associated uncertainty **DO NOT USE**

        **DO NOT USE**

        I think this was intended as a standalone evaluation of
        self-emission but this is rather built into calculate_radiance

        **DO NOT USE**

        Parameters
        ----------
            
        ds_core : xarray.Dataset

            Data for which to estimate self-emission.  Should be an
            xarray Dataset covering the period for which the
            self-emission shall be evaluated.

        ds_context : xarray.Dataset

            Context data.  Should be an xarray Dataset containing a
            longer period than ds_core, will be used to estimate the
            self-emission model parameters and uncertainty.  Must
            contain for calibration lines (space and IWCT views)
            including temperatures and space/IWCT counts.

        Returns
        -------

        nothing, will raise an exception before it gets anywhere
        """
        
        if self.rself_model is None:
            self.rself_model = models.RSelf(self)

        raise NotImplementedError("Not implemented yet")


    def calc_u_for_variable(self, var, quantities, all_effects,
                            cached_uncertainties, return_more=False):
        """Calculate total uncertainty for variable

        Calculate the total uncertainty for a particular variable /
        element of the measurement equation.  For example, to get the
        total uncertainty in Earth radiance, use
        ``fcdr.calc_u_for_variable("R_e", ...)``.

        This method requires that the magnitude of the components of the
        measurement equation has already been calculated.  To get those,
        make sure you calculate the Earth radiance first, using
        `HIRSFCDR.calculate_radiance`, while passing ``tuck=True``.  This
        will store the required quantities in ``self._quantities``.

        Parameters
        ----------

        var : str or :class:`sympy.core.symbol.Symbol`

            Variable for which to calculate uncertainty

        quantities : Mapping

            Dictionary with numerical values for quantities.  This can be
            obtained from ``self._quantities`` if Earth radiance has
            already been calculated using `calculate_radiance` while
            setting ``tuck=True``.

        all_effects : Mapping

            Dictionary with sets of effects (:class:`effects.Effect` objects)
            with magnitudes filled in.  This can be obtained from
            ``self._effects`` if Earth radiance has already been
            calculated with ``tuck=True``.

        cached_uncertainties : Mapping

            Dictionary with cached uncertainties for quantities that
            for which we do not directly estimate uncertainties, but
            that are expressions of other quantities including
            uncertainties and effects (i.e. ``R_IWCT`` uncertainty results
            from uncertainties in ``T_IWCT``, ``ε``, ``φ``, etc.).  Note that this
            dictionary will be changed by this function!  I recommend to
            define an empty dictionary and pass that in.  The same object
            will be passed on recursively as `calc_u_for_variable` calls
            itself.

        return_more : bool, optional

            Optional.  If true, also return a xarray.Dataset with all
            uncertainty components.

        Returns
        -------

        u : `typhon.physics.units.tools.UnitsAwareDataArray`
            Uncertainties for ``var``
        sub_sensitivities : `measurement_equation.ExpressionDict`, if ``return_more`` is True
            Nested dictionary containing recursively the values for the
            sensitivity coefficients, with a form of::

                Dict[Symbol,
                     Tuple[ndarray,
                           Dict[Symbol,
                                Tuple[ndarray,
                                      Dict[Symbol, Tuple[...]]]]]

            where each level of the dictionary corresponds to a level
            within the measurement equation, the keys correspond to
            symbols occurring within the measurement equation.  The values
            are 2-tuples, where the first element is the magnitude of the
            sensitivity coefficient for this parameter, and the second
            element is a - possibly empty - dictionary which describes the
            evaluated sensitivities for the expression describing that
            parameter; the dictionary is empty if there is none.  At each
            level, the sensivitity coefficients are relative to the level
            immediately above.

            This nested dictionary is filled recursively within this
            method, and used, among other places, in
            `metrology.accum_sens_coef`, which calculates the _total_
            sensitivity coefficient all the way from the Earth radiance to
            arbitrarily deep within the measurement equation.  It is also
            used by `HIRSFCDR.propagate_uncertainty_components`.
        sub_components : `measurement_equation.ExpressionDict`, if ``return_more`` is True
            Nested dictionary containing recursively the uncertainty for
            each variable broken down in the parts due to each component
            of the measurement equation, with a form similar to
            ``sub_sensitivities``::

                Dict[Symbol,
                     Tuple[ndarray,
                           Dict[Symbol,
                                Tuple[ndarray,
                                      Dict[Symbol, Tuple[...]]]]]]

            where each level of the dictionary corresponds again to a
            level within the measurement equation, the symbols to symbols
            within the measurement equation.  Each tuple has as its first
            element the magnitude of the uncertainty of the parent
            quantity due to the child quantity, and as a second element
            its own uncertainty broken down in its components.

            This does not include covariant components.

            You need to pass this mapping if you call
            `HIRSFCDR.propagate_uncertainty_components`.
        cov_comps : `measurement_equation.ExpressionDict`, if ``return_more`` is True
            Mapping that describes the covariant components.

        Example
        -------

        To get uncertainties::

            cu = {}
            (uRe, sensRe, compRe, covcomps) = fcdr.calc_u_for_variable(
                "R_e", fcdr._quantities, fcdr.effects, cu,
                return_more=True)

        For a more detailed example, study the source code of
        :meth:`FCDR_HIRS.processing.generate_fcdr.FCDRGenerator.get_piece`.
            
        """

        # Traversing down the uncertainty expression for the measurement
        # equation.  Example expression for u²(R_e):
        #
#    4  2          2  2                      2  2         2          2             2    
# C_E ⋅u (a₂) + C_E ⋅u (a₁) + (2⋅C_E⋅a₂ + a₁) ⋅u (C_E) + u (O_Re) + u (R_selfE) + u (a₀)
        #
        # Here:
        #
        #   1. C_E, a₂ are values that gets directly substituted, #
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
        sbase = s.args[0].args[0] if isinstance(s, sympy.Indexed) else s

        if s not in me.expressions:
            # If there is no expression for this value, the uncertainty is
            # simply what should have already been calculated
            #all_effects = effects.effects()


            if s in all_effects:
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
                    u = UADA(0, name="u_{!s}".format(sbase),
                        attrs={
                            "quantity": str(s),
                            "note": "No uncertainty quantified for: {:s}".format(
                                ';'.join(eff.name for eff in baddies)),
                            "units": self._data_vars_props[
                                        me.names[sbase]][2]["units"],
                            "encoding": self._data_vars_props[
                                        me.names[sbase]][3]})
                cached_uncertainties[s] = u
                return (u, {}, {}, {}) if return_more else u
            else:
                u = UADA(0, name="u_{!s}".format(sbase),
                    attrs={
                        "quantity": str(s),
                        "note": "No documented effect associated with this "
                                "quantity",
                        "units": self._data_vars_props[
                                    me.names[sbase]][2]["units"],
                        "encoding": self._data_vars_props[
                                    me.names[sbase]][3]})
                cached_uncertainties[s] = u
                return (u, {}, {}, {}) if return_more else u

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
        
        # any summation argument must be concrete and explicit for uncertainty
        # propagation to work correctly
        for sumarg in {foo.args[1][2] for foo in typhon.physics.metrology.recursive_args(e,
                       stop_at=sympy.concrete.expr_with_limits.ExprWithLimits)}:
            if isinstance(sumarg, sympy.Symbol):
                sumelems = {sumarg}
            else:
                sumelems = typhon.physics.metrology.recursive_args(sumarg,
                    stop_at=sympy.Symbol)
            for sumel in sumelems:
                e = e.subs(sumel, quantities[sumel].values.item()).doit()
        failures = set()
        (u_e, sensitivities, components) = typhon.physics.metrology.express_uncertainty(
            e, on_failure="warn", collect_failures=failures,
            return_sensitivities=True,
            return_components=True,
            correlated_terms=(
                {me.symbols[f"a_{i:d}"] for i in {2,3,4}}
                if me.symbols["a_2"] in e.free_symbols
                else set())
            )

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
            return (u, {}, {}, {}) if return_more else u

        fu = sympy.Function("u")
        args = typhon.physics.metrology.recursive_args(u_e,
            stop_at=(sympy.Symbol, sympy.Indexed, fu))

        # Before I proceed, I want to check for zero arguments; this
        # might mean that I need to evaluate less.  Hence two runs through
        # args: first to check the zeroes, then to see what's left.
        for v in sorted(args, key=str):
            if isinstance(v, fu) and len(v.args)==1:
                # comparing .values to avoid entering
                # xarray.core.nputils.array_eq which has a catch_warnings
                # context manager destroying the context registry, see
                # http://bugs.python.org/issue29672
                if (v.args[0] in cached_uncertainties and
                        numpy.all(cached_uncertainties[v.args[0]].values == 0)):
                    u_e = u_e.subs(v, 0)
                    del sensitivities[v.args[0]]
                    del components[v.args[0]]
                elif ((v.args[0] not in me.expressions or
                       isinstance(me.expressions[v.args[0]], sympy.Number)) and
                      v.args[0] not in all_effects):
                    # FIXME/BUG: this elif gets wrongly entered when
                    # v==u(T_PRT[0]) and I have an expression for
                    # T_PRT[n].  Gets replaced by 0 when it should not!
                    #
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
        sub_sensitivities = me.ExpressionDict()
        sub_components = me.ExpressionDict()
        cov_comps = me.ExpressionDict()
        sortedargs = sorted(args, key=str)
        # move covariances to end, so that I have the sensitivities by the
        # time I reach them
        sortedargscopy = sortedargs.copy()
        for v in sortedargs.copy():
            if isinstance(v, fu) and len(v.args)==2:
                idx = sortedargs.index(v)
                sortedargs.append(sortedargs.pop(idx))

        for v in sortedargs:
            # check which one of the four aforementioned applies
            if isinstance(v, fu) and len(v.args)==1:
                # this covers both cases (2) and (3); if there is no
                # expression for the uncertainty, it will be read from the
                # effects tables (see above)
                if v.args[0] in cached_uncertainties:
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
                    (adict[v], subsens, subcomp, subcov) = self.calc_u_for_variable(
                        v.args[0], quantities, all_effects,
                        cached_uncertainties, return_more=True)
                    if len(subcov) > 0:
                        raise ValueError("I expected covariances to only "
                            "occur at the top-level.  Something's wrong.")
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
                                assert ddto[kk][1] is not None, \
                                       "Should not be None now :("
                    cached_uncertainties[v.args[0]] = adict[v]
                # NB: We should have sensitivities.keys() ==
                # components.keys() == our current loop
                sub_sensitivities[v.args[0]] = (
                    sensitivities[v.args[0]], subsens)
                sub_components[v.args[0]] = (
                    components[v.args[0]], subcomp)
            elif isinstance(v, fu) and len(v.args) == 2:
                effs = all_effects[v.args[0]]
                if len(effs) != 1:
                    raise NotImplementedError("Covariances with "
                        "variables with more or less than 1 effect "
                        "not implemented.")
                eff = effs.copy().pop()
                adict[v] = eff.covariances[v.args[1]]
                cov_comps[v.args] = (sub_sensitivities[v.args[0]][0] *
                                     sub_sensitivities[v.args[1]][0],
                                     eff.covariances[v.args[1]])
            elif isinstance(v, fu):
                raise ValueError(
                    f"uncertainty function with {len(v.args):d} arguments?!")
            else:
                # it's a quantity
                if v not in quantities:
                    if v not in me.expressions:
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
                    
        hope = True
        if any([v.size==0 for (k, v) in adict.items()]):
            logger.error("FATAL! One or more components have size zero. "
                "I cannot propagate uncertainties!")
            adict = {k: UADA(0, dims=(), coords={}, attrs={"units": v.units}) for (k, v) in adict.items()}
            self._flags["scanline"] |= (_fcdr_defs.FlagsScanline.DO_NOT_USE|_fcdr_defs.FlagsScanline.UNCERTAINTY_SUSPICIOUS)
            hope = False

        # now I have adict with values for uncertainties and other
        # quantities, that I need to substitute into the expression
        # I expect I'll have to do some trick to substitute u(x)? no?
        ta = tuple(args)
        # dummify=True because some args are not valid identifiers
        f = sympy.lambdify(ta, u_e, numpy, dummify=True)
        adict = self._make_adict_dims_consistent_if_needed(adict, sbase)
        # verify/convert dimensions
        u = f(*[typhon.math.common.promote_maximally(
                    adict[x]).to_root_units() for x in ta])
        u = u.to(self._data_vars_props[me.names[sbase]][2]["units"])
        u = u.rename("u_"+me.names[sbase])
        cached_uncertainties[s] = u
        if return_more:
            var_unit = self._data_vars_props[me.names[sbase]][2]["units"]
            # turn expressions into data for the dictionairies
            # sub_sensitivities and sub_components
            for k in sub_sensitivities:
                # I already verified that sub_sensitivities and
                # sub_components have the same keys
                # NB 2018-10-10: should u_e here be dd[k][0]?
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
            for (pair, (sens_pair, cov_pair)) in cov_comps.items():
                args = typhon.physics.metrology.recursive_args(sens_pair,
                    stop_at=(sympy.Symbol, sympy.Indexed, fu))
                ta = tuple(args)
                f = sympy.lambdify(ta, sens_pair, numpy, dummify=True)
                cov_comps[pair] = (f(
                        *[typhon.math.common.promote_maximally(adict[x]).to_root_units()
                            for x in ta]),
                        cov_pair)
                
            # make units nicer.  This may prevent loss of precision
            # problems when values become impractically large or small
            for (k, v) in sub_sensitivities.items():
                kbase = k.args[0].args[0] if isinstance(k, sympy.Indexed) else k
                if isinstance(sub_sensitivities[k][0], numbers.Number):
                    # sensitivity is a scalar / constant
                    k_unit = self._data_vars_props[me.names[kbase]][2]["units"]
                    if not ureg.Unit(k_unit) == ureg.Unit(var_unit):
                        raise ValueError("∂{s!s}/∂{k!s} = {sens:g} should be "
                            "dimensionless, but {s!s} is in "
                            "{var_unit!s} and {k!s} is in "
                            "{k_unit!s}".format(sens=sub_sensitivities[k][0],
                                **vars()))
                    # nothing else to do
                else: # must be xarray.DataArray
                    sub_sensitivities[k] = (
                        sub_sensitivities[k][0].to(
                            ureg.Unit(self._data_vars_props[me.names[sbase]][2]["units"])/
                            ureg.Unit(self._data_vars_props[me.names[k]][2]["units"])),
                        sub_sensitivities[k][1])
            for (k, v) in sub_components.items():
                sub_components[k] = (
                    sub_components[k][0].to(
                        self._data_vars_props[me.names[sbase]][2]["units"]),
                   sub_components[k][1])
            # FIXME: perhaps I need to add prefixes such that the
            # magnitude becomes close to 1?

            return (u, sub_sensitivities, sub_components, cov_comps)
        else:
            return u
    
    def _make_adict_dims_consistent_if_needed(self, adict, var):
        """Helper function for calc_u_for_variable

        Internal helper function for `HIRSFCDR.calc_u_for_variable`.
        Check whether making dimensions consistent is needed, and make it
        so if it is.  This derives from the fact that some values, such as
        space counts, are only occurring once per calibration cycle,
        whereas others are once per scanline, so we do a zero-order spline
        interpolation to allow elementwise processing.

        The components of adict are the contents to calculate var.  Make
        sure dimensions are consistent, through interpolation, averaging,
        etc.  Requires that the desired dimensions of var occur in at
        least one of the values for adict so that coordinates can be read
        from it.
        
        See `make_debug_fcdr_dims_consistent` for details.

        Parameters
        ----------

        adict : Mapping[Symbol, UnitsAwareDataArray]
            adict is the dictionary of everything (uncertainties and
            quantities) that needs to be substituted to evaluate the
            magnitude of the uncertainty
        var : `Symbol`
            Variable for which it needs to be made consistent.

        Returns
        -------

        adict : Mapping
            New adict where all values have the same dimensions.
        """

        # multiple dimensions with time coordinates:
        # - calibration_cycle
        # - scanline_earth
        # Any dimension other than calibration_cycle needs to be
        # interpolated to have dimension scanline_earth before further
        # processing.
        src_dims = set().union(itertools.chain.from_iterable(
            x.dims for x in adict.values() if hasattr(x, 'dims')))
        dest_dims = set(self._data_vars_props[me.names[var]][1])
        if not dest_dims <= src_dims: # problem!
            warnings.warn("Cannot correctly estimate uncertainty u({!s}). "
                "Destination has dimensions {!s}, arguments (between them) "
                "have {!s}!".format(var, dest_dims, src_dims or "none"),
                FCDRWarning)
        if not src_dims <= dest_dims: # needs reducing
            adict = self._make_adict_dims_consistent(adict)
        return adict

    def _make_adict_dims_consistent(self, adict):
        """Ensure adict dims are consistent with earth counts dimensions

        Internal helper function for `HIRSFCDR.calc_u_for_variable`.
        Check whether making dimensions consistent is needed, and make it
        so if it is.  This derives from the fact that some values, such as
        space counts, are only occurring once per calibration cycle,
        whereas others are once per scanline, so we do a zero-order spline
        interpolation to allow elementwise processing.

        The components of adict are the contents to calculate var.  Make
        sure dimensions are consistent, through interpolation, averaging,
        etc.  Requires that the desired dimensions of var occur in at
        least one of the values for adict so that coordinates can be read
        from it.

        Currently hardcoded for:
            
            - calibration_cycle → interpolate → scanline_earth
            - calibration_position → average → ()

        Parameters
        ----------

        adict : Mapping[Symbol, UnitsAwareDataArray]
            adict is the dictionary of everything (uncertainties and
            quantities) that needs to be substituted to evaluate the
            magnitude of the uncertainty

        Returns
        -------

        adict : Mapping
            New adict where all values have the same dimensions.        
        """
        new_adict = {}

        for (k, v) in adict.items():
            new_adict[k] = self._make_dims_consistent(
                adict[me.symbols["C_E"]],
                v)

        return new_adict

    def _make_dims_consistent(self, dest, src):
        """Helper to make dims consistent for single dest/src

        See `make_adict_dims_consistent` and
        `make_debug_fcdr_dims_consistent`.

        Parameters
        ----------

        dest : `typhon.physics.units.tools.UnitsAwareDataArray`
            Target data array
        src : `typhon.physics.units.tools.UnitsAwareDataArray`
            Source data array

        Returns
        -------

        newsrc : `typhon.physics.units.tools.UnitsAwareDataArray`
            Source data array, if needed expanded to have same dimensions
            as dest data array.

        """
        return make_debug_fcdr_dims_consistent(
            dest, src, impossible="warn", flags=self._flags)

    def numerically_propagate_DeltaL(self, L, ΔL):
        """Temporary method to numerically propagate ΔL to ΔTb

        Until I find a proper solution for the exploding Tb uncertainties
        (see https://github.com/FIDUCEO/FCDR_HIRS/issues/78 ) approximate
        these numerically.

        Uses the standard SRFs as stored in ``self.srfs``.

        Parameters
        ----------

        L : xarray.DataArray
            Radiances for all channels
        ΔL : xarray.DataArray
            Radiance uncertainties for all channels

        Returns
        -------

        ΔTb : xarray.DataArray
            Brightness temperature uncertainties for all channels
        """
        ΔTb = xarray.zeros_like(L).drop(("scanline", "lat", "lon"))
        ΔTb.encoding = _fcdr_defs._debug_bt_coding
        ΔTb.attrs["units"] = "K"
        for ch in range(1, 20):
            srf = self.srfs[ch-1]
            Lch = L.sel(calibrated_channel=ch)
            ΔLch = ΔL.sel(calibrated_channel=ch)
            low = (Lch-ΔLch).to("K", "radiance", srf=srf)
            high = (Lch+ΔLch).to("K", "radiance", srf=srf)
            ΔTb.loc[{"calibrated_channel": ch}] = (high-low)/2
        return ΔTb

    def estimate_channel_correlation_matrix(self, ds_context, calpos=20,
            type="spearman"):
        """Estimate channel correlation matrix

        Calculates correlation coefficients between space view anomalies
        accross channels.  

        As per :issue:`87`.  

        Parameters
        ----------

        ds_context : xarray.Dataset
            xarray.Dataset containing the context over which the channel
            correlation matrix is to be calculated.
        calpos : int, optional
            Calibration position to use for the correlation calculations.
            Defaults to 20.
        type : str, optional
            What type of correlation to use.  Can be "pearson" or
            "spearman".  Defaults to "spearman".

        Returns
        -------

        da : xarray.DataArray
            Channel correlation matrix.
        """
        Cs = ds_context["counts"].isel(
            time=ds_context["scantype"].values == self.typ_space,
            scanpos=slice(8, None))
        bad = self.filter_calibcounts.filter_outliers(Cs.values)
        ok = ~bad[:, calpos, :].any(1)

        ΔCs = (Cs - Cs.mean("scanpos"))
        if ok.sum() < 3:
            logger.warning("Not enough valid values, filling channel "
                "correlation matrix with NaN")
            S = numpy.full((ds_context.dims["channel"],)*2, numpy.nan)
        elif type == "pearson":
            S = numpy.corrcoef(ΔCs.isel(time=ok).sel(scanpos=calpos).T)
        elif type == "spearman":
            S = scipy.stats.spearmanr(ΔCs.isel(time=ok).sel(scanpos=calpos))[0]
        else:
            raise ValueError(f"Unknown type: {type:s}")
        da = xarray.DataArray(S,
            coords={"channel": ds_context.coords["channel"]},
            dims=("channel", "channel"))
        da.name = "channel_correlation_matrix"
        da.attrs = self._data_vars_props[da.name][2]
        da.encoding = self._data_vars_props[da.name][3]
        da.attrs["note"] = "covers only crosstalk effect"
        if da.min().item() < -1 or da.max().item() > 1:
            raise ValueError("Found correlations out of range!")
        return da

    def get_BT_to_L_LUT(self):
        """Returns LUT to translate BT to LUT

        Create a lookup table for the translation between brightness
        temperatures and radiances, for channels 1–19.

        Parameters
        ----------

        none

        Returns
        -------

        lookup_table_BT : xarray.DataArray
            Brightness temperatures for lookup table
        lookup_table_radiance
            Radiances for lookup table
        """

        n = 100
        lookup_table_BT = xarray.DataArray(
            numpy.tile(numpy.linspace(200, 300, 101)[:, numpy.newaxis],
                       19),
            coords={"calibrated_channel": range(1, 20)},
            dims=("lut_size", "calibrated_channel"),
            name="lookup_table_BT")
        lookup_table_radiance = xarray.DataArray(
            numpy.zeros(shape=(101, 19), dtype="f4"),
            coords=lookup_table_BT.coords,
            dims=lookup_table_BT.dims,
            name="lookup_table_radiance")
        for ch in lookup_table_BT.calibrated_channel.values:
            srf = self.srfs[ch-1]
            lookup_table_radiance.loc[{"calibrated_channel": ch}] = (
                srf.blackbody_radiance(lookup_table_BT.sel(calibrated_channel=ch)).to(
                    rad_u["ir"], "radiance"))
        return (lookup_table_BT, lookup_table_radiance)

    def _reset_flags(self, ds):
        """Reset cached flags for scanline, channel, and minor frame.

        Should be called at the beginning of each new set of radiance
        calculations.

        Parameters
        ----------

        ds : xarray.Dataset
            Dataset from which new radiances will be calculated.
        """

        views_Earth = xarray.DataArray(ds["scantype"].values == self.typ_Earth, coords=ds["scantype"].coords)

        flags_scanline = xarray.DataArray(
            numpy.zeros(
                shape=int(views_Earth.sum()),
                dtype=self._data_vars_props["quality_scanline_bitmask"][3]["dtype"]),
            dims=("scanline_earth",),
            coords={"scanline_earth": ds["time"][(ds["scantype"].values == self.typ_Earth)].values}, 
            name="quality_scanline_bitmask",
            attrs=self._data_vars_props["quality_scanline_bitmask"][2]
        )

        flags_channel = xarray.DataArray(
            numpy.zeros(
                shape=(flags_scanline["scanline_earth"].size,  ds["calibrated_channel"].size),
                dtype=self._data_vars_props["quality_channel_bitmask"][3]["dtype"]),
            dims=("scanline_earth", "calibrated_channel"),
            coords={"scanline_earth": flags_scanline.coords["scanline_earth"],
                    "calibrated_channel": ds.coords["calibrated_channel"]},
            name="quality_channel_bitmask",
            attrs=self._data_vars_props["quality_channel_bitmask"][2]
        )

        # need to semi-hardcode number 64 here: HIRS/2 does not have any
        # thing with dimension minor_frame, and indeed the minor_frame
        # related flags are set only once per scanline!
        flags_minorframe = xarray.DataArray(
            numpy.zeros(
                shape=(flags_scanline["scanline_earth"].size, 64),
                dtype=self._data_vars_props["quality_minorframe_bitmask"][3]["dtype"]),
            dims=("scanline_earth", "minor_frame"),
            coords={"scanline_earth": flags_scanline.coords["scanline_earth"]},
            name="quality_minorframe_bitmask",
            attrs=self._data_vars_props["quality_minorframe_bitmask"][2]
        )

        # most generic one of all
        flags_pixel = xarray.DataArray(
            numpy.zeros(
                shape=(flags_scanline["scanline_earth"].size,
                       self.n_perline,
                       ds["calibrated_channel"].size),
                dtype=self._data_vars_props["quality_pixel_bitmask"][3]["dtype"]),
            dims=("scanline_earth", "scanpos", "calibrated_channel"),
            coords={"scanline_earth": flags_scanline.coords["scanline_earth"],
                    "scanpos": ds.coords["scanpos"],
                    "calibrated_channel": ds.coords["calibrated_channel"]},
            name="quality_pixel_bitmask",
            attrs=self._data_vars_props["quality_pixel_bitmask"][2]
        )


        self._flags["scanline"] = flags_scanline
        self._flags["channel"] = flags_channel
        self._flags["minorframe"] = flags_minorframe
        self._flags["pixel"] = flags_pixel

    def get_flags(self, ds, context, R_E):
        """Extract flags 

        Extract flags, as available after the radiance calculations have
        been completed.

        Parameters
        ----------

        ds : xarray.Dataset
            L1B data   
        context : xarray.Dataset
            Context around L1B data (not used)
        R_E : xarray.DataArray
            Calibrated FIDUCEO L1B radiances.  Used for coordinates to
            extract Earth view scanlines.

        Returns
        -------

        flags_scanline : xarray.DataArray
            Scanline-specific flags.
        flags_channel : xarray.DataArray
            Channel-specific flags.
        flags_minorframe : xarray.DataArray
            Minor frame specific flags.
        flags_pixel : xarray.DataArray
            Pixel specific flags.
            
        """

#        flags_scanline = xarray.DataArray(
#            numpy.zeros(
#                shape=R_E["scanline_earth"].size,
#                dtype=self._data_vars_props["quality_scanline_bitmask"][3]["dtype"]),
#            dims=("scanline_earth",),
#            coords={"scanline_earth": R_E.coords["scanline_earth"]},
#            name="quality_scanline_bitmask",
#            attrs=self._data_vars_props["quality_scanline_bitmask"][2]
#            )
#
#        flags_channel = xarray.DataArray(
#            numpy.zeros(
#                shape=(R_E["scanline_earth"].size, R_E["calibrated_channel"].size),
#                dtype=self._data_vars_props["quality_channel_bitmask"][3]["dtype"]),
#            dims=("scanline_earth", "calibrated_channel"),
#            coords={"scanline_earth": R_E.coords["scanline_earth"],
#                    "calibrated_channel": R_E.coords["calibrated_channel"]},
#            name="quality_channel_bitmask",
#            attrs=self._data_vars_props["quality_channel_bitmask"][2]
#            )
        flags_scanline = self._flags["scanline"]
        flags_channel = self._flags["channel"]
        flags_minorframe = self._flags["minorframe"]
        flags_pixel = self._flags["pixel"]

        
#        for (ch, flag) in self._flags["channel"].items():
#            flags_channel.loc[{"calibrated_channel": ch}] |= flag

        da_qfb = ds["quality_flags_bitfield"].sel(
            time=R_E.coords["scanline_earth"])
        fd_qif = typhon.datasets._tovs_defs.QualIndFlagsHIRS[self.version]
        fs = _fcdr_defs.FlagsScanline


        # pass .values to each to avoid xarrays array_ne which uses catch_warnings
        # which triggers http://bugs.python.org/issue29672
        flags_scanline[{"scanline_earth":((da_qfb & fd_qif.qidonotuse).values!=0)}] |= fs.DO_NOT_USE
        flags_scanline[{"scanline_earth":((da_qfb & fd_qif.qitimeseqerr).values!=0)}] |= fs.SUSPECT_TIME
        flags_scanline[{"scanline_earth":((da_qfb & fd_qif.qinofullcalib).values!=0)}] |= fs.SUSPECT_CALIB
        flags_scanline[{"scanline_earth":((da_qfb & fd_qif.qinoearthloc).values!=0)}] |= fs.SUSPECT_GEO


        # do not touch flags_channel here; HIRS/2 does not have any
        # channel-specific flags, so anything specific goes into HIRSKLM

        return (flags_scanline, flags_channel, flags_minorframe, flags_pixel)

    def propagate_uncertainty_components(self, u, sens, comp, sens_above=1):
        """Propagate individual uncertainty components seperately

        To investigate or otherwise communicate exactly how much each
        effect contributes to the total uncertainty in destination
        coordinates (radiance units or brightness temperatures), need to
        propagate each individually.  calc_u_for_variable only calculates
        it per sub-measurement-equation, then returns dictionaries with
        components and sensitivities.  Here, we propagate those and
        calculate the total magnitude for each uncertainty effect.
        Note that this does not include any covariant components.

        Returns a generator that yields the components one by one.

        Parameters
        ----------

        u : ndarray

            Total uncertainty, only used to determine output dimensions

        sub_sensitivities : `measurement_equation.ExpressionDict`

            Nested dictionary containing sensitivities per sub-measurement
            equation.  Obtained from `HIRSFCDR.calc_u_for_variable`, see
            docstring there for details on the specification.

        sub_components : `measurement_equation.ExpressionDict`, if ``return_more`` is True
        
            Nested dictionary containing components of sub-measurement.
            Obtained from `HIRSFCDR.calc_u_for_variable`, see docstring
            there for details on the specification.

        sens_above : array_like

            Used internally in the recursive algorithm, do not pass.  This
            is the cumulative sensitivity from the top-level down to the
            level of the current call.

        Yields
        ------

        :class:`sympy.core.symbol.Symbol`

            Component of measurement equation

        :class:`typhon.physics.units.tools.UnitsAwareDataArray`

            Magnitude of total uncertainty due to the aforementioned
            symbol.
        """
        # FIXME: also consider covariances as inputs
        # FIXME: how to return covariances?  They fall outside the logic
        # as they are naturally squared.  Should be returned separately.

        if not sens.keys() == comp.keys():
            raise ValueError("Must have same keys in sensitivity dict "
                "as in component dict!")

        for k in comp:
            # 'comp' is already a dictionary with 'what component of
            # uncertainty comes from this?', i.e. we only need to multiply
            # with sens_above to get correctness...
            # multiplication with three operands:
            # sens_above * sens[k][0] * uncertainty_comp[k][0]
            # first, sens_above is the dest, then it is the src
            sensk0 = self._make_dims_consistent(u, sens[k][0])
            sens_to_here = numpy.sqrt(sensk0**2 * sens_above**2)
            compk0 = self._make_dims_consistent(u, comp[k][0])
            # This is WRONG because I need to multiply with u_x, not with x!
#            compun = self._make_dims_consistent(sens_to_here, self._quantities[k])
#            if not src_dims <= dest_dims:
#                ipsens = self._make_dims_consistent(comp[k][0], sens_to_here)
#            else:
#                ipsens = sens_above
#            yield (k, numpy.sqrt(compk0**2 * sens_to_here**2))
            # FIXME: seperately yield covariant components?
            yield (k, numpy.sqrt(compk0**2 * sens_above**2))
            yield from self.propagate_uncertainty_components(u,
                sens[k][1], comp[k][1], sens_to_here)

    def calc_angles(self, ds):
        """Calculate satellite and solar angles

        Calculate satellite zenith angle, satellite azimuth angle, solar
        zenith angle, solar azimuth angle.

        Parameters
        ----------

        ds : xarray.Dataset

            Segment of L1B data

        Returns
        -------

        sat_za : xarray.DataArray
            Satellite zenith angle in degrees.
        sat_aa : xarray.DataArray
            Satellite azimuth angle in degrees, defined clockwise with the
            north at 0°.
        sol_za : xarray.DataArray
            Solar zenith angle in degrees.
        sol_aa : xarray.DataArray
            Solar azimuth angle in degrees.
        """

        # satellite angles
        satlatlon = ds[["lat","lon"]].sel(
            scanpos=[28, 29]).reset_coords(["lat", "lon"])
        # check for crossing antimeridian
        Δlon = (satlatlon.isel(scanpos=1) - satlatlon.isel(scanpos=0))["lon"]
        crossing = Δlon>180
        satlatlon["lon"][{"scanline_earth":crossing.values,"scanpos":0}] += 360
        satlatlon = satlatlon.mean("scanpos")
        satelev = ds["platform_altitude"]
        (sat_aa, sat_ea) = pyorbital.orbital.get_observer_look(
            satlatlon["lon"],
            satlatlon["lat"],
            satelev,
            ds["time"], # FIXME: Who cares?
            ds["lon"],
            ds["lat"],
            numpy.zeros(ds["lat"].shape)) # elevations
            
        (sun_el_rad, sun_az_rad) = pyorbital.astronomy.get_alt_az(
            ds["time"],
            ds["lon"],
            ds["lat"])

        sun_za = self._quantity_to_xarray(
            90 - numpy.rad2deg(sun_el_rad),
            "solar_zenith_angle")
        sun_aa = self._quantity_to_xarray(
            numpy.rad2deg(sun_az_rad),
            "solar_azimuth_angle")
        sat_za = self._quantity_to_xarray(
            90 - sat_ea,
            "platform_zenith_angle")
        sat_aa = self._quantity_to_xarray(
            sat_aa,
            "platform_azimuth_angle")

        return (sat_za, sat_aa, sun_za, sun_aa)


    #####################################################################
    #
    #   DEPRECATED!
    #
    # The remaining methods should no longer be used but legacy code such
    # as in timeseries.py still depends on them
    #
    #####################################################################

    def calc_sens_coef(self, typ, M, ch, srf): 
        """Calculate sensitivity coefficient.

        .. deprecated::
            **DEPRECATED --- DO NOT USE!**

        Actual work is delegated to calc_sens_coef_{name}
        """
        warnings.warn("Deprecated, use self.calc_u_for_variable", DeprecationWarning)

        f = getattr(self, "calc_sens_coef_{:s}".format(typ))

        (L_iwct, C_iwct, C_space, C_Earth) = (
            self.extract_and_interp_calibcounts_and_temp(M, ch, srf))

        return f(L_iwct[:, numpy.newaxis], C_iwct[:, numpy.newaxis],
                 C_space[:, numpy.newaxis], C_Earth)
    
    @typhon.math.common.calculate_precisely
    def calc_sens_coef_C_Earth(self, L_iwct, C_iwct, C_space, C_Earth):
        """**DEPRECATED --- DO NOT USE**"""
        warnings.warn("Deprecated, use self.calc_u_for_variable", DeprecationWarning)
        return L_iwct / (C_iwct - C_space)

    @typhon.math.common.calculate_precisely
    def calc_sens_coef_C_iwct(self, L_iwct, C_iwct, C_space, C_Earth):
        """**DEPRECATED --- DO NOT USE**"""
        warnings.warn("Deprecated, use self.calc_u_for_variable", DeprecationWarning)
        return - L_iwct * (C_Earth - C_space) / (C_iwct - C_space)**2

    @typhon.math.common.calculate_precisely
    def calc_sens_coef_C_iwct_slope(self, L_iwct, C_iwct, C_space):
        """Sensitivity coefficient for C_IWCT for slope (a₁) calculation

        .. deprecated::
            **DEPRECATED --- DO NOT USE!**

        Parameters
        ----------

            L_iwct : ndarray
                Radiance for IWCT.  Can be obtained with
                self.extract_calibcounts_and_temp.  Should
                be 1-D [N].
            C_iwct : ndarray
                Counts for IWCTs.  Should be 2-D [N × 48]
            C_space : ndarray
                Counts for space views.  Same shape as C_iwct.

        Returns
        -------

            sens : ndarray
                Sensitivity coefficient.

        """
        warnings.warn("Deprecated, use self.calc_u_for_variable", DeprecationWarning)
        return L_iwct[:, numpy.newaxis] / (C_iwct - C_space)**2

    @typhon.math.common.calculate_precisely
    def calc_sens_coef_C_space(self, L_iwct, C_iwct, C_space, C_Earth):
        """**DEPRECATED --- DO NOT USE!**"""
        warnings.warn("Deprecated, use self.calc_u_for_variable", DeprecationWarning)
        return L_iwct * (C_Earth - C_iwct) / (C_iwct - C_space)**2

    @typhon.math.common.calculate_precisely
    def calc_sens_coef_C_space_slope(self, L_iwct, C_iwct, C_space):
        """Sensitivity coefficient for C_space for slope (a₁) calculation
        Input as for calc_sens_coef_C_iwct_slope

        .. deprecated::
            **DEPRECATED --- DO NOT USE!**
        """
        warnings.warn("Deprecated, use self.calc_u_for_variable", DeprecationWarning)
        return -L_iwct[:, numpy.newaxis] / (C_iwct - C_space)**2


    def calc_urad(self, typ, M, ch, *args, srf=None):
        """Calculate uncertainty

        .. deprecated::
            **DEPRECATED --- DO NOT USE!**

        Parameters
        ----------

        typ : str
            Sort of uncertainty.  Currently implemented: "noise" and
            "calib".
        M : ndarray
        ch : int
        *args : any
            Depends on the sort of uncertainty, but should pass all
            the "base" uncertainties needed for propagation.  For
            example, for calib, must be u_C_iwct and u_C_space.
        srf : SRF
            Only if different from the nominal

        """
        warnings.warn("Deprecated, use self.calc_u_for_variable", DeprecationWarning)

        srf = srf or self.srfs[ch-1]
        f = getattr(self, "calc_urad_{:s}".format(typ))
        (L_iwct, C_iwct, C_space, C_Earth) = (
            self.extract_and_interp_calibcounts_and_temp(M, ch, srf))
        return f(L_iwct[:, numpy.newaxis],
                 C_iwct[:, numpy.newaxis],
                 C_space[:, numpy.newaxis], C_Earth, *args)

    def calc_urad_noise(self, L_iwct, C_iwct, C_space, C_Earth, u_C_Earth):
        """Calculate uncertainty due to random noise

        .. deprecated::
            **DEPRECATED --- DO NOT USE!**
        """
        warnings.warn("Deprecated, use self.calc_u_for_variable", DeprecationWarning)

        s = self.calc_sens_coef_C_Earth(L_iwct, C_iwct, C_space, C_Earth)
        return abs(s) * u_C_Earth

    def calc_urad_calib(self, L_iwct, C_iwct, C_space, C_Earth,
                              u_C_iwct, u_C_space):
        """Calculate radiance uncertainty due to calibration

        .. deprecated::
            **DEPRECATED --- DO NOT USE!**
        """
        warnings.warn("Deprecated, use self.calc_u_for_variable", DeprecationWarning)
        s_iwct = self.calc_sens_coef_C_iwct(
                    L_iwct, C_iwct, C_space, C_Earth)
        s_space = self.calc_sens_coef_C_space(
                    L_iwct, C_iwct, C_space, C_Earth)
        return numpy.sqrt((s_iwct * u_C_iwct)**2 +
                    (s_space * u_C_space)**2)

    def calc_uslope(self, M, ch, srf=None):
        """Direct calculation of slope uncertainty
        Such as for purposes of visualising uncertainties in slope/gain

        .. deprecated::
            **DEPRECATED --- DO NOT USE!**
        """
        warnings.warn("Deprecated, use self.calc_u_for_variable", DeprecationWarning)
        srf = srf or self.srfs[ch-1]
        # legacy code does not handle xarray dataarrays, move back to
        # legacy ureg form
        (time, L_iwct, C_iwct, C_space, u_C_iwct,
            u_C_space) = [
                (ureg.Quantity(numpy.ma.masked_invalid(x.values), x.attrs["units"])
                    if "units" in x.attrs
                    else numpy.ma.MaskedArray(x.values, mask=numpy.zeros_like(x.values)))
                          for x in self.extract_calibcounts_and_temp(
                          M, ch, srf, return_u=True)]
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

        .. deprecated::
            **DEPRECATED --- DO NOT USE!**
        """
        warnings.warn("Deprecated, use self.calc_u_for_variable", DeprecationWarning)

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

        .. deprecated::
            **DEPRECATED --- DO NOT USE!**

        Parameters
        ----------
            
            u : numpy.ndarray
                Vector of uncertainties.  Last dimension must be the
                dimension to estimate covariance matrix for.
            c_id : numpy.ndarray
                Vector with identifier for what calibration cycle was used
                in each.  Most commonly, the time.  Shape must match u.
        """
        warnings.warn("Deprecated, use self.calc_u_for_variable", DeprecationWarning)

        u = ureg.Quantity(numpy.atleast_2d(u), u.u)
        u_cross = u[..., numpy.newaxis] * u[..., numpy.newaxis].swapaxes(-1, -2)

        # r = 1 when using same calib, 0 otherwise...
        c_id = numpy.atleast_2d(c_id)
        r = (c_id[..., numpy.newaxis] == c_id[..., numpy.newaxis].swapaxes(-1, -2)).astype("f4")

        S = u_cross * r

        #S.mask |= (u[:, numpy.newaxis].mask | u[numpy.newaxis, :].mask) # redundant

        return S.squeeze()

class HIRSPODFCDR:
    """Mixin for HIRS POD FCDRs
    """
    def get_flags(self, ds, context, R_E):
        (flags_scanline, flags_channel, flags_minorframe, flags_pixel) = super().get_flags(ds, context, R_E)

        # might add stuff here later

        return (flags_scanline, flags_channel, flags_minorframe, flags_pixel)

class HIRSKLMFCDR:
    """Mixin for HIRS KLM FCDRs
    """
    def get_flags(self, ds, context, R_E):
        (flags_scanline, flags_channel, flags_minorframe, flags_pixel) = super().get_flags(ds, context, R_E)

        da_qfb = ds["quality_flags_bitfield"].sel(
            time=R_E.coords["scanline_earth"])
        da_lqfb = ds["line_quality_flags_bitfield"].sel(
            time=R_E.coords["scanline_earth"])
        da_mqfb = ds["minorframe_quality_flags_bitfield"].sel(
            time=R_E.coords["scanline_earth"])
        fd_qif = typhon.datasets._tovs_defs.QualIndFlagsHIRS[self.version]
        fd_qfb = typhon.datasets._tovs_defs.LinQualFlagsHIRS[self.version]
        fd_mff = typhon.datasets._tovs_defs.MinorFrameFlagsHIRS[self.version]
        fs = _fcdr_defs.FlagsScanline
        fc = _fcdr_defs.FlagsChannel
        fmf = _fcdr_defs.FlagsMinorFrame

        probs = {st:
            functools.reduce(operator.or_,
                (v for (k, v) in fd_qfb.__members__.items() if k.startswith(st)))
                    for st in {"tm", "el", "ca"}}

        # pass .values to each; xarrays array_ne uses catch_warnings,
        # which triggers http://bugs.python.org/issue29672
        flags_scanline[{"scanline_earth":((da_lqfb & probs["el"]).values!=0)}] |= fs.SUSPECT_GEO
        flags_scanline[{"scanline_earth":((da_lqfb & probs["tm"]).values!=0)}] |= fs.SUSPECT_TIME
        flags_scanline[{"scanline_earth":((da_lqfb & probs["ca"]).values!=0)}] |= fs.SUSPECT_CALIB

        flags_channel[{"scanline_earth":((da_qfb & fd_qif.qidonotuse).values!=0)}] |= fc.DO_NOT_USE

        mirprob = (functools.reduce(operator.or_,
                    (v for (k, v) in fd_mff.__members__.items() if
                        k.startswith("mfmir"))))

        flags_minorframe.values[(da_mqfb & mirprob).values!=0] |= fmf.SUSPECT_MIRROR

        # easy FCDR does not contain minorframe flags.  To contain at
        # least some mirror information, flag entire scanline when there
        # is a mirror flag for any minor frame.  Should consider if this
        # is the best way: see
        # https://github.com/FIDUCEO/FCDR_HIRS/issues/133
        flags_scanline[(flags_minorframe.any("minor_frame") & fmf.SUSPECT_MIRROR).values!=0] |= fs.SUSPECT_MIRROR_ANY

        return (flags_scanline, flags_channel, flags_minorframe,
                flags_pixel)

class HIRS2FCDR(HIRSPODFCDR, HIRSFCDR, HIRS2):
    """HIRS2 FCDR class

    For documentation, see `HIRSFCDR`
    """

    l1b_base = HIRS2

class HIRS3FCDR(HIRSKLMFCDR, HIRSFCDR, HIRS3):
    """HIRS3 FCDR class

    For documentation, see `HIRSFCDR`
    """

    l1b_base = HIRS3

class HIRS4FCDR(HIRSKLMFCDR, HIRSFCDR, HIRS4):
    """HIRS4 FCDR class

    For documentation, see `HIRSFCDR`
    """

    l1b_base = HIRS4

def which_hirs_fcdr(satname, *args, **kwargs):
    """Given a satellite, return right HIRS object

    From a satellite name, return an object from the correct HIRSFCDR
    class.  This function support various different spellings of the same
    satellite, such as "noaa15", "NOAA-15", or "N15".

    Parameters
    ----------

    satname : str
        Name of the satellite for which a `HIRSFCDR` object is desired.
    *args
        Remaining arguments passed on to the applicable `HIRSFCDR` class.
        The satellite name ``satname`` is already passed on.
    **kwargs
        Remaining arguments passed on to the applicable `HIRSFCDR` class.
        The satellite name ``satname`` is already passed on.

    Returns
    -------

    `HIRSFCDR`
        Object from the correct `HIRSFCDR` class with the correct
        satellite name passed on to the constructor.
    """
    for h in {HIRS2FCDR, HIRS3FCDR, HIRS4FCDR}:
        for (k, v) in h.satellites.items():
            if satname in {k}|v:
                return h(*args, satname=k, **kwargs)
    else:
        raise ValueError("Unknown HIRS satellite: {:s}".format(satname))

def list_all_satellites_chronologically():
    """Return a list of all satellite names, sorted

    List all satellites, starting with tirosn, followed by all the NOAA
    satellites, then followed by the Metop satellites.

    Returns
    -------

    List[str]
        List of all satellite names.
    """
    return ["tirosn"] + [f"noaa{i:02d}" for i in range(6, 20) if i!=13] + [
        "metopa", "metopb"]

def _recursively_search_for(sub, var):
    """Internal helper with sub_sensitivities or sub_components
    
    Internal helper function to search if a variable is already defined in
    the recursive tree that it sub_sensitivities or sub_components, and
    return it if it is.


    Parameters
    ----------

    sub : Dict
        Either sub_sensitivities or sub_components.  For a detailed
        definition, see the documentation for the
        :meth:`HIRSFCDR.calc_u_for_variable` return values.
    var : Symbol
        Symbol to look up.

    Returns
    -------

    xarray.DataArray
        Sensitivity or uncertainty component for ``var``.
    """

    for (k, v) in sub.items():
        if k is var:
            return sub[k][1]
        elif sub[k][1] is not None:
            res = _recursively_search_for(sub[k][1], var)
            if res is not None:
                return res

def make_debug_fcdr_dims_consistent(dest, src, impossible="warn",
                                    flags=None):
    """From debug FCDR, expand and restrict (temporal) dimensions

    Make the dimension in ``src`` equal to the dimension in ``dest``, by
    interpolating the dimensions ``calibration_cycle`` and
    ``rself_update_time`` and collapsing the dimension
    ``calibration_position``
    
    See :meth:`HIRSFCDR._make_adict_dims_consistent` and
    :meth:`HIRSFCDR._make_adict_dims_consistent_if_needed`.

    Parameters
    ----------

    dest : xarray.DataArray
        Array to which the dimensions shall be matched.
    src: xarray.DataArray
        Array for which the dimensions shall be adapted.
    impossible : str, optional
        Flag on what to do if the task is impossible.  Should be either
        "warn", which means issue a warning, or "error", which means raise
        an exception (`ValueError`)
    flags : Mapping, optional
        Mapping on which to set flags if uncertainties were extrapolated

    Returns
    -------

    xarray.DataArray
        like src but with dimensions now matching dest
    """

    srcdims = getattr(src, "dims", ())
    if "calibration_position" in srcdims:
        src = src.mean(dim="calibration_position", keep_attrs=True)
    if (("calibration_cycle" in srcdims
        or "rself_update_time" in srcdims)
            and hasattr(dest, "scanline_earth")):
        dest_time = dest.scanline_earth.astype("u8")
        d = ("calibration_cycle" if "calibration_cycle" in srcdims else "rself_update_time")
        src_time = src[d].astype("u8")
        if (src[d][0] <= dest.scanline_earth[0] and
            src[d][-1] >= dest.scanline_earth[-1]):
            kind="zero"
            bounds_error=True
            fill_value=None
        else:
            msg = (
                "Problem propagating uncertainties "
                "from {!s} to {!s}. "
                "Calibration cycles ({:s}) cover "
                "{:%Y-%m-%d %H:%M}--{:%Y-%m-%d %H:%M}, "
                "Earth views cover "
                "{:%Y-%m-%d %H:%M}--{:%Y-%m-%d %H:%M}, "
                "reduced context means calibration cycles "
                "do not fully cover earth views, so I cannot "
                "use the former to interpolate uncertainties "
                "on the latter.  Extrapolating instead. "
                "Use with care.  I will flag the data.".format(
                dest.name, src.name,
                d,
                src[d].values[0].astype("M8[ms]").astype(datetime.datetime),
                src[d].values[-1].astype("M8[ms]").astype(datetime.datetime),
                dest.scanline_earth.values[0].astype("M8[ms]").astype(datetime.datetime),
                dest.scanline_earth.values[-1].astype("M8[ms]").astype(datetime.datetime)))
            if impossible=="warn": # reduced context, this is dangerous
                logger.warning(msg)
            else:
                raise ValueError(msg)
            kind="zero"
            bounds_error=False
            fill_value="extrapolate"
            flags["scanline"][{"scanline_earth": dest.scanline_earth<src[d][0]}] |= _fcdr_defs.FlagsScanline.UNCERTAINTY_SUSPICIOUS
            flags["scanline"][{"scanline_earth": dest.scanline_earth>src[d][-1]}] |= _fcdr_defs.FlagsScanline.UNCERTAINTY_SUSPICIOUS
            # FIXME: flag - but only part that I have extrapolated.
            # And need to think, what happened when I applied the
            # calibration in the first place?  This sort of stuff
            # happens twice in my code, first at initial calculation,
            # then at uncertainty calculation!

        fnc = scipy.interpolate.interp1d(
            src_time, src,
            kind=kind,
            bounds_error=bounds_error,
            axis=src.dims.index(d),
            fill_value=fill_value)
        src = UADA(fnc(dest_time),
            dims=[x.replace(d, "scanline_earth") for
                    x in src.dims],
            attrs=src.attrs,
            encoding=src.encoding)

        src = src.assign_coords(
            **{k: v
                for (k, v) in dest.coords.items()
                if all(d in src.dims for d in v.dims)})
    return src

# Patch xarray.core._ignore_warnings_if to avoid repeatedly hearing the
# same warnings.  This function contains the catch_warnings contextmanager
# which is buggy, see http://bugs.python.org/issue29672
import contextlib
import xarray.core.ops

@contextlib.contextmanager
def _do_nothing(*args, **kwargs):
    yield
xarray.core.ops._ignore_warnings_if = _do_nothing
xarray.core.duck_array_ops._ignore_warnings_if = _do_nothing

# from xarray/core/duck_array_ops.py
def _new_array_equiv(arr1, arr2):
    """Like np.array_equal, but also allows values to be NaN in both arrays
    """
    arr1, arr2 = xarray.core.duck_array_ops.as_like_arrays(arr1, arr2)
    if arr1.shape != arr2.shape:
        return False

    with _do_nothing():
        flag_array = (arr1 == arr2)
        flag_array |= (xarray.core.duck_array_ops.isnull(arr1) & xarray.core.duck_array_ops.isnull(arr2))

        return bool(flag_array.all())



xarray.core.duck_array_ops.array_equiv = _new_array_equiv
xarray.core.variable.Variable.equals.__defaults__ = (_new_array_equiv,)
xarray.core.variable.Variable.broadcast_equals.__defaults__ = (_new_array_equiv,)

def _new_array_notnull_equiv(arr1, arr2):
    """Like np.array_equal, but also allows values to be NaN in either or both
    arrays
    """
    arr1, arr2 = xarray.core.duck_array_ops.as_like_arrays(arr1, arr2)
    if arr1.shape != arr2.shape:
        return False

    with _do_nothing():
        flag_array = (arr1 == arr2)
        flag_array |= xarray.core.duck_array_ops.isnull(arr1)
        flag_array |= xarray.core.duck_array_ops.isnull(arr2)

        return bool(flag_array.all())

xarray.core.duck_array_ops.array_notnull_equiv = _new_array_notnull_equiv

def _new_array_eq(self, other):
    with _do_nothing():
        return xarray.core.nputils._ensure_bool_is_ndarray(self == other, self, other) 

def _new_array_ne(self, other):
    with _do_nothing():
        return xarray.core.nputils._ensure_bool_is_ndarray(self != other, self, other) 

xarray.core.nputils.array_eq = _new_array_eq
xarray.core.nputils.array_ne = _new_array_ne
