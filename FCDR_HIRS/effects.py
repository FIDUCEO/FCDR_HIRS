"""Classes and objects related to uncertainty effects

The effects module contains classes, functions, and objects describing the
uncertainty effects affecting the HIRS FCDR.  The centrepiece is the
`Effect` class, which is an attempt to encode into Python abstraction the
effects table such as described in FIDUCEO deliverible D4.2, albeit
currently an outdated version of it.  Each HIRS FCDR source of uncertainty
is a specific instance of this `Effect` class, which are all included
within this module.  The remaining classes and functions are essentially
helper classes for the functionality within each effect.

The existance of an `Effect` object in this module does not magically
include it in the uncertainty budget.  The user/developer still needs to
populate the ``.magnitude`` attribute as documented in the `Effect` class
documentation, then pass it to
:meth:`FCDR_HIRS.fcdr.HIRSFCDR.calc_u_for_variables` inside the
``all_effects`` argument.

To get an impression of which ones are actually implemented, have a look
at calls to `FCDR_HIRS.fcdr._tuck_effect_channel`, which are littered
about here and there, which is currently being used to populate a
dictionary in preparation for the uncertainty calculation and CURUC.
"""

import math
import abc
import collections
import copy
import numbers
import warnings
import itertools

import numpy
import xarray
import sympy
import docrep

from typing import (Tuple, Mapping, Set)

from typhon.physics.units.common import (radiance_units, ureg)
from typhon.physics.units.tools import UnitsAwareDataArray as UADA

from . import measurement_equation as meq
from . import _fcdr_defs
from .exceptions import FCDRWarning

WARNING = ("VERY EARLY TRIAL VERSION! "
           "DO NOT USE THE CONTENTS OF THIS PRODUCT FOR ANY PURPOSE UNDER ANY CIRCUMSTANCES! "
            "This serves exclusively as a file format demonstration!")
dst = docrep.DocstringProcessor()

#: Form of tuple to describe correlation type on four scales, used in `Effect` class.
CorrelationType = collections.namedtuple("CorrelationType",
    ["within_scanline", "between_scanlines", "between_orbits",
    "across_time"])
#: Form of tuple to describe correlation scale on four scales, used in `Effect` class.
CorrelationScale = collections.namedtuple("CorrelationScale",
    CorrelationType._fields)

class Rmodel(metaclass=abc.ABCMeta):
    """Abstract class describing the interface to calculate R

    This class defines the interface for an Rmodel that each effect needs
    to describe to calculate R for that effect, as an input to the CURUC
    recipes.  Rather than each effect implementing those from scratch, in
    practice, several Rmodels may be shared between different effects,
    such that `Effect.calc_R_eΛlk` just delegates to the
    `Rmodel.calc_R_eΛlk` for the corresponding Rmodel.  The module defines
    several `Rmodel`s.
    """

    #@dst.get_full_descriptionf("R_eΛk")
    #@dst.get_sectionsf("R_eΛlk")
    #@dst.with_indent(8)
    @abc.abstractmethod
    def calc_R_eΛlk(self, ds,
            sampling_l=1, sampling_e=1):
        """Return R_eΛlk for single k

        Return the cross-element correlation matrix for a single effect,
        for all lines and for all channels.

        Parameters
        ----------

        ds : xarray.Dataset
            Dataset (debug version) describing FCDR segment, as calculated
            by `processing.generate_fcdr.FCDRGenerator.get_piece`.
        sampling_l : int, optional
            Sampling rate along scanlines.  Defaults to 1.
        sampling_e : int, optional
            Sampling rate along elements.  Defaults to 1.

        Returns
        -------

        xarray.DataArray [n_c, n_l, n_e, n_e]
            Value for ``R_eΛlk`` for each channel, line, and element, for the
            effect to which this `Rmodel` belongs.
        """

    #@dst.with_indent(8)
    @abc.abstractmethod
    def calc_R_lΛek(self, ds,
        sampling_l=1, sampling_e=1):
        """Return R_lΛek for single k

        Return the cross-line correlation matrix for a single effect, for
        all elements and for all channels.

        Parameters
        ----------
        %(R_eΛk.parameters)s

        Returns
        -------

        xarray.DataArray [n_c, n_e, n_l, n_l]
            Value for ``R_lΛek`` for each channel, line, and element, for the
            effect to which this `Rmodel` belongs.
        """

    #@dst.with_indent(8)
    @abc.abstractmethod
    def calc_R_cΛpk(self, ds,
        sampling_l=1, sampling_e=1):
        """Return R_cΛpk for single k

        Return the cross-channel correlation matrix for a single effect,
        for all elements and for all channels.

        Parameters
        ----------
        %(R_eΛk.parameters)s

        Returns
        -------

        xarray.DataArray [n_l, n_e, n_c, n_c]
            Value for ``R_lΛek`` for each channel, line, and element, for the
            effect to which this `Rmodel` belongs.
        """

#@dst.with_indent(4)
def _calc_R_eΛlk_allones(ds, sampling_l=1, sampling_e=1):
    """Return R_eΛlk for single k with all ones

    Return the cross-element error correlation matrix ``R_eΛk`` for the
    situation where the correlation is total, i.e., a matrix of ones.

    This is a helper function that is not intended to be used externally,
    but documented here nevertheless.

    Parameters
    ----------
    %(R_eΛk.parameters)s

    Returns
    -------

    xarray.DataArray [n_c, n_l, n_e, n_e]
        Matrix of said dimensions, completely filled with ones.
    """
    return numpy.ones(
        (ds.dims["calibrated_channel"],
         math.ceil(ds.dims["scanline_earth"]/sampling_l),
         math.ceil(ds.dims["scanpos"]/sampling_e),
         math.ceil(ds.dims["scanpos"]/sampling_e)), dtype="f4")

class RModelCalib(Rmodel): # docstring in parent
    """R Model implementation for calibration effects

    Implementation of the R model for sources of uncertainty that repeat
    along with a calibration cycle, such as IWCT or deep space view
    uncertainties.
    """

    # docstring in parent
    def calc_R_eΛlk(self, ds,
        sampling_l=1, sampling_e=1):
        return _calc_R_eΛlk_allones(ds, sampling_l=sampling_l,
            sampling_e=sampling_e)

    # docstring in parent
    def calc_R_lΛek(self, ds,
            sampling_l=1, sampling_e=1):

        # wherever scanline_earth shares a calibration_cycle the
        # correlation is 1; anywhere else, it's 0.
        ccid = (ds["scanline_earth"]>ds["calibration_cycle"]).sum("calibration_cycle").values
        R = (ccid[:, numpy.newaxis] == ccid[numpy.newaxis, :]).astype("f4")
        return numpy.tile(R[::sampling_l, ::sampling_l][
                numpy.newaxis, numpy.newaxis, :, :],
            (ds.dims["calibrated_channel"],
            math.ceil(ds.dims["scanpos"]/sampling_e),
            1, 1))

    # docstring in parent
    def calc_R_cΛpk(self, ds,
        sampling_l=1, sampling_e=1):
        # ERROR WARNING FIXME: This needs to be updated.  See #223
        warnings.warn("Inter-channel correlation not implemented "
            "for calibration-scale correlations.  See #223.",
            FCDRWarning)
        return numpy.tile(
            numpy.eye(ds.dims["calibrated_channel"], dtype="f4"),
            [math.ceil(ds.dims["scanline_earth"]/sampling_l),
             math.ceil(ds.dims["scanpos"]/sampling_e), 1, 1])

#: `Rmodel` implementation for effects per calibration cycle
rmodel_calib = RModelCalib()

class RModelCalibPRT(RModelCalib):
    """R Model implementation for IWCT PRT effects

    Implementation of the R model for sources of uncertainty due to IWCT
    PRT views, such as the PRT uncertainties, but also temperature
    gradients (not implemented).  These are a special case of general
    calibration uncertainties, because those are shared between channels.
    """
    # docstring in parent
    def calc_R_cΛpk(self, ds,
        sampling_l=1, sampling_e=1):
        return numpy.ones(
            (math.ceil(ds.dims["scanline_earth"]/sampling_l),
             math.ceil(ds.dims["scanpos"]/sampling_e),
             ds.dims["calibrated_channel"]),
             dtype="f4")
        
#: `Rmodel` implementation for effects due to IWCT PRTs
rmodel_calib_prt = RModelCalibPRT()

class RModelRandom(Rmodel):
    """R Model for fully random effects.

    Implementation of the R model where all correlation matrices between lines,
    between elements, and between channels are the identity matrix, i.e.
    errors are completely random.  That includes correlations between
    channels.

    See :issue:`224`.
    """

    # docstring in parent
    def calc_R_eΛlk(self, ds,
        sampling_l=1, sampling_e=1):
        return numpy.tile(
            numpy.eye(math.ceil(ds.dims["scanpos"]/sampling_e), dtype="f4"),
            [ds.dims["calibrated_channel"], 
             math.ceil(ds.dims["scanline_earth"]/sampling_l), 1, 1])

    # docstring in parent
    def calc_R_lΛek(self, ds,
            sampling_l=1, sampling_e=1):
        return numpy.tile(
            numpy.eye(math.ceil(ds.dims["scanline_earth"]/sampling_l), dtype="f4"),
            [ds.dims["calibrated_channel"],
            math.ceil(ds.dims["scanpos"]/sampling_e), 1, 1])

    # docstring in parent
    def calc_R_cΛpk(self, ds,
        sampling_l=1, sampling_e=1):
        return numpy.tile(
            numpy.eye(ds.dims["calibrated_channel"], dtype="f4"),
            [math.ceil(ds.dims["scanline_earth"]/sampling_l),
             math.ceil(ds.dims["scanpos"]/sampling_e), 1, 1])

#: `Rmodel` implemented for fully random case
rmodel_random = RModelRandom()

class RModelCommon(Rmodel):
    """RModel for common case.  Unconditional error.

    There is no `Rmodel` implementation for the fully common case, as the
    CURUC recipes only calculate this for random and systematic effects,
    not common effects.  Calling methods on this class will be an
    unconditional error.
    """

    # docstring in parent
    def calc_R_eΛlk(self, ds,
            sampling_l=1, sampling_e=1):
        raise ValueError(
            "We do not calculate error correlation matrices for common effects")
    calc_R_lΛek = calc_R_eΛlk

    # docstring in parent
    def calc_R_cΛpk(self, ds,
        sampling_l=1,
        sampling_e=1):
        raise NotImplementedError("Not implemented yet")

#: `Rmodel` implementation for common case --- unconditional error!
rmodel_common = RModelCommon()

class RModelPeriodicError(Rmodel):
    """RModel for "periodic noise"

    RModel implementation for "periodic error", "periodic noise", such as
    observed in HIRS, perhaps due to the filter wheel.

    This is currently not implemented, see :issue:`224`.
    """

    # docstring in parent
    def calc_R_eΛlk(self, ds,
            sampling_l=1, sampling_e=1):
        raise NotImplementedError()

    # docstring in parent
    def calc_R_lΛek(self, ds,
            sampling_l=1, sampling_e=1):
        raise NotImplementedError()

    # docstring in parent
    def calc_R_cΛpk(self, ds,
        sampling_l=1,
        sampling_e=1):
        raise NotImplementedError("Not implemented yet")

#: `Rmodel` implementation for periodic errors
rmodel_periodicerror = RModelPeriodicError()

class RModelRSelf(Rmodel):
    """Rmodel implementation for self-emission uncertainty

    This class implements the correlation model (`Rmodel`) for the
    uncertainty due to the self-emission model.  It was written to
    correspond with the self-emission model in `FCDR_HIRS.models.RSelf`,
    but may also apply to other self-emission models.  It assumes:

    * Self emission is perfectly correlated between elements.  The
      parameters as calculated by `FCDR_HIRS.models.RSelf` is a function
      of temperature, which is only measured once per scanline.  Therefore,
      the self emission estimate is constant within a scanline, as is any
      error in the self emission estimate.
    * Self emission correlation between scanlines reduces linearly from 1
      to 0 over a period of 25 minutes.  If we were to ignore
      self-emission completely (null model), then we are overestimating
      the Earth radiance during the half-orbit that the satellite is
      getting colder, and underestimating the Earth radiance during the
      half-orbit that the satellite is getting warmer.  We would therefore
      expect the self emission error correlation for the null model to
      vary between +1 and -1.  Metrologically, we must correct for all
      known errors before estimating any uncertainty, and this class makes
      that assumption.  In the absence of knowledge on how the self
      emission model error correlations are affected by our attempts to
      eliminate self emission errors, we will assume that this has
      remained the same.  However, the implementation does not currently
      consider negative correlations, because the CURUC recipes include an
      exponential fit to calculate the correlation length scale and
      assumes positive correlations only.  Therefore, we assume an error
      correlation that drops off from 1 to 0 during a quarter orbit, which
      is here rounded to 25 minutes.

      A better approach would be to consider the three components
      individually, as described (but not implemented) in the following
      bullet point.
    * Self emission between different channels is 0.5.  This is a poor
      estimate.  The self emission model actually has three sources of
      error, each with a different error correlation behaviour:

      - The model fitted parameters, ``d_n``.  In a perfect world, I would
        fit the self emission model at the same time for all channels.
        This would yield an error covariance from the regression model
        used to fit the parameters, which could feed straight into the
        CURUC.
      - The temperatures used to estimate self emission.  Those are
        identical between channels, and so are their errors.
      - The model error.  I don't know the error correlation between
        channels for the model error.

      In the absence of a reliable estimate and in the knowledge that
      assuming either 0 or 1 is wrong, the current implementation sets all
      inter-channel correlations to 0.5.

    See also
    --------

    `FCDR_HIRS.models.RSelf`
        Implementation of self-emission model.
    """

    # docstring in parent
    def calc_R_eΛlk(self, ds,
            sampling_l=1, sampling_e=1):
        return _calc_R_eΛlk_allones(ds, sampling_l=sampling_l,
            sampling_e=sampling_e)
    # docstring in parent
    def calc_R_lΛek(self, ds,
            sampling_l=1, sampling_e=1):
        # same for all channels anyway
        # linearly decreasing from 1 (at t=0) to 0 (at t=25m).
        # We'd rather include negative correlatinos (going to -1 at t=50m)
        # but the CURUC recipes don't support those anyway.  This
        # assumption is derived from the idea that if the self-emission
        # overestimates by x at time t, it may underestimate by x half an
        # orbit later.
        sz = ds.dims["scanline_earth"]
        R = numpy.zeros((sz, sz), dtype="f4")
        r = 1
        i = itertools.count()
        while True:
            d = next(i)
            try:
                r = 1 - (ds["scanline_earth"][d] - ds["scanline_earth"][0])/numpy.timedelta64(25, 'm')
            except IndexError: # not enough lines in orbit
                break
            if r <= 0:
                break
            for s in (+1, -1):
                diag = numpy.diagonal(R, s*d)
                diag.setflags(write=True)
                diag[:] = r

        return R[::sampling_l, ::sampling_l]

    # docstring in parent
    def calc_R_cΛpk(self, ds,
        sampling_l=1, sampling_e=1):

        warnings.warn("Inter-channel correlation not implemented "
            "for self-emission model.  See "
            "https://github.com/FIDUCEO/FCDR_HIRS/labels/self-emission . "
            "Arbitrarily assuming inter-channel correlation = 0.5.")
        (nc, nl, ne) = (ds.dims["calibrated_channel"],
                        ds.dims["scanline_earth"],
                        ds.dims["scanpos"])
        return numpy.tile(
            (5*numpy.ones((nc,nc), dtype="f4")
             + 5*numpy.eye(nc, dtype="f4"))/10,
            [math.ceil(ds.dims["scanline_earth"]/sampling_l),
             math.ceil(ds.dims["scanpos"]/sampling_e), 1, 1])

#: implementation of self-emission model error
rmodel_rself = RModelRSelf()

class Effect:
    """Class to represent an effect / source of uncertainty.

    The `Effect` class is a representation of an effect that causes an
    uncertainty.  It is an in-code representation of the effects tables
    from FIDUCEO D4.2, although it is not entirely up to date with the
    latest version.  The information contained in the `Effect` class and
    the methods defined in it are used to populate the debug FCDR and
    calculate inputs to the CURUC recipes, although the latter is in
    practice often delegated to an `Rmodel` class, an instance of which
    may be an attribute here.

    The module defines a set of effects, but defining an `Effect` does not
    automagically add it to the uncertainty budget.  A copy of the table with all
    effects can be obtained using the `effects` function.  It is up to the
    developer to populate the ``.magnitude`` attribute for all effects, and
    then pass it on to functions requiring it, in particular the
    all-important method `FCDR_HIRS.fcdr.HIRSFCDR.calc_u_for_variable`.

    Attributes
    ----------
    
    These attributes are generally set upon definition:

    name : str
        Short name.  This will be used to, for example, refer to the
        effect in a global dictionary with all effects.
    description : str
        Long name, human readable description of the effect.
    parameter : `sympy.Symbol`
        The `sympy.Symbol` within the `measurement_equation` that this
        effect relates to.  Each effect must relate to exactly one
        measurement equation parameter.
    unit : `pint.unit._Unit`
        The pint unit in which the magnitude of the effect is measured.
        Must be compatible with the physical dimensions for the
        measurement that the measurement equation parameter represents.
    pdf_shape : str
        Shape of the PDF corresponding to the uncertainty.  Defaults to "Gaussian".
    channels_affected : str
        What channels are affected.  Defaults to "all".
    correlation_type : tuple[Str]
        Property, 4-tuple of strings, or `CorrelationType` named tuple.
        Each member of the tuple must be one of:
            * "undefined"
            * "random"
            * "rectangular_absolute"
            * "triangular_relative"
            * "truncated_gaussian_relative"
            * "repeated_rectangles"
            * "repeated_truncated_gaussians"
        The four tuple elements refer to (and, in case of the named tuple,
        can be referred to as) ``within_scanline``, ``between_scanlines``,
        ``between_orbits``, and ``across_time``.
    channel_correlations : (n_c, n_c), numpy.ndarray
        The correlation matrix between channels.  Sometimes this is
        calculated on the fly.  See also the `Effect.calc_R_cΛpk` method.
        Currently (2018-12-20), channel correlations is used to set an
        attribute on the effect in the debug FCDR, but the result of
        `Effect.calc_R_cΛpk` is used it the CURUC recipes.
    dimensions : Tuple[str] or None
        Normally, the data dimensions for an uncertainty quantity should
        be the same as the dimensions for the quantity in the measurement
        equation the uncertainty corresponds to.  The latter are defined
        in the `FCDR_HIRS._fcdr_defs` module and as the
        `FCDR_HIRS.fcdr.FCDR._data_vars_props` attribute.  In some cases,
        the data dimensions for the uncertainty may differ from the
        quantity.  For example, IWCT type B uncertainty is scalar, even
        though it belongs to a quantity that is not.  This attribute can
        be set to a sequence of strings describing the dimensions for the
        uncertainty type.
    rmodel : `Rmodel`
        Instance of the `Rmodel` class, or rather one of its
        implementation, which in turn describes the correlation
        scructures.  This may appear to duplicate the `correlation_type`
        and `correlation_scale` attributes, and perhaps it does, but the
        attributes are stored in the debug FCDR whereas the `rmodel`
        attribute is used to calculate the inputs to CURUC.

    There are some attributes that are set during FCDR calculation:

    magnitude : `typhon.physics.units.tools.UnitsAwareDataArray`
        This will be set to an instance of
        `typhon.physics.units.tools.UnitsAwareDataArray`, which will
        describe the magnitude and the units of the uncertainty.
    correlation_scale : `CorrelationScale`
        This is a `CorrelationScale` namedtuple that is supposed to
        contain the magnitudes of the correlation along different types.
        The types are as for ``correlation_type`` above, but are not
        actually set by the CURUC recipes and this `correlation_scale` is
        not really used.
    covariances : None
        Not currently (2018-12-20) used for anything.  Covariances and
        correlations are calculated through the CURUC recipes, which call
        the `calc_R_eΛlk`, `calc_R_lΛek`, and `calc_R_cΛpk` methods on
        the `Effect` class which in turn delegates those calculations to
        the `Rmodel` of choice.  This property is a placeholder for
        covariances between different effects, but those are not currently
        considered.

    See also
    --------

    The `measurement_equation` module used to calculate sensitivity
    coefficients.
    """

    #: `measurement_equation.ExpressionDict` containing all effects defined so far
    _all_effects = meq.ExpressionDict()
    #: name of the effect
    name = None
    #: description of the effect
    description = None
    #: parameter that the effect relates to
    parameter = None
    #: unit of the quantity quantifying the effect
    unit = None
    #: shape of the uncertainty PDF
    pdf_shape = "Gaussian"
    #: describes what channels are affected
    channels_affected = "all"
    #: describes channel correlations
    channel_correlations = None
    #: describes data dimensions, if different from source
    dimensions = None
    #: implementation of `Rmodel` for the calculation of correlations, inputs to CURUC
    rmodel = None
    #: holds covariances but not currently used
    _covariances = None

    def __init__(self, **kwargs):
        """Create new Effect

        All keyword arguments are used to set attributes to the resulting
        object.  Properties are defined last.
        """
        later_pairs = []
        while len(kwargs) > 0:
            (k, v) = kwargs.popitem()
            if isinstance(getattr(self.__class__, k), property):
                # setter may depend on other values, do last
                later_pairs.append((k, v))
            else:
                setattr(self, k, v)
        while len(later_pairs) > 0:
            (k, v) = later_pairs.pop()
            setattr(self, k, v)
        if not self.parameter in self._all_effects:
            self._all_effects[self.parameter] = set()
        self._all_effects[self.parameter].add(self)
        self._covariances = {}

    def __setattr__(self, k, v):
        if not hasattr(self, k):
            raise AttributeError("Unknown attribute: {:s}".format(k))
        super().__setattr__(k, v)

    def __repr__(self):
        return "<Effect {!s}:{:s}>\n".format(self.parameter, self.name) + (
            "{description:s} {dims!s} [{unit!s}]\n".format(
                description=self.description, dims=self.dimensions, unit=self.unit) +
            "Correlations: {!s} {!s}\n".format(self.correlation_type,
                self.correlation_scale) +
            "Magnitude: {!s}".format(self.magnitude))

    _magnitude = None
    @property
    def magnitude(self):
        """Magnitude of the uncertainty

        This should be a DataArray with dimensions depending on the
        effect.  Assumed to be constant along any dimensions not given.
        Note that this means the magnitude of the uncertainty is constant;
        it does not mean anything about the error correlation, which is
        treated separately.
        """
        return self._magnitude

    @magnitude.setter
    def magnitude(self, da):
        if not isinstance(da, xarray.DataArray):
            try:
                unit = da.u
            except AttributeError:
                unit = None

            da = xarray.DataArray(da)
            if unit is not None:
                da.attrs.setdefault("units", unit)

        if da.name is None:
            da.name = "u_{:s}".format(self.name)

        # make sure dimensions match
        # FIXME: make sure short names for effects always match the short
        # names used in _fcdr_defs so that the dictionary lookup works
        # EDIT 2017-02-13: Commenting this because I don't understand
        # why this is needed.  If I uncomment it later I should explain
        # clearly what is going on here.  It fails because uncertainty
        # magnitudes may have less dimensions than the quantities they relate to, in
        # particular when relating to systematic errors; for example, PRT
        # type B uncertainty has magnitude 0.1 across all dimensions.
        #da = da.rename(dict(zip(da.dims, _fcdr_defs.FCDR_data_vars_props[self.name][1])))

        da.attrs.setdefault("long_name", self.description)
        da.attrs["short_name"] = self.name
        da.attrs["parameter"] = str(self.parameter)
        da.attrs["pdf_shape"] = self.pdf_shape
        da.attrs["channels_affected"] = self.channels_affected
        for (k, v) in self.correlation_type._asdict().items():
            da.attrs["correlation_type_" + k] = v
            # FIXME: can an attribute have dimensions?  Or does this need to
            # be stored as a variable?  See
            # https://github.com/FIDUCEO/FCDR_HIRS/issues/47
            da.attrs["correlation_scale_" + k] = getattr(self.correlation_scale, k)
        da.attrs["channel_correlations"] = self.channel_correlations
        da.attrs["sensitivity_coefficient"] = str(self.sensitivity())
        da.attrs["WARNING"] = WARNING

        if not self.name.startswith("O_") or self.name in _fcdr_defs.FCDR_data_vars_props:
            da.encoding.update(_fcdr_defs.FCDR_data_vars_props[self.name][3])
        da.encoding.update(_fcdr_defs.FCDR_uncertainty_encodings.get(self.name, {}))

        self._magnitude = da


    _corr_type = CorrelationType("undefined", "undefined", "undefined",
                                "undefined")
    _valid_correlation_types = ("undefined", "random",
                                "rectangular_absolute",
                                "triangular_relative",
                                "truncated_gaussian_relative",
                                "repeated_rectangles",
                                "repeated_truncated_gaussians")

    def set_covariance(self, other, channel, da_ch, _set_other=True):
        """Set covariance between this and other effect.

        Parameters
        ----------

        other : Effect
            The other `Effect` that we have a coviarance with.
        da_ch : `typhon.physics.units.tools.UnitsAwareDataArray`
            The magnitude of the covariance.
        """


        da_ch = UADA(da_ch)
        da_ch = da_ch.assign_coords(calibrated_channel=channel)
        da_ch.attrs["units"] = str(
            (_fcdr_defs.FCDR_data_vars_props[self.name][2]["units"] *
             _fcdr_defs.FCDR_data_vars_props[other.name][2]["units"]))

        da_ch.name = f"u_{self.name:s}_{other.name:s}"
#        da.attrs["long_name"] = f"error covariance {self.magnitude.attrs['long_name']:s} with {other.magnitude.attrs['long_name']:s}"
        da_ch.attrs["short_name"] = da_ch.name
        da_ch.attrs["parameters"] = (str(self.parameter), str(other.parameter))

        da_old = self._covariances.get(other.parameter)
        if da_old is None:
            da = da_ch
        elif channel in da_old.coords["calibrated_channel"].values:
            da = da_old
        else:
            da = xarray.concat([da_old, da_ch], dim="calibrated_channel", compat="identical")
        
        self._covariances[other.parameter] = da
        if _set_other:
            other.set_covariance(self, channel, da_ch, _set_other=False)

    @property
    def covariances(self):
        """Holds covariances between different effects.

        This property holds covariances between different effects.

        It is not currently (2018-12-20) used.
        """

        return self._covariances

    @property
    def correlation_type(self):
        """Holds the type of correlation per scale.

        Property, 4-tuple of strings, or `CorrelationType` named tuple.
        Each member of the tuple must be one of:
            * "undefined"
            * "random"
            * "rectangular_absolute"
            * "triangular_relative"
            * "truncated_gaussian_relative"
            * "repeated_rectangles"
            * "repeated_truncated_gaussians"
        The four tuple elements refer to (and, in case of the named tuple,
        can be referred to as) ``within_scanline``, ``between_scanlines``,
        ``between_orbits``, and ``across_time``.
        """
        return self._corr_type

    @correlation_type.setter
    def correlation_type(self, v):
        if not isinstance(v, CorrelationType):
            v = CorrelationType(*v)
        for x in v:
            if not x in self._valid_correlation_types:
                raise ValueError("Unknown correlation type: {:s}. "
                    "Expected one of: {:s}".format(
                        x, ", ".join(self._valid_correlation_types)))
        self._corr_type = v
            
    _corr_scale = CorrelationScale(*[0]*4)
    @property
    def correlation_scale(self):
        """`CorrelationScale` namedtuple describing correlation scales

        This is a `CorrelationScale` namedtuple that is supposed to
        contain the magnitudes of the correlation along different types.
        The types are as for ``correlation_type`` above, but are not
        actually set by the CURUC recipes and this `correlation_scale` is
        not really used.
        """
        return self._corr_scale

    @correlation_scale.setter
    def correlation_scale(self, v):
        if not isinstance(v, CorrelationScale):
            v = CorrelationScale(*v)
        for x in v:
            if not isinstance(x, (numbers.Number, numpy.ndarray)):
                raise TypeError("correlation scale must be numeric, "
                    "found {:s}".format(type(v)))
        self._corr_scale = v

    def sensitivity(self, s="R_e"):
        """Get expression for sensitivity coefficient

        Normally starting at R_e, but can provide other.

        Parameters
        ----------

        s : str or `sympy.Symbol`
            Parameter for which to calculate the sensitivity coefficients.
            Defaults to `R_e`, i.e. the Earth radiance.

        Returns
        -------

        sympy.Expr
            The symbolic expression for the sensitivity coefficient.
        """

        return meq.calc_sensitivity_coefficient(s, self.parameter)

    def is_independent(self):
        """True if this effect is independent.

        An effect is independent if the correlation type is random for all
        types.
        """

        return all(x=="random" for x in self.correlation_type)

    def is_common(self):
        """True if this effect is common

        An effect is common if the correlation type is
        rectangular_absolute for all scales and the magnitude is infinite
        for all scales.

        Note that common effects are not considered in CURUC.
        """
        return all(x=="rectangular_absolute" for x in
                self.correlation_type) and all(numpy.isinf(i) for i in
                self.correlation_scale)

    def is_structured(self):
        """True if this effect is structured.

        An effect is structured if it is neither independent nor common.
        """

        return not self.is_independent() and not self.is_common()

    def calc_R_eΛlk(self, ds,
            sampling_l=1, sampling_e=1):
        return self.rmodel.calc_R_eΛlk(ds,
            sampling_l=sampling_l,
            sampling_e=sampling_e)
    calc_R_eΛlk.__doc__ = Rmodel.calc_R_eΛlk.__doc__

    def calc_R_lΛek(self, ds,
            sampling_l=1, sampling_e=1):
        return self.rmodel.calc_R_lΛek(ds,
            sampling_l=sampling_l,
            sampling_e=sampling_e)
    calc_R_lΛek.__doc__ = Rmodel.calc_R_lΛek.__doc__

    def calc_R_cΛpk(self, ds,
            sampling_l=1, sampling_e=1):
        return self.rmodel.calc_R_cΛpk(ds,
            sampling_l=sampling_l,
            sampling_e=sampling_e)
    calc_R_cΛpk.__doc__ = Rmodel.calc_R_cΛpk.__doc__

def effects() -> Mapping[sympy.Symbol, Set[Effect]]:
    """Return a copy of the dictionary with all effects per symbol.

    Returns
    -------

    Mapping[symbol, Set[Effect]]
        Mapping containing keys which are symbols in the measurement
        equation, and values with are a set of `Effect` instances.
    """
    return copy.deepcopy(Effect._all_effects)

#: Identity matrix for all channels (for uncorrelated)
_I = numpy.eye(19, dtype="f4")
#: Matrix of ones for all channels (for fully correlated)
_ones = numpy.ones(shape=(19, 19), dtype="f4")
#: Tuple for correlation description of fully random effects
_random = ("random",)*4
#: Tuple for correlation description of effects due to calibration
_calib = ("rectangular_absolute", "rectangular_absolute",
          "random", "triangular_relative")
#: Tuple for correlation description of common effects
_systematic = ("rectangular_absolute",)*4
#: Tuple for correlation magnitude of common effects
_inf = (numpy.inf,)*4

#: Effect object describing Earth counts noise
earth_counts_noise = Effect(name="C_Earth",
    description="noise on Earth counts",
    parameter=meq.symbols["C_E"],
    correlation_type=_random,
    unit=ureg.count,
    channel_correlations=_I,
    dimensions=["calibration_cycle"], # FIXME: update if interpolated (issue#10)
    rmodel=rmodel_random,
    ) 

#: Effect object describing space counts noise
space_counts_noise = Effect(name="C_space",
    description="noise on Space counts",
    parameter=meq.symbols["C_s"],
    correlation_type=_calib,
    unit=ureg.count,
    channel_correlations=_I,
    dimensions=["calibration_cycle"],
    rmodel=rmodel_calib)

#: Effect describing IWCT counts noise
IWCT_counts_noise = Effect(name="C_IWCT",
    description="noise on IWCT counts",
    parameter=meq.symbols["C_IWCT"],
    correlation_type=_calib,
    unit=ureg.count,
    channel_correlations=_I,
    dimensions=["calibration_cycle"],
    rmodel=rmodel_calib)

#: Effect describing SRF uncertainty
SRF_calib = Effect(name="SRF_calib",
    description="Spectral response function calibration",
    parameter=meq.symbols["νstar"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=ureg.nm,
    dimensions=(),
    channel_correlations=_I,
    rmodel=rmodel_common)

# This one does not fit in measurement equation, how to code?
#
#SRF_RtoBT = Effect(description="Spectral response function radiance-to-BT",
#    parameter=meq.symbols["T_b"],
#    correlation_type=_systematic,
#    correlation_scale=_inf,
#    unit=ureg.nm,
#    channel_correlations=_I)

#: Effect describing PRT counts noise
PRT_counts_noise = Effect(name="C_PRT",
    description="IWCT PRT counts noise",
    parameter=meq.symbols["C_PRT"],
    correlation_type=_calib,
    unit=ureg.count,
    dimensions=(),
    channel_correlations=_ones,
    rmodel=rmodel_calib_prt)

#: Effect describing PRT representation
IWCT_PRT_representation = Effect(
    name="O_TIWCT",
    description="IWCT PRT representation",
    parameter=meq.symbols["O_TIWCT"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=ureg.K,
    dimensions=(),
    channel_correlations=_ones,
    rmodel=rmodel_calib_prt)

#: Effect describing uncertainty due to IWCT PRT calibration
IWCT_PRT_counts_to_temp = Effect(
    name="d_PRT",
    description="IWCT PRT counts to temperature",
    parameter=meq.symbols["d_PRT"], # Relates to free_symbol but actual
        # parameter in measurement equation to be replaced relates to as
        # returned by typhon.physics.metrology.recursive_args; need to
        # translate in some good way (see also #129)
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=ureg.counts/ureg.K, # FIXME WARNING: see https://github.com/FIDUCEO/FCDR_HIRS/issues/43
    dimensions=(),
    channel_correlations=_ones,
    rmodel=rmodel_calib_prt)

#: Effect describing uncertainty due to IWCT type b
IWCT_type_b = Effect(
    name="O_TPRT",
    description="IWCT type B",
    parameter=sympy.IndexedBase(meq.symbols["O_TPRT"])[meq.symbols["n"],meq.symbols["m"]],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=ureg.K,
    dimensions=(),
    channel_correlations=_ones,
    rmodel=rmodel_calib_prt)
# set magnitude when I'm sure everything else has been set (order of
# kwargs not preserved before Python 3.6)
IWCT_type_b.magnitude=UADA(0.1, name="uncertainty", attrs={"units": "K"})

#: Helper for effects correlated within same detector
_blockmat = numpy.vstack((
            numpy.hstack((
                numpy.ones(shape=(12,12)),
                numpy.zeros(shape=(12,9)))),
            numpy.hstack((
                numpy.zeros(shape=(9,12)),
                numpy.ones(shape=(9,9))))))

#: Effect describing uncertainty due to non-linearity harmonsiation parameter
nonlinearity = Effect(
    name="a_2",
    description="Nonlinearity",
    parameter=meq.symbols["a_2"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=radiance_units["si"]/ureg.count**2,
    dimensions=(),
    channel_correlations=_blockmat,
    rmodel=rmodel_common)

#: Effect describing uncertainty due to emissity correction harmonsiation parameter
emissivitycorrection = Effect(
    name="a_3",
    description="Emissivity correction",
    parameter=meq.symbols["a_3"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=ureg.dimensionless,
    dimensions=(),
    channel_correlations=_ones,
    rmodel=rmodel_common)

#: Effect describing uncertainty due to self-emission bias harmonisation parameter
selfemissionbias = Effect(
    name="a_4",
    description="Self-emission bias",
    parameter=meq.symbols["a_4"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=ureg.dimensionless,
    dimensions=(),
    channel_correlations=_ones,
    rmodel=rmodel_common)

#: Effect describing uncertainty due to non-quadratic non-linearity
nonnonlinearity = Effect(
    name="O_Re",
    description="Wrongness of nonlinearity",
    parameter=meq.symbols["O_Re"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=radiance_units["ir"],
    dimensions=(),
    channel_correlations=nonlinearity.channel_correlations,
    rmodel=rmodel_common)

#: Effect describing uncertainty due to Earthshine model
Earthshine = Effect(
    name="Earthshine",
    description="Earthshine",
    parameter=meq.symbols["R_refl"],
    correlation_type=("rectangular_absolute", "rectangular_absolute",
          "repeated_rectangles", "triangular_relative"),
    channel_correlations=_blockmat,
    dimensions=(),
    unit=radiance_units["ir"],
    rmodel=rmodel_calib)

#: Effect describing uncertainty due to self-emission model
Rself = Effect(
    name="Rself",
    dimensions=("rself_update_time",),
    description="self-emission",
    parameter=meq.symbols["R_selfE"],
    correlation_type=("rectangular_absolute", "triangular_relative",
        "triangular_relative", "repeated_rectangles"),
    channel_correlations=_blockmat,
    unit=radiance_units["ir"],
    rmodel=rmodel_rself)

#: Effect describing uncertainty due to self-emission model parameters
Rselfparams = Effect(
    name="Rselfparams",
    description="self-emission parameters",
    parameter=Rself.parameter,
    correlation_type=Rself.correlation_type,
    channel_correlations=_blockmat,
    dimensions=(),
    unit=Rself.unit,
    rmodel=rmodel_rself)

#: Effect describing uncertainty due to unknown electronics effect
electronics = Effect(
    name="electronics",
    description="unknown electronics effects",
    parameter=meq.symbols["O_Re"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    channel_correlations=_blockmat,
    dimensions=(),
    unit=radiance_units["ir"],
    rmodel=rmodel_common)

#: Effect describing uncertainty due to "correlated noise"
unknown_periodic = Effect(
    name="extraneous_periodic",
    description="extraneous periodic signal",
    parameter=meq.symbols["O_Re"],
    #correlation_type=_systematic,
    #correlation_scale=_inf,
    #channel_correlations=_blockmat,
    dimensions=(),
    unit=radiance_units["ir"],
    rmodel=rmodel_periodicerror)

#: Effect describing uncertainty in band correction factor α
Δα = Effect(
    name="α",
    description="uncertainty in band correction factor α (ad-hoc)",
    parameter=meq.symbols["α"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    channel_correlations=_I,
    dimensions=(),
    unit="1",
    rmodel=rmodel_common)

#: Effect describing uncertainty in band correction factor β
Δβ = Effect(
    name="β",
    description="uncertainty in band correction factor β (ad-hoc)",
    parameter=meq.symbols["β"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    channel_correlations=_I,
    dimensions=(),
    unit="1/K",
    rmodel=rmodel_common),

#: Effect describing uncertainty in effective channel frequency
Δf_eff = Effect(
    name="f_eff",
    description="uncertainty in band correction centroid",
    parameter=meq.symbols["fstar"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    channel_correlations=_I,
    dimensions=(),
    unit="THz",
    rmodel=rmodel_common)
