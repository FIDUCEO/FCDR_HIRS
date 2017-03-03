"""For the uncertainty effects
"""

import abc
import collections
import copy
import numbers

import numpy
import xarray
import sympy

from typing import (Tuple, Mapping, Set)

from typhon.physics.units.common import (radiance_units, ureg)

from . import measurement_equation as meq
from . import _fcdr_defs

CorrelationType = collections.namedtuple("CorrelationType",
    ["within_scanline", "between_scanlines", "between_orbits",
    "across_time"])

CorrelationScale = collections.namedtuple("CorrelationScale",
    CorrelationType._fields)

class Effect:
    """For uncertainty effects.

    Needs to have (typically set on creation):
    
    - name: short name 
    - description: description of effect
    - parameter: what it relates to
    - unit: pint unit
    - pdf_shape: str, defaults to "Gaussian"
    - channels_affected: str, defaults "all"
    - correlation_type: what the form of the correlation is (4×)
    - channel_correlations: channel correlation matrix
    - dimensions: list of dimension names or None, which means same as
      parameter it relates to.

    Additionally needs to have (probably set only later):

    - magnitude
    - correlation_scale

    Sensitivity coefficients are calculated on-the-fly using the
    measurement_equation module.
    """

    _all_effects = {}
    name = None
    description = None
    parameter = None
    unit = None
    pdf_shape = "Gaussian"
    channels_affected = "all"
    channel_correlations = None
    dimensions = None

    def __init__(self, **kwargs):
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
        if not self.parameter in self._all_effects.keys():
            self._all_effects[self.parameter] = set()
        self._all_effects[self.parameter].add(self)

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

        This should be a DataArray with dimensions matching the dimensions
        of the FCDR.  Assumed to be constant along any others.  Note that
        this means the magnitude of the uncertainty is constant; it does
        not mean anything about the error correlation, which is treated
        separately.
        """
#        if self._magnitude.identical(self._init_magnitude):
#            logging.warning("uncertainty magnitude not set for " +
#                self.name)
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

        if not self.name.startswith("O_") or self.name in _fcdr_defs.FCDR_data_vars_props.keys():
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

    @property
    def correlation_type(self):
        """Form of correlation
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
        """Scale for correlation
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

        Returns sympy expression.
        """

        return meq.calc_sensitivity_coefficient(s, self.parameter)

def effects() -> Mapping[sympy.Symbol, Set[Effect]]:
    """Initialise a new dictionary with all effects per symbol.

    Returns: Mapping[symbol, Set[Effect]]
    """
    return copy.deepcopy(Effect._all_effects)

_I = numpy.eye(19, dtype="f4")
_ones = numpy.ones(shape=(19, 19), dtype="f4")
_random = ("random",)*4
_calib = ("rectangular_absolute", "rectangular_absolute",
          "random", "triangular_relative")
_systematic = ("rectangular_absolute",)*4
_inf = (numpy.inf,)*4

earth_counts_noise = Effect(name="C_Earth",
    description="noise on Earth counts",
    parameter=meq.symbols["C_E"],
    correlation_type=_random,
    unit=ureg.count,
    channel_correlations=_I,
    dimensions=["calibration_cycle"]) # FIXME: update if interpolated (issue#10)

space_counts_noise = Effect(name="C_space",
    description="noise on Space counts",
    parameter=meq.symbols["C_s"],
    correlation_type=_calib,
    unit=ureg.count,
    channel_correlations=_I,
    dimensions=["calibration_cycle"])

IWCT_counts_noise = Effect(name="C_IWCT",
    description="noise on IWCT counts",
    parameter=meq.symbols["C_IWCT"],
    correlation_type=_calib,
    unit=ureg.count,
    channel_correlations=_I,
    dimensions=["calibration_cycle"])

SRF_calib = Effect(name="SRF_calib",
    description="Spectral response function calibration",
    parameter=meq.symbols["νstar"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=ureg.nm,
    channel_correlations=_I)

# This one does not fit in measurement equation, how to code?
#
#SRF_RtoBT = Effect(description="Spectral response function radiance-to-BT",
#    parameter=meq.symbols["T_b"],
#    correlation_type=_systematic,
#    correlation_scale=_inf,
#    unit=ureg.nm,
#    channel_correlations=_I)

PRT_counts_noise = Effect(name="C_PRT",
    description="IWCT PRT counts noise",
    parameter=meq.symbols["C_PRT"],
    correlation_type=_calib,
    unit=ureg.count,
    channel_correlations=_ones)

IWCT_PRT_representation = Effect(
    name="O_TIWCT",
    description="IWCT PRT representation",
    parameter=meq.symbols["O_TIWCT"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=ureg.K,
    channel_correlations=_ones)

IWCT_PRT_counts_to_temp = Effect(
    name="d_PRT",
    description="IWCT PRT counts to temperature",
    parameter=meq.symbols["d_PRT"], # Relates to free_symbol but actual
        # parameter in measurement equation to be replaced relates to as
        # returned by typhon.physics.metrology.recursive_args; need to
        # translate in some good way
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=ureg.counts/ureg.K, # FIXME WARNING: see https://github.com/FIDUCEO/FCDR_HIRS/issues/43
    channel_correlations=_ones)

IWCT_type_b = Effect(
    name="O_TPRT",
    description="IWCT type B",
    parameter=meq.symbols["O_TPRT"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=ureg.K,
    channel_correlations=_ones)
# set magnitude when I'm sure everything else has been set (order of
# kwargs not preserved before Python 3.6)
IWCT_type_b.magnitude=xarray.DataArray(0.1, name="uncertainty", attrs={"units": "K"})

blockmat = numpy.vstack((
            numpy.hstack((
                numpy.ones(shape=(12,12)),
                numpy.zeros(shape=(12,9)))),
            numpy.hstack((
                numpy.zeros(shape=(9,12)),
                numpy.ones(shape=(9,9))))))
nonlinearity = Effect(
    name="nonlinearity",
    description="Nonlinearity",
    parameter=meq.symbols["a_2"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=radiance_units["ir"]/ureg.count**2,
    channel_correlations=blockmat)

nonnonlinearity = Effect(
    name="O_Re",
    description="Wrongness of nonlinearity",
    parameter=meq.symbols["O_Re"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=radiance_units["ir"],
    channel_correlations=nonlinearity.channel_correlations)

Earthshine = Effect(
    name="Earthshine",
    description="Earthshine",
    parameter=meq.symbols["R_refl"],
    correlation_type=("rectangular_absolute", "rectangular_absolute",
          "repeated_rectangles", "triangular_relative"),
    channel_correlations=blockmat,
    unit=radiance_units["ir"])

Rself = Effect(
    name="Rself",
    description="self-emission",
    parameter=meq.symbols["R_selfE"],
    correlation_type=("rectangular_absolute", "triangular_relative",
        "triangular_relative", "repeated_rectangles"),
    channel_correlations=blockmat,
    unit=radiance_units["ir"])

Rselfparams = Effect(
    name="Rselfparams",
    description="self-emission parameters",
    parameter=Rself.parameter,
    correlation_type=Rself.correlation_type,
    channel_correlations=blockmat,
    unit=Rself.unit)

electronics = Effect(
    name="electronics",
    description="unknown electronics effects",
    parameter=meq.symbols["O_Re"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    channel_correlations=blockmat,
    unit=radiance_units["ir"])

unknown_periodic = Effect(
    name="extraneous_periodic",
    description="extraneous periodic signal",
    parameter=meq.symbols["O_Re"],
    #correlation_type=_systematic,
    #correlation_scale=_inf,
    #channel_correlations=blockmat,
    unit=radiance_units["ir"])
