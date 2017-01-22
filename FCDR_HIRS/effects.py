"""For the uncertainty effects
"""

import abc
import numpy
import xarray
import collections

from typhon.physics.units.common import (radiance_units, ureg)

from . import measurement_equation as meq

CorrelationType = collections.namedtuple("CorrelationType",
    ["within_scanline", "between_scanlines", "between_orbits",
    "across_time"])

CorrelationScale = collections.namedtuple("CorrelationScale",
    CorrelationType._fields)

class Effect:
    """For uncertainty effects.

    Needs to have (typically set on creation):
    
    - name: description of effect
    - parameter: what it relates to
    - unit: pint unit
    - pdf_shape: str, defaults to "Gaussian"
    - channels_affected: str, defaults "all"
    - correlation_type: what the form of the correlation is (4×)
    - channel_correlations: channel correlation matrix

    Additionally needs to have (probably set only later):

    - magnitude
    - correlation_scale

    Sensitivity coefficients are calculated on-the-fly using the
    measurement_equation module.
    """

    all_effects = {}
    name = None
    parameter = None
    unit = None
    pdf_shape = "Gaussian"
    channels_affected = "all"
    channel_correlations = None

    def __init__(self, **kwargs):
        for (k, v) in kwargs.items():
            setattr(self, k, v)
        if not self.parameter in self.all_effects.keys():
            self.all_effects[self.parameter] = set()
        self.all_effects[self.parameter].add(self)

    def __setattr__(self, k, v):
        if not hasattr(self, k):
            raise AttributeError("Unknown attribute: {:s}".format(k))
        super().__setattr__(k, v)

    def __repr__(self):
        return "<Effect {!s}:{:s}>".format(self.parameter, self.name)

    # default: 0-dimensional nan
    _magnitude = xarray.DataArray(numpy.nan, name="uncertainty")
    @property
    def magnitude(self):
        """Magnitude of the uncertainty

        This should be a DataArray with dimensions matching the dimensions
        of the FCDR.  Assumed to be constant along any others.  Note that
        this means the magnitude of the uncertainty is constant; it does
        not mean anything about the error correlation, which is treated
        separately.
        """
        return self._magnitude

    @magnitude.setter
    def magnitude(self, v):
        if isinstance(v, xarray.DataArray):
            self._magnitude = v
        else:
            raise TypeError("uncertainty magnitude must be DataArray. "
                "Found {:s}".format(type(v)))

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
            
    _corr_scale = CorrelationScale(*[xarray.DataArray(0)]*4)
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
            if not numpy.isinf(x) or isinstance(x, xarray.DataArray):
                raise TypeError("correlation scale must be inf or DataArray, "
                    "found {:s}".format(type(v)))
        self._corr_scale = v

    def sensitivity(self, s="R_e"):
        """Get expression for sensitivity coefficient

        Normally starting at R_e, but can provide other.

        Returns sympy expression.
        """

        return meq.calc_sensitivity_coefficient(s, self.parameter)


_I = numpy.eye(19, dtype="f2")
_ones = numpy.ones(shape=(19, 19), dtype="f2")
_random = ("random",)*4
_calib = ("rectangular_absolute", "rectangular_absolute",
          "random", "triangular_relative")
_systematic = ("rectangular_absolute",)*4
_inf = (numpy.inf,)*4

earth_counts_noise = Effect(name="noise on Earth counts",
    parameter=meq.symbols["C_E"],
    correlation_type=_random,
    unit=ureg.count,
    channel_correlations=_I)

space_counts_noise = Effect(name="noise on Space counts",
    parameter=meq.symbols["C_s"],
    correlation_type=_calib,
    unit=ureg.count,
    channel_correlations=_I)

IWCT_counts_noise = Effect(name="noise on IWCT counts",
    parameter=meq.symbols["C_IWCT"],
    correlation_type=_calib,
    unit=ureg.count,
    channel_correlations=_I)

SRF_calib = Effect(name="Spectral response function calibration",
    parameter=meq.symbols["φ"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=ureg.nm,
    channel_correlations=_I)

# This one does not fit in measurement equation, how to code?
#
#SRF_RtoBT = Effect(name="Spectral response function radiance-to-BT",
#    parameter=meq.symbols["T_b"],
#    correlation_type=_systematic,
#    correlation_scale=_inf,
#    unit=ureg.nm,
#    channel_correlations=_I)

PRT_counts_noise = Effect(name="IWCT PRT counts noise",
    parameter=meq.symbols["C_PRT"],
    correlation_type=_calib,
    unit=ureg.count,
    channel_correlations=_ones)

IWCT_PRT_representation = Effect(
    name="IWCT PRT representation",
    parameter=meq.symbols["O_TIWCT"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=ureg.K,
    channel_correlations=_ones)

IWCT_PRT_counts_to_temp = Effect(
    name="IWCT PRT counts to temperature",
    parameter=meq.symbols["d_PRT"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=ureg.counts/ureg.K, # FIXME WARNING: see https://github.com/FIDUCEO/FCDR_HIRS/issues/43
    channel_correlations=_ones)

IWCT_type_b = Effect(
    name="IWCT type B",
    parameter=meq.symbols["O_TIWCT"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=ureg.K,
    magnitude=xarray.DataArray(0.1, name="uncertainty"),
    channel_correlations=_ones)

nonlinearity = Effect(
    name="Nonlinearity",
    parameter=meq.symbols["a_2"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=radiance_units["ir"]/ureg.count**2,
    channel_correlations=
        numpy.vstack((
            numpy.hstack((
                numpy.ones(shape=(12,12)),
                numpy.zeros(shape=(12,9)))),
            numpy.hstack((
                numpy.zeros(shape=(9,12)),
                numpy.ones(shape=(9,9)))))))

nonnonlinearity = Effect(
    name="Wrongness of nonlinearity",
    parameter=meq.symbols["O_Re"],
    correlation_type=_systematic,
    correlation_scale=_inf,
    unit=radiance_units["ir"],
    channel_correlations=nonlinearity.channel_correlations)

Earthshine = Effect(
    name="Earthshine",
    parameter=meq.symbols["R_refl"],
    correlation_type=("rectangular_absolute", "rectangular_absolute",
          "repeated_rectangles", "triangular_relative"),
    unit=radiance_units["ir"])

#Rself = Effect(
#    name="self-emission",
#    parameter=meq.symbols["Rself"],
