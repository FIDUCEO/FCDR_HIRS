"""Module with metrology-related functions

This module contains metrology-related functions.  Some of those are
helpers often used in conjunction with the functions in the
`measurement_equation` module or with `typhon.physics.metrology`, but this
module also contains functions related to the ``CURUC`` recipes.  The
highest-level functions a user will want to use are `allocate_curuc` and
`apply_curuc`.
"""

import math
import functools
import operator
import collections
import itertools
import logging
import numbers
import copy

from typing import (List, Dict, Tuple, Deque, Optional)

import numexpr
import numpy
import scipy.optimize
import scipy.interpolate
import xarray
import sympy

import typhon.physics.metrology
from . import effects
from . import measurement_equation as me
from .fcdr import make_debug_fcdr_dims_consistent
from . import _fcdr_defs
from .exceptions import FCDRError

logger = logging.getLogger(__name__)

def evaluate_uncertainty(e, unset="raise"):
    """Evaluate uncertainty for expression.

    This function does not work and should not be used.  Use
    `fcdr.HIRSFCDR.calc_u_for_variable` instead.

    Requires effects tables to be populated including quantified
    uncertainties.  Any variable which does not have any effects
    associated with it is assumed to have uncertainty 0.

    Parameters
    ----------

    e : sympy.Symbol
        Symbol for which to calculate the uncertainty.
    unset : str, optional
        If set to "raise", which is the default, any effects with
        undefined uncertainty will result in raising `ValueError`.

    Returns
    -------

    dict
        Dictionary with uncertainties
    set
        Effects without uncertainty quantified
    """

    e = me.recursive_substitution(e)
    u_e = typhon.physics.metrology.express_uncertainty(e)
    no_uncertainty = set()
    all_effects = effects.effects()
    magnitudes = {}
    for sym in u_e.free_symbols:
        magnitudes[sym] = []
        if sym in all_effects:
            for effect in all_effects[sym]:
                if numpy.isnan(effect.magnitude).all():
                    if unset=="raise":
                        raise ValueError("No magnitude set for: {!s}".format(
                            effect))
                else:
                    magnitudes[sym].append(effect.magnitude)
        else:
            no_uncertainty.add(sym)
    # FIXME: sum over magnitudes
    return magnitudes, no_uncertainty

def prepare():
    """**DO NOT USE**

    I don't know what this does and I don't think it works.
    """
    
    # before proceeding, need to substitute N (no. PRTs) and K (no.
    # components in IWCT PRT polynomial) and expand sum, such that I can
    # calculate the uncertainty
    ee = e.subs("K", 3).subs("N", 3).doit()
    # when lambdify()ing, can only use base names, but I want to keep the
    # indexed variables as d_PRT[3, 1] etc, so make it dPRT31 instead.
    newnames = {a: str(a).replace('[','').replace(', ','_').replace(']','') for a in args}



##### Functions related to Chris recipes' document #####
    
#    Chris Merchant, Emma Woolliams and Jonathan Mittaz,
#    Uncertainty and Error Correlation Quantification for FIDUCEO “easy-
#    FCDR” Products: Mathematical Recipes.  Hereunder referred to as
#    "Recipes".  Section and page numbers refer to document version 0.9.3.

# Within generate_fcdr.FCDRGenerator.get_piece, the fcdr object stored in
# self.fcdr collects effects in fcdr._effects.  Once this is populated, it
# contains:
#
#   Dict[sympy.Symbol, Set[effects.Effect]]
#
# which translates nicely to the recipes:  Each Symbol corresponds to a
# term j in the measurement equation, and each Effect to an effect k
# (either independent i, or structured s).  After the uncertainties have
# been calculated, those Effect objects have been populated with
# uncertainties in their 'magnitude' attribute.  Note that terms without
# associated uncertainties are not included here.
#
# Sensitivity expressions are included with the effects, but at this
# point, not their values.  The values of sensitivities are stored in
# sensRe but only per sub-measurement-equation and thus need to be
# multiplied together to get a direct "R_e-to-end" sensitivity.
# I should be already doing that when I propagate individual components
# for debugging purposes?
#

def calc_S_from_CUR(R_xΛyt: xarray.DataArray,
                    U_xΛyt_diag: xarray.DataArray,
                    C_xΛyt_diag: xarray.DataArray,
                    per_channel: bool=None):
    """Calculate S_esΛl, S_lsΛe, S_ciΛp, or S_csΛp

    Calculate either of those:
    
    - S_esΛl, the total cross-element error covariance from the structured
      effects per channel evaluated at a single line.  To get S_esΛl, pass
      in R_eΛls, U_eΛls, and C_eΛlj.
    - S_lsΛe, the total cross-line error covariance from the structured
      effects per channel evaluated at a single element.  To get S_lsΛe,
      pass in R_lΛes, U_lΛes, and C_lΛej.
    - S_ciΛp, the total cross-channel error covariance from the
      independent effects evaluated at a single pixel.  To get S_ciΛp, pass
      in R_cΛpi, U_cΛpi, and C_cΛpj.
    - S_csΛp, like S_ciΛp but for structured effects.

    NB: Λ indicates superscript.

    Follows recipe from:
    
    Chris Merchant, Emma Woolliams and Jonathan Mittaz,
    Uncertainty and Error Correlation Quantification for FIDUCEO “easy-
    FCDR” Products: Mathematical Recipes.  Hereunder referred to as
    "Recipes".  Section and page numbers refer to document version 0.9.3.

    As defined by §3.3.3 and §3.3.6.

    Parameters
    ----------

    R_eΛls, R_lΛes, R_cΛpi, or R_cΛps : xarray.DataArray
        with dimensions [n_c, n_s, n_e, n_l, n_l] or [n_c, n_s, n_l,
        n_e, n_e] or .
        For each channel (c) and each effect (k), either a collection
        of cross-element error correlation matrices for each line, or
        of cross-line error correlation matrices for each element.
        Defined by §3.2.3.

    Diagonals of one of U_eΛls, U_lΛes, U_cΛpi, or U_cΛps : xarray.DataArray
        Same dimensions as previous but minus the final dimension,
        because it only stores the diagonals.
        Considering §3.2.6, consider that the final dimension shows
        the diagonals of any U_eΛls, U_lΛes, U_cΛpi, or U_cΛps.
        Defined by §3.2.6.

    One of C_eΛls, C_lΛes, C_cΛps, C_cΛpi : xarray.DataArray
        Contains the sensitivity diagonals *per effect*.  Although
        sensitivity is defined per term and not per effect, I need
        them per effect.  Most terms have exactly one effect defined
        anyway.  Dimensions therefore the same as U_eΛls and friends.
        Defined by §3.2.9.

    per_channel : bool, optional
        Boolean "per_channel".  If not given, this will be inferred from
        the presence of a dimension "n_c" within the leading ndim-1
        dimensions of U.

    You probably want to vectorise this over an entire image.  Probable
    dimensions:

    One per term:
    
        * C_eΛls [n_c, n_s, n_l, n_e, n_e] (last 2 diagonal, not explicitly calculated)
        * C_lΛes [n_c, n_s, n_e, n_l, n_l] (last 2 diagonal)
        * C_cΛps [n_s, n_l, n_e, n_c, n_c] (last 2 diagonal)
        * C_cΛpi [n_i, n_l, n_e, n_c, n_c] (last 2 diagonal)
        
    One per effect:
        
        * S_esΛl [n_c, n_l, n_e, n_e]
        * S_lsΛe [n_c, n_e, n_l, n_l]
        * S_ciΛp [n_l, n_e, n_c, n_c]
        * S_csΛp [n_l, n_e, n_c, n_c]
        
        * R_eΛls [n_c, n_l, n_s, n_e, n_e]
        * R_lΛes [n_c, n_e, n_s, n_l, n_l]
        * R_cΛpi [n_l, n_e, n_i, n_c, n_c]
        * R_cΛps [n_l, n_e, n_s, n_c, n_c]
        
        * U_eΛls [n_c, n_l, n_s, n_e, n_e] (last 2 diagonal)
        * U_lΛes [n_c, n_e, n_s, n_l, n_l] (last 2 diagonal)
        * U_cΛpi [n_l, n_e, n_i|j, n_c, n_c] (last 2 diagonal)
        * U_cΛps [n_l, n_e, n_s|j, n_c, n_c] (last 2 diagonal)

    One total:
        
        * S_es [n_c, n_e, n_e]
        * S_ls [n_c, n_l, n_l]
        * S_ci [n_c, n_c]
        * S_cs [n_c, n_c]
        
        * R_es [n_c, n_e, n_e]
        * R_ls [n_c, n_l, n_l]
        * R_ci [n_c, n_c]
        * R_cs [n_c, n_c]

    Returns
    -------

    S_esΛl, S_lsΛe, S_ciΛp, or S_csΛp: numpy.ndarray
        as described above
    """

    if not C_xΛyt_diag.dims == U_xΛyt_diag.dims == R_xΛyt.dims[:-1]:
        raise ValueError("R, U, C wrong dimensions")

    if per_channel is None:
        per_channel = "n_c" in U_xΛyt_diag.dims[:-1]

    # can work with diagonals only:
    #
    # U @ R @ U.T == (diag(U)[:, newaxis]*R)*diag(U)
    #
    # Prepare for numexpr
    U = U_xΛyt_diag.values[..., numpy.newaxis]
    UT = U.swapaxes(-1, -2)
    C = C_xΛyt_diag.values[..., numpy.newaxis]
    CT = C.swapaxes(-1, -2)
    R = R_xΛyt
    S_xΛyt = numexpr.evaluate("C * U * R * UT * CT")
    S_xtΛy = S_xΛyt.sum(int(per_channel))
    newcoords = copy.deepcopy(R_xΛyt.coords)
    del newcoords[R_xΛyt.dims[int(per_channel)]]
    return xarray.DataArray(S_xtΛy,
        dims=R_xΛyt.dims[0:int(per_channel)] + R_xΛyt.dims[(1+int(per_channel)):],
        coords=newcoords)

def calc_S_xt(S_xtΛy: List[numpy.ndarray],
              per_channel: bool=None) -> numpy.ndarray:
    """Calculate S_es, S_ls, S_ci, or S_cs

    Calculate either of:
    
    - S_es, the average cross-element error covariance from the
      structured effects per channel (Eq. 20)
    - S_ls, the average cross-line error covariance from the structured
      effects per channel (Eq. 24)
    - S_ci, the average cross-channel error covariance matrix from the
      spatially independent effects, per channel (Eq. 27)
    - S_cs, the average cross-channel error covariance matrix from the
      structured effects, per channel (Eq. 30)

    Follows recipe with same document source as calc_S_from_CUR.

    Parameters
    ----------

    S_esΛl, S_lsΛe, S_ciΛp, or S_csΛp : List[numpy.ndarray]
        List of values of relevant matrix, or ndarray with outermost dimension
        being the scanline.  You can obtain those from calc_S_from_CUR

    per_channel : bool
        Boolean "per_channel".  If not given, this will be inferred from
        the presence of a dimension "n_c" within the leading ndim-1
        dimensions of U.

    Returns
    -------

    S_es, S_el, S_ci, or S_cs : numpy.ndarray
        as described above

    """

    if per_channel is None:
        per_channel = S_xtΛy.dims[0] == "n_c"
    return S_xtΛy.mean(S_xtΛy.dims[int(per_channel)]) # Eq. 20, 24, 27, or 30 (FIXME: or n_e or n_p)

def calc_R_xt(S_xt: numpy.ndarray):
    """Calculate R_es, R_ls, R_ci, or R_cs

    Calculate either of:
    
    - R_es, the cross-element radiance error correlation matrix,
      structured effects, per channel (Eq. 22)
    - R_ls, the cross-line radiance error correlation matrix, structured
      effects, per channel (Eq. 26)
    - R_ci, cross-channel error correlation matrix, independent effects (Eq. 29)
    - R_cs, cross-channel error correlation matrix, structured effects (Eq. 32)

    Follows recipe with same document source as calc_S_from_CUR.

    Parameters
    ----------

    S_es or S_el : numpy.ndarray

        Average cross-element error covariance from the structured
        effects per channel.  Can be obtained from calc_S_xt.

    Returns
    -------

    R_es or R_el : numpy.ndarray, as described. 
    """

    U_xt_diag = numpy.sqrt(numpy.diagonal(S_xt, axis1=-2, axis2=-1))
    dUi = (1/U_xt_diag)[..., numpy.newaxis]
    dUiT = dUi.swapaxes(-1, -2)
    R_xt = numexpr.evaluate("dUi * S_xt * dUiT") # equivalent to Ui@S@Ui.T when written fully
    return xarray.DataArray(R_xt, dims=S_xt.dims, coords=S_xt.coords)

def calc_Delta_x(R_xt: xarray.DataArray,
             return_vector: bool):
    """Calculate optimum Δ_e or Δ_l

    Calculate optimum correlation length scale, either across elements,
    Δ_e or across lines, Δ_l.
    Structured effects, per channel.

    Recipe source as for calc_S_from_CUR, now §3.3.4.

    Parameters
    ----------

    R_es or R_ls : xarray.DataArray

        Either cross-element or cross-line radiance error correlation matrix, structured
        effects, per channel.  Can be obtained from calc_R_xt.

    return_vector : bool

        If true, return the full vector of average correlation length
        scales per distance.

    Returns
    -------

    popt : xarray.DataArray

        xarray.DataArray object containing in one column the optimal
        correlation length scales, and in the other column the
        corresponding covariances, ``pcov``, such as returned by
        `scipy.optimize.curve_fit`.  For each channel.

    r_xΔ : xarray.DataArray

        Only returned if return_vector is True.  xarray.DataArray
        object that describes, for each element or line separation,
        the average correlation length.  Note that this is still
        subsampled by sampling_l and/or sampling_e.  For each channel.
        Note that the dimension along pixel is always Δ_p, changed
        from Δ_e or Δ_l.  That may change in the future.
    """

    dim = R_xt.dims[-1]
    Δ_ref = xarray.DataArray(
        R_xt.coords[dim],
        dims=(dim,))
    r_xΔ = xarray.DataArray(
        numpy.array(
            [numpy.ma.masked_invalid(
                numpy.diagonal(R_xt, i, -2, -1)).mean(-1)
                for i in range(R_xt.shape[-1])]),
        dims=("Δp", "n_c"),
        coords={"Δp": R_xt.coords[R_xt.dims[1]].values,
                "n_c": R_xt.coords["n_c"]})

    def f(Δ, Δ_e):
        return numpy.exp(-Δ/Δ_e)

    # I don't suppose I can vectorise this one…
    popt = xarray.DataArray(
        numpy.array([scipy.optimize.curve_fit(f, Δ_ref, r_xΔ.sel(n_c=c),
            p0=1) for c in r_xΔ["n_c"]])[..., 0],
        dims=("n_c", "val"),
        coords={"n_c": r_xΔ["n_c"],
                "val": ["popt", "pcov"]})

    if return_vector:
        return (popt, r_xΔ)
    else:
        return popt


def allocate_curuc(n_c, n_l, n_e, n_s, n_i, sampling_l=1, sampling_e=1):
    """Allocate empty xarray DataArrays for CURUC recipes.

    Allocate empty xarray DataArrays that are needed to calculate all
    necessary CURUC recipe inputs.  You will need to fill all resulting
    DataArrays.  The DataArrays are subsampled according to the desired
    sampling.

    Parameters
    ----------

    n_c : int

        Number of channels.

    n_l : int

        Number of scanlines.

    n_e : int

        Number of elements in a scanline.

    n_s : int

        Number of systematic effects.

    n_i : int

        Number if independent effects.

    sampling_l : int

        Sampling rate per line.  Defaults to 1.

    sampling_e : int

        Sampling rate per element.  Defaults to 1.

    Returns
    -------

    Tuple with 13 elements:

    R_eΛls : (n_c, n_s, n_l, n_e, n_e) xarray.DataArray

        Cross-element correlation matrix for each line, structured effect,
        and channel.

    R_lΛes : (n_c, n_s, n_e, n_l, n_l) xarray.DataArray

        Cross-line correlation matrix for each element, structured effect,
        and channel.

    R_cΛpi : (n_i, n_l, n_e, n_c, n_c) xarray.DataArray

        Cross-channel correlation matrix for each element, line, and
        independent effect.  To get the same data in the form [n_i, n_p,
        n_c, n_c], call
        ``typhon.utils.stack_xarray_repdim(R_cΛpi, n_p=("n_l", "n_e"))``.

    R_cΛps : (n_s, n_l, n_e, n_c, n_c) xarray.DataArray

        Cross-channel correlation matrix for each element, line, and
        structured effect.  Get a view per pixel analogously to R_cΛpi
        above.

    U_eΛls_diag : (n_c, n_s, n_l, n_e) xarray.DataArray

        Diagonals for the uncertainties corresponding to the cross-element
        correlation matrix for each line, structured effect, and channel.
        Note that this DataArray is a view of the same memory as
        U_lΛes_diag or U_cΛps_diag, so if you fill one the other ones will
        appear filled as well.

    U_lΛes_diag : (n_c, n_s, n_e, n_l) xarray.DataArray

        Diagonals for the uncertainties corresponding to the cross-line
        correlation matrix for each element, structured effect, and
        channel.  Note that this DataArray is a view of the same memory as
        U_eΛls_diag or U_cΛps_diag, so if you fill one the other ones will
        be filled as well.

    U_cΛps_diag : (n_l, n_e, n_s, n_c) xarray.DataArray

        Diagonals for the uncertainties corresponding to the cross-channel
        correlation matrix for each element, line, and structured effect.
        Note that this DataArray is a view of the same memory as
        U_lΛes_diag or U_eΛls_diag, so if you fill one the other ones will
        appear filled as well.

    U_cΛpi_diag : (n_i, n_l, n_e, n_c) xarray.DataArray

        Diagonals for the uncertainties corresponding to the cross-channel
        correlation matrix for each element, line, and independent effect.

    C_eΛls_diag : (n_c, n_s, n_l, n_e) xarray.DataArray

        Diagonals for the sensitivities corresponding to the cross-element
        correlation matrix for each line, structured effect, and channel.
        Note that this DataArray is a view of the same memory as
        C_lΛes_diag and C_cΛps_diag.

    C_lΛes_diag : (n_c, n_s, n_e, n_l) xarray.DataArray

        Diagonals for the sensitivities corresponding to the cross-line
        correlation matrix for each element, structured effect, and
        channel.  Note that this DataArray is a view of the same memory as
        C_eΛls_diag and C_cΛps_diag.

    C_cΛps_diag : (n_s, n_l, n_e, n_c) xarray.DataArray

        Diagonals for the sensitivities corresponding to the cross-channel
        correlation matrix for each element, line, and structured effect.
        Note that this DataArray is a view of the same memory as
        C_eΛls_diag and C_lΛes_diag.

    - C_cΛpi_diag : (n_i, n_l, n_e, n_c) xarray.DataArray

        Diagonals for the sensitivities corresponding to the cross-channel
        correlation matrix for each element, line, and independent effect.

    - all_coords : dict

        Dictionary with coordinates for n_c, n_s, n_l, n_e, n_i.  In
        principle redundant as all of those coordinates are also contained
        in the different `xarray.DataArray` objects, but only here are
        they all in one place.
    """

    ## Allocation ##

    logger.debug("Allocating arrays for correlation calculations")

    all_coords = {
        "n_c": numpy.arange(1, n_c+1),
        "n_s": numpy.arange(0, n_s),
        "n_l": numpy.arange(0, n_l, sampling_l),
        "n_e": numpy.arange(0, n_e, sampling_e),
        "n_i": numpy.arange(0, n_i)}

    R_eΛls = xarray.DataArray(
        numpy.zeros((n_c, n_s, 
            math.ceil(n_l/sampling_l),
            math.ceil(n_e/sampling_e),
            math.ceil(n_e/sampling_e)), dtype="f4"),
        dims=("n_c", "n_s", "n_l", "n_e", "n_e"),
        coords={k:v for (k, v) in all_coords.items()
                if k in {"n_c", "n_s", "n_l", "n_e"}})

    R_lΛes = xarray.DataArray(
        numpy.zeros((n_c, n_s, 
            math.ceil(n_e/sampling_e),
            math.ceil(n_l/sampling_l),
            math.ceil(n_l/sampling_l)), dtype="f4"),
        dims=("n_c", "n_s", "n_e", "n_l", "n_l"),
        coords={k:v for (k, v) in all_coords.items()
                if k in {"n_c", "n_s", "n_l", "n_e"}})

    R_cΛpi = xarray.DataArray(
        numpy.zeros(
           (n_i,
            math.ceil(n_l/sampling_l),
            math.ceil(n_e/sampling_e),
            n_c,
            n_c), dtype="f4"),
        dims=("n_i", "n_l", "n_e", "n_c", "n_c"),
        coords={k:v for (k, v) in all_coords.items()
                if k in {"n_c", "n_i", "n_l", "n_e"}})

    R_cΛps = xarray.DataArray(
        numpy.zeros(
           (n_s,
            math.ceil(n_l/(sampling_l)),
            math.ceil(n_e/(sampling_e)),
            n_c,
            n_c), dtype="f4"),
        dims=("n_s", "n_l", "n_e", "n_c", "n_c"),
        coords={k:v for (k, v) in all_coords.items()
                if k in {"n_c", "n_s", "n_l", "n_e"}})

    # store only diagonals for optimised memory consumption and
    # calculation speed
    U_eΛls_diag = xarray.DataArray(
        numpy.zeros((n_c, n_s,
            math.ceil(n_l/sampling_l),
            math.ceil(n_e/sampling_e)), dtype="f4"),
        dims=("n_c", "n_s", "n_l", "n_e"),
        coords={k:v for (k, v) in all_coords.items()
                if k in {"n_c", "n_s", "n_l", "n_e"}})

    U_lΛes_diag = U_eΛls_diag.transpose("n_c", "n_s", "n_e", "n_l")
    U_cΛps_diag = U_eΛls_diag.transpose("n_l", "n_e", "n_s", "n_c")

    U_cΛpi_diag = xarray.DataArray(
        numpy.zeros((
            n_i,
            math.ceil(n_l/sampling_l),
            math.ceil(n_e/sampling_e),
            n_c), dtype="f4"),
        dims=("n_i", "n_l", "n_e", "n_c"),
        coords={k:v for (k, v) in all_coords.items()
                if k in {"n_c", "n_i", "n_l", "n_e"}})
    # Equivalent to:
    #
    # U_lΛes_diag = xarray.DataArray(
    #     numpy.zeros((n_c, n_s,
    #         math.ceil(n_e/sampling_e),
    #         math.ceil(n_l/sampling_l)), dtype="f4"),
    #     dims=("n_c", "n_s", "n_e", "n_l")) # last n_l superfluous
    #
    # U_eΛls_diag and U_lΛes_diag are views of the same data, thus saving
    # memory.

    # Actual dimension of C_eΛlj would be [n_c, n_j, n_l, n_e, n_e], but
    # I'm storing it on the dimensions of [n_c, n_k, n_l, n_e] (diagonals)
    # so I can apply it directly with the corresponding U_eΛlk and R_eΛlk.
    C_eΛls_diag = xarray.DataArray(
        numpy.zeros((n_c, n_s, math.ceil(n_l/sampling_l),
            math.ceil(n_e/sampling_e)), dtype="f4"),
        dims=("n_c", "n_s", "n_l", "n_e"), # last n_e superfluous
        coords={k:v for (k, v) in all_coords.items()
                if k in {"n_c", "n_s", "n_l", "n_e"}})
    C_lΛes_diag = C_eΛls_diag.transpose("n_c", "n_s", "n_e", "n_l")
    C_cΛps_diag = C_eΛls_diag.transpose("n_s", "n_l", "n_e", "n_c")

    C_cΛpi_diag = xarray.DataArray(
        numpy.zeros_like(U_cΛpi_diag.values),
        dims=("n_i", "n_l", "n_e", "n_c"),
        coords={k:v for (k, v) in all_coords.items()
                if k in {"n_c", "n_i", "n_l", "n_e"}})

    return (R_eΛls, R_lΛes, R_cΛpi, R_cΛps,
            U_eΛls_diag, U_lΛes_diag, U_cΛps_diag, U_cΛpi_diag,
            C_eΛls_diag, C_lΛes_diag, C_cΛps_diag, C_cΛpi_diag, all_coords)

def apply_curuc(R_eΛls, R_lΛes, R_cΛpi, R_cΛps,
        U_eΛls_diag, U_lΛes_diag, U_cΛps_diag, U_cΛpi_diag,
        C_eΛls_diag, C_lΛes_diag, C_cΛps_diag, C_cΛpi_diag,
        all_coords, brokenchan, brokenline, return_vectors=False,
        interpolate_lengths=False, cutoff_l=None, cutoff_e=None,
        return_locals=False):
    """Apply CURUC recipes.

    Arguments correspond to the ones returned by allocate_curuc.


    Parameters
    ----------

    R_eΛls : (n_c, n_s, n_l, n_e, n_e) xarray.DataArray

        Cross-element correlation matrix for each line, structured effect,
        and channel.

    R_lΛes : (n_c, n_s, n_e, n_l, n_l) xarray.DataArray

        Cross-line correlation matrix for each element, structured effect,
        and channel.

    R_cΛpi : (n_i, n_l, n_e, n_c, n_c) xarray.DataArray

        Cross-channel correlation matrix for each element, line, and
        independent effect.

    R_cΛps : (n_s, n_l, n_e, n_c, n_c) xarray.DataArray

        Cross-channel correlation matrix for each element, line, and
        structured effect.

    U_eΛls_diag : (n_c, n_s, n_l, n_e) xarray.DataArray

        Diagonals for the uncertainties corresponding to the cross-element
        correlation matrix for each line, structured effect, and channel.

    U_lΛes_diag : (n_c, n_s, n_e, n_l) xarray.DataArray

        Diagonals for the uncertainties corresponding to the cross-line
        correlation matrix for each element, structured effect, and
        channel.

    U_cΛps_diag : (n_l, n_e, n_s, n_c) xarray.DataArray

        Diagonals for the uncertainties corresponding to the cross-channel
        correlation matrix for each element, line, and structured effect.

    U_cΛpi_diag : (n_i, n_l, n_e, n_c) xarray.DataArray

        Diagonals for the uncertainties corresponding to the cross-channel
        correlation matrix for each element, line, and independent effect.

    C_eΛls_diag : (n_c, n_s, n_l, n_e) xarray.DataArray

        Diagonals for the sensitivities corresponding to the cross-element
        correlation matrix for each line, structured effect, and channel.

    C_lΛes_diag : (n_c, n_s, n_e, n_l) xarray.DataArray

        Diagonals for the sensitivities corresponding to the cross-line
        correlation matrix for each element, structured effect, and
        channel.

    C_cΛps_diag : (n_l, n_e, n_s, n_c) xarray.DataArray

        Diagonals for the sensitivities corresponding to the cross-channel
        correlation matrix for each element, line, and structured effect.

    C_cΛpi_diag : (n_i, n_l, n_e, n_c) xarray.DataArray

        Diagonals for the sensitivities corresponding to the cross-channel
        correlation matrix for each element, line, and independent effect.

    all_coords : dict

        Dictionary with coordinates for n_c, n_s, n_l, n_e, n_i.

    brokenchan : (n_c,) xarray.DataArray
        
        xarray.DataArray, dtype bool, 1-D, dimension "n_c", True for
        channels that should be skipped.

    brokenline : (n_l,) xarray.DataArray

        xarray.DataArray, dtype bool, 1-D, dimension "n_l", True for
        lines that should be skipped.

    return_vectors : bool, optional

        Optional, defaults to False.  If True, in addition of returning
        optimal length scales, also return the full vectors with average
        correlation per separation length.

    interpolate_lengths : bool, optional

        Only needed if return_vectors is True.  Interpolate skipped lines.
        If False, average correlation is only given for lengths according
        to the sampling interval.  For example, with sampling_l=5, it's
        only given every 5 lines.  If True, a spline interpolation is
        applied and average correlation length is returned at every
        separation.

    cutoff_l : int, conditionally optional

        Only needed if return_vectors is True.  Cutoff lengths for
        lengths.  Note that `apply_curuc` may not know that the total
        n_l and n_e of the image is, because sampling may have cut off
        the tail.  Therefore, if you want to get full vectors returned,
        you must pass a value here.  I suggest you pass the same value you
        put in to n_l when calling `allocate_curuc`.

    cutoff_e : int, conditionally optional

        As cutoff_l, but for number of elements.

    return_locals : bool, optional

        If True, return full locals() dictionary.  This is very ugly.
        Please don't be like Gerrit who actually used this for a plot.

    Returns
    -------

    Δ_l_all : (n_c) xarray.DataArray

        Cross-line correlation length scale for each channel.

    Δ_e_all : (n_c) xarray.DataArray

        Cross-element correlation length scale for each channel.

    R_ci : (n_c, n_c) xarray.DataArray

        Cross-channel correlation matrix for independent effects.

    R_cs : (n_c, n_c) xarray.DataArray

        Cross-channel correlation matrix for structured effects.

    Δ_l_all_full : (n_c, n_l) xarray.DataArray

        Only returned if return_vectors input argument is True.

        Average cross-line correlation length for each channel and
        length.

    Δ_e_all_full : (n_c, n_e) xarray.DataArray

        Only returned if return_vectors input argument is True.

        Average cross-line correlation length for each channel and
        element.

    locals() : Dict
        
        Only returned if ``return_locals`` is True.  Do not do this.
        Please.  It's so ugly.
        
    """


    # FIXME: verify input correctness?

    n_c = brokenchan.size

    ## Apply recipes ##

    S_lsΛe = calc_S_from_CUR(R_lΛes, U_lΛes_diag, C_lΛes_diag)
    S_ls = calc_S_xt(S_lsΛe)
    R_ls = calc_R_xt(S_ls)
    R_ls_goodchans = xarray.DataArray(
        R_ls.values[~brokenchan.values, :, :],
        dims=R_ls.dims,
        coords={"n_c": all_coords["n_c"][~brokenchan.values],
                "n_l": all_coords["n_l"]})
    # setting bad lines to nan.  Simply removing them is incorrect,
    # because it will mean that the k-diagonal may contain correlations
    # corresponding to points more than k scanlines apart from the
    # diagonal.
    R_ls_goodchans.values[:, brokenline.values, :] = numpy.nan
    R_ls_goodchans.values[:, :, brokenline.values] = numpy.nan
    (Δ_l, Δ_l_full) = calc_Delta_x(R_ls_goodchans, return_vector=True)

    # verify that bad data in result is due to known bad data in input.
    # We only need to check a single row or column in S_lsΛe because this
    # matrix is symmetric.
#    if not numpy.array_equal(
#            numpy.isnan(C_lΛes_diag).any("n_s"),
#            numpy.isnan(S_lsΛe.values[:, :, :, 0])):
#        raise ValueError("Unexpected nan propagation")

    S_esΛl = calc_S_from_CUR(R_eΛls, U_eΛls_diag, C_eΛls_diag)
    S_es = calc_S_xt(S_esΛl)
    R_es = calc_R_xt(S_es)
    R_es_goodchans = xarray.DataArray(
        R_es.values[~brokenchan.values, :, :],
        dims=R_es.dims,
        coords={"n_c": all_coords["n_c"][~brokenchan.values],
                "n_e": all_coords["n_e"]})
    (Δ_e, Δ_e_full) = calc_Delta_x(R_es_goodchans, return_vector=True)

    R_cΛpi_stacked = typhon.utils.stack_xarray_repdim(R_cΛpi, n_p=("n_l", "n_e"))
    S_ciΛp = calc_S_from_CUR(
        xarray.DataArray(
            R_cΛpi_stacked.values.transpose(0, 3, 1, 2),
            dims=("n_i", "n_p", "n_c", "n_c"),
            coords=R_cΛpi_stacked.coords),
        U_cΛpi_diag.stack(n_p=("n_l", "n_e")).transpose("n_i", "n_p", "n_c"),
        C_cΛpi_diag.stack(n_p=("n_l", "n_e")).transpose("n_i", "n_p", "n_c"))
    S_ci = calc_S_xt(S_ciΛp)
    R_ci = calc_R_xt(S_ci)

    R_cΛps_stacked = typhon.utils.stack_xarray_repdim(R_cΛps, n_p=("n_l", "n_e"))
    S_csΛp = calc_S_from_CUR(
        xarray.DataArray(
            R_cΛps_stacked.values.transpose(0, 3, 1, 2),
            dims=("n_s", "n_p", "n_c", "n_c"),
            coords=R_cΛps_stacked.coords),
        U_cΛps_diag.stack(n_p=("n_l", "n_e")).transpose("n_s", "n_p", "n_c"),
        C_cΛps_diag.stack(n_p=("n_l", "n_e")).transpose("n_s", "n_p", "n_c"))
    S_cs = calc_S_xt(S_csΛp)
    R_cs = calc_R_xt(S_cs)

    # fill missing channels
    #
    # FIXME: those Δ_l_all, Δ_e_all, Δ_l_full_all, Δ_e_full_all fillings
    # violate DRY and need to be put in a function of some sorts.

    Δ_l_all = xarray.DataArray(
        numpy.zeros((n_c, 2)),
        dims=Δ_l.dims,
        coords={"n_c": all_coords["n_c"], "val": Δ_l["val"]})
    Δ_l_all.loc[{"n_c": Δ_l["n_c"]}] = Δ_l
    Δ_l_all[{"n_c": brokenchan}] = numpy.nan

    Δ_e_all = xarray.DataArray(
        numpy.zeros((n_c, 2)),
        dims=Δ_e.dims,
        coords={"n_c": all_coords["n_c"], "val": Δ_e["val"]})
    Δ_e_all.loc[{"n_c": Δ_e["n_c"]}] = Δ_e
    Δ_e_all[{"n_c": brokenchan}] = numpy.nan

    if interpolate_lengths:
        if cutoff_l is None or cutoff_e is None:
            raise TypeError("If interpolate_lengths is True, you must pass "
                "both cutoff_l and cutoff_e.")
        try:
            Δ_l_full = interpolate_Delta_x(Δ_l_full, cutoff_l)
            Δ_e_full = interpolate_Delta_x(Δ_e_full, cutoff_e)
        except ValueError as e:
            if e.args[0] == "x and y arrays must have at least 2 entries":
                raise FCDRError("Too few valid lines to interpolate "
                    "correlation lengths")
            else:
                raise

        # those still don't contain all the channels; fill the other
        # channels with nans again
        #
        # it's also not at all pretty that those n_e and n_l are changed
        # to n_p by calc_Delta_x

        Δ_l_full_all = xarray.DataArray(
            numpy.zeros((cutoff_l, n_c)),
            dims=Δ_l_full.dims,
            coords={"Δp": Δ_l_full.coords["Δp"],
                    "n_c": all_coords["n_c"]})
        Δ_l_full_all.loc[{"n_c": Δ_l["n_c"]}] = Δ_l_full
        Δ_l_full_all[{"n_c": brokenchan}] = numpy.nan

        Δ_e_full_all = xarray.DataArray(
            numpy.zeros((cutoff_e, n_c)),
            dims=Δ_e_full.dims,
            coords={"Δp": Δ_e_full.coords["Δp"],
                    "n_c": all_coords["n_c"]})
        Δ_e_full_all.loc[{"n_c": Δ_e["n_c"]}] = Δ_e_full
        Δ_e_full_all[{"n_c": brokenchan}] = numpy.nan
    
    return (Δ_l_all, Δ_e_all, R_ci, R_cs) + (
        (Δ_l_full_all, Δ_e_full_all) if return_vectors else ()) + (
        (locals(),) if return_locals else ())

def interpolate_Delta_x(Δ_x, cutoff):
    """Interpolate Δ_e or Δ_l vectors to fill sampling gaps

    When the full vectors for Δ_e or Δ_l are calculated, this is only done
    at intervals corresponding to sampling_e and sampling_l.  This
    function applies a spline interpolation to have correlation estimates
    at intermediate values.  If the last element of Δ_x is smaller than
    the cutoff, the remainder are filled with zeroes (this happens if
    n_l%sampling_l≠0).
    
    You usually don't need to call this function directly, as it is called
    by `apply_curuc` if you pass ``interpolate_lengths=True``.

    Parameters
    ----------

    Δ_x : (n_p, n_c) xarray.DataArray

        As returned by `calc_Delta_x` if ``return_vectors`` is True.

    cutoff : int

        Total desired lengths, at most equal to n_l or n_e.

    Returns
    -------

    (cutoff, ) xarray.DataArray
        
        New Δ_x, now interpolated such as having lengths ``cutoff``.
    """
   
    rv = xarray.DataArray(
        numpy.zeros((cutoff, Δ_x["n_c"].size), dtype="f4"),
        dims=Δ_x.dims,
        coords={"n_c": Δ_x.coords["n_c"],
                "Δp": numpy.arange(cutoff)})

    for c in Δ_x["n_c"]:
        yref = Δ_x.sel(n_c=c).values
        xref = Δ_x["Δp"].values
        # use slinear to ensure we always remain between -1 and 1
        f = scipy.interpolate.interp1d(xref, yref, kind="slinear",
            fill_value=(-2, 0), assume_sorted=False, bounds_error=False)
        rv.loc[{"n_c":c}] = f(rv["Δp"])
        if (rv.loc[{"n_c":c}].values==-2).any():
            raise ValueError("Could not interpolate correlation lengths, "
                "it appears I had no information on correlation length 0, "
                "which is very weird indeed and almost certainly either a "
                "bug or the consequence of very weird input data.")

    return rv

def accum_sens_coef(sensdict: Dict[sympy.Symbol, Tuple[numpy.ndarray, Dict[sympy.Symbol, Tuple[numpy.ndarray, Dict[sympy.Symbol, Tuple]]]]],
        sym: sympy.Symbol,
        _d: Optional[Deque]=None) -> Deque:
    """Given a sensitivity coefficient dictionary, accumulate them for term

    Given a dictionary of sensitivity coefficients (see function
    annotation) such as returned by calc_u_for_variable), accumulate
    recursively the sensitivity coefficients for symbol ``sym``.  The
    sensitivity coefficient dictionary is a nested dictionary with a
    structure documented in the return values of
    `fcdr.HIRSFCDR.calc_u_for_variable`.

    Parameters
    ----------

    sensdict : Dict[Symbol, Tuple[ndarray, Dict[Symbol, Tuple[ndarray, Dict[Symbol, Tuple[...]]]]]]

        Collection of sensitivities of the form returned by
        `fcdr.HIRSFCDR.calc_u_for_variable`.

    sym : sympy.Symbol

        Symbol for which to calculate total sensitivity
        coefficient

    _d: Deque

        **THOU SHALT NOT PASS**!  Internal recursive use only.


    Returns
    -------

    ndarray or sympy.Expr
        Total sensitivity down the chain
    """

    if _d is None:
        _d = collections.deque([1])

    if sym in sensdict:
        _d.append(sensdict[sym][0])
        return _d

    for (subsym, (sensval, sub_sensdict)) in sensdict.items():
        _d.append(sensval)
        try:
            return accum_sens_coef(sub_sensdict, sym, _d)
        except KeyError: # not found
            _d.pop()
    raise KeyError(f"Term not found: {sym!s}")

def calc_corr_scale_channel(effects, sensRe, ds, 
        sampling_l=8, sampling_e=1, flags=None,
        robust=False, return_vectors=False,
        interpolate_lengths=False, return_locals=False):
    """Calculate correlation length scales per channel

    Note that this function expects quite specific data structured
    corresponding to what happens to be the FCDR_HIRS implementation.
    Consider if using the lower-level functions `allocate_curuc` and
    `apply_curuc` may be easier.

    Parameters
    ----------

    effects : Mapping[symbol, Collection[effect]]

        Dictionary containing, for each term in the measurement
        equation (sympy symbols), a collection (such a set) of all
        effects (instances of the Effect class) for this particular
        symbol.

    sensRe : Dict[Symbol, Tuple[ndarray,
                Dict[Symbol, Tuple[ndarray,
                    Dict[Symbol, Tuple[...]]]]]]

        Collection of sensitivities such as returned by
        the :meth:`fcdr.HIRSFCDR.calc_u_for_variable` method.

    ds : xarray.Dataset

        xarray Dataset containing the debug version of the FCDR.

    sampling_l : int, optional

        Sampling level between lines.  Defaults to 8.

    sampling_e : int, optional

        Sampling level between elements.  Defaults to 1 (i.e. consider
        all elements).

    flags : Mapping[str, DataArray], optional

        Flags such as collected during FCDR generation.  This is used
        to know what scanlines, elements, or channels to skip (missing
        data).

    robust : bool, optional

        If True and nothing can be calculated, log a warning and
        return objects full of fill values.  If False and nothing can
        be calculated, raise an error.

    return_vectors : bool, optional

        Optional, defaults to False.  If True, in addition of returning
        optimal length scales, also return the full vectors with average
        correlation per separation length.

    interpolate_lengths : bool, optional

        Only needed if return_vectors is True.  Interpolate skipped lines.
        If False, average correlation is only given for lengths according
        to the sampling interval.  For example, with sampling_l=5, it's
        only given every 5 lines.  If True, a spline interpolation is
        applied and average correlation length is returned at every
        separation.

    return_locals : bool, optional

        Return complete dictionary of locals.  When deep inspection is
        a must.

    Returns
    -------

        As for `apply_curuc`.
    """

    # For suggested dimensions per term, see docstring of
    # metrology.calc_S_from_CUR

    n_s = n_i = 0
    for k in itertools.chain.from_iterable(effects.values()):
        if k.is_structured():
            n_s += 1
        elif k.is_independent():
            n_i += 1
        elif k.is_common():
            continue
        else:
            raise RuntimeError(f"Effect {k!s} neither structured nor "
                                "independent?! Impossible!")

    # for the full n_l this is too memory-intensive.  Need to split in
    # smaller parts.
    n_l = ds.dims["scanline_earth"]
    n_e = ds.dims["scanpos"]
    n_c = ds.dims["calibrated_channel"]

    (R_eΛls, R_lΛes, R_cΛpi, R_cΛps, U_eΛls_diag, U_lΛes_diag,
        U_cΛps_diag, U_cΛpi_diag, C_eΛls_diag, C_lΛes_diag,
        C_cΛps_diag, C_cΛpi_diag, all_coords) = allocate_curuc(
            n_c, n_l, n_e, n_s, n_i, sampling_l, sampling_e)
    
    # comparing .values to avoid triggering
    # http://bugs.python.org/issue29672
    # the following are equivalent, but the first version does and the
    # second version does not trigger the issue
#    bad = (((flags["channel"].isel(scanline_earth=all_coords["n_l"]) &
#            _fcdr_defs.FlagsChannel.DO_NOT_USE)!=0) |
#           ((flags["scanline"].isel(scanline_earth=all_coords["n_l"]) &
#            _fcdr_defs.FlagsScanline.DO_NOT_USE)!=0) |
#           ((flags["pixel"].isel(scanline_earth=all_coords["n_l"]) &
#            _fcdr_defs.FlagsPixel.DO_NOT_USE)!=0))

    bad = (xarray.DataArray(((flags["channel"].isel(scanline_earth=all_coords["n_l"]) & _fcdr_defs.FlagsChannel.DO_NOT_USE).values!=0), dims=flags["channel"].dims, coords=flags["channel"].isel(scanline_earth=all_coords["n_l"]).coords) |
           xarray.DataArray((flags["scanline"].isel(scanline_earth=all_coords["n_l"]) & _fcdr_defs.FlagsScanline.DO_NOT_USE).values!=0, dims=flags["scanline"].dims, coords=flags["scanline"].isel(scanline_earth=all_coords["n_l"]).coords) |
           xarray.DataArray((flags["pixel"].isel(scanline_earth=all_coords["n_l"]) & _fcdr_defs.FlagsPixel.DO_NOT_USE).values!=0, dims=flags["pixel"].dims, coords=flags["pixel"].isel(scanline_earth=all_coords["n_l"]).coords))

    # I want to set bad pixels not to nan, but to some interpolated value
    # or the median (https://github.com/FIDUCEO/FCDR_HIRS/issues/322).
    # The 'bad' array is based on the sampled resolution, but the sampling
    # operation counts as advanced indexing (see
    # https://docs.scipy.org/doc/numpy-1.15.1/reference/arrays.indexing.html#advanced-indexing
    # or http://xarray.pydata.org/en/stable/indexing.html#copies-vs-views). 
    # Advanced indexing returns a copy, although __setitem__ still works
    # (https://stackoverflow.com/a/37842121/974555), this means I can't
    # assign to a chained advanced assignment
    # (ds["T_b"]{"scanline:earth":all_coords["n_l"]}[bad] = 42 will have
    # no result).  Therefore, I'm generating a special version of 'bad'
    # that applies before the sampling.

    fullbad = (xarray.DataArray(((flags["channel"] & _fcdr_defs.FlagsChannel.DO_NOT_USE).values!=0), dims=flags["channel"].dims, coords=flags["channel"].coords) |
           xarray.DataArray((flags["scanline"] & _fcdr_defs.FlagsScanline.DO_NOT_USE).values!=0, dims=flags["scanline"].dims, coords=flags["scanline"].coords) |
           xarray.DataArray((flags["pixel"] & _fcdr_defs.FlagsPixel.DO_NOT_USE).values!=0, dims=flags["pixel"].dims, coords=flags["pixel"].coords))

    ds = ds.copy() # don't want to change it for the caller…
    for (k, v) in ((k, v) for (k, v) in ds.data_vars.items()
                    if set(v.dims) == set(fullbad.dims)):
        v.values[fullbad.transpose(*v.dims).values] = numpy.median(
            v.values[(~fullbad).transpose(*v.dims).values])

    bad = bad.rename(
        {"scanline_earth": "n_l",
         "scanpos": "n_e",
         "calibrated_channel": "n_c"}).assign_coords(
            n_l=all_coords["n_l"],
            n_e=all_coords["n_e"],
            n_c=all_coords["n_c"])

    fracbad = (bad.sum()/bad.size).item()
    (logger.error if fracbad > 0.9 
     else logger.warning if 0.1 < fracbad < 0.9
     else logger.info)(f"CURUC: {fracbad:.2%} of pixels in segment bad")

    # Decide how to treat 
    brokenchan = bad.any("n_e").all("n_l")
    brokenline = bad.sel(n_c=~brokenchan).all("n_e").all("n_c")
    if brokenchan.all() or brokenline.all():
        errmsg = ("No valid data found, cannot calculate "
            "correlation length scales")
        if robust:
            logger.error(errmsg)
            Δ_e = Δ_l = xarray.DataArray(
                numpy.zeros((n_c, 2))*numpy.nan,
                dims=("n_c", "val"),
                coords={"n_c": all_coords["n_c"], "val": ["popt", "pcov"]})
            R_ci = R_cs = xarray.DataArray(
                numpy.zeros((n_c, n_c))*numpy.nan,
                dims=("n_c", "n_c"),
                coords={"n_c": all_coords["n_c"]})
        
            Δ_l_full_all = xarray.DataArray(
                numpy.zeros((n_l, n_c)),
                dims=("Δp", "n_c"),
                coords={"Δp": numpy.arange(n_l),
                        "n_c": all_coords["n_c"]})

            Δ_e_full_all = xarray.DataArray(
                numpy.zeros((n_e, n_c)),
                dims=("Δp", "n_c"),
                coords={"Δp": numpy.arange(n_e),
                        "n_c": all_coords["n_c"]})

            return (Δ_l, Δ_e, R_ci, R_cs) + (
                (Δ_l_full_all, Δ_e_full_all) if return_vectors else ()) + (
                (locals(),) if return_locals else ())
        else:
            raise FCDRError(errmsg)
    else:
        rat = brokenline.sum()/brokenline.size
        if rat > 0.75:
            lev = logger.warning
        elif 0.1 < rat < 0.75:
            lev = logger.info
        else:
            lev = logger.debug
        lev(f"In correlation calculation, "
            f"{brokenline.sum().item():d}/{brokenline.size:d} "
            "lines invalid")

    ## Copying data to correct format ##

    ccs = itertools.count()
    cci = itertools.count()
    for (cj, j) in enumerate(effects.keys()): # loop over terms
        logger.debug(f"Processing term {cj:d}, {j!s}")
        try:
            C = accum_sens_coef(sensRe, j)
        except KeyError as k:
            logger.error(f"I have {len(effects[j]):d} effects associated "
                f"with term {j!s}, but I have no sensitivity coefficient "
                "for this term.  I don't think I used it in the "
                "measurement equation.  For the purposes of the " 
                "correlation length calculation, I will ignore it.  "
                "Sorry!")
            continue

        # before the mul, ensure consistent dimensions

        if all(isinstance(x, numbers.Number) for x in C): # scalar, no units
            CC = numpy.product(C)
        else:
            CC = functools.reduce(operator.mul, 
                [make_debug_fcdr_dims_consistent(ds, x, impossible="error")
                  for x in C]).sel(scanline_earth=ds["scanline_earth"]).sel(
                    scanline_earth=slice(None, None, sampling_l),
                    scanpos=slice(None, None, sampling_e)).values
#        # Thanks to https://stackoverflow.com/a/48628917/974555
#        expanded = numpy.zeros(CC.shape + CC.shape[-1:], dtype=CC.dtype)
#
#        diagonals = numpy.diagonal(expanded, axis1=-2, axis2=-1)
#        diagonals.setflags(write=True)
#        diagonals[:] = CC

        if len(effects[j]) == 0:
            warnings.warn(f"Zero effects for term {j!s}!", UserWarning)
        for k in effects[j]: # loop over effects for term (usually exactly one)
            if k.magnitude is None:
                logger.warn(f"Magnitude for {k.name:s} is None, not "
                    "considering for correlation scale calculations.")
                continue

            if k.is_common():
                continue # don't bother with common, should all be
                         # harmonised anyway

            # Make sure we have one estimate for every scanline.
            new_u = make_debug_fcdr_dims_consistent(
                ds, k.magnitude, impossible="error").sel(
                    scanline_earth=slice(None, None, sampling_l))

            if k.is_independent():
                # for independent, still need to consider
                # inter-channel: R_cΛpi 
                ci = next(cci)
                R_cΛpi[{"n_i": ci}].values[...] = k.calc_R_cUpk(ds,
                    sampling_l=sampling_l,
                    sampling_e=sampling_e)
                U_cΛpi_diag[{"n_i": ci}].values[...] = new_u.T.values[:, numpy.newaxis, :]
                C_cΛpi_diag[{"n_i": ci}].values[...] = CC.transpose((1, 2, 0))
                continue

            try:
                R_eΛlk = k.calc_R_eUlk(ds,
                        sampling_l=sampling_l, sampling_e=sampling_e)
                R_lΛek = k.calc_R_lUek(ds,
                        sampling_l=sampling_l, sampling_e=sampling_e)
            except NotImplementedError:
                logger.error("No method to estimate R_eΛlk or R_lΛek "
                    f"implemented for effect {k.name:s}")
                continue

            cs = next(ccs)

            R_eΛls[{"n_s": cs}].values[...] = R_eΛlk
            R_lΛes[{"n_s": cs}].values[...] = R_lΛek

            # We have at most one estimate of U per scanline, so not
            # only is U diagonal; for U_eΛlk, the value along the diagonal
            # is constant too.
            U_eΛls_diag[{"n_s": cs}].values[...] = new_u.values[..., numpy.newaxis]
#            U_eΛlk[{"n_k": ck}].values[...] = (
#                new_u.values[:, :, numpy.newaxis, numpy.newaxis] *
#                numpy.eye(math.ceil(n_e/sampling_e))[numpy.newaxis, numpy.newaxis, :, :])
            # not sure if next line is correct
#            U_lΛek[{"n_k": ck}].values[...] = (
#                new_u.values[:, :, numpy.newaxis, numpy.newaxis] *
#                numpy.eye(math.ceil(n_l/sampling_l))[numpy.newaxis, numpy.newaxis, :, :])
            C_eΛls_diag[{"n_s": cs}].values[...] = CC

            R_cΛps[{"n_s": cs}].values[...] = k.calc_R_cUpk(ds,
                sampling_l=sampling_l,
                sampling_e=sampling_e)

            # FIXME: U_cΛps_diag, C_cΛps_diag
    # use value of cs to consider how many to pass on
    tcs = next(ccs)
    tci = next(cci)
    R_lΛes = R_lΛes.sel(n_s=slice(tcs))
    U_lΛes_diag = U_lΛes_diag.sel(n_s=slice(tcs))
    C_lΛes_diag = C_lΛes_diag.sel(n_s=slice(tcs))
    R_eΛls = R_eΛls.sel(n_s=slice(tcs))
    U_eΛls_diag = U_eΛls_diag.sel(n_s=slice(tcs))
    C_eΛls_diag = C_eΛls_diag.sel(n_s=slice(tcs))
    R_cΛps = R_cΛps.sel(n_s=slice(tcs))
    U_cΛps_diag = U_cΛps_diag.sel(n_s=slice(tcs))
    C_cΛps_diag = C_cΛps_diag.sel(n_s=slice(tcs))
    R_cΛpi = R_cΛpi.sel(n_i=slice(tci))
    U_cΛpi_diag = U_cΛpi_diag.sel(n_i=slice(tci))
    C_cΛpi_diag = C_cΛpi_diag.sel(n_i=slice(tci))

    return apply_curuc(R_eΛls, R_lΛes, R_cΛpi, R_cΛps,
        U_eΛls_diag, U_lΛes_diag, U_cΛps_diag, U_cΛpi_diag,
        C_eΛls_diag, C_lΛes_diag, C_cΛps_diag, C_cΛpi_diag,
        all_coords, brokenchan, brokenline, return_vectors=return_vectors,
        interpolate_lengths=interpolate_lengths, cutoff_l=n_l,
        cutoff_e=n_e, return_locals=return_locals)
