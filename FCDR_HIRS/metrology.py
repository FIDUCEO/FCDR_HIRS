"""For any metrology-related functions
"""

import math
import functools
import operator
import collections
import itertools
import logging
import numbers
import numexpr
from typing import (List, Dict, Tuple, Deque, Optional)

import numpy
import scipy.optimize
import xarray
import sympy

import typhon.physics.metrology
from . import effects
from . import measurement_equation as me
from .fcdr import make_debug_fcdr_dims_consistent

def evaluate_uncertainty(e, unset="raise"):
    """Evaluate uncertainty for expression.

    Requires effects tables to be populated including quantified
    uncertainties.  Any variable which does not have any effects
    associated with it is assumed to have uncertainty 0.
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

def calc_S_from_CUR(R_xΛyt: numpy.ndarray,
                    U_xΛyt_diag: numpy.ndarray,
                    C_xΛyt_diag: numpy.ndarray,
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

    Arguments:

        One of R_eΛls, R_lΛes, R_cΛpi, or R_cΛps: xarray.DataArray
            with dimensions [n_c, n_s, n_e, n_l, n_l] or [n_c, n_s, n_l,
            n_e, n_e] or .
            For each channel (c) and each effect (k), either a collection
            of cross-element error correlation matrices for each line, or
            of cross-line error correlation matrices for each element.
            Defined by §3.2.3.

        Diagonals of one of U_eΛls, U_lΛes, U_cΛpi, or U_cΛps: xarray.DataArray
            Same dimensions as previous but minus the final dimension,
            because it only stores the diagonals.
            Considering §3.2.6, consider that the final dimension shows
            the diagonals of any U_eΛls, U_lΛes, U_cΛpi, or U_cΛps.
            Defined by §3.2.6.

        One of C_eΛls, C_lΛes, C_cΛps, C_cΛpi: xarray.DataArray
            Contains the sensitivity diagonals _per effect_.  Although
            sensitivity is defined per term and not per effect, I need
            them per effect.  Most terms have exactly one effect defined
            anyway.  Dimensions therefore the same as U_eΛls and friends.
            Defined by §3.2.9.

        Boolean "per_channel".  If not given, this will be inferred from
        the presence of a dimension "n_c" within the leading ndim-1
        dimensions of U.

    You probably want to vectorise this over an entire image.  Probable
    dimensions:

        One per term:
        
        C_eΛls [n_c, n_s, n_l, n_e, n_e] (last 2 diagonal, not explicitly calculated)
        C_lΛes [n_c, n_s, n_e, n_l, n_l] (last 2 diagonal)
        C_cΛps [n_s, n_l, n_e, n_c, n_c] (last 2 diagonal)
        C_cΛpi [n_i, n_l, n_e, n_c, n_c] (last 2 diagonal)
        
        One per effect:
        
        S_esΛl [n_c, n_l, n_e, n_e]
        S_lsΛe [n_c, n_e, n_l, n_l]
        S_ciΛp [n_l, n_e, n_c, n_c]
        S_csΛp [n_l, n_e, n_c, n_c]
        
        R_eΛls [n_c, n_l, n_s, n_e, n_e]
        R_lΛes [n_c, n_e, n_s, n_l, n_l]
        R_cΛpi [n_l, n_e, n_i, n_c, n_c]
        R_cΛps [n_l, n_e, n_s, n_c, n_c]
        
        U_eΛls [n_c, n_l, n_s, n_e, n_e] (last 2 diagonal)
        U_lΛes [n_c, n_e, n_s, n_l, n_l] (last 2 diagonal)
        U_cΛpi [n_l, n_e, n_i|j, n_c, n_c] (last 2 diagonal)
        U_cΛps [n_l, n_e, n_s|j, n_c, n_c] (last 2 diagonal)

        One total:
        
        S_es [n_c, n_e, n_e]
        S_ls [n_c, n_l, n_l]
        S_ci [n_c, n_c]
        S_cs [n_c, n_c]
        
        R_es [n_c, n_e, n_e]
        R_ls [n_c, n_l, n_l]
        R_ci [n_c, n_c]
        R_cs [n_c, n_c]

    Returns:

        S_esΛl, S_lsΛe, S_ciΛp, or S_csΛp: numpy.ndarray, as described above
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
    return xarray.DataArray(S_xtΛy,
        dims=R_xΛyt.dims[0:1] + R_xΛyt.dims[2:])

def calc_S_xt(S_xtΛy: List[numpy.ndarray],
              per_channel: bool=None) -> numpy.ndarray:
    """Calculate S_es, S_ls, S_ci, or S_cs

    Calculate either of:
    
    - S_es, the average cross-element error covariance from the
    structured effects per channel (Eq. 20)
    - S_ls, the average cross-line error covariance from the structured
      effects per channel (Eq. 24)
    covariance from the structured effects per channel.
    - S_ci, the average cross-channel error covariance matrix from the
      spatially independent effects, per channel (Eq. 27)
    - S_cs, the average cross-channel error covariance matrix from the
      structured effects, per channel (Eq. 30)

    Follows recipe with same document source as calc_S_from_CUR.

    Arguments:

    S_esΛl, S_lsΛe, S_ciΛp, or S_csΛp: List[numpy.ndarray]

        List of values of relevant matrix, or ndarray with outermost dimension
        being the scanline.  You can obtain those from calc_S_from_CUR

    per_channel: bool

        Boolean "per_channel".  If not given, this will be inferred from
        the presence of a dimension "n_c" within the leading ndim-1
        dimensions of U.

    Returns:

        S_es, S_el, S_ci, or S_cs: numpy.ndarray, as described.
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

    Arguments:

        S_es or S_el: numpy.ndarray

            Average cross-element error covariance from the structured
            effects per channel.  Can be obtained from calc_S_xt.

    Returns:

        R_es or R_el: numpy.ndarray, as described. 
    """

    U_xt_diag = numpy.sqrt(numpy.diagonal(S_xt, axis1=-2, axis2=-1))
    dUi = (1/U_xt_diag)[..., numpy.newaxis]
    dUiT = dUi.swapaxes(-1, -2)
    R_xt = numexpr.evaluate("dUi * S_xt * dUiT") # equivalent to Ui@S@Ui.T when written fully
    return xarray.DataArray(R_xt, dims=S_xt.dims)

def calc_Δ_x(R_xt: numpy.ndarray):
    """Calculate optimum Δ_e or Δ_l

    Calculate optimum correlation length scale, either across elements,
    Δ_e or across lines, Δ_l.
    Structured effects, per channel.

    Recipe source as for calc_S_from_CUR, now §3.3.4.

    Arguments:

        R_es or R_ls: numpy.ndarray

            Either cross-element or cross-line radiance error correlation matrix, structured
            effects, per channel.  Can be obtained from calc_R_xt.
    """

    dim = R_xt.dims[-1]
    Δ_ref = xarray.DataArray(
        numpy.arange(R_xt[dim].size),
        dims=(dim,))
    r_xΔ = xarray.DataArray(
        numpy.array([numpy.diagonal(R_xt, i, -2, -1).mean(-1) for i in Δ_ref]),
        dims=("n_p", "n_c"))

    def f(Δ, Δ_e):
        return numpy.exp(-Δ/Δ_e)

    # I don't suppose I can vectorise this one…
    popt = xarray.DataArray(
        numpy.array([scipy.optimize.curve_fit(f, Δ_ref, r_xΔ.sel(n_c=c),
            p0=1) for c in r_xΔ["n_c"]]).squeeze(),
        dims=("n_c", "val"),
        coords={"n_c": r_xΔ["n_c"],
                "val": ["popt", "pcov"]})

    return popt



def accum_sens_coef(sensdict: Dict[sympy.Symbol, Tuple[numpy.ndarray, Dict[sympy.Symbol, Tuple[numpy.ndarray, Dict[sympy.Symbol, Tuple]]]]],
        sym: sympy.Symbol,
        _d: Optional[Deque]=None) -> Deque:
    """Given a sensitivity coefficient dictionary, accumulate them for term

    Given a dictionary of sensitivity coefficients (see function
    annotation) such as returned by calc_u_for_variable), accumulate
    recursivey the sensitivity coefficients for symbol `sym`.

    Arguments:

        sensdict (Dict[Symbol, Tuple[ndarray,
                Dict[Symbol, Tuple[ndarray,
                  Dict[Symbol, Tuple[...]]]]]])

            Collection of sensitivities.  Returned by
            calc_u_for_variable.

        sym: sympy.Symbol

            Symbol for which to calculate total sensitivity
            coefficient

        _d: Deque

            THOU SHALT NOT PASS!  Internal recursive use only.

    """

    if _d is None:
        _d = collections.deque([1])

    if sym in sensdict.keys():
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
        sampling_l=5, sampling_e=1):
    """Calculate correlation length scales per channel
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
    n_j = len(effects.keys())

    logging.debug("Allocating arrays for correlation calculations")

    R_eΛls = xarray.DataArray(
        numpy.zeros((n_c, n_s, 
            math.ceil(n_l/sampling_l),
            math.ceil(n_e/sampling_e),
            math.ceil(n_e/sampling_e)), dtype="f4"),
        dims=("n_c", "n_s", "n_l", "n_e", "n_e"))

    R_lΛes = xarray.DataArray(
        numpy.zeros((n_c, n_s, 
            math.ceil(n_e/sampling_e),
            math.ceil(n_l/sampling_l),
            math.ceil(n_l/sampling_l)), dtype="f4"),
        dims=("n_c", "n_s", "n_e", "n_l", "n_l"))

    R_cΛpi = xarray.DataArray(
        numpy.zeros(
           (n_i,
            math.ceil(n_l/sampling_l),
            math.ceil(n_e/sampling_e),
            n_c,
            n_c), dtype="f4"),
        dims=("n_i", "n_l", "n_e", "n_c", "n_c"))

    R_cΛps = xarray.DataArray(
        numpy.zeros(
           (n_s,
            math.ceil(n_l/(sampling_l)),
            math.ceil(n_e/(sampling_e)),
            n_c,
            n_c), dtype="f4"),
        dims=("n_s", "n_l", "n_e", "n_c", "n_c"))

    # store only diagonals for optimised memory consumption and
    # calculation speed
    U_eΛls_diag = xarray.DataArray(
        numpy.zeros((n_c, n_s,
            math.ceil(n_l/sampling_l),
            math.ceil(n_e/sampling_e)), dtype="f4"),
        dims=("n_c", "n_s", "n_l", "n_e"))

    U_lΛes_diag = U_eΛls_diag.transpose("n_c", "n_s", "n_e", "n_l")
    U_cΛps_diag = U_eΛls_diag.transpose("n_l", "n_e", "n_s", "n_c")

    U_cΛpi_diag = xarray.DataArray(
        numpy.zeros((
            n_i,
            math.ceil(n_l/sampling_l),
            math.ceil(n_e/sampling_e),
            n_c), dtype="f4"),
        dims=("n_i", "n_l", "n_e", "n_c"))
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
        dims=("n_c", "n_s", "n_l", "n_e")) # last n_e superfluous
    C_lΛes_diag = C_eΛls_diag.transpose("n_c", "n_s", "n_e", "n_l")
    C_cΛps_diag = C_eΛls_diag.transpose("n_s", "n_l", "n_e", "n_c")

    C_cΛpi_diag = xarray.DataArray(
        numpy.zeros_like(U_cΛpi_diag.values),
        dims=("n_i", "n_l", "n_e", "n_c"))

    # should the next one be skipped?
#    R_eΛli = xarray.DataArray(
#        numpy.zeros((n_c, n_i, n_l//sampling_l,
#                     n_e//sampling_e,
#                     n_e//sampling_e), dtype="f4"),
#        dims=("n_c", "n_k", "n_l", "n_e", "n_e")) # FIXME: change dim names

    #U_eΛli = xarray.zeros_like(R_eΛli)

#    U_lΛes: ...
#    C_lΛej: ...
#    R_lΛei: ...
#    U_eΛei: ...

    ccs = itertools.count()
    cci = itertools.count()
    for (cj, j) in enumerate(effects.keys()): # loop over terms
        logging.debug(f"Processing term {cj:d}, {j!s}")
        try:
            C = accum_sens_coef(sensRe, j)
        except KeyError as k:
            logging.error(f"I have {len(effects[j]):d} effects associated "
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

        #C_eΛlj[{"n_j": cj}].values[...] = expanded
        if len(effects[j]) == 0:
            warnings.warn(f"Zero effects for term {j!s}!", UserWarning)
        for k in effects[j]: # loop over effects for term (usually exactly one)
            if k.magnitude is None:
                logging.warn(f"Magnitude for {k.name:s} is None, not "
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
                R_cΛpi[{"n_i": ci}].values[...] = k.calc_R_cΛpk(ds,
                    sampling_l=sampling_l,
                    sampling_e=sampling_e)
                U_cΛpi_diag[{"n_i": ci}].values[...] = new_u.T.values[:, numpy.newaxis, :]
                C_cΛpi_diag[{"n_i": ci}].values[...] = CC.transpose((1, 2, 0))
                continue

            try:
                R_eΛlk = k.calc_R_eΛlk(ds,
                        sampling_l=sampling_l, sampling_e=sampling_e)
                R_lΛek = k.calc_R_lΛek(ds,
                        sampling_l=sampling_l, sampling_e=sampling_e)
            except NotImplementedError:
                logging.error("No method to estimate R_eΛlk or R_lΛek "
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

            R_cΛps[{"n_s": cs}].values[...] = k.calc_R_cΛpk(ds,
                sampling_l=sampling_l,
                sampling_e=sampling_e)

            # FIXME: U_cΛps_diag, C_cΛps_diag
    # use value of cs to consider how many to pass on
    R_lΛes = R_lΛes.sel(n_s=slice(cs))
    U_lΛes_diag = U_lΛes_diag.sel(n_s=slice(cs))
    C_lΛes_diag = C_lΛes_diag.sel(n_s=slice(cs))
    R_eΛls = R_eΛls.sel(n_s=slice(cs))
    U_eΛls_diag = U_eΛls_diag.sel(n_s=slice(cs))
    C_eΛls_diag = C_eΛls_diag.sel(n_s=slice(cs))

    S_lsΛe = calc_S_from_CUR(R_lΛes, U_lΛes_diag, C_lΛes_diag)
    S_ls = calc_S_xt(S_lsΛe)
    R_ls = calc_R_xt(S_ls)
    Δ_l = calc_Δ_x(R_ls)*sampling_l

    S_esΛl = calc_S_from_CUR(R_eΛls, U_eΛls_diag, C_eΛls_diag)
    S_es = calc_S_xt(S_esΛl)
    R_es = calc_R_xt(S_es)
    Δ_e = calc_Δ_x(R_es)*sampling_e

    S_ciΛp = calc_S_from_CUR(
        xarray.DataArray(
        typhon.utils.stack_xarray_repdim(R_cΛpi, n_p=("n_l", "n_e"))
            .values.transpose(0, 3, 1, 2), dims=("n_i", "n_p", "n_c", "n_c")),
        U_cΛpi_diag.stack(n_p=("n_l", "n_e")).transpose("n_i", "n_p", "n_c"),
        C_cΛpi_diag.stack(n_p=("n_l", "n_e")).transpose("n_i", "n_p", "n_c"))
    S_ci = calc_S_xt(S_ciΛp)
    R_ci = calc_R_xt(S_ci)

    S_csΛp = calc_S_from_CUR(
        xarray.DataArray(
        typhon.utils.stack_xarray_repdim(R_cΛps, n_p=("n_l", "n_e"))
            .values.transpose(0, 3, 1, 2), dims=("n_s", "n_p", "n_c", "n_c")),
        U_cΛps_diag.stack(n_p=("n_l", "n_e")).transpose("n_s", "n_p", "n_c"),
        C_cΛps_diag.stack(n_p=("n_l", "n_e")).transpose("n_s", "n_p", "n_c"))
    S_cs = calc_S_xt(S_csΛp)
    R_cs = calc_R_xt(S_cs)

    return (Δ_l, Δ_e, R_ci, R_cs)
