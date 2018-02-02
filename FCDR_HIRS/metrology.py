"""For any metrology-related functions
"""

import functools
import operator
import collections

import numpy
import scipy.optimize
import xarray
import sympy

import typhon.physics.metrology
from . import effects
from . import measurement_equation as me
from typing import (List, Dict, Tuple, Deque, Optional)

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

def calc_S_from_CUR(R_xΛyt: List[List[numpy.ndarray]],
                    U_xΛyt: List[List[numpy.ndarray]],
                    C_xΛyj: List[numpy.ndarray]) -> numpy.ndarray:
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

        One of R_eΛls, R_lΛes, R_cΛpi, or R_cΛps: List[List[numpy.ndarray]],
            for each terms, either the list of all
            cross-element error correlation matrices, or the list of all
            cross-line error correlation matrices, for one effect, that
            all affect that particular term in the measurement equation. 
            The outer list has the length corresponding to the number of
            terms (n_j), the inner list the number of structured effects for each
            particular term (n_s|j), except in the case of R_cΛpi, where
            it's the number of independent effects for that term.
            Defined by §3.2.3.

        One of U_eΛls, U_lΛes, U_cΛpi, or U_cΛps: List[List[numpy.ndarray]],
            for all terms, either a list of all
            cross-element term uncertainty matrices or a list of all
            cross-line term uncertainty matrices, for one effect, that
            all affect the same term in the measurement equation.  Those
            matrices are diagonal.  Defined by §3.2.6.

        One of C_eΛlj, C_lΛej, C_cΛpj: List[numpy.ndarray], for all terms,
            either cross-element
            sensitivity matrices per term or cross-line sensitivity matrices per
            term.  These matrices are diagonal.  Defined by §3.2.9.

    You probably want to vectorise this over an entire image.  Probable
    dimensions:

        One per term:
        
        C_eΛlj [n_c, n_l, n_j, n_e, n_e]
        C_lΛej [n_c, n_e, n_j, n_l, n_l]
        C_cΛpj [n_p, n_j, n_c, n_c]
        
        One per effect:
        
        S_esΛl [n_c, n_l, n_e, n_e]
        S_lsΛe [n_c, n_e, n_l, n_l]
        S_ciΛp [n_p, n_c, n_c]
        S_csΛp [n_p, n_c, n_c]
        
        R_eΛls [n_c, n_l, n_s|j, n_e, n_e]
        R_lΛes [n_c, n_e, n_s|j, n_l, n_l]
        R_cΛpi [n_p, n_i|j, n_c, n_c]
        R_cΛps [n_p, n_s|j, n_c, n_c]
        
        U_eΛls [n_c, n_l, n_s|j, n_e, n_e]
        U_lΛes [n_c, n_e, n_s|j, n_l, n_l]
        U_cΛpi [n_p, n_i|j, n_c, n_c]
        U_cΛps [n_p, n_s|j, n_c, n_c]

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

    if not (len(R_xΛyt) == len(U_xΛyt) == len(C_xΛyj)):
        raise ValueError("R, U, C must have same length")

    # How much can be vectorised here?  The arrays may be jagged?  And
    # this function itself also needs to be called many times (for every
    # line) — expensive?
    agg = []
    for j in range(len(C_xΛyj)):
        if not (len(R_xΛyt[j]) == len(U_xΛyt[j])):
            raise ValueError(f"R and U nr. {j:d} must have same length")
        for s in range(len(R_xΛyt[j])):
            # equation 20 or 24
            agg.append(C_xΛyj[j] @ U_xΛyt[j][s] @ R_xΛyt[j][s] @ U_xΛyt[j][s].T @ C_xΛyj[j].T)

    S_xtΛy = functools.reduce(operator.add, agg)

    return S_xtΛy

def calc_S_xt(S_xtΛy: List[numpy.ndarray]) -> numpy.ndarray:
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

    Returns:

        S_es, S_el, S_ci, or S_cs: numpy.ndarray, as described.
    """

    return numpy.average(S_xtΛy, 0) # Eq. 20, 24, 27, or 30

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

    U_xt = numpy.diag(numpy.sqrt(S_xt)) # Eq. 21 or 25
    R_xt = numpy.inv(U_xt) @ S_xt @ numpy.inv(U_xt).T # Eq. 22 or 26

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

    Δ_ref = numpy.arange(R_xt.shape[0])
    r_xΔ = numpy.array([numpy.diag(M, k=-i).mean() for i in Δ_ref])

    def f(Δ, Δ_e):
        return numpy.exp(-Δ/Δ_e)

    (popt, pcov) = scipy.optimize.curve_fit(f, Δ, r_xΔ, p0=1)
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

def calc_corr_scale_channel(effects, sensRe, ds):
    """Calculate correlation length scale for channel
    """

    # For suggested dimensions per term, see docstring of
    # metrology.calc_S_from_CUR

    for j in effects.keys(): # loop over terms
        C = accum_sens_coef(sensRe, j)
        for k in effects[j]: # loop over effects for term
            R_eΛlkx = k.calc_R_eΛlkx(ds)

            U_eΛls = ... # diagonal

            C_eΛlj = ... # diagonal

            raise NotImplementedError("And now?")

