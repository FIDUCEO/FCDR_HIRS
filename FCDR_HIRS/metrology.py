"""For any metrology-related functions
"""

import numpy
import xarray

import typhon.physics.metrology
from . import effects
from . import measurement_equation as me
from typing import List

import functools
import operator

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



def calc_S_esl(R_els: List[List[numpy.ndarray]],
               U_els: List[List[numpy.ndarray]],
               C_elj: List[numpy.ndarray]):
    """Calculate S_esl
    
    Calculate S_esl, the total cross-element error covariance from the structured
    effects per channel evaluated at a single line.

    Follows recipe from:
    
    Chris Merchant, Emma Woolliams and Jonathan Mittaz,
    Uncertainty and Error Correlation Quantification for FIDUCEO “easy-
    FCDR” Products: Mathematical Recipes.  Hereunder referred to as
    "Recipes".  Section and page numbers refer to version 0.9.3.

    As defined by §3.3.3.

    Arguments:

        R_els: List[List[numpy.ndarray]], for all terms, list of all
            cross-element error correlation matrices for one effect, that
            all affect that particular term in the measurement equation. 
            The outer list has the length corresponding to the number of
            terms (n_j), the inner list the number of structured effects for each
            particular term (n_s|j).  Defined by §3.2.3.

        U_els: List[List[numpy.ndarray]], for all terms, list of all
            cross-element term uncertainty matrices for one effect, that
            all affect the same term in the measurement equation.  Those
            matrices are diagonal.  Defined by §3.2.6.

        C_elj: numpy.ndarray, for all terms, Cross-element sensitivity
            matrices per term.  These matrices are diagonal.  Defined by §3.2.9.

    Returns:

        S_esl as described above
    """

    if not (len(R_els) == len(U_els) == len(C_elj)):
        raise ValueError("R, U, C must have same length")

    # how much can be vectorised here?  The arrays may be jagged.
    agg = []
    for j in range(len(C_elj)):
        if not (len(R_els[j]) == len(U_els[j])):
            raise ValueError(f"R and U nr. {j:d} must have same length")
        for s in range(len(R_els[j])):
            agg.append(C_elj[j] @ U_els[j][s] @ R_els[j][s] @ U_els[j][s].T @ C_elj[j].T)

    S_esl = functools.reduce(operator.add, agg)

    return S_esl
