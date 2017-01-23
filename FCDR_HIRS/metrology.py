"""For any metrology-related functions
"""

import numpy
import xarray

import typhon.physics.metrology
from . import effects
from . import measurement_equation as me

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
