"""Measurement equation and associated functionality
"""

import numbers

import numpy
import scipy.constants

import sympy
from sympy.core.symbol import Symbol

import typhon.physics.metrology

version = "β"

names = ("R_e a_0 a_1 a_2 C_s R_selfIWCT C_IWCT C_E R_selfE R_selfs ε λ Δλ "
         "a_3 R_refl d_PRT C_PRT k n K N h c k_b T_PRT T_IWCT B φn "
         "R_IWCT ε O_Re O_TIWCT O_TPRT α β T* λ* O_RIWCT")

symbols = sym = dict(zip(names.split(), sympy.symbols(names)))

expressions = {}
expressions[sym["R_e"]] = (
    sym["a_0"] + sym["a_1"]*sym["C_E"] + sym["a_2"]*sym["C_E"]**2 - sym["R_selfE"] + sym["O_Re"])
expressions[sym["a_0"]] = (
    -sym["a_2"] * sym["C_s"]**2 - sym["a_1"]*sym["C_s"])
expressions[sym["a_1"]] = (
    sym["R_IWCT"] + sym["R_selfIWCT"] - sym["R_selfs"] -
    sym["a_2"]*(sym["C_IWCT"]**2-sym["C_s"]**2))/(sym["C_IWCT"]-sym["C_s"])
if version == "β":
    expressions[sym["R_IWCT"]] = (
        (sym["ε"] + sym["a_3"]) * sym["B"] +
        (1 - sym["ε"] - sym["a_3"]) * sym["R_refl"]
        + sym["O_RIWCT"])
else:
    expressions[sym["R_IWCT"]] = (
        (sympy.Integral(((sym["ε"] + sym["a_3"]) * sym["B"] +
        (1-sym["ε"]-sym["a_3"])*sym["R_refl"]) * sym["φn"], sym["λ"]))) # /
#    sympy.Integral(sym["φ"], sym["λ"]))
expressions[sym["B"]] = (
    (2*sym["h"]*sym["c"]**2)/((sym["λ"])**5) *
    1/(sympy.exp((sym["h"]*sym["c"])/((sym["λ"])*sym["k_b"]*sym["T_IWCT"]))-1))
if version == "β":
    expressions[sym["B"]] = expressions[sym["B"]].subs(
        {sym["T_IWCT"]: sym["T*"], sym["λ"]: sym["λ*"]})
    expressions[sym["T*"]] = sym["α"] + sym["β"]*sym["T_IWCT"]
expressions[sym["T_IWCT"]] = (
    sympy.Sum(sympy.IndexedBase(sym["T_PRT"])[sym["n"]], (sym["n"], 0, sym["N"]))/sym["N"] + sym["O_TIWCT"])
expressions[sympy.IndexedBase(sym["T_PRT"])[sym["n"]]] = (
    sympy.Sum(sympy.IndexedBase(sym["d_PRT"])[sym["n"],sym["k"]] *
        sympy.IndexedBase(sym["C_PRT"])[sym["n"]]**sym["k"], (sym["k"], 0, sym["K"]-1))
    + sym["O_TPRT"])
expressions[sym["φn"]] = (sympy.Function("φn")(sym["λ"]+sym["Δλ"]))

expressions[sym["c"]] = sympy.sympify(scipy.constants.speed_of_light)
expressions[sym["h"]] = sympy.sympify(scipy.constants.Planck)
expressions[sym["k_b"]] = sympy.sympify(scipy.constants.Boltzmann)

aliases = {}
aliases[sym["T_PRT"]] = sympy.IndexedBase(sym["T_PRT"])[sym["n"]]
aliases[sym["C_PRT"]] = sympy.IndexedBase(sym["C_PRT"])[sym["n"]]
aliases[sym["d_PRT"]] = sympy.IndexedBase(sym["d_PRT"])[sym["n"],sym["k"]]

def recursive_substitution(e, stop_at=None, return_intermediates=False):
    """For expression 'e', substitute all the way down.

    Using the dictionary `expressions`, repeatedly substitute all symbols
    into the expression until there is nothing left to substitute.
    """
    o = None
    intermediates = set()
    if isinstance(e, sympy.Symbol) and e in expressions.keys():
        return recursive_substitution(expressions[e])
    while o != e:
        o = e
        for sym in typhon.physics.metrology.recursive_args(e):
            if sym != stop_at:
                # subs only works for simple values but is faster
                # replace works for arbitrarily complex expressions but is
                # slower and may yield false positives
                # see http://stackoverflow.com/a/41808652/974555
                e = getattr(e, ("replace" if isinstance(sym, sympy.Indexed) else "subs"))(
                    sym, expressions.get(sym, sym))
                if sym in expressions:
                    intermediates.add(sym)
    return (e, intermediates) if return_intermediates else e
#
#dependencies = {aliases.get(e, e):
#                typhon.physics.metrology.recursive_args(
#                    recursive_substitution(
#                        expressions.get(
#                            aliases.get(e,e),
#                            e)))
#        for e in symbols.values()}

dependencies = {}
for s in symbols.values():
    (e, im) = recursive_substitution(
                expressions.get(aliases.get(s,s),s),
                return_intermediates=True)
    dependencies[aliases.get(s, s)] = typhon.physics.metrology.recursive_args(e) | im

functions = {}
for (sn, s) in symbols.items():
    if s in dependencies.keys():
        if dependencies[s]:
            functions[s] = sympy.Function(sn)(*(aliases.get(sm, sm) for sm in dependencies[s]))

def substitute_until_explicit(expr, s2):
    oldexpr = None
    # expand expression until no more sub-expressions and s2 is explicit
    while expr != oldexpr:
        oldexpr = expr
        for sym in expr.free_symbols - {s2}:
            if aliases.get(s2, s2) in dependencies[aliases.get(sym,sym)]:
                here = aliases.get(sym, sym)
                expr = getattr(expr, ("replace" if sym in aliases else
                    "subs"))(here, expressions.get(here, here))
    return expr

def calc_sensitivity_coefficient(s1, s2):
    """Calculate sensitivity coefficient ∂s1/∂s2

    Arguments:

        s1: Symbol
        s2: Symbol
    """

    if not isinstance(s1, Symbol):
        s1 = symbols[s1]
    if not isinstance(s2, Symbol):
        s2 = symbols[s2]

    if s1 == s2:
        expr = s1
    else:
        expr = expressions[aliases.get(s1, s1)]
    expr = substitute_until_explicit(expr, s2)

# NB: see also https://github.com/sympy/sympy/issues/12134
def evaluate_quantity(v, quantities,
        stop_at=(numpy.ndarray, sympy.Number, numbers.Number)):
    """Evaluate numerical value of `v` using `quantities`

    Get the numerical value of variable `v`, using a dictionary of
    quantities `quantities` containing the values of other variables.

    Arguments:

        v [Symbol]
        quantities [Mapping[Symbol, Expr]]
    """
    e = expressions.get(v, v)

    values = {}
    for arg in typhon.physics.metrology.recursive_args(e,
            stop_at=(sympy.Symbol, sympy.Indexed)):
        #
        try:
            values[arg] = quantities[arg]
        except KeyError:
            values[arg] = evaluate_quantity(arg, quantities) # if this fails `arg` should be added to quantities

    # substitute numerical values into expression
    if isinstance(e, stop_at):
        return e
    elif not e.args:
        raise ValueError("I don't know the value for: {!s}".format(e))
    else:
        smb = tuple(e.free_symbols)
        return sympy.lambdify(smb, e, dummify=False)(*[values[x] for x in smb])


            
#    oldexpr = None
#    # expand expression until no more sub-expressions and s2 is explicit
#    while expr != oldexpr:
#        oldexpr = expr
#        for sym in expr.free_symbols - {s2}:
#            if aliases.get(s2, s2) in dependencies[aliases.get(sym,sym)]:
#                here = aliases.get(sym, sym)
#                expr = getattr(expr, ("replace" if sym in aliases else
#                    "subs"))(here, expressions.get(here, here))
    return expr.diff(aliases.get(s2,s2))
