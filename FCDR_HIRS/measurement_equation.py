"""Measurement equation and associated functionality
"""

import sympy
from sympy.core.symbol import Symbol

names = ("R_e a_0 a_1 a_2 C_s R_selfIWCT C_IWCT C_E R_selfE R_selfs ε λ "
         "a_3 R_refl d_PRT C_PRT k n K N h c k_b T_PRT T_IWCT B φ "
         "R_IWCT ε O_Re O_TIWCT O_TPRT")

symbols = sym = dict(zip(names.split(), sympy.symbols(names)))

expressions = {}
expressions[sym["R_e"]] = (
    sym["a_0"] + sym["a_1"]*sym["C_E"]**2 - sym["R_selfE"] + sym["O_Re"])
expressions[sym["a_0"]] = (
    -sym["a_2"] * sym["C_s"]**2 - sym["a_1"]*sym["C_s"])
expressions[sym["a_1"]] = (
    sym["R_IWCT"] + sym["R_selfIWCT"] - sym["R_selfs"] -
    sym["a_2"]*(sym["C_IWCT"]**2-sym["C_s"]**2))/(sym["C_IWCT"]-sym["C_s"])
expressions[sym["R_IWCT"]] = (
    (sympy.Integral(((sym["ε"] + sym["a_3"]) * sym["B"] +
    (1+sym["ε"]-sym["a_3"])*sym["R_refl"]) * sym["φ"], sym["λ"])) /
    sympy.Integral(sym["φ"], sym["λ"]))
expressions[sym["B"]] = (
    (2*sym["h"]*sym["c"]**2)/(sym["λ"]**5) *
    1/(sympy.exp((sym["h"]*sym["c"])/(sym["λ"]*sym["k_b"]*sym["T_IWCT"]))-1))
expressions[sym["T_IWCT"]] = (
    sympy.Sum(sympy.IndexedBase(sym["T_PRT"])[sym["n"]], (sym["n"], 0, sym["N"]))/sym["N"] + sym["O_TIWCT"])
expressions[sympy.IndexedBase(sym["T_PRT"])[sym["n"]]] = (
    sympy.Sum(sympy.IndexedBase(sym["d_PRT"])[sym["n"],sym["k"]] *
        sympy.IndexedBase(sym["C_PRT"])[sym["n"]]**sym["k"], (sym["k"], 0, sym["K"]-1))
    + sym["O_TPRT"])
expressions[sym["φ"]] = (sympy.Function("φ")(sym["λ"]))

aliases = {}
aliases[sym["T_PRT"]] = sympy.IndexedBase(sym["T_PRT"])[sym["n"]]

def recursive_substitution(e):
    """For expression 'e', substitute all the way down.

    Using the dictionary `expressions`, repeatedly substitute all symbols
    into the expression until there is nothing left to substitute.
    """
    o = None
    while o != e:
        o = e
        for sym in e.free_symbols:
            # subs only works for simple values but is faster
            # replace works for arbitrarily complex expressions but is
            # slower and may yield false positives
            e = getattr(e, ("replace" if sym in aliases else "subs"))(aliases.get(sym,sym), expressions.get(aliases.get(sym,sym), sym))
    return e

dependencies = {e: recursive_substitution(expressions.get(aliases.get(e,e), e)).free_symbols-{e}
        for e in symbols.values()}

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

    expr = expressions[aliases.get(s1, s1)]
    oldexpr = None
    # expand expression until no more sub-expressions and s2 is explicit
    while expr != oldexpr:
        oldexpr = expr
        for sym in expr.free_symbols - {s2}:
            if s2 in dependencies[sym]:
                here = aliases.get(sym, sym)
                expr = getattr(expr, ("replace" if sym in aliases else
                    "subs"))(here, expressions.get(here, here))
    return expr.diff(s2)
