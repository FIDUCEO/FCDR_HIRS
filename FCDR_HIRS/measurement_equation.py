"""Measurement equation and associated functionality
"""

import numbers

import numpy
import scipy.constants
import xarray

import sympy
from sympy.core.symbol import Symbol

import typhon.physics.metrology
from typhon.physics.units.common import ureg
from typhon.physics.units.tools import UnitsAwareDataArray as UADA

version = "β"

names = ("R_e a_0 a_1 a_2 C_s R_selfIWCT C_IWCT C_E R_selfE R_selfs ε λ Δλ "
         "a_3 R_refl d_PRT C_PRT a n m A N M h c k_b T_PRT T_IWCT B φn "
         "R_IWCT O_Re O_TIWCT O_TPRT α β Tstar λstar O_RIWCT f Δf fstar "
         "ν Δν νstar T_bstar T_b a_4 S h_0 h_1 h_2 h_3")

symbols = sym = dict(zip(names.split(), sympy.symbols(names)))

class ExpressionDict(dict):
    """Special version of dictionary where keys and values are sympy expressions

    Special behaviour: When there is an IndexedBase key, this will match
    even if the index is different.  For example, if the key is T_PRT[n]
    and we request T_PRT[0], it will return D[T_PRT[n]] substituting n by
    0.
    """

    def __getitem__(self, k):
        if isinstance(k, sympy.tensor.indexed.Indexed):
            try:
                return super().__getitem__(k)
            except KeyError:
                # search through indexed keys
                matches = {km for km in self.keys()
                        if isinstance(km, sympy.tensor.indexed.Indexed)
                    and k.args[0].args[0] == km.args[0].args[0]
                    and len(k.args) == len(km.args)}
                if len(matches) == 1:
                    km = matches.pop()
                    v = self[km]
                    try:
                        return v.subs(
                            {str(k):v for (k,v) in zip(km.args[1:],
                                k.args[1:])})
                    except AttributeError: # not an expression
                        return v
                elif len(matches) == 0:
                    raise
                else:
                    raise KeyError("Multiple keys fulfill "
                        "fuzzy expression match")
        else:
            return super().__getitem__(k)

    def __contains__(self, k):
        try:
            self[k]
        except KeyError:
            return False
        else:
            return True

    def __repr__(self):
        return "ExpressionDict({" + super().__repr__() + "})"

    def get(self, k, *args):
        if k in self:
            return self[k]
        else:
            return super().get(k, *args)

    def __setitem__(self, k, v):
        super().__setitem__(k, v) # here only so I can set breakpoint

expressions = ExpressionDict()
expressions[sym["R_e"]] = (
    sym["a_0"] + sym["a_1"]*sym["C_E"] + sym["a_2"]*sym["C_E"]**2 -
    sym["R_selfE"] + sym["O_Re"] + sym["a_4"])
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
# NB 2017-03-02: B(λ) results in spectral radiance per wavelength, which
# is a different quantity than spectral radiance per frequency or
# wavenumber and cannot be directly converted.  Specrad per wavenumber can
# be converted but when evaluating uncertainties we have squares of that,
# for which I don't have a conversion rule, so let's just keep it per
# frequency for now.
# converted, but when evaluating 
#expressions[sym["B"]] = (
#    (2*sym["h"]*sym["c"]**2)/((sym["λ"])**5) *
#    1/(sympy.exp((sym["h"]*sym["c"])/((sym["λ"])*sym["k_b"]*sym["T_IWCT"]))-1))
#expressions[sym["B"]] = (
#     (2 * sym["h"] * sym["c"]**2 * sym["ν"]**3) / (
#     sympy.exp(sym["h"]*sym["c"]*sym["ν"]/(sym["k_b"]*sym["T_IWCT"])-1)))
expressions[sym["B"]] = (
    (2 * sym["h"] * sym["f"]**3 / sym["c"]**2) *
    (1 / (sympy.exp((sym["h"]*sym["f"])/(sym["k_b"]*sym["T_IWCT"]))-1)))
if version == "β":
    expressions[sym["B"]] = expressions[sym["B"]].subs(
        {sym["T_IWCT"]: sym["Tstar"],
         sym["λ"]: sym["λstar"],
         sym["f"]: sym["fstar"],
         sym["ν"]: sym["νstar"]})
    expressions[sym["Tstar"]] = sym["α"] + sym["β"]*sym["T_IWCT"]
expressions[sym["T_IWCT"]] = (
    sympy.Sum(sympy.IndexedBase(sym["T_PRT"])[sym["n"]], (sym["n"], 0, sym["N"]))/sym["N"] + sym["O_TIWCT"])
expressions[sympy.IndexedBase(sym["T_PRT"])[sym["n"]]] = (
    sympy.Sum(sympy.IndexedBase(sym["T_PRT"])[sym["n"],sym["m"]], (sym["m"], 0, sym["M"]))/sym["M"])
expressions[sympy.IndexedBase(sym["T_PRT"])[sym["n"],sym["m"]]] = (
    sympy.Sum(sympy.IndexedBase(sym["d_PRT"])[sym["n"],sym["a"]] *
        sympy.IndexedBase(sym["C_PRT"])[sym["n"],sym["m"]]**sym["a"], (sym["a"], 0, sym["A"]-1))
    + sympy.IndexedBase(sym["O_TPRT"])[sym["n"],sym["m"]])

#expressions[sym["φn"]] = (sympy.Function("φn")(sym["λ"]+sym["Δλ"]))
#expressions[sym["φn"]] = (sympy.Function("φn")(sym["ν"]+sym["Δν"]))
expressions[sym["φn"]] = (sympy.Function("φn")(sym["f"]+sym["Δf"]))

# not strictly part of measurement equation, but to convert from radiance
# to BT at the end
expressions[sym["T_bstar"]] = sym["fstar"]*sym["h"]/(sym["k_b"]*sympy.log(
    1 + 2*sym["fstar"]**3*sym["h"]/(sym["R_e"]*sym["c"]**2)))
expressions[sym["T_b"]] = (sym["T_bstar"] - sym["α"])/sym["β"]

expressions[sym["c"]] = sympy.sympify(scipy.constants.speed_of_light)
expressions[sym["h"]] = sympy.sympify(scipy.constants.Planck)
expressions[sym["k_b"]] = sympy.sympify(scipy.constants.Boltzmann)
expressions[sym["M"]] = sympy.Number(5)
expressions[sym["N"]] = sympy.Number(5) # FIXME: actually depends on HIRS version...
expressions[sym["A"]] = sympy.Number(6)

units = {}
units[sym["c"]] = ureg.c
units[sym["h"]] = ureg.h
units[sym["k_b"]] = ureg.k
units[sym["N"]] = units[sym["M"]] = ureg.dimensionless

aliases = {}
#aliases[sym["T_PRT"]] = sympy.IndexedBase(sym["T_PRT"])[sym["n"]]
#aliases[sym["C_PRT"]] = sympy.IndexedBase(sym["C_PRT"])[sym["n"]]
#aliases[sym["d_PRT"]] = sympy.IndexedBase(sym["d_PRT"])[sym["n"],sym["k"]]

def recursive_substitution(e, stop_at=None, return_intermediates=False,
        expressions=expressions):
    """For expression 'e', substitute all the way down.

    Using the dictionary `expressions`, repeatedly substitute all symbols
    into the expression until there is nothing left to substitute.
    """
    o = None
    intermediates = set()
    if isinstance(e, sympy.Symbol) and e in expressions:
        return recursive_substitution(expressions[e],
            stop_at=stop_at, return_intermediates=return_intermediates,
            expressions=expressions)
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

# make sure no symbol occurs in multiple sub-measurement equations
# and all the harmonisation parameters occur directly
expressions[sym["R_e"]] = recursive_substitution(sym["R_e"],
    stop_at=sym["B"])
del expressions[sym["a_0"]]
del expressions[sym["a_1"]]
del expressions[sym["R_IWCT"]]
all_args = set()
for (s, e) in expressions.items():
    if s in (sym["T_b"], sym["T_bstar"]): # exempted
        continue
    new_args = typhon.physics.metrology.recursive_args(e)
    if new_args & all_args:
        raise ValueError("Duplicate symbols found")
    all_args |= new_args

dependencies = {}
for s in symbols.values():
    (e, im) = recursive_substitution(
                expressions.get(aliases.get(s,s),s),
                return_intermediates=True)
    dependencies[aliases.get(s, s)] = typhon.physics.metrology.recursive_args(e) | im

functions = {}
for (sn, s) in symbols.items():
    if s in dependencies:
        if dependencies[s]:
            functions[s] = sympy.Function(sn)(*(aliases.get(sm, sm) for sm in dependencies[s]))

names = {
    sym["R_selfE"]: "Rself",
    sym["R_selfIWCT"]: "RselfIWCT",
    sym["R_selfs"]: "Rselfspace",
    sym["C_E"]: "C_Earth",
    sym["R_e"]: "R_Earth",
    sym["R_refl"]: "R_refl",
    sym["α"]: "α",
    sym["β"]: "β",
    #sym["λstar"]: "λ_eff",
    #sym["νstar"]: "ν_eff",
    sym["fstar"]: "f_eff",
    sym["ε"]: "ε",
    sym["a_3"]: "a_3",
    sym["a_4"]: "a_4",
    sym["C_s"]: "C_space",
    sym["C_IWCT"]: "C_IWCT",
    sym["R_IWCT"]: "R_IWCT",
    sym["a_0"]: "offset",
    sym["a_1"]: "slope",
    sym["a_2"]: "a_2",
    sym["T_IWCT"]: "T_IWCT_calib_mean",
    sym["N"]: "prt_number_iwt",
    sym["M"]: "prt_reading",
    sym["A"]: "prt_iwct_polynomial_order",
    sym["Tstar"]: "Tstar",
    sym["B"]: "B",
    sym["O_Re"]: "O_Re",
    sym["O_TIWCT"]: "O_TIWCT",
    sym["O_RIWCT"]: "O_RIWCT",
    sym["h"]: "planck_constant",
    sym["c"]: "speed_of_light",
    sym["k_b"]: "boltzmann_constant",
    sym["T_b"]: "T_b",
    sym["T_bstar"]: "T_bstar",
    sym["C_PRT"]: "prt_iwct_counts",
    sym["d_PRT"]: "prt_iwct_coefficients",
    sym["T_PRT"]: "prt_iwct_temperature",
    sym["O_TPRT"]: "O_TPRT",
}


# version for simplified harmonisation

expression_Re_simplified = recursive_substitution(
    expressions[symbols["R_e"]],
    expressions=expressions,
    stop_at=symbols["T_IWCT"]).subs(
        {symbols["R_selfIWCT"]: 0,
         symbols["O_RIWCT"]: 0,
         symbols["O_Re"]: 0,
         symbols["R_refl"]: 0,
         symbols["R_selfs"]: 0,
         symbols["a_4"]: symbols["h_1"],
         symbols["a_3"]: symbols["h_3"],
         symbols["a_2"]: symbols["h_2"]})

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

        s1: sympy.Expr
        s2: sympy.Expr
    """

    if not isinstance(s1, sympy.Expr):
        s1 = symbols[s1]
    if not isinstance(s2, sympy.Expr):
        s2 = symbols[s2]

    if s1 == s2:
        expr = s1
    else:
        expr = expressions[aliases.get(s1, s1)]
    expr = substitute_until_explicit(expr, s2)
    return expr.diff(s2)

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
    if isinstance(e, sympy.Number):
        return UADA(float(e),
            name = names.get(v, str(v)),
            dims = (),
            attrs = {"units": str(units[v])})
    elif isinstance(e, stop_at):
        return e
    elif not e.args:
        raise ValueError("I don't know the value for: {!s}".format(e))
    else:
        smb = tuple(e.free_symbols)
        return sympy.lambdify(smb, e, dummify=False, modules=numpy)(
            *[values[x].to_root_units() for x in smb])
