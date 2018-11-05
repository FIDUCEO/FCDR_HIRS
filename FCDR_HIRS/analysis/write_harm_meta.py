"""Write harmonisation meta info
"""

import datetime
now = datetime.datetime.now
import sympy
import typhon.physics.units
from typhon.datasets.tovs import norm_tovs_name

from .. import fcdr
from .. import measurement_equation as me

table_file = "/group_workspaces/cems2/fiduceo/Data/Harmonisation_matchups/HIRS/coef_ch{ch:d}.dat"

def write_table_for_channel(ch, fn):
    """Write table for channel to file
    """

    with open(fn, "wt", encoding="utf-8") as fp:
        fp.write(f"{'sensor':<6s} "
                 f"{'fstar':<10s} {'alpha':<10s} {'beta':<10s} "
                 f"{'Δfstar':<10s} {'Δalpha':<10s} {'Δbeta':<10s}\n")
        for satname in sorted(
                (fcdr.HIRS2FCDR.satellites.keys()|
                 fcdr.HIRS3FCDR.satellites.keys()|
                 fcdr.HIRS4FCDR.satellites.keys())-{'noaa13'}):
            srf = typhon.physics.units.SRF.fromRTTOV(
                norm_tovs_name(satname, "RTTOV"), "hirs", ch)
            (α, β, λ_eff, Δα, Δβ, Δλ_eff) = srf.estimate_band_coefficients(
                satname, "fcdr_hirs", ch, include_shift=False)
            f_eff = λ_eff.to("Hz", "sp")
            Δf_eff= ((λ_eff+Δλ_eff).to("Hz", "sp") -
                     (λ_eff-Δλ_eff).to("Hz", "sp"))/2
            short = norm_tovs_name(satname, "BC")
            fp.write(f"{short:<6s} "
                     f"{float(f_eff):<10.3e} {float(α):<10.5f} {float(β):<10.6f} "
                     f"{float(Δf_eff):<10.3e} {float(Δα):<10.4e} {float(Δβ):<10.4e}\n")

def main():
    expr = me.expression_Re_simplified
    print("Full expression:")
    sympy.pprint(expr)
    print(sympy.latex(expr))
    print(expr)
    print("Free symbols:")
    free = expr.free_symbols.copy()
    print(free)
    provided = {me.sym["C_E"], me.sym["T_IWCT"], me.sym["C_s"],
              me.sym["C_IWCT"], me.sym["R_selfE"]}
    consts = free & me.units.keys()
    harms = {e for e in free if str(e).startswith("h_")}
    assumed = {me.sym["ε"], me.sym["fstar"], me.sym["α"], me.sym["β"]}
    print("Provided in data", provided)
    print("Fundamental constants", consts)
    print("Harmonisation", harms)
    print("Assumed", assumed)
    remaining = free - provided - consts - assumed - harms
    if remaining:
        raise ValueError(f"Not determined: {remaining!s}")
    for s in {me.sym["C_E"], me.sym["T_IWCT"], me.sym["C_s"],
              me.sym["C_IWCT"], me.sym["R_selfE"]}:
        print(f"Sensitivity coefficient for {s!s}:")
        D = expr.diff(s)
#        sympy.pprint(D)
        print(sympy.latex(D))
        print(D)

    for s in consts:
        if s in me.units:
            print(s, "=", me.expressions[s], me.units[s])

    # values for the assumed
    for ch in range(1, 20):
        fn = table_file.format(ch=ch)
        print(now(), f"Writing for channel {ch:d} to {fn:s}")
        write_table_for_channel(ch, fn)
