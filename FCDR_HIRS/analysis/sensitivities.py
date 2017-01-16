#!/usr/bin/env python

"""Symbolically calculate sensitivity parameters
"""

import sympy

(Re, a2, Cs, RIWCT, RselfIWCT, CIWCT, CE, RselfE, Rselfs, ε, ν, TIWCT, a1,
    Rrefl, dprtn, Cprtkn, k, n, K, N) = sympy.symbols(
        "R_e a_2 C_s R_IWCT R_selfIWCT C_IWCT C_E R_selfE R_selfs ε ν "
        "T_IWCT a_1 R_refl d_prtn C_prtk k n K N")
ε = sympy.Function("ε")
Β = sympy.Function("B")
φ = sympy.Function("φ")

T_IWCT = Sum(Sum(dprtn * Cprtk**n, (n, 1, N)), (k, 1, K)) / (K*N)

RIWCT = (Integral(((ε(ν, TIWCT) + a1) * B(ν, TIWCT) + (1-ε(ν,
            TIWCT)-a1)*Rrefl) * φ(TIWCT, ν), ν)) / Integral(
                φ(TIWCT, ν), ν)

S = (RIWCT + RselfIWCT - Rselfs - a2*(CIWCT**2-Cs**2))/(CIWCT-Cs)
R_e = -a2*Cs**2 - S * Cs * S * CE + a2*CE**2 - RselfE

def main():
    pass
