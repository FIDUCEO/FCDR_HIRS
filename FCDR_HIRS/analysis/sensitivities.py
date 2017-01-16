#!/usr/bin/env python

"""Symbolically calculate sensitivity parameters
"""

import sympy
import logging

(Re, a2, Cs, RIWCT, RselfIWCT, CIWCT, CE, RselfE, Rselfs, ε, λ, TIWCT, a3,
    Rrefl, d_PRTnk, Cprtn, k, n, K, N) = sympy.symbols(
        "R_e a_2 C_s R_IWCT R_selfIWCT C_IWCT C_E R_selfE R_selfs ε λ "
        "T_IWCT a_3 R_refl d_prtn C_prtk k n K N")
# secondary versions for easier representation
(T_prta, T_IWCTa, R_IWCTa, a1a, a0a) = sympy.symbols(
    "T_prta T_IWCTa R_IWCTa a1a a0a")
ε = sympy.Function("ε")
B = sympy.Function("B")
φ = sympy.Function("φ")

T_PRT = sympy.Sum(d_PRTnk * Cprtn**k, (k, 0, K-1))
T_IWCT = sympy.Sum(T_PRT, (n, 1, N))/N

RIWCT = (sympy.Integral(((ε(λ, TIWCT) + a3) * B(λ, TIWCT) + (1-ε(λ,
            TIWCT)-a3)*Rrefl) * φ(TIWCT, λ), λ)) / sympy.Integral(
                φ(TIWCT, λ), λ)


a1 = (RIWCT + RselfIWCT - Rselfs - a2*(CIWCT**2-Cs**2))/(CIWCT-Cs)
a0 = -2*Cs**2 -a1*Cs
R_e = a0 + a1*CE + a2*CE**2 - RselfE
#R_e = -a2*Cs**2 - S * Cs * S * CE + a2*CE**2 - RselfE

def main():
    # sm.diff(a0d + a1d*s.CE + a2d*s.CE**2, s.Cs)
    pass
