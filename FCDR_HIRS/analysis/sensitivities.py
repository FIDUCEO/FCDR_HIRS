#!/usr/bin/env python

"""Symbolically calculate sensitivity parameters
"""

import pathlib
import sympy
import sympy.printing
import typhon.config
import argparse
import datetime
import signal
now = datetime.datetime.now

from .. import measurement_equation as me

template="""
<!DOCTYPE html>
<html>
<head>
<script type="text/javascript"
  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
<title>tex texample</title>
</head>
<body>

$${latex:s}$$

</body>
</html>
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Write sensitivity coefficients")

    parser.add_argument("start",
        action="store",
        type=str,
        help="Write sensitivity coefficients relative to this",
        choices=me.symbols.keys(),
        default="R_e")

    parser.add_argument("--assume_constant",
        action="store",
        type=str,
        nargs="*",
        help="Assume those to be constant",
        choices=me.symbols.keys(),
        default=["K", "N", "c", "h", "k", "n", "k_b"])

    parser.add_argument("--time_limit",
        action="store",
        type=int,
        default=30,
        help="Maximum time in seconds to try .doit() per factor")

    return parser.parse_args()

class TimeOut(Exception):
    pass

def handler(signum, frame):
    raise TimeOut("Giving up")
signal.signal(signal.SIGALRM, handler)

def print_and_write(eq, fn):
    print(now(), "Writing to {!s}".format(fn))
    fn.parent.mkdir(parents=True, exist_ok=True)
    with fn.open('w') as fp:
        fp.write(template.format(latex=
            sympy.printing.latex(eq).replace(
                "ε", r"\epsilon").replace(
                "λ", r"\lambda").replace(
                "φ", r"\phi").replace(
                "T_{IWCT}", "{T_{IWCT}}").replace(
                "T_{PRT}", "{T_{PRT}}").replace(
                "d_{PRT}", "{d_{PRT}}").replace(
                "C_{PRT}", "{C_{PRT}}")))
#    with fn_mathml.open('w') as fp:
#        fp.write(sympy.
    sympy.pprint(eq)
        
outdir = pathlib.Path(
    typhon.config.conf["main"]["myscratchdir"]) / "sensitivity"

def main():
    p = parse_args()
    #e_main = me.expressions[me.symbols[p.start]]
    for symname in me.symbols.keys() - set(p.assume_constant):
        #sym = me.symbols[symname]
        print(now(), "Evaluating ∂{:s}/∂{:s}".format(p.start, symname))
        e = me.calc_sensitivity_coefficient(p.start, symname)
        if e==0:
            print("Zero… not writing")
            continue
        try:
            print(now(), "Simplifying")
            es_mod = e.simplify(ratio=2)
            es_extr = e.simplify(ratio=50)
            # doit is veeeery slow
            try:
                signal.alarm(p.time_limit)
                print(now(), "“Doit”")
                e = es_extr.doit()
            except TimeOut:
                print(now(), "no result after {:d} seconds, giving up".format(p.time_limit))
                e = es_mod
            else:
                signal.alarm(0) # reset alarm
                print(now(), "Simplify again")
                e = e.simplify(ratio=2)
        except TypeError as t:
            print(now(), "Failed with", t)
        print(now(), "Writing")
        print_and_write(e,
                        outdir / "latex_d{:s}d{:s}.html".format(p.start, symname),
#                        mathml=outdir / "mathml_d{:s}d{:s}.xml".format(p_start, symname)
                        )
