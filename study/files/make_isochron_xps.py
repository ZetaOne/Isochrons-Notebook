#!/usr/bin/env python3
"""Write the XPPAUT 8.4 script (and its BVP .ode) for isochrons of the reduced
Hodgkin-Huxley model by Osinga & Moehlis's continuation method.

Pseudocode:
  1. once, beforehand (pyauto-xpp): T, gamma0, the adjoint
     for each theta: gamma(theta) and n_s(theta) (unit vector along the straight isochron)
  2. write isochron_bvp.ode:  u' = per F(u) on [0,1],  u(1) = gamma + d n_s
  3. write isochrons.xps: for each theta, XPPAUT itself does
       stage 1: start on Γ from gamma(theta), AUTO Bdry Value (6 steps), then
                continue in d both ways (inside and outside Γ); save the diagram
       stage k: from an end of stage k-1, integrate k periods, AUTO Bdry Value, continue
  4. run:  xvfb-run ./xppaut-8.4/xppaut isochron_bvp.ode -script isochrons.xps -scriptexit

    python make_isochron_xps.py THETA [THETA ...]      (default: 0)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = Path.home() / "Documents/isochron-shell-scripts/isochron_continuation.py"

# --- 1. T, gamma0, adjoint, phase_point(): the code of isochron_continuation.py, unchanged
code = SRC.read_text()
exec(code[:code.index("def bvp_builder")])          # defines T, gamma0, phase_point, FUN, FV, FN, PARS

thetas = [float(a) for a in sys.argv[1:]] or [0.0]
ETA, DS = 1e-4, 1e-5

# --- 2. the BVP .ode (d is the first parameter, so it is AUTO's continuation parameter)
g0, n0 = phase_point(thetas[0])
ode = ["# isochron_bvp.ode -- orbit segments u' = per*F(u) on [0,1] that end on the line",
       "# through gamma along n_s:  u(1) = gamma + d*n_s   (Osinga & Moehlis 2010)"]
ode += [f"{s}={e}" for s, e in FUN]
ode += [f"par d=0,per={T:.10g},gv={g0[0]:.10g},gn={g0[1]:.10g},tv={n0[0]:.10g},tn={n0[1]:.10g}",
        "par " + ",".join(f"{k}={v}" for k, v in PARS.items()),
        f"v'=per*({FV})", f"n'=per*({FN})",
        "bndry v'-gv-d*tv", "bndry n'-gn-d*tn",
        f"init v={g0[0]:.10g},n={g0[1]:.10g}",
        "@ total=1,dt=1e-4,maxstor=400000,meth=rk4,bound=1e4,jac_eps=1e-7,newt_tol=1e-10",
        "@ ntst=100,nmax=6,npr=100000,ds=1e-5,dsmin=1e-8,dsmax=1e-6,ncol=4",
        "@ epsl=1e-8,epsu=1e-8,epss=1e-6,parmin=-1,parmax=1,normmax=1e6",
        "@ autovar=v,autoxmin=-1e-4,autoxmax=1e-4,autoymin=-90,autoymax=52",
        "done"]
(HERE / "isochron_bvp.ode").write_text("\n".join(ode) + "\n")

# --- 3. the script
START = "Ntst=100|Nmax=6|NPr=100000|Ds=1e-05|Dsmin=1e-08|Dsmax=1e-06|Par Min=-1|Par Max=1|ITNW=5"
GO = "Nmax=5000|Ds={ds}|Dsmax=0.1|Par Min=-0.0001|Par Max=0.0001"
xps = ["# isochrons.xps -- written by make_isochron_xps.py; run with",
       "#   xvfb-run ./xppaut-8.4/xppaut isochron_bvp.ode -script isochrons.xps -scriptexit",
       "keys f a                               # open the AUTO window"]
for th in thetas:
    g, ns = phase_point(th)
    tag = f"{th:.3f}"
    xps += [f"# ---- theta = {th}",
            f"set d=0;per={T:.10g};gv={g[0]:.10g};gn={g[1]:.10g};tv={ns[0]:.10g};tn={ns[1]:.10g}",
            f"set v={g[0]:.10g};n={g[1]:.10g}",
            "keys i g                               # Γ from gamma(theta), one period",
            "try                                    # AUTO File -> Reset diagram (asks only",
            "  autokeys f r                         #  when there is a diagram)",
            "  ans y",
            "end",
            "autokeys n                             # AUTO Numerics: the start",
            f"ans {START}",
            "autokeys r b                           # Run -> Bdry Value (6 steps from Γ)",
            "auto goto last                         # its end (d back near 0)",
            "autokeys n",
            f"ans {GO.format(ds=DS)}",
            "autokeys r                             # continue one way",
            "auto goto last branch 1                # the start's end again",
            "autokeys n",
            f"ans {GO.format(ds=-DS)}",
            "autokeys r                             # and the other way",
            "autokeys f s                           # File -> Save diagram",
            f"ans stage1_{tag}.dgm"]
xps += ["quit"]
(HERE / "isochrons.xps").write_text("\n".join(xps) + "\n")
print(f"T = {T}; wrote isochron_bvp.ode and isochrons.xps for theta = {thetas}")
