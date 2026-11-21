"""Data for chapters 11 and 13: asymptotic phase of the reduced Hodgkin-Huxley model (paper eq. 4.1)
on a fine grid and on zoom windows, each point integrated with XPPAUT (pyauto-xpp) until it is
within 1e-6 (scaled units) of the cycle.
Run:  .venv/bin/python scripts/isochron/figs/rhh_phasemaps.py"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import iso_lib as L
import figkit as K

SC = (150.0, 0.5)
WINDOWS = {
    "full": ((-90, 60), (0.3, 0.8), 601),
    "wide": ((-200, 150), (0.0, 1.1), 401),
    "left": ((-80, -40), (0.36, 0.56), 601),
    "eq": ((-62, -57), (0.39, 0.415), 501),
    "eq2": ((-60.1, -59.1), (0.4006, 0.4046), 401),
}

if __name__ == "__main__":
    m = L.load("rhh")
    t, X, T = L.periodic_orbit(m, (-60, 0.45), dt=0.0002)
    K.savedata("rhh_cycle", t=t, X=X, T=T)
    which = sys.argv[1:] or list(WINDOWS)
    for name in which:
        (v0, v1), (n0, n1), N = WINDOWS[name]
        xs = np.linspace(v0, v1, N); ys = np.linspace(n0, n1, N)
        t0 = time.time()
        th, nper = L.phase_map("rhh", {}, X, T, xs, ys, scale=SC, tol=1e-6, maxper=400, procs=7)
        K.savedata(f"rhh_phase_{name}", xs=xs, ys=ys, th=th, nper=nper)
        print(f"{name}: {N}x{N} in {time.time() - t0:.0f} s, NaN {np.isnan(th).sum()}, max periods {nper.max()}", flush=True)
