#!/usr/bin/env python3
"""
fhn_isochrons.py -- the isochrons of the FitzHugh-Nagumo oscillator, the simple way.

IDEA
  Every point x in the plane eventually lands on the limit cycle Gamma and then moves around
  it in step with one particular point of Gamma. That point's phase is the ASYMPTOTIC PHASE
  of x. An ISOCHRON is a set of points with the same asymptotic phase: a contour line of the
  phase. Here we measure the asymptotic phase on a grid by simulation (brute force) and draw
  its contour lines.

HOW THE PHASE IS MEASURED
  1. Phase 0 is the top of the spike on Gamma (the point gamma0 where v is largest).
  2. Starting at gamma0, the cycle's first spike (v crossing 0 upward) comes at time t_c0.
  3. Starting at x, wait until the trajectory has settled on Gamma, and take the time t_s of a
     spike. If x spikes earlier than the cycle by a time D, it is ahead by D/T of a cycle:
         theta(x) = ((t_c0 - t_s) / T) mod 1

Runs XPPAUT 8.2 through pyauto-xpp:
    pip install pyauto-xpp numpy matplotlib tqdm \
        --find-links https://github.com/ZetaOne/pyauto-xpp/releases/expanded_assets/v0.3.0
    python fhn_isochrons.py
Writes fhn_isochrons.png and the .ode files it used (fhn.ode, fhn_network.ode).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyxpp.builder import ModelBuilder
from tqdm import tqdm

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------------------
# 1. The model:  v' = v - v^3/3 - w + I,   w' = eps (v + a - b w)
#    With I = 0.5 the equilibrium is an unstable spiral and every trajectory goes to one
#    stable limit cycle: the neuron fires periodically.
# ---------------------------------------------------------------------------------------
PARS = dict(i=0.5, a=0.7, b=0.8, eps=0.08)
fhn = (ModelBuilder("fhn")
       .ode("v", "v-v^3/3-w+i")
       .ode("w", "eps*(v+a-b*w)")
       .parameter(**PARS)
       .initial(v=-1, w=1)
       .option(total=400, dt=0.01, meth="rk4", maxstor=200000))
print(fhn.render())                                  # the .ode file XPPAUT reads
m = fhn.load(HERE / "fhn.ode")

# ---------------------------------------------------------------------------------------
# 2. The limit cycle, its period T, and phase 0.
#    XPP: Initialconds > Go (integrate onto the cycle), then Initialconds > Periodic (sets
#    total = T). Phase 0 = the point of the cycle where v is largest.
# ---------------------------------------------------------------------------------------
m.run()
m.buttons.xi_vs_t("v")
m.buttons.initialconds.periodic()
T = m.numerics["total"]
cycle = m.run(keep=False, total=T, dt=0.001)
k = int(np.argmax(cycle["v"]))
gamma0 = (cycle["v"][k], cycle["w"][k])
m.set(v=gamma0[0], w=gamma0[1])
cycle = m.run(keep=False, total=T, dt=0.001)         # one period, starting at phase 0
print(f"period T = {T:.4f}, phase 0 at (v, w) = ({gamma0[0]:.4f}, {gamma0[1]:.4f})")


def spike_times(t, v, threshold=0.0):
    """times where v crosses the threshold upward (linear interpolation between samples)"""
    i = np.nonzero((v[:-1] < threshold) & (v[1:] >= threshold))[0]
    return t[i] + (threshold - v[i]) / (v[i + 1] - v[i]) * (t[i + 1] - t[i])


ref = m.run(keep=False, total=3 * T, dt=0.01)        # the cycle from gamma0
t_c0 = spike_times(ref.t, ref["v"])[0]               # its first spike

# ---------------------------------------------------------------------------------------
# 3. Many starting points at once: an XPP network of N uncoupled copies of the model,
#    v[0..N-1] and w[0..N-1]. One XPP run integrates N starting points.
# ---------------------------------------------------------------------------------------
N = 200
net = (ModelBuilder("fhn_network")
       .ode(f"v[0..{N - 1}]", "v[j]-v[j]^3/3-w[j]+i")
       .ode(f"w[0..{N - 1}]", "eps*(v[j]+a-b*w[j])")
       .parameter(**PARS)
       .option(dt=0.01, meth="rk4", maxstor=400000)
       .load(HERE / "fhn_network.ode"))


def asymptotic_phase(points, periods=8):
    """the asymptotic phase of each point (NaN if it has not settled on the cycle)"""
    phase = np.full(len(points), np.nan)
    for start in range(0, len(points), N):
        chunk = points[start:start + N]
        ics = {}
        for j in range(N):                           # unused copies repeat the first point
            p = chunk[j] if j < len(chunk) else chunk[0]
            ics[f"v{j}"], ics[f"w{j}"] = p[0], p[1]
        net.ics.update(ics)
        # integrate for `periods` periods, keep only the last 2.5 (after the transient)
        data = net.run(total=periods * T, trans=(periods - 2.5) * T).array
        t, V = data[:, 0], data[:, 1:1 + len(chunk)]   # columns: t, v0..v(N-1), w0..w(N-1)
        for c in range(len(chunk)):
            s = spike_times(t, V[:, c])
            settled = len(s) >= 2 and abs(s[-1] - s[-2] - T) < 1e-3 * T
            if settled:
                phase[start + c] = ((t_c0 - s[-1]) / T) % 1
    return phase


# ---------------------------------------------------------------------------------------
# 4. The phase map on a grid of the (v, w) plane.
# ---------------------------------------------------------------------------------------
V, W = np.meshgrid(np.linspace(-2.5, 2.5, 101), np.linspace(-1.0, 2.0, 76))
grid = np.c_[V.ravel(), W.ravel()]
TH = np.concatenate([asymptotic_phase(grid[s:s + 1000])
                     for s in tqdm(range(0, len(grid), 1000), desc="phase map")]).reshape(V.shape)


def without_jumps(F):
    """F with the grid points next to a jump > 0.5 hidden: there the phase wraps from 1 to 0,
    and a contour line drawn across the wrap would be a false line"""
    bad = np.zeros(F.shape, bool)
    dx, dy = np.abs(np.diff(F, axis=1)) > 0.5, np.abs(np.diff(F, axis=0)) > 0.5
    bad[:, :-1] |= dx; bad[:, 1:] |= dx; bad[:-1, :] |= dy; bad[1:, :] |= dy
    return np.ma.masked_where(bad | np.isnan(F), F)


# ---------------------------------------------------------------------------------------
# 5. The picture: the phase as colour, the isochrons as its contour lines.
#    Isochrons 0.3 ... 0.7 are contours of theta; 0.8, 0.9, 0, 0.1, 0.2 are contours of
#    theta shifted by half a cycle, so that none of them sits on the 1 -> 0 wrap.
# ---------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
pm = ax.pcolormesh(V, W, TH, cmap="hsv", shading="auto", vmin=0, vmax=1, alpha=0.7)
ax.contour(V, W, without_jumps(TH), [0.3, 0.4, 0.5, 0.6, 0.7], colors="k", linewidths=1)
ax.contour(V, W, without_jumps((TH + 0.5) % 1), [0.3, 0.4, 0.5, 0.6, 0.7], colors="k", linewidths=1)
ax.plot(cycle["v"], cycle["w"], color="k", lw=3, label="limit cycle")
ax.plot(*gamma0, "o", color="k", ms=8, label="phase 0")
fig.colorbar(pm, label="asymptotic phase θ")
ax.set(xlabel="v", ylabel="w", title="FitzHugh-Nagumo: isochrons θ = 0, 0.1, …, 0.9 (black lines)")
ax.legend(loc="lower right", fontsize=8)
fig.savefig(HERE / "fhn_isochrons.png", dpi=130)
print("wrote fhn_isochrons.png")
plt.show()
