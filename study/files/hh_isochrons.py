#!/usr/bin/env python3
"""
hh_isochrons.py -- the isochrons of the reduced Hodgkin-Huxley neuron (Osinga & Moehlis 2010,
eq. 4.1), the simple way: by measuring the asymptotic phase on a grid.

Same method as fhn_isochrons.py (read that first):
  theta(x) = ((t_c0 - t_s) / T) mod 1
where t_s is a spike of the trajectory from x after it has settled on the limit cycle, and
t_c0 the first spike of the cycle after phase 0 (the top of the spike, gamma0). A spike is
V crossing -20 mV upward.

What you will see: outside the cycle the isochrons are almost horizontal; inside, almost all
phases are squeezed into a thin strip along the middle branch of the V-nullcline, much thinner
than the grid. That strip is why Osinga & Moehlis need continuation (reproduce_isochrons.py).

Runs XPPAUT 8.2 through pyauto-xpp:
    pip install pyauto-xpp numpy matplotlib tqdm \
        --find-links https://github.com/ZetaOne/pyauto-xpp/releases/expanded_assets/v0.3.0
    python hh_isochrons.py
Writes hh_isochrons.png and the .ode files it used (reduced_hh.ode, reduced_hh_network.ode).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyxpp.builder import ModelBuilder
from tqdm import tqdm

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------------------
# 1. The model (the paper's eq. 4.1): V and the potassium gate n. The fast sodium gate m is
#    replaced by m_inf(V) and the inactivation h by 0.8 - n.
# ---------------------------------------------------------------------------------------
FUNCTIONS = [("am(v)", "0.1*(v+40)/(1-exp(-(v+40)/10))"), ("bm(v)", "4*exp(-(v+65)/18)"),
             ("an(v)", "0.01*(v+55)/(1-exp(-(v+55)/10))"), ("bn(v)", "0.125*exp(-(v+65)/80)"),
             ("minf(v)", "am(v)/(am(v)+bm(v))")]
PARS = dict(iapp=10, gna=120, gk=36, gl=0.3, ena=50, ek=-77, el=-54.4)
VDOT = "iapp-gna*minf({v})^3*(0.8-{n})*({v}-ena)-gk*{n}^4*({v}-ek)-gl*({v}-el)"
NDOT = "an({v})*(1-{n})-bn({v})*{n}"

hh = ModelBuilder("reduced_hh")
for sig, expr in FUNCTIONS:
    hh.function(sig, expr)
hh = (hh.ode("v", VDOT.format(v="v", n="n")).ode("n", NDOT.format(v="v", n="n"))
      .parameter(**PARS).initial(v=-60, n=0.3)
      .option(total=300, dt=0.01, meth="rk4", maxstor=200000))
print(hh.render())                                   # the .ode file XPPAUT reads
m = hh.load(HERE / "reduced_hh.ode")

# ---------------------------------------------------------------------------------------
# 2. The limit cycle, its period T (paper: 11.8463), and phase 0 (the maximum of V).
#    XPP: Initialconds > Go, then Initialconds > Periodic.
# ---------------------------------------------------------------------------------------
m.run()
m.buttons.xi_vs_t("v")
m.buttons.initialconds.periodic()
T = m.numerics["total"]
cycle = m.run(keep=False, total=T, dt=0.001)
k = int(np.argmax(cycle["v"]))
gamma0 = (cycle["v"][k], cycle["n"][k])
m.set(v=gamma0[0], n=gamma0[1])
cycle = m.run(keep=False, total=T, dt=0.001)
print(f"period T = {T:.4f} (paper 11.8463), phase 0 at (V, n) = ({gamma0[0]:.3f}, {gamma0[1]:.4f})")


def spike_times(t, v, threshold=-20.0):
    """times where V crosses the threshold upward (linear interpolation between samples)"""
    i = np.nonzero((v[:-1] < threshold) & (v[1:] >= threshold))[0]
    return t[i] + (threshold - v[i]) / (v[i + 1] - v[i]) * (t[i + 1] - t[i])


ref = m.run(keep=False, total=3 * T, dt=0.01)
t_c0 = spike_times(ref.t, ref["v"])[0]               # the cycle's first spike after phase 0

# ---------------------------------------------------------------------------------------
# 3. An XPP network of N uncoupled copies (v[j], n[j]): one run, N starting points.
# ---------------------------------------------------------------------------------------
N = 200
nb = ModelBuilder("reduced_hh_network")
for sig, expr in FUNCTIONS:
    nb.function(sig, expr)
net = (nb.ode(f"v[0..{N - 1}]", VDOT.format(v="v[j]", n="n[j]"))
       .ode(f"n[0..{N - 1}]", NDOT.format(v="v[j]", n="n[j]"))
       .parameter(**PARS).option(dt=0.01, meth="rk4", maxstor=400000)
       .load(HERE / "reduced_hh_network.ode"))


def asymptotic_phase(points, periods=8):
    """the asymptotic phase of each point (NaN if it has not settled on the cycle)"""
    phase = np.full(len(points), np.nan)
    for start in range(0, len(points), N):
        chunk = points[start:start + N]
        ics = {}
        for j in range(N):
            p = chunk[j] if j < len(chunk) else chunk[0]
            ics[f"v{j}"], ics[f"n{j}"] = p[0], p[1]
        net.ics.update(ics)
        data = net.run(total=periods * T, trans=(periods - 2.5) * T).array
        t, V = data[:, 0], data[:, 1:1 + len(chunk)]
        for c in range(len(chunk)):
            s = spike_times(t, V[:, c])
            if len(s) >= 2 and abs(s[-1] - s[-2] - T) < 1e-3 * T:     # settled on the cycle
                phase[start + c] = ((t_c0 - s[-1]) / T) % 1
    return phase


# ---------------------------------------------------------------------------------------
# 4. The phase map on the window of the paper's Figure 2.
# ---------------------------------------------------------------------------------------
Vg, Ng = np.meshgrid(np.linspace(-90, 52, 121), np.linspace(0.3, 0.8, 91))
grid = np.c_[Vg.ravel(), Ng.ravel()]
TH = np.concatenate([asymptotic_phase(grid[s:s + 1000])
                     for s in tqdm(range(0, len(grid), 1000), desc="phase map")]).reshape(Vg.shape)


def without_jumps(F):
    """hide grid points next to a jump > 0.5 (the 1 -> 0 wrap, or the unresolved strip)"""
    bad = np.zeros(F.shape, bool)
    dx, dy = np.abs(np.diff(F, axis=1)) > 0.5, np.abs(np.diff(F, axis=0)) > 0.5
    bad[:, :-1] |= dx; bad[:, 1:] |= dx; bad[:-1, :] |= dy; bad[1:, :] |= dy
    return np.ma.masked_where(bad | np.isnan(F), F)


# ---------------------------------------------------------------------------------------
# 5. The picture, with the V-nullcline (green) and n-nullcline (red) for reference.
# ---------------------------------------------------------------------------------------
with np.errstate(all="ignore"):
    am = 0.1 * (Vg + 40) / (1 - np.exp(-(Vg + 40) / 10)); bm = 4 * np.exp(-(Vg + 65) / 18)
    an = 0.01 * (Vg + 55) / (1 - np.exp(-(Vg + 55) / 10)); bn = 0.125 * np.exp(-(Vg + 65) / 80)
vdot = 10 - 120 * (am / (am + bm)) ** 3 * (0.8 - Ng) * (Vg - 50) - 36 * Ng ** 4 * (Vg + 77) - 0.3 * (Vg + 54.4)
ndot = an * (1 - Ng) - bn * Ng

fig, ax = plt.subplots(figsize=(8.5, 6))
pm = ax.pcolormesh(Vg, Ng, TH, cmap="hsv", shading="auto", vmin=0, vmax=1, alpha=0.7)
ax.contour(Vg, Ng, without_jumps(TH), [0.3, 0.4, 0.5, 0.6, 0.7], colors="k", linewidths=1)
ax.contour(Vg, Ng, without_jumps((TH + 0.5) % 1), [0.3, 0.4, 0.5, 0.6, 0.7], colors="k", linewidths=1)
ax.contour(Vg, Ng, vdot, [0], colors="#2e8b2e", linewidths=1.2)
ax.contour(Vg, Ng, ndot, [0], colors="#c0201a", linewidths=1.2)
ax.plot(cycle["v"], cycle["n"], color="k", lw=3)
ax.plot(*gamma0, "o", color="k", ms=8)
fig.colorbar(pm, label="asymptotic phase θ")
ax.set(xlabel="V", ylabel="n",
       title="Reduced HH: isochrons θ = 0, 0.1, …, 0.9 (black); V- and n-nullclines (green, red)")
fig.savefig(HERE / "hh_isochrons.png", dpi=130)
print("wrote hh_isochrons.png")
plt.show()
