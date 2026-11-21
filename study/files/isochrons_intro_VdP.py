#!/usr/bin/env python3
# Isochrons, step by step, for the van der Pol oscillator.
#   python isochrons_intro_VdP.py          -> figures pop up, and are saved in isochrons_intro_figures/VdP/
# Needs: pip install pyauto-xpp numpy matplotlib tqdm
#        --find-links https://github.com/ZetaOne/pyauto-xpp/releases/expanded_assets/v0.3.0
import re
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyxpp.builder import ModelBuilder
from tqdm import tqdm

WORK = Path(tempfile.mkdtemp())

# ------------------------------------------------------------------------------------------
# STEP 0. The model (replace this block for another oscillator; first variable = "voltage")
#   x' = f(x, y),  y' = g(x, y)   with one stable limit cycle around an unstable equilibrium
# ------------------------------------------------------------------------------------------
NAME = "VdP"                           # van der Pol, mu = 2
X, Y = "x", "y"
FUNCTIONS = []
RHS_X = "y"
RHS_Y = "mu*(1-x^2)*y-x"
PARS = dict(mu=2)
START = (0.5, 0.0)
DT = 0.005
WINDOW = (-3.0, 3.0, -4.5, 4.5)        # (xmin, xmax, ymin, ymax) for the pictures
X0 = (0.5, 0.5)                        # a state off Γ, for step 3
N = 200                                # copies of the model integrated together in one XPP run

model = ModelBuilder("model")
copies = ModelBuilder("copies")        # N uncoupled copies: v[0..N-1], w[0..N-1]
for sig, expr in FUNCTIONS:
    model.function(sig, expr)
    copies.function(sig, expr)
model = (model.ode(X, RHS_X).ode(Y, RHS_Y).parameter(**PARS).initial(**{X: START[0], Y: START[1]})
         .option(total=20000 * DT, dt=DT, meth="rk4", bound=1e6, maxstor=400000))
idx = lambda e: re.sub(rf"\b({X}|{Y})\b", r"\1[j]", e)     # v -> v[j], w -> w[j] (copy j)
copies = (copies.ode(f"{X}[0..{N - 1}]", idx(RHS_X)).ode(f"{Y}[0..{N - 1}]", idx(RHS_Y))
          .parameter(**PARS).option(dt=DT, meth="rk4", bound=1e6, maxstor=400000))
OUT = Path("isochrons_intro_figures") / NAME
OUT.mkdir(parents=True, exist_ok=True)
print(model.render())
m = model.load(WORK / "model.ode")
net = copies.load(WORK / "copies.ode")

# ------------------------------------------------------------------------------------------
# STEP 1. The limit cycle Γ, its period T, and phase 0
#   integrate long from START             -> on Γ
#   XPP Initialconds > Periodic           -> T
#   gamma0 = point of Γ with largest x    -> phase 0
#   one period from gamma0                -> Γ as a table (t, x, y), t in [0, T]
#   threshold = middle of x on Γ;  t_c0 = first upward crossing of threshold from gamma0
# ------------------------------------------------------------------------------------------
m.run()
m.buttons.xi_vs_t(X)
m.buttons.initialconds.periodic()
T = m.numerics["total"]
c = m.run(keep=False, total=T, dt=T / 20000)
k = int(np.argmax(c[X]))
gamma0 = np.array([c[X][k], c[Y][k]])
m.set(**{X: gamma0[0], Y: gamma0[1]})
c = m.run(keep=False, total=T, dt=T / 20000)
CT, CX, CY = np.asarray(c.t), np.asarray(c[X]), np.asarray(c[Y])
THR = (CX.min() + CX.max()) / 2
SX, SY = CX.max() - CX.min(), CY.max() - CY.min()      # size of Γ (to scale distances)
r = m.run(keep=False, total=3 * T, dt=DT)
rt, rx = np.asarray(r.t), np.asarray(r[X])
i = np.nonzero((rx[:-1] < THR) & (rx[1:] >= THR))[0][0]
t_c0 = rt[i] + (THR - rx[i]) / (rx[i + 1] - rx[i]) * (rt[i + 1] - rt[i])
eq = m.equilibrium({X: CX.mean(), Y: CY.mean()})
xstar = np.array([eq.x[X.upper()], eq.x[Y.upper()]])
gamma = lambda th: np.array([np.interp((th % 1) * T, CT, CX), np.interp((th % 1) * T, CT, CY)])
print(f"T = {T:.4f}, phase 0 at {gamma0}, equilibrium at {xstar}")


# ------------------------------------------------------------------------------------------
# The clock reading (asymptotic phase) of many states at once
#   for each block of N states:
#       put them in the N copies, integrate 8 periods, keep the last 2.5
#       for each copy: last two upward crossings s1, s2 of THR
#           if s2 - s1 = T (settled):  theta = ((t_c0 - s2) / T) mod 1   else: NaN
# ------------------------------------------------------------------------------------------
def clock_reading(states, periods=8):
    states = np.atleast_2d(states)
    theta = np.full(len(states), np.nan)
    for b in range(0, len(states), N):
        block = states[b:b + N]
        ics = {}
        for j in range(N):
            s = block[j] if j < len(block) else block[0]
            ics[f"{X}{j}"], ics[f"{Y}{j}"] = s[0], s[1]
        net.ics.update(ics)
        A = net.run(total=periods * T, trans=(periods - 2.5) * T).array
        t = A[:, 0]
        for j in range(len(block)):
            x = A[:, 1 + j]
            up = np.nonzero((x[:-1] < THR) & (x[1:] >= THR))[0]
            if len(up) < 2:
                continue
            s = t[up] + (THR - x[up]) / (x[up + 1] - x[up]) * (t[up + 1] - t[up])
            if abs(s[-1] - s[-2] - T) < 1e-3 * T:
                theta[b + j] = ((t_c0 - s[-1]) / T) % 1
    return theta


# ------------------------------------------------------------------------------------------
# Draw the isochron of phase `level` as a contour line of a phase map TH on the grid (GX, GY)
#   hide grid points next to a jump > 0.5 (the 1 -> 0 wrap), else the contour draws a false line
#   levels near 0 or 1: contour the phase shifted by half a cycle instead
# ------------------------------------------------------------------------------------------
def draw_isochron(ax, GX, GY, TH, level, **kw):
    F = TH if 0.25 <= level <= 0.75 else (TH + 0.5) % 1
    L = level if 0.25 <= level <= 0.75 else (level + 0.5) % 1
    bad = np.isnan(F)
    jx, jy = np.abs(np.diff(F, axis=1)) > 0.5, np.abs(np.diff(F, axis=0)) > 0.5
    bad[:, :-1] |= jx; bad[:, 1:] |= jx; bad[:-1, :] |= jy; bad[1:, :] |= jy
    ax.contour(GX, GY, np.ma.masked_where(bad, F), [L], **kw)


# ------------------------------------------------------------------------------------------
# STEP 2. The clock on Γ:  theta = t / T, measured from gamma0
#   mark Γ every 0.05 of the clock: equal steps in time, not in distance
# ------------------------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
a1.plot(CX, CY, "k", lw=2.5); a1.plot(*xstar, "kx", ms=9)
for th in np.arange(0, 1, 0.05):
    a1.plot(*gamma(th), "o", color=plt.cm.hsv(th), ms=6)
    a1.annotate(f"{th:.2f}", gamma(th), fontsize=7, xytext=(4, 3), textcoords="offset points")
a1.set(xlabel=X, ylabel=Y, title="the clock on Γ: marks every 0.05 of θ = t/T")
a2.plot(CT / T, CX, "k"); a2.set(xlabel="θ = t / T", ylabel=X, title="one period from phase 0")
fig.tight_layout(); fig.savefig(OUT / "step2_clock.png", dpi=120)

# ------------------------------------------------------------------------------------------
# STEP 3. The clock reading of one state x0 off Γ
#   theta0 = clock_reading(x0)
#   integrate x0 and gamma(theta0) side by side: same spikes after the transient, distance -> 0
# ------------------------------------------------------------------------------------------
x0 = np.array(X0)
theta0 = clock_reading(x0)[0]
m.set(**{X: x0[0], Y: x0[1]}); a = m.run(keep=False, total=4 * T, dt=DT)
g = gamma(theta0)
m.set(**{X: g[0], Y: g[1]}); b = m.run(keep=False, total=4 * T, dt=DT)
dist = np.hypot((np.asarray(a[X]) - np.asarray(b[X])) / SX, (np.asarray(a[Y]) - np.asarray(b[Y])) / SY)
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.2))
a1.plot(CX, CY, "k", lw=2.5); a1.plot(a[X], a[Y], "C0", lw=1)
a1.plot(*x0, "o", color="C0", ms=9); a1.plot(*g, "o", color="C3", ms=9)
a1.set(xlabel=X, ylabel=Y, title=f"x0 (blue) keeps step with γ({theta0:.3f}) (red)")
a2.plot(np.asarray(a.t) / T, a[X], "C0", label="from x0")
a2.plot(np.asarray(b.t) / T, b[X], "C3--", label=f"from γ({theta0:.3f})")
a2.set(xlabel="t / T", ylabel=X, title="same spikes after the transient"); a2.legend(fontsize=8)
a3.semilogy(np.asarray(a.t) / T, dist + 1e-12); a3.set(xlabel="t / T", ylabel="distance", title="distance → 0")
fig.tight_layout(); fig.savefig(OUT / "step3_clock_reading.png", dpi=120)
print(f"clock reading of x0 = {x0}: theta = {theta0:.4f}")

# ------------------------------------------------------------------------------------------
# STEP 4. The phase map and the isochrons
#   TH[i, j] = clock_reading(grid point)        (stored in phase_map.npz: computed only once)
#   isochron of phase L = contour line TH = L
# ------------------------------------------------------------------------------------------
MAP = OUT / "phase_map.npz"
if MAP.exists():
    d = np.load(MAP); GX, GY, TH = d["GX"], d["GY"], d["TH"]
else:
    GX, GY = np.meshgrid(np.linspace(WINDOW[0], WINDOW[1], 121), np.linspace(WINDOW[2], WINDOW[3], 91))
    pts = np.c_[GX.ravel(), GY.ravel()]
    TH = np.concatenate([clock_reading(pts[s:s + 1000]) for s in tqdm(range(0, len(pts), 1000),
                                                                       desc="phase map")]).reshape(GX.shape)
    np.savez(MAP, GX=GX, GY=GY, TH=TH)
fig, ax = plt.subplots(figsize=(8.5, 6))
ax.pcolormesh(GX, GY, TH, cmap="hsv", shading="auto", vmin=0, vmax=1, alpha=0.45)
for L in np.arange(0.05, 1, 0.1):
    draw_isochron(ax, GX, GY, TH, L, colors="k", linewidths=0.9)
ax.plot(CX, CY, "k", lw=2.5); ax.plot(*xstar, "kx", ms=9)
ax.set(xlabel=X, ylabel=Y, title="clock reading (colour) and isochrons θ = 0.05, 0.15, … (lines)")
fig.savefig(OUT / "step4_isochrons.png", dpi=120)

# ------------------------------------------------------------------------------------------
# STEP 5. States on one isochron move together
#   take the grid points with TH ~ 0.2, integrate them all
#   at time t they lie on the isochron 0.2 + t/T and shrink onto gamma(0.2 + t/T)
# ------------------------------------------------------------------------------------------
sel = np.c_[GX.ravel(), GY.ravel()][np.abs(TH.ravel() - 0.2) < 0.004][:N]
ics = {}
for j in range(N):
    s = sel[j] if j < len(sel) else sel[0]
    ics[f"{X}{j}"], ics[f"{Y}{j}"] = s[0], s[1]
net.ics.update(ics)
A = net.run(total=1.1 * T, trans=0).array
fig, axs = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
for ax, tt in zip(axs, (0, 0.05, 0.2, 1.0)):
    k = int(np.argmin(np.abs(A[:, 0] - tt * T)))
    ax.plot(CX, CY, "k", lw=2)
    draw_isochron(ax, GX, GY, TH, (0.2 + tt) % 1, colors="C0", linewidths=1.5)
    ax.plot(A[k, 1:1 + len(sel)], A[k, 1 + N:1 + N + len(sel)], "k.", ms=3)
    ax.plot(*gamma(0.2 + tt), "o", color="C3", ms=9)
    ax.set(xlim=WINDOW[:2], ylim=WINDOW[2:], xlabel=X, title=f"t = {tt:g} T: isochron {(0.2 + tt) % 1:.2f}")
axs[0].set_ylabel(Y)
fig.tight_layout(); fig.savefig(OUT / "step5_isochron_moves.png", dpi=120)

# ------------------------------------------------------------------------------------------
# STEP 6. Near Γ: isochrons are straight lines; their direction gives the infinitesimal PRC
#   XPP Numerics > Averaging > New adj      -> adjoint Z(t) along Γ (gradient of the clock reading)
#   isochron direction at gamma(theta) = perpendicular to Z(theta)
#   infinitesimal PRC for kicks in x:  dtheta ~ Z_x(theta) * kick / T
#   zoom in: a fine phase map around gamma(theta) (the big map is too coarse at this scale)
#   check: real small kicks, theta_new = clock_reading(gamma(theta) + (kick, 0))
# ------------------------------------------------------------------------------------------
m.set(**{X: gamma0[0], Y: gamma0[1]})
m.run(keep=False, total=T, dt=T / 20000)
m.buttons.numerics.averaging.parameters(4000)
m.buttons.numerics.averaging.new_adj()
adj = m.data
ZT, ZX, ZY = np.asarray(adj.t), np.asarray(adj[X]), np.asarray(adj[Y])
th = 0.25
g = gamma(th)
z = np.array([np.interp(th * T, ZT, ZX), np.interp(th * T, ZT, ZY)])
tangent = np.array([z[1], -z[0]])                              # perpendicular to Z
tangent /= np.hypot(tangent[0] / SX, tangent[1] / SY)          # one unit = the size of Γ
ths = np.linspace(0, 1, 60, endpoint=False)
kick = 0.002 * SX
new = clock_reading(np.array([gamma(t) + [kick, 0] for t in ths]))
LX, LY = np.meshgrid(np.linspace(g[0] - 0.1 * SX, g[0] + 0.1 * SX, 41), np.linspace(g[1] - 0.1 * SY, g[1] + 0.1 * SY, 41))
LTH = clock_reading(np.c_[LX.ravel(), LY.ravel()]).reshape(LX.shape)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
a1.pcolormesh(LX, LY, LTH, cmap="hsv", shading="auto", vmin=0, vmax=1, alpha=0.45)
draw_isochron(a1, LX, LY, LTH, th, colors="C3", linewidths=2.5)
a1.plot(CX, CY, "k", lw=2.5)
s = np.linspace(-0.1, 0.1, 2)
a1.plot(g[0] + s * tangent[0], g[1] + s * tangent[1], "k--", lw=1.5)
a1.plot(*g, "ko")
a1.set(xlim=(g[0] - 0.1 * SX, g[0] + 0.1 * SX), ylim=(g[1] - 0.1 * SY, g[1] + 0.1 * SY), xlabel=X, ylabel=Y,
       title=f"near γ({th}): isochron (red) touches the line ⊥ Z (dashed)")
a2.plot(ths, ((new - ths + 0.5) % 1 - 0.5) / kick, "o", ms=4, label=f"real kicks of {kick:.3g}: Δθ / kick")
a2.plot(ZT / T, ZX / T, "k", label="infinitesimal PRC Z_x / T")
a2.axhline(0, color="0.7", lw=0.8)
a2.set(xlabel="phase θ of the kick", ylabel="Δθ per unit kick", title="small kicks follow the infinitesimal PRC")
a2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "step6_near_cycle_prc.png", dpi=120)

# ------------------------------------------------------------------------------------------
# STEP 7. Far from Γ: isochrons bend, fold and pile up -> large kicks
#   one large kick: gamma(0.6) + (0.35*width, 0) lands on a far isochron
#   PRC for growing kicks (1% ... 40% of the width of Γ) against the infinitesimal PRC
# ------------------------------------------------------------------------------------------
big = 0.35 * SX
landed = gamma(0.6) + [big, 0]
th_landed = clock_reading(landed)[0]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
a1.pcolormesh(GX, GY, TH, cmap="hsv", shading="auto", vmin=0, vmax=1, alpha=0.45)
for L in np.arange(0.05, 1, 0.1):
    draw_isochron(a1, GX, GY, TH, L, colors="k", linewidths=0.8)
draw_isochron(a1, GX, GY, TH, th_landed, colors="C0", linewidths=2.5)
a1.plot(CX, CY, "k", lw=2.5)
a1.annotate("", landed, gamma(0.6), arrowprops=dict(arrowstyle="->", lw=2.5))
a1.plot(*landed, "o", color="C0", ms=10)
a1.set(xlim=WINDOW[:2], ylim=WINDOW[2:], xlabel=X, ylabel=Y,
       title=f"kick {big:.3g} at θ = 0.6 lands on isochron {th_landed:.3f}")
for f, col in zip((0.01, 0.1, 0.2, 0.4), plt.cm.viridis(np.linspace(0, 0.9, 4))):
    new = clock_reading(np.array([gamma(t) + [f * SX, 0] for t in ths]))
    a2.plot(ths, (new - ths + 0.5) % 1 - 0.5, ".-", color=col, ms=3, label=f"kick = {f:g} × width")
a2.plot(ZT / T, ZX / T * 0.4 * SX, "k--", lw=1, label="infinitesimal PRC × (0.4 × width)")
a2.axhline(0, color="0.7", lw=0.8)
a2.set(xlabel="phase θ of the kick", ylabel="phase shift Δθ (cycles)", ylim=(-0.55, 0.55),
       title="large kicks: far from the infinitesimal prediction")
a2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "step7_large_kicks.png", dpi=120)
print(f"kick {big:.3g} at theta 0.6: shift {(th_landed - 0.6 + 0.5) % 1 - 0.5:+.4f}, "
      f"infinitesimal PRC predicts {np.interp(0.6 * T, ZT, ZX) * big / T:+.4f}")

print(f"figures saved in {OUT}/")
plt.show()                                   # all figures pop up
