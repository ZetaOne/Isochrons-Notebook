"""Chapter 11 figures: the Osinga-Moehlis method (continuation of orbit segments), re-implemented
with scipy collocation; cycles and stable directions from XPPAUT / chapter 9.
Run:  .venv/bin/python scripts/isochron/figs/ch11_figs.py   (needs rhh_phasemaps.py data for the check)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import pyxpp
import iso_lib as L
import figkit as K

SC = np.array([150.0, 0.5])
A, B, W = 0.1, 0.25, 1.0          # weakly attracting twisted oscillator (chapter 5)


def F_twist(u):
    x, y = u[0], u[1]; r2 = x * x + y * y
    return np.array([A * x * (1 - r2) - y * (W - B * r2), A * y * (1 - r2) + x * (W - B * r2)])


def twist_k():
    """k increases each time the end point u(1) reaches the end of L_eta; exact isochron theta = (b/a) ln r."""
    T = 2 * np.pi / (W - B)
    th = np.linspace(0, 1, 4001)
    X = np.column_stack([np.cos(2 * np.pi * th), np.sin(2 * np.pi * th)]); t = th * T
    vs = np.array([1.0, B / A]); vs /= np.linalg.norm(vs)          # tangent of theta = (b/a) ln r at r = 1
    out = {}
    for side in (+1, -1):
        P, segs = L.isochron_bvp(F_twist, t, X, T, vs, eta=0.02, arclen=4.0 if side > 0 else 2.2, kmax=8, side=side,
                                 hmax=0.03, tol=1e-7, record=6)
        out[side] = (P, segs)
        r = np.hypot(*P.T); ang = np.arctan2(P[:, 1], P[:, 0])
        err = np.abs(((ang - (B / A) * np.log(r)) + np.pi) % (2 * np.pi) - np.pi)
        print(f"twist side {side}: {len(P)} points, r from {r.min():.3f} to {r.max():.3f}, k up to {max(s[0] for s in segs) if segs else 1}, max angle error {err.max():.1e}")
    K.savedata("ch11_twist", P1=out[1][0], P2=out[-1][0])
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 5.2))
    ax = axs[0]
    rr = np.geomspace(0.05, 3.5, 2000)
    ax.plot(rr * np.cos(B / A * np.log(rr)), rr * np.sin(B / A * np.log(rr)), color=K.GREY, lw=5, alpha=0.5, label="exact isochron")
    cc = np.linspace(0, 2 * np.pi, 300); ax.plot(np.cos(cc), np.sin(cc), color="k", lw=1.6)
    for side in (+1, -1):
        P, segs = out[side]
        ax.plot(P[:, 0], P[:, 1], ".", ms=3, color=K.ACC2 if side > 0 else K.ACC, label="continuation: $\\vec u(0)$" if side > 0 else None)
    ax.plot(1, 0, "*", color=K.ORANGE, ms=14, zorder=6)
    ax.set_aspect("equal"); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.legend(fontsize=9, loc="lower left")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_title("isochron computed by continuation of orbit segments")
    ax = axs[1]
    P, segs = out[+1]
    shown = {}
    for (k, dl, s, u) in segs:
        if k not in shown and k <= 4:
            shown[k] = (s, u)
    cols = {1: K.ACC, 2: K.BLUE, 3: K.ORANGE, 4: K.ACC2}
    ax.plot(np.cos(cc), np.sin(cc), color="k", lw=1.2)
    for k, (s, u) in shown.items():
        ax.plot(u[0], u[1], color=cols[k], lw=1.4, label=f"a segment with $k={k}$ ({k} period{'s' if k > 1 else ''})")
        ax.plot(u[0, 0], u[1, 0], "o", color=cols[k], ms=6)
    ax.plot(1, 0, "*", color=K.ORANGE, ms=14, zorder=6)
    ax.set_aspect("equal"); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.legend(fontsize=9, loc="lower left")
    ax.set_xlabel("$x$"); ax.set_title("orbit segments from $\\vec u(0)$ (dot) to $L_\\eta$ near $\\gamma_0$")
    fig.tight_layout()
    K.save(fig, "ch11_twist", wide=True)


def rhh_isochron(theta, side, arclen, record=None, hmax=0.004):
    d = K.loaddata("ch09_bundle")
    cyc = K.loaddata("rhh_cycle")
    t, X, T = cyc["t"], cyc["X"], float(cyc["T"])
    k0 = int(round(theta * (len(X) - 1)))
    Xr = np.vstack([X[k0:-1], X[:k0 + 1]]); tr = np.arange(len(Xr)) * (T / (len(X) - 1))
    # stable direction at that phase: from the cycle+bundle BVP of chapter 9
    w = np.array([np.interp(theta, d["s"], d["w"][i]) for i in range(2)])
    F = L.rhs_np("rhh", {})
    return L.isochron_bvp(F, tr, Xr, T, w, eta=1e-4, arclen=arclen, kmax=4, side=side, scale=tuple(SC),
                          hmax=hmax, tol=1e-6, max_nodes=60000, record=record)


def rhh_segments_and_growth():
    P, segs = rhh_isochron(0.0, +1, 0.9, record=1)
    print("rhh gamma0 inner branch:", len(P), "points; last", P[-1])
    cyc = K.loaddata("rhh_cycle"); X = cyc["X"]
    K.savedata("ch11_rhh_inner", P=P)
    # static figure: four orbit segments like the paper's Figure 4
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(X[:, 0], X[:, 1], color="k", lw=1.2, alpha=0.5)
    ax.plot(P[:, 0], P[:, 1], color=K.ACC2, lw=2.2, label="isochron $I(\\gamma_0)$ traced by $\\vec u(0)$")
    picks = np.linspace(len(segs) * 0.15, len(segs) - 1, 4).astype(int)
    for j, col in zip(picks, [K.ACC, K.BLUE, K.ORANGE, "#6b3fa0"]):
        k, dl, s, u = segs[j]
        ax.plot(u[0], u[1], color=col, lw=1.2)
        ax.plot(u[0, 0], u[1, 0], "o", color=col, ms=7, mec="white", zorder=6)
    ax.plot(*X[0], "*", color=K.ORANGE, ms=15, zorder=7)
    ax.annotate("$\\gamma_0$: every segment ends on\n$L_\\eta(\\gamma_0)$, $\\eta = 10^{-4}$", X[0], xytext=(5, 0.33), textcoords="data",
                fontsize=10, arrowprops=dict(arrowstyle="-|>", color=K.INK, lw=0.8), bbox=K.BOX)
    ax.set_xlim(-90, 60); ax.set_ylim(0.3, 0.8); ax.set_xlabel("$V$"); ax.set_ylabel("$n$")
    ax.legend(fontsize=9, loc="upper right"); ax.set_title("four orbit segments of the continuation (cf. paper Fig. 4)")
    K.save(fig, "ch11_segments")
    # animation: the continuation growing the isochron
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(X[:, 0], X[:, 1], color="k", lw=1.2, alpha=0.5)
    iso, = ax.plot([], [], color=K.ACC2, lw=2.4)
    seg, = ax.plot([], [], color=K.BLUE, lw=1.1)
    dot, = ax.plot([], [], "o", color=K.BLUE, ms=7)
    ax.plot(*X[0], "*", color=K.ORANGE, ms=15, zorder=7)
    lab = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=11, bbox=K.BOX)
    ax.set_xlim(-90, 60); ax.set_ylim(0.3, 0.8); ax.set_xlabel("$V$"); ax.set_ylabel("$n$")
    ax.set_title("continuation grows the isochron one orbit segment at a time")
    idx = np.unique(np.linspace(0, len(segs) - 1, min(len(segs), 140)).astype(int))

    def upd(i):
        j = idx[i]
        k, dl, s, u = segs[j]
        iso.set_data(P[:j + 2, 0], P[:j + 2, 1]); seg.set_data(u[0], u[1]); dot.set_data([u[0, 0]], [u[1, 0]])
        lab.set_text(f"step {j + 1}:  $\\delta = {dl:+.1e}$,  $k = {k}$")
    K.save_gif(fig, upd, len(idx), "ch11_grow", fps=14)


def rhh_validation():
    """Isochrons of 8 phases by continuation, on top of the brute-force phase map."""
    d = K.loaddata("rhh_phase_full"); cyc = K.loaddata("rhh_cycle"); X = cyc["X"]
    xs, ys, th = d["xs"], d["ys"], d["th"]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    phases = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    curves = L.isochron_curves(xs, ys, th, phases)
    for c in phases:
        for seg in curves[c]:
            ax.plot(seg[:, 0], seg[:, 1], color=K.GREY, lw=4, alpha=0.5)
    ax.plot(X[:, 0], X[:, 1], color="k", lw=1.5)
    for c in phases:
        for side, arc in [(+1, 0.9), (-1, 0.9)]:
            try:
                P = rhh_isochron(c, side, arc, hmax=0.006)
            except Exception as e:
                print("phase", c, side, "failed", e); continue
            ax.plot(P[:, 0], P[:, 1], color=K.PHASE_CMAP(c), lw=1.3)
        k = int(round(c * (len(X) - 1)))
        ax.plot(*X[k], "o", color=K.PHASE_CMAP(c), ms=6, mec="white", zorder=6)
    ax.plot([], [], color=K.GREY, lw=4, alpha=0.5, label="phase map (XPPAUT, 601×601 grid)")
    ax.plot([], [], color=K.ACC2, lw=1.3, label="continuation of orbit segments")
    ax.set_xlim(-90, 60); ax.set_ylim(0.3, 0.8); ax.set_xlabel("$V$"); ax.set_ylabel("$n$")
    ax.legend(fontsize=9, loc="upper right"); ax.set_title("two independent methods, the same isochrons")
    K.save(fig, "ch11_validate")


if __name__ == "__main__":
    which = sys.argv[1:] or ["twist", "segments", "validate"]
    if "twist" in which:
        twist_k()
    if "segments" in which:
        rhh_segments_and_growth()
    if "validate" in which:
        rhh_validation()
