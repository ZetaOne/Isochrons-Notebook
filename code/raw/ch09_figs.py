"""Chapter 9 figures: why computing isochrons is hard; the paper's section 2.2 boundary-value problem.
Run:  .venv/bin/python scripts/isochron/figs/ch09_figs.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import iso_lib as L
import figkit as K


def rhh_cycle(dt=0.0005):
    m = L.load("rhh")
    t, X, T = L.periodic_orbit(m, (-60, 0.45), dt=dt)
    return m, t, X, T


def backward_failure():
    """(a) vdP eps=1: backward images of L_eta trace the isochron.  (b) reduced HH: integrating
    backwards from a point ON the cycle falls off it within one period."""
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.8))
    # (a) van der Pol
    d = K.loaddata("ch04_vdp1"); xs, ys, th, X, T = d["xs"], d["ys"], d["th"], d["X"], float(d["T"])
    m = L.load("vdp", eps=1.0)
    F = L.rhs_np("vdp", {"eps": 1.0})
    M = L.monodromy(F, X[0], T, steps=20000); w, V = np.linalg.eig(M); vs = np.real(V[:, np.argmin(np.abs(w))])
    ax = axs[0]
    curves = L.isochron_curves(xs, ys, th, [0.0])[0.0]
    for seg in curves:
        ax.plot(seg[:, 0], seg[:, 1], color=K.GREY, lw=5, alpha=0.5)
    ax.plot(X[:, 0], X[:, 1], color="k", lw=1.5)
    eta = 1e-3
    deltas = np.linspace(-eta, eta, 41)
    for k, col in [(1, K.ACC), (2, K.ACC2)]:
        P = []
        for dl in deltas:
            q = X[0] + dl * vs
            tt, Y = L.traj(m, q, k * T, dt=T / 400, back=True)
            P.append(Y[-1])
        P = np.array(P)
        ax.plot(P[:, 0], P[:, 1], "o", ms=3.5, color=col, label=f"$\\Phi_{{-{k}T}}(L_\\eta)$, 41 points")
    ax.plot(*X[0], "*", color=K.ORANGE, ms=13, zorder=6)
    ax.set_xlim(-3, 3); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.legend(loc="lower left", fontsize=9); ax.set_title("van der Pol ($\\lambda \\approx 10^{-3}$): it works, unevenly")
    # (b) reduced HH
    m2, t2, X2, T2 = rhh_cycle()
    ax = axs[1]
    ax.plot(X2[:, 0], X2[:, 1], color="k", lw=2.0, label="$\\Gamma$")
    starts = [X2[0], X2[len(X2) // 3], X2[2 * len(X2) // 3]]
    for s0, col in zip(starts, [K.ACC2, K.BLUE, K.ACC]):
        tt, Y = L.traj(m2, s0, T2, dt=0.001, back=True)
        ok = np.all(np.isfinite(Y), axis=1) & (np.abs(Y[:, 0]) < 300)
        Y = Y[ok]
        ax.plot(Y[:, 0], Y[:, 1], color=col, lw=1.4)
        ax.plot(*s0, "o", color=col, ms=7, mec="white", zorder=6)
        dev = np.min(np.hypot((X2[:, None, 0] - Y[None, ::200, 0]) / 150, (X2[:, None, 1] - Y[None, ::200, 1]) / 0.5), axis=0)
        print(f"start {s0}: backward run lasted t = {abs(tt[ok][-1]):.3f} before leaving the window; max scaled distance from Gamma {dev.max():.2e}")
    e = m2.equilibrium({"v": -59.6, "n": 0.40})
    ax.plot(e.x["V"], e.x["N"], "o", mfc="white", mec="k", ms=7, zorder=6)
    ax.set_xlim(-90, 60); ax.set_ylim(0.3, 0.8); ax.set_xlabel("$V$"); ax.set_ylabel("$n$")
    ax.set_title("reduced Hodgkin–Huxley ($\\lambda \\approx 10^{-20}$): backwards from $\\Gamma$")
    fig.tight_layout()
    K.save(fig, "ch09_backward", wide=True)


def sensitivity():
    """Distance after one backward period between two starting points delta apart, vdP eps=0.3/1 vs rhh (scipy, double precision)."""
    from scipy.integrate import solve_ivp
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rows = []
    for name, pars, x0, col, lab in [("vdp", {"eps": 1.0}, (2, 0), K.ACC, "van der Pol, $\\varepsilon=1$ ($1/\\lambda \\approx 1.2\\times10^3$)"),
                                     ("vdp", {"eps": 0.3}, (2, 0), K.BLUE, "van der Pol, $\\varepsilon=0.3$ ($1/\\lambda \\approx 7.6\\times10^6$)"),
                                     ("rhh", {}, (-60, 0.45), K.ACC2, "reduced Hodgkin–Huxley ($1/\\lambda \\approx 1.5\\times10^{20}$)")]:
        m = L.load(name, **pars)
        t, X, T = L.periodic_orbit(m, x0, dt=0.0005)
        F = L.rhs_np(name, pars)
        g0 = X[0].copy()
        sc = np.array([150.0, 0.5]) if name == "rhh" else np.array([1.0, 1.0])
        ds = np.logspace(-16, -2, 29)
        out = []
        for dl in ds:
            ends = []
            for q in (g0, g0 + dl * np.array([1.0, 0.0]) * sc):
                sol = solve_ivp(lambda tt, u: -F(u), (0, T), q, method="Radau", rtol=1e-12, atol=1e-14)
                ends.append(sol.y[:, -1] if sol.success else np.array([np.nan, np.nan]))
            out.append(np.linalg.norm((ends[1] - ends[0]) / sc))
        out = np.array(out)
        if np.all(~np.isfinite(out)):
            # every backward run blows up before t = -T: draw the linear prediction delta / lambda instead
            ax.loglog(ds, ds * 1.5e20, "--", color=col, lw=2, label=lab + ": prediction; every run fails")
        else:
            ax.loglog(ds, out, "o-", ms=3.5, color=col, label=lab)
        rows.append((name, pars, ds, out))
        m.close()
    ax.loglog([1e-16, 1e-2], [1e-16, 1e-2], ":", color=K.GREY, label="no amplification")
    ax.axvline(2.2e-16, color=K.INK, lw=0.8, ls="--")
    ax.text(3e-16, 1e-11, "double\nprecision", fontsize=9)
    ax.set_xlabel("initial separation $\\delta$ (scaled units)"); ax.set_ylabel("separation after $\\Phi_{-T}$")
    ax.set_title("backward integration amplifies differences by $1/\\lambda$")
    ax.legend(fontsize=8.5, loc="lower right"); ax.set_ylim(1e-17, 1e9)
    K.save(fig, "ch09_sensitivity")
    K.savedata("ch09_sensitivity", rows=np.array(rows, dtype=object))


def bundle_bvp():
    """Paper section 2.2: Gamma and its stable normal bundle from one boundary-value problem (scipy collocation)."""
    m, t, X, T = rhh_cycle()
    F = L.rhs_np("rhh", {})
    s, u, w, T2, kap, sol = L.cycle_bundle_bvp(F, t, X, T, [-1.0, 1e-4], -46.47 / T)
    print("bundle BVP:", sol.success, "T =", T2, " kappa =", kap, " ln mu =", kap * T2, " nodes", len(s))
    K.savedata("ch09_bundle", s=s, u=u, w=w, T=T2, kappa=kap)
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax = axs[0]
    ax.plot(u[0], u[1], color="k", lw=1.8)
    sc = np.array([150.0, 0.5])
    idx = np.unique(np.searchsorted(s, np.linspace(0, 1, 41)[:-1]))
    for j in idx:
        p = u[:, j]; q = w[:, j] / sc; q = q / np.linalg.norm(q) * 0.045
        ax.plot([p[0] - q[0] * 150, p[0] + q[0] * 150], [p[1] - q[1] * 0.5, p[1] + q[1] * 0.5], color=K.ORANGE, lw=2.4, solid_capstyle="round")
    ax.set_xlim(-90, 60); ax.set_ylim(0.3, 0.8); ax.set_xlabel("$V$"); ax.set_ylabel("$n$")
    ax.set_title("stable direction $\\vec w$ at 40 phases (orange)")
    ax = axs[1]
    ang = np.degrees(np.arctan2(w[1] / 0.5, w[0] / 150))
    ang = (ang + 90) % 180 - 90
    ax.plot(s, ang, color=K.ACC2, lw=1.8)
    ax.axhline(0, color=K.GREY, lw=0.8)
    ax.set_xlabel("phase $\\vartheta$"); ax.set_ylabel("angle of $\\vec w$ to the $V$-axis (degrees, scaled plot)")
    ax.set_title(f"one BVP gives $T = {T2:.4f}$ and $\\ln\\mu = {kap * T2:.3f}$")
    fig.tight_layout()
    K.save(fig, "ch09_bundle", wide=True)


if __name__ == "__main__":
    backward_failure()
    sensitivity()
    bundle_bvp()
