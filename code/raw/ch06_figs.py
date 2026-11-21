"""Chapter 6 figures: phase response curves (direct method with XPP, adjoint with XPP's Averaging).
Run:  .venv/bin/python scripts/isochron/figs/ch06_figs.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import iso_lib as L
import figkit as K


def kick_gif():
    """A state on the cycle is kicked across the isochrons; it settles with a shifted phase."""
    d = K.loaddata("ch04_vdp1")
    xs, ys, th, X, T = d["xs"], d["ys"], d["th"], d["X"], float(d["T"])
    m = L.load("vdp", eps=1.0)
    look = L.PhaseLookup(X)
    k0 = int(0.30 * (len(X) - 1))
    p0 = X[k0]; kick = np.array([0.9, 0.0])
    p1 = p0 + kick
    t, P = L.traj(m, p1, 2 * T, dt=T / 300)
    th_new, _ = look(P[-1])
    shift = ((th_new - 2.0) - (0.30)) % 1
    shift = shift - 1 if shift > 0.5 else shift
    ghost = np.vstack([X[(k0 + j * (len(X) - 1) // 300) % (len(X) - 1)] for j in range(len(t))])
    K.savedata("ch06_kick", P=P, ghost=ghost, p0=p0, p1=p1, X=X, T=T, shift=shift)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    levels = np.arange(20) / 20
    curves = L.isochron_curves(xs, ys, th, levels)
    for c, segs in curves.items():
        for seg in segs:
            ax.plot(seg[:, 0], seg[:, 1], color=K.PHASE_CMAP(c), lw=1, alpha=0.7)
    ax.plot(X[:, 0], X[:, 1], color="k", lw=1.8)
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=K.ORANGE, lw=2.5))
    path, = ax.plot([], [], color=K.ACC2, lw=1.3)
    a, = ax.plot([], [], "o", color=K.ACC2, ms=10, label="kicked state")
    g, = ax.plot([], [], "o", color=K.GREY, ms=9, label="unkicked state")
    lab = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=11, bbox=K.BOX, zorder=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-3, 3); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("a kick moves the state to another isochron")
    ks = np.arange(0, len(t), 2)

    def upd(i):
        k = ks[i]
        path.set_data(P[:k + 1, 0], P[:k + 1, 1]); a.set_data([P[k, 0]], [P[k, 1]]); g.set_data([ghost[k, 0]], [ghost[k, 1]])
        lab.set_text(f"$t/T = {t[k] / T:.2f}$;   phase shift $\\Delta\\vartheta = {shift:+.3f}$")
    K.save_gif(fig, upd, len(ks), "ch06_kick", fps=20, still_frame=-1)
    print("kick phase shift", shift)


def prc_direct(name, pars, x0, amps, var=0, nph=60, settle=4, cyc_dt=0.002):
    m = L.load(name, **pars)
    t, X, T = L.periodic_orbit(m, x0, dt=cyc_dt)
    adjoint = L.floquet(m)[2]            # before any other run replaces XPP's one-period orbit
    look = L.PhaseLookup(X)
    phs = np.arange(nph) / nph
    out = {}
    for A in amps:
        res = []
        for p in phs:
            k = int(round(p * (len(X) - 1)))
            q = X[k].copy(); q[var] += A
            tt, P = L.traj(m, q, settle * T, dt=T / 200)
            thn, dist = look(P[-1])
            sh = (thn - p) % 1
            res.append(sh if sh <= 0.5 else sh - 1)
        out[A] = np.array(res)
    return m, t, X, T, phs, out, adjoint


def prc_figure():
    """Direct-method PRCs (kick in x, measure phase shift) vs the infinitesimal PRC from XPP's adjoint."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (name, pars, x0, title, amps) in zip(axs, [
            ("sl", {"b": 0.6}, (0.5, 0.0), "twisted oscillator ($b=0.6$)", [0.01, 0.2, 0.5]),
            ("vdp", {"eps": 1.0}, (2.0, 0.0), "van der Pol ($\\varepsilon=1$)", [0.01, 0.3, 0.9])]):
        m, t, X, T, phs, out, adj = prc_direct(name, pars, x0, amps)
        z = np.asarray(adj[adj.names[1]], float)       # x-component of the adjoint (time units)
        tz = np.asarray(adj.t, float) / T
        for A, col in zip(amps, [K.ACC, K.BLUE, K.ACC2]):
            ax.plot(phs, out[A] / A, "o", ms=3.5, color=col, label=f"direct: kick {A}, shift / kick")
        ax.plot(tz, z / T, color=K.INK, lw=2, label="adjoint $Z_x/T$ (infinitesimal PRC)")
        if name == "sl":
            b = pars["b"]
            a = 2 * np.pi * tz
            exact = (-np.sin(a) - b * np.cos(a)) / (2 * np.pi)       # d(theta - b ln r)/dx / 2pi on r = 1
            ax.plot(tz, exact, "--", color=K.ORANGE, lw=1.5, label="exact $\\partial_x\\vartheta$")
        ax.axhline(0, color=K.GREY, lw=0.8)
        ax.set_xlabel("phase of the kick $\\vartheta$"); ax.set_ylabel("$\\Delta\\vartheta$ / kick size")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        K.savedata(f"ch06_prc_{name}", phs=phs, amps=np.array(amps), shifts=np.array([out[A] for A in amps]), tz=tz, z=z / T, T=T)
    fig.tight_layout()
    K.save(fig, "ch06_prc", wide=True)


def gradient_figure():
    """grad(theta) along Gamma: perpendicular to the isochrons; its length is the isochron density."""
    d = K.loaddata("ch04_vdp1")
    xs, ys, th, X, T = d["xs"], d["ys"], d["th"], d["X"], float(d["T"])
    m = L.load("vdp", eps=1.0)
    L.periodic_orbit(m, (2, 0), dt=0.002)
    kap, nu, adj = L.floquet(m)
    tz = np.asarray(adj.t, float); Z = np.column_stack([adj[adj.names[1]], adj[adj.names[2]]]).astype(float) / T
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    levels = np.arange(40) / 40
    curves = L.isochron_curves(xs, ys, th, levels)
    for c, segs in curves.items():
        for seg in segs:
            ax.plot(seg[:, 0], seg[:, 1], color=K.PHASE_CMAP(c), lw=0.9, alpha=0.7)
    ax.plot(X[:, 0], X[:, 1], color="k", lw=1.8)
    for j in range(0, len(tz) - 1, len(tz) // 24):
        k = int(round(tz[j] / T * (len(X) - 1))) % (len(X) - 1)
        p = X[k]; z = Z[j]
        ax.annotate("", xy=p + 1.2 * z, xytext=p, arrowprops=dict(arrowstyle="-|>", color=K.ORANGE, lw=1.8))
    ax.set_xlim(-3, 3); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("$\\nabla\\vartheta$ on $\\Gamma$ (orange): perpendicular to the isochrons")
    K.save(fig, "ch06_gradient")


def resetting_types():
    """Radial clock, kick of size A in x: new phase = angle of (cos 2 pi th + A, sin 2 pi th). Type 1 for A < 1, type 0 for A > 1."""
    th = np.linspace(0, 1, 801)
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.3))
    ax = axs[0]
    for A, col in [(0.3, K.ACC), (0.7, K.BLUE), (0.95, K.GREY), (1.05, K.ORANGE), (1.6, K.ACC2)]:
        new = (np.arctan2(np.sin(2 * np.pi * th), np.cos(2 * np.pi * th) + A) / (2 * np.pi)) % 1
        jumps = np.where(np.abs(np.diff(new)) > 0.5)[0]
        y = new.copy(); y[jumps] = np.nan
        ax.plot(th, y, color=col, lw=2, label=f"$A = {A}$")
    ax.plot([0, 1], [0, 1], ":", color=K.GREY)
    ax.set_xlabel("old phase $\\vartheta$"); ax.set_ylabel("new phase $\\vartheta'$")
    ax.set_title("phase transition curves, radial clock"); ax.legend(fontsize=9, loc="upper left")
    ax = axs[1]
    c = np.linspace(0, 2 * np.pi, 400)
    for k in range(12):
        a = 2 * np.pi * k / 12
        ax.plot([0, 2.8 * np.cos(a)], [0, 2.8 * np.sin(a)], color=K.PHASE_CMAP(k / 12), lw=1.1)
    ax.plot(np.cos(c), np.sin(c), color="k", lw=2)
    for A, col in [(0.6, K.BLUE), (1.6, K.ACC2)]:
        ax.plot(np.cos(c) + A, np.sin(c), "--", color=col, lw=2, label=f"$\\Gamma$ shifted by $A={A}$")
    ax.plot(0, 0, "o", mfc="white", mec="k", ms=8, zorder=5)
    ax.set_aspect("equal"); ax.set_xlim(-1.6, 2.9); ax.set_ylim(-1.8, 1.8); ax.legend(fontsize=9, loc="lower left")
    ax.set_title("kicked cycle: around the phaseless point or not")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    K.save(fig, "ch06_types", wide=True)


def rhh_prc():
    """Infinitesimal PRC of the reduced Hodgkin-Huxley neuron (XPP adjoint)."""
    m = L.load("rhh")
    t, X, T = L.periodic_orbit(m, (-60, 0.45), dt=0.001)
    kap, nu, adj = L.floquet(m)
    tz = np.asarray(adj.t, float) / T
    zv = np.asarray(adj[adj.names[1]], float) / T
    zn = np.asarray(adj[adj.names[2]], float) / T
    K.savedata("ch06_rhh_prc", tz=tz, zv=zv, zn=zn, T=T, X=X, t=t)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    axs[0].plot(t / T, X[:, 0], color=K.INK, lw=1.8)
    axs[0].set_xlabel("phase $\\vartheta$"); axs[0].set_ylabel("$V$ (mV)"); axs[0].set_title("one period of the spike train")
    axs[1].plot(tz, zv, color=K.ACC2, lw=2)
    axs[1].axhline(0, color=K.GREY, lw=0.8)
    axs[1].set_xlabel("phase $\\vartheta$"); axs[1].set_ylabel("$\\partial\\vartheta/\\partial V$ (1/mV)")
    axs[1].set_title("infinitesimal PRC to voltage kicks (XPP adjoint)")
    fig.tight_layout()
    K.save(fig, "ch06_rhh_prc", wide=True)
    print("rhh PRC: min", zv.min(), "max", zv.max(), "at", tz[np.argmax(zv)])


if __name__ == "__main__":
    kick_gif()
    prc_figure()
    gradient_figure()
    resetting_types()
    rhh_prc()
