"""Chapter 1 figures: flows, linear phase portraits, a limit cycle, the Hopf bifurcation.
Run:  .venv/bin/python scripts/isochron/figs/ch01_figs.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import iso_lib as L
import figkit as K


def linear_portraits():
    cases = [("stable node", [[-1, 0], [0, -3]]), ("saddle", [[1, 0], [0, -1]]),
             ("unstable spiral", [[0.2, -1], [1, 0.2]]), ("stable spiral", [[-0.3, -1], [1, -0.3]]),
             ("centre", [[0, -1], [1, 0]]), ("unstable node", [[1, 0], [0, 2.5]])]
    fig, axs = plt.subplots(2, 3, figsize=(10, 6.6))
    for ax, (name, A) in zip(axs.flat, cases):
        A = np.array(A, float)
        F = lambda U, A=A: np.tensordot(A, U, axes=1)
        K.streams(ax, F, (-2, 2), (-2, 2), n=30, color="#7c8794", density=1.0, lw=0.9)
        w, V = np.linalg.eig(A)
        if np.all(np.isreal(w)):
            for k in range(2):
                v = np.real(V[:, k]); col = K.ACC2 if np.real(w[k]) > 0 else K.ACC
                ax.plot([-3 * v[0], 3 * v[0]], [-3 * v[1], 3 * v[1]], color=col, lw=2.2)
        ax.plot(0, 0, "o", color=K.INK, ms=6)
        tr, det = np.trace(A), np.linalg.det(A)
        ax.set_title(f"{name}\n$\\tau={tr:.1f},\\ \\Delta={det:.2f}$", fontsize=11)
        ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    K.save(fig, "ch01_linear", wide=True)


def flow_blob():
    """A disc of initial conditions carried by the van der Pol flow (Phi_t of a set)."""
    m = L.load("vdp", eps=1.0)
    F = L.rhs_np("vdp", {"eps": 1.0})
    th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    starts = np.column_stack([0.3 + 0.45 * np.cos(th), 0.6 + 0.45 * np.sin(th)])
    T = 14.0; dt = 0.02
    paths = np.array([L.traj(m, s, T, dt=dt)[1] for s in starts])      # (60, N, 2)
    _, cyc, Tc = L.periodic_orbit(m, (2, 0))
    K.savedata("ch01_blob", paths=paths, cyc=cyc, dt=dt)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    K.streams(ax, F, (-3, 3), (-3.2, 3.2), n=30, color="#d4d9de")
    ax.plot(cyc[:, 0], cyc[:, 1], color=K.CYCLE, lw=1.6)
    blob, = ax.fill([], [], color=K.ACC2, alpha=0.55, lw=0)
    edge, = ax.plot([], [], color=K.ACC2, lw=1.2)
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=12, bbox=K.BOX, zorder=10)
    ax.set_xlim(-3, 3); ax.set_ylim(-3.2, 3.2); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("A set of initial conditions carried by the flow $\\Phi_t$")
    steps = np.arange(0, paths.shape[1], 6)

    def upd(i):
        k = steps[i]
        P = paths[:, k, :]
        blob.set_xy(P); edge.set_data(np.r_[P[:, 0], P[0, 0]], np.r_[P[:, 1], P[0, 1]])
        txt.set_text(f"$t = {k * dt:4.1f}$")
    def still():
        blob.set_xy(paths[:, 0, :]); edge.set_data([], []); txt.set_text("snapshots at $t = 0, 0.6, 1.2, 2.4, 4.8$")
        for j, k in enumerate([30, 60, 120, 240]):
            P = paths[:, k, :]
            ax.fill(P[:, 0], P[:, 1], color=K.ACC2, alpha=0.25 + 0.15 * j, lw=0)
    K.save_gif(fig, upd, len(steps), "ch01_blob", fps=18, still_frame=0, still_draw=still)


def polar_cycle():
    m = L.load("sl", b=0.0)
    F = L.rhs_np("sl", {"b": 0.0})
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    K.streams(ax, F, (-2, 2), (-2, 2), n=30, color="#cfd5db")
    for s, c in [((0.05, 0), K.ACC2), ((1.9, 1.2), K.ACC), ((-1.7, -1.6), K.ACC), ((0.0, -0.1), K.ACC2)]:
        t, X = L.traj(m, s, 12, dt=0.01)
        ax.plot(X[:, 0], X[:, 1], color=c, lw=1.3)
        ax.plot(*s, "o", color=c, ms=4)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=K.CYCLE, lw=2.4)
    ax.plot(0, 0, "o", mfc="white", mec=K.INK, ms=7)
    ax.set_aspect("equal"); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("$\\dot r = r(1-r^2),\\ \\dot\\theta = 1$")
    ax = axs[1]
    r = np.linspace(0, 1.6, 300)
    ax.axhline(0, color=K.GREY, lw=0.8)
    ax.plot(r, r * (1 - r ** 2), color=K.INK, lw=2)
    ax.plot([0], [0], "o", mfc="white", mec=K.INK, ms=8); ax.plot([1], [0], "o", color=K.INK, ms=8)
    for x0, d in [(0.45, 1), (1.3, -1)]:
        ax.annotate("", xy=(x0 + 0.12 * d, 0), xytext=(x0, 0), arrowprops=dict(arrowstyle="-|>", color=K.ACC2, lw=2))
    ax.set_xlabel("$r$"); ax.set_ylabel("$\\dot r$"); ax.set_title("radial equation: $r=1$ attracts")
    fig.tight_layout()
    K.save(fig, "ch01_polar", wide=True)


def hopf_normal_form():
    """Supercritical Hopf bifurcation of the normal form, computed with AUTO from pyauto-xpp."""
    import pyxpp
    m = pyxpp.load(text="""# Hopf normal form (supercritical)
x'=mu*x-y-x*(x^2+y^2)
y'=x+mu*y-y*(x^2+y^2)
par mu=-0.5
init x=0.01,y=0
@ total=200,dt=0.05,meth=rk4
done
""")
    m.run(); m.run_last(total=20)
    eq = m.auto.run("mu", start="steady", ds=0.01, dsmax=0.02, parmin=-0.5, parmax=1.0, nmax=500)
    hb = eq.special("HB")[0]
    po = m.auto.run(grab=hb.label, choice="periodic", ntst=30, dsmax=0.02, nmax=300, parmin=-0.5, parmax=1.0)
    deq = eq.to_pandas("x"); dpo = po.to_pandas("x")
    K.savedata("ch01_hopfnf", eq=deq[["par", "ymax", "stable"]].to_numpy(float),
               po=dpo[["par", "ymax", "ymin", "stable", "period"]].to_numpy(float), hb=np.array([hb.x]))
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axs[0]
    for st, ls in [(True, "-"), (False, "--")]:
        sub = deq[deq.stable == st]
        ax.plot(sub.par, sub.ymax, ls, color=K.INK, lw=2, label="equilibrium, " + ("stable" if st else "unstable"))
    sub = dpo[dpo.stable == True]
    ax.plot(sub.par, sub.ymax, "o", ms=3, color=K.ACC, label="limit cycle (max and min of $x$)")
    ax.plot(sub.par, sub.ymin, "o", ms=3, color=K.ACC)
    mus = np.linspace(0, 1, 100)
    ax.plot(mus, np.sqrt(mus), color=K.ACC2, lw=1, label="$\\pm\\sqrt{\\mu}$ (theory)")
    ax.plot(mus, -np.sqrt(mus), color=K.ACC2, lw=1)
    ax.plot(hb.x, 0, "s", color=K.ORANGE, ms=9, zorder=5, label="Hopf point (HB)")
    ax.set_xlabel("$\\mu$"); ax.set_ylabel("$x$"); ax.legend(fontsize=9, loc="upper left")
    ax.set_title("AUTO: a cycle is born at $\\mu = 0$")
    ax = axs[1]
    for mu, col in [(-0.3, K.GREY), (0.25, K.ACC), (0.8, K.BLUE)]:
        m.set(mu=mu)
        t, X = L.traj(m, (1.4, 0.0), 60, dt=0.02)
        ax.plot(X[:, 0], X[:, 1], color=col, lw=1.2, alpha=0.8, label=f"$\\mu = {mu}$")
    ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.legend(fontsize=9, loc="upper right")
    ax.set_title("one trajectory from $(1.4, 0)$")
    fig.tight_layout()
    K.save(fig, "ch01_hopf", wide=True)


def hopf_auto():
    """FitzHugh-Nagumo in the current I, computed with AUTO (used in chapter 8)."""
    m = L.load("fhn", i=0.0)
    m.set(meth="rk4", dt=0.05)
    m.run(total=400); m.run_last(total=20)
    eq = m.auto.run("i", start="steady", ds=0.01, dsmax=0.02, parmin=0, parmax=1.6, nmax=2000)
    hbs = eq.special("HB")
    po = m.auto.run(grab=hbs[0].label, choice="periodic", ntst=60, dsmax=0.02, nmax=500, parmin=0, parmax=1.6)
    deq = eq.to_pandas("v"); dpo = po.to_pandas("v")
    K.savedata("ch01_hopf", eq=deq[["par", "ymax", "stable"]].to_numpy(float), po=dpo[["par", "ymax", "ymin", "stable", "period"]].to_numpy(float),
               hb=np.array([[h.x, h.u0["v"]] for h in hbs]))
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axs[0]
    for st, ls in [(True, "-"), (False, "--")]:
        s = deq[deq.stable == st]
        ax.plot(s.par, s.ymax, ls, color=K.INK, lw=2, label="equilibrium " + ("(stable)" if st else "(unstable)"))
    for st, col in [(True, K.ACC), (False, K.ACC2)]:
        s = dpo[dpo.stable == st]
        ax.plot(s.par, s.ymax, "o", ms=2.5, color=col, label="max / min of cycle" if st else None)
        ax.plot(s.par, s.ymin, "o", ms=2.5, color=col)
    for h in hbs:
        ax.plot(h.x, h.u0["v"], "s", color=K.ORANGE, ms=8, zorder=5)
    ax.set_xlabel("applied current $I$"); ax.set_ylabel("$v$")
    ax.set_title("FitzHugh–Nagumo: AUTO bifurcation diagram")
    ax.legend(fontsize=9, loc="upper left")
    ax = axs[1]
    for I, col in [(0.25, K.GREY), (0.5, K.ACC), (1.0, K.BLUE)]:
        mm = L.load("fhn", i=I)
        t, X = L.traj(mm, (-1.0, -0.5), 300, dt=0.05)
        k = X.shape[0] // 2
        ax.plot(X[k:, 0], X[k:, 1], color=col, lw=2, label=f"$I = {I}$")
        mm.close()
    ax.set_xlabel("$v$"); ax.set_ylabel("$w$"); ax.legend(fontsize=9)
    ax.set_title("long-time behaviour for three currents")
    fig.tight_layout()
    K.save(fig, "ch08_fhn_auto", wide=True)
    return hbs


def nullclines_fhn():
    m = L.load("fhn", i=0.5)
    F = L.rhs_np("fhn", {"i": 0.5})
    _, cyc, T = L.periodic_orbit(m, (-1, -0.5), dt=0.01)
    v = np.linspace(-2.5, 2.5, 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    K.streams(ax, F, (-2.5, 2.5), (-1, 2.2), n=30, color="#d4d9de", scale=(1, 6))
    ax.plot(v, v - v ** 3 / 3 + 0.5, color=K.NULL_X, lw=2, label="$\\dot v = 0$")
    ax.plot(v, (v + 0.7) / 0.8, color=K.NULL_Y, lw=2, label="$\\dot w = 0$")
    ax.plot(cyc[:, 0], cyc[:, 1], color=K.CYCLE, lw=2.2, label="limit cycle")
    e = m.equilibrium({"v": 0.4, "w": 1.3})
    ev, ew = e.x["V"], e.x["W"]
    ax.plot(ev, ew, "o", mfc="white", mec=K.INK, ms=8, zorder=6, label="unstable fixed point")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-1, 2.2); ax.set_xlabel("$v$"); ax.set_ylabel("$w$")
    ax.legend(fontsize=9, loc="upper left"); ax.set_title("FitzHugh–Nagumo, $I=0.5$: nullclines and the cycle")
    K.savedata("ch01_fhn", cyc=cyc, T=T)
    K.save(fig, "ch01_fhn")


if __name__ == "__main__":
    linear_portraits()
    polar_cycle()
    flow_blob()
    nullclines_fhn()
    hopf_normal_form()
