"""Chapter 8 figures: slow-fast systems, slow manifolds, canards, slow-fast isochrons.
Run:  .venv/bin/python scripts/isochron/figs/ch08_figs.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import pyxpp
import iso_lib as L
import figkit as K

VDPA = """# van der Pol with a constant input a (Lienard form)
x'=(y-x^3/3+x)/eps
y'=a-x
par eps=0.1,a=0
init x=2,y=0
@ meth=cvode,tol=1e-11,atol=1e-11,dt=0.001,total=100,maxstor=2000000,bound=1000
done
"""


def relaxation():
    xx = np.linspace(-2.6, 2.6, 400)
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    ax.plot(xx, xx ** 3 / 3 - xx, color=K.GREY, lw=5, alpha=0.5, label="critical manifold $C_0$")
    cols = [K.BLUE, K.ACC, K.ORANGE, K.ACC2]
    store = {}
    for eps, col in zip([1.0, 0.3, 0.1, 0.02], cols):
        m = L.load("vdp", eps=eps)
        t, X, T = L.periodic_orbit(m, (2, 0), dt=min(0.002, eps / 100))
        store[eps] = (t, X, T)
        ax.plot(X[:, 0], X[:, 1], color=col, lw=1.6, label=f"$\\varepsilon = {eps}$  ($T = {T:.2f}$)")
        m.close()
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.4, 2.4); ax.set_xlabel("$x$ (fast)"); ax.set_ylabel("$y$ (slow)")
    ax.legend(fontsize=9, loc="upper left"); ax.set_title("van der Pol cycles approach the singular cycle")
    ax = axs[1]
    for eps, col in zip([1.0, 0.1, 0.02], [K.BLUE, K.ORANGE, K.ACC2]):
        t, X, T = store[eps]
        ax.plot(t / T, X[:, 0], color=col, lw=1.6, label=f"$\\varepsilon={eps}$")
    ax.set_xlabel("$t/T$"); ax.set_ylabel("$x$"); ax.legend(fontsize=9, loc="lower left")
    ax.set_title("slow drifts and fast jumps")
    fig.tight_layout()
    K.save(fig, "ch08_relaxation", wide=True)
    T3 = 3 - 2 * np.log(2)
    print("vdP periods:", {e: round(v[2], 4) for e, v in store.items()}, " singular limit (3-2 ln 2) =", round(T3, 4))
    K.savedata("ch08_relax", **{f"X{e}": v[1] for e, v in store.items()})


def slow_manifolds():
    """Attracting and repelling slow manifolds near the critical manifold, for the
    van der Pol system with input a just below the Hopf value (canard regime)."""
    eps = 0.05
    m = pyxpp.load(text=VDPA); m.set(eps=eps, a=0.0)
    xx = np.linspace(-2.4, 2.4, 400)
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.6))
    ax = axs[0]
    c0 = xx ** 3 / 3 - xx
    for sel, col, lab in [(xx < -1, K.ACC, "attracting"), ((xx > -1) & (xx < 1), K.ACC2, "repelling"), (xx > 1, K.ACC, None)]:
        ax.plot(xx[sel], c0[sel], color=col, lw=3, label=(f"$C_0$, {lab}" if lab else None))
    for x0, y0 in [(-1.6, 1.3), (2.2, -1.2), (0.0, 1.0), (0.3, -1.1), (-0.3, 0.6), (0.8, 0.2), (-0.6, -0.9)]:
        t, X = L.traj(m, (x0, y0), 0.6, dt=0.0005)
        ax.plot(X[:, 0], X[:, 1], color=K.INK, lw=0.9)
        ax.plot(x0, y0, "o", color=K.INK, ms=3)
    ax.plot([-1, 1], [2 / 3, -2 / 3], "s", color=K.ORANGE, ms=8, label="fold points")
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-1.5, 1.5); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.legend(fontsize=9, loc="upper left"); ax.set_title("fast jumps onto the attracting branches")
    ax = axs[1]
    e2 = 0.1
    m2 = pyxpp.load(text=VDPA); m2.set(eps=e2, a=0.0)
    ax.plot(xx, c0, color=K.GREY, lw=1.2, ls="--", label="$C_0$")
    C = lambda x: x ** 3 / 3 - x
    for x0, stop in [(-2.3, lambda X: X[:, 0] > -1.0), (2.3, lambda X: X[:, 0] < 1.0)]:
        t, Sa = L.traj(m2, (x0, C(x0)), 6, dt=0.0005)
        k = np.argmax(stop(Sa)) or len(Sa)
        ax.plot(Sa[:k, 0], Sa[:k, 1], color=K.ACC, lw=2.4, label="$S^a_\\varepsilon$ (forward in time)" if x0 < 0 else None)
    for x0 in (0.97, -0.97):
        t, Sr = L.traj(m2, (x0, C(x0)), 8, dt=0.0005, back=True)
        k = np.argmax(np.hypot(Sr[:, 0], Sr[:, 1]) < 0.02) or len(Sr)
        ax.plot(Sr[:k, 0], Sr[:k, 1], color=K.ACC2, lw=2.4, label="$S^r_\\varepsilon$ (backward in time)" if x0 > 0 else None)
    ax.plot([-1, 1], [2 / 3, -2 / 3], "s", color=K.ORANGE, ms=7)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-1.5, 1.5); ax.set_xlabel("$x$"); ax.legend(fontsize=9, loc="upper left")
    ax.set_title("slow manifolds ($\\varepsilon = 0.1$) lie $O(\\varepsilon)$ from $C_0$")
    fig.tight_layout()
    K.save(fig, "ch08_slowmanifolds", wide=True)


def canard_explosion():
    """AUTO: van der Pol with input a; Hopf at a = 1, explosive growth of the cycle just below it."""
    eps = 0.05
    m = pyxpp.load(text=VDPA); m.set(eps=eps, a=1.2, meth="cvode")
    m.run(total=50); m.run_last(total=20)
    eq = m.auto.run("a", start="steady", ds=-0.002, dsmax=0.01, parmin=0.5, parmax=1.3, nmax=400)
    hb = eq.special("HB")[0]
    po = m.auto.run(grab=hb.label, choice="periodic", ntst=200, ncol=4, ds=-0.0005, dsmin=1e-9, dsmax=0.005, nmax=3000,
                    parmin=0.5, parmax=1.3)
    df = po.to_pandas("x")
    print("Hopf at a =", hb.x, " points on periodic branch", len(df))
    amp = df.ymax - df.ymin
    # draw some cycles: integrate from AUTO's starting point at chosen parameter values
    picks = []
    targets = [0.15, 0.6, 1.2, 2.0, 2.9, 3.6]
    for A in targets:
        j = int(np.argmin(np.abs(amp.values - A)))
        picks.append(j)
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.5))
    ax = axs[0]
    ax.plot(df.par, amp, "o-", ms=2, lw=1, color=K.INK)
    for j, col in zip(picks, plt.cm.viridis(np.linspace(0, 0.9, len(picks)))):
        ax.plot(df.par.values[j], amp.values[j], "o", ms=8, color=col, zorder=5)
    ax.axvline(hb.x, color=K.ORANGE, lw=1, ls="--")
    ax.text(hb.x, 0.2, " Hopf", color=K.ORANGE, fontsize=10)
    ax.set_xlabel("input $a$"); ax.set_ylabel("cycle amplitude (max $x$ − min $x$)")
    ax.set_title("AUTO: the canard explosion")
    span = df.par[(amp > 0.3) & (amp < 3.5)]
    ax.set_xlim(span.min() - 0.004, hb.x + 0.003)
    ax = axs[1]
    xx = np.linspace(-2.3, 2.3, 300)
    ax.plot(xx, xx ** 3 / 3 - xx, color=K.GREY, lw=4, alpha=0.5)
    cyc = []
    for j, col in zip(picks, plt.cm.viridis(np.linspace(0, 0.9, len(picks)))):
        p = po[j] if hasattr(po, "__getitem__") else None
        a = float(df.par.values[j])
        mm = pyxpp.load(text=VDPA); mm.set(eps=eps, a=a)
        u0 = (float(df.ymax.values[j]) * 0 + p.u0["x"], p.u0["y"]) if p is not None else (0.5, 0.0)
        Tp = float(df.period.values[j])
        t, X = L.traj(mm, u0, Tp, dt=Tp / 4000)
        cyc.append(X)
        ax.plot(X[:, 0], X[:, 1], color=col, lw=1.6)
        mm.close()
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-1.0, 1.25); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("canard cycles follow the repelling branch")
    fig.tight_layout()
    K.save(fig, "ch08_canards", wide=True)
    K.savedata("ch08_canards", par=df.par.values, amp=amp.values, hb=hb.x, cyc=np.array(cyc, dtype=object))


def vdp_slowfast_isochrons():
    """Isochrons of van der Pol with eps = 0.1: crowded along the slow branches; inside the cycle they
    are squeezed against the repelling middle branch of the critical manifold."""
    eps = 0.1
    m = L.load("vdp", eps=eps)
    t, X, T = L.periodic_orbit(m, (2, 0), dt=0.0005)
    xs = np.linspace(-2.6, 2.6, 321); ys = np.linspace(-1.4, 1.4, 321)
    th, nper = L.phase_map("vdp", {"eps": eps}, X, T, xs, ys, scale=(1, 1), tol=1e-6, maxper=300)
    K.savedata("ch08_vdp01", xs=xs, ys=ys, th=th, X=X, T=T)
    print("vdp eps=0.1 phase map: NaN", np.isnan(th).sum(), " max periods", nper.max())
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.8))
    xx = np.linspace(-2.6, 2.6, 300)
    for ax, lim, nlev in [(axs[0], (-2.6, 2.6, -1.4, 1.4), 30), (axs[1], (-1.0, 1.0, -0.75, 0.75), 30)]:
        ax.pcolormesh(xs, ys, th, cmap=K.CYCLIC, shading="auto", alpha=0.3, rasterized=True)
        curves = L.isochron_curves(xs, ys, th, np.arange(nlev) / nlev)
        for c, segs in curves.items():
            for seg in segs:
                ax.plot(seg[:, 0], seg[:, 1], color=K.PHASE_CMAP(c), lw=0.9)
        ax.plot(xx, xx ** 3 / 3 - xx, color=K.INK, lw=1, ls="--")
        ax.plot(X[:, 0], X[:, 1], color="k", lw=1.8)
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    axs[0].set_title("van der Pol, $\\varepsilon = 0.1$: 30 isochrons")
    axs[1].set_title("inside: isochrons hug the repelling branch (dashed)")
    fig.tight_layout()
    K.save(fig, "ch08_vdpiso", wide=True)


if __name__ == "__main__":
    relaxation()
    slow_manifolds()
    canard_explosion()
    vdp_slowfast_isochrons()
