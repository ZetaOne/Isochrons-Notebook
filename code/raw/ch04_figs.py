"""Chapter 4 figures: asymptotic phase and isochrons.
Phase maps are computed by integrating every grid point with XPPAUT (pyauto-xpp)
for whole periods and reading off the phase where it lands on the cycle.
Run:  .venv/bin/python scripts/isochron/figs/ch04_figs.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import iso_lib as L
import figkit as K


def sl_phase(b, n=161, R=2.0):
    m = L.load("sl", b=b)
    t, X, T = L.periodic_orbit(m, (0.5, 0.0), dt=0.002)
    xs = np.linspace(-R, R, n); ys = np.linspace(-R, R, n)
    th, _ = L.phase_map("sl", {"b": b}, X, T, xs, ys, tol=1e-6)
    return m, t, X, T, xs, ys, th


def draw_isochrons(ax, xs, ys, th, nlev=20, lw=1.3, cmap=None, zorder=3):
    cmap = cmap or K.PHASE_CMAP
    levels = np.arange(nlev) / nlev
    curves = L.isochron_curves(xs, ys, th, levels)
    for c in levels:
        for seg in curves[c]:
            ax.plot(seg[:, 0], seg[:, 1], color=cmap(c), lw=lw, zorder=zorder)
    return curves


def definition_gif():
    """A point off the cycle and the point on the cycle with the same asymptotic phase end up together."""
    b = 0.0
    eps = 1.0
    m = L.load("vdp", eps=eps)
    t, X, T = L.periodic_orbit(m, (2, 0), dt=0.002)
    look = L.PhaseLookup(X)
    x0 = np.array([-0.4, 2.8])
    tt, P = L.traj(m, x0, 3 * T, dt=T / 400)
    th0, _ = look(P[-1])                        # 3 whole periods: phase of x0 = phase of the landing point
    k0 = int(round(th0 * (len(X) - 1)))
    g = np.vstack([X[(k0 + j * (len(X) - 1) // 400) % (len(X) - 1)] for j in range(len(tt))])
    # a second point on the cycle with a different phase, for contrast
    k1 = (k0 + (len(X) - 1) // 6) % (len(X) - 1)
    g1 = np.vstack([X[(k1 + j * (len(X) - 1) // 400) % (len(X) - 1)] for j in range(len(tt))])
    K.savedata("ch04_def", P=P, g=g, g1=g1, X=X, T=T, th0=th0)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(X[:, 0], X[:, 1], color=K.CYCLE, lw=1.3, alpha=0.5)
    path, = ax.plot([], [], color=K.ACC2, lw=1.2)
    a, = ax.plot([], [], "o", color=K.ACC2, ms=10, label="$\\vec x_0$ (off the cycle)")
    c, = ax.plot([], [], "o", color=K.ACC, ms=8, label=f"on $\\Gamma$, phase $\\vartheta(\\vec x_0)={th0:.3f}$")
    d, = ax.plot([], [], "o", color=K.GREY, ms=8, label="on $\\Gamma$, a different phase")
    fig.subplots_adjust(bottom=0.25)
    lab = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=11, bbox=K.BOX, zorder=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9.5)
    ax.set_xlim(-3, 3); ax.set_ylim(-3.2, 3.2); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("asymptotic phase: whom does $\\vec x_0$ end up with?")
    frames = len(tt)
    ks = np.arange(0, frames, 3)

    def upd(i):
        k = ks[i]
        path.set_data(P[:k + 1, 0], P[:k + 1, 1])
        a.set_data([P[k, 0]], [P[k, 1]]); c.set_data([g[k, 0]], [g[k, 1]]); d.set_data([g1[k, 0]], [g1[k, 1]])
        lab.set_text(f"$t/T = {tt[k] / T:.2f}$,  distance to the teal point: {np.linalg.norm(P[k] - g[k]):.1e}")
    def still():
        for q, col in [(P[0], K.ACC2), (g[0], K.ACC), (g1[0], K.GREY)]:
            ax.plot(*q, "o", mfc="white", mec=col, mew=2, ms=10, zorder=7)
        ax.annotate("start", P[0], xytext=(8, 6), textcoords="offset points", fontsize=10, color=K.ACC2)
        lab.set_text("open circles: $t=0$;  filled: $t=3T$ (magenta and teal coincide)")
    K.save_gif(fig, upd, len(ks), "ch04_def", fps=20, still_frame=-1, still_draw=still)


def radial_and_spiral():
    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, b, title in [(axs[0], 0.0, "radial isochron clock ($b=0$): rays"),
                         (axs[1], 0.6, "with twist $b=0.6$: logarithmic spirals")]:
        m, t, X, T, xs, ys, th = sl_phase(b)
        out[b] = (xs, ys, th, X, T)
        ax.pcolormesh(xs, ys, th, cmap=K.CYCLIC, shading="auto", alpha=0.35, rasterized=True)
        draw_isochrons(ax, xs, ys, th, nlev=12)
        ax.plot(X[:, 0], X[:, 1], color=K.CYCLE, lw=2.2)
        ax.plot(0, 0, "o", mfc="white", mec=K.INK, ms=7, zorder=6)
        ax.plot(*X[0], "*", color=K.ORANGE, ms=13, zorder=6)
        ax.set_aspect("equal"); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_title(title)
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
        # check against the exact phase  theta - b ln r
        XX, YY = np.meshgrid(xs, ys); R = np.hypot(XX, YY)
        ex = ((np.arctan2(YY, XX) - b * np.log(np.where(R > 0, R, 1))) / (2 * np.pi)) % 1
        err = np.abs(((th - ex) + 0.5) % 1 - 0.5)[R > 0.05]
        print(f"b={b}: max |phase - exact| = {np.nanmax(err):.2e}")
        K.savedata(f"ch04_sl_b{b}", xs=xs, ys=ys, th=th, X=X, T=T)
    fig.tight_layout()
    K.save(fig, "ch04_radial_spiral", wide=True)
    return out


def cloud_gif():
    """Points spread along one isochron (b = 0.6) flow together: the curve they form is again an isochron,
    and they all land on the cycle at the same point."""
    b = 0.6
    m = L.load("sl", b=b)
    t, X, T = L.periodic_orbit(m, (0.5, 0.0), dt=0.002)
    r = np.geomspace(0.12, 2.2, 26)
    ang = b * np.log(r)                                    # isochron of phase 0: theta = b ln r
    starts = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
    paths = np.array([L.traj(m, s, 0.35 * T, dt=T / 600)[1] for s in starts])
    xs = np.linspace(-2.4, 2.4, 2); rr = np.geomspace(0.03, 3, 400)
    K.savedata("ch04_cloud", paths=paths, X=X, T=T, b=b)
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    ax.plot(X[:, 0], X[:, 1], color=K.CYCLE, lw=1.5)
    iso, = ax.plot([], [], color=K.ACC2, lw=1, alpha=0.6)
    dots = ax.scatter(starts[:, 0], starts[:, 1], c=np.log(r), cmap="viridis", s=26, zorder=5)
    lab = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=11, bbox=K.BOX, zorder=10)
    ax.set_aspect("equal"); ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.4, 2.4)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("26 states on one isochron move as one")
    frames = paths.shape[1]

    def upd(i):
        P = paths[:, i, :]
        dots.set_offsets(P)
        tt = i * T / 600
        phi = (tt * (1.0 - b))                      # the isochron rotates rigidly at rate (w - b)
        a2 = b * np.log(rr) + phi
        iso.set_data(rr * np.cos(a2), rr * np.sin(a2))
        lab.set_text(f"$t/T = {tt / T:.2f}$")

    def still():
        for j, k in enumerate([0, 15, 45, 110]):
            P = paths[:, k, :]
            ax.plot(P[:, 0], P[:, 1], "o-", ms=3, lw=0.8, color=[K.ACC2, K.ORANGE, K.BLUE, K.ACC][j], alpha=0.9,
                    label=f"$t = {k / 600:.3f}\\,T$")
        ax.legend(loc="lower left", fontsize=10, title="same 26 states at", title_fontsize=10)
        lab.set_text("")
        dots.set_offsets(np.empty((0, 2))); iso.set_data([], [])
    K.save_gif(fig, upd, frames, "ch04_cloud", fps=24, still_draw=still)


def vdp_isochrons():
    eps = 1.0
    m = L.load("vdp", eps=eps)
    t, X, T = L.periodic_orbit(m, (2, 0), dt=0.002)
    xs = np.linspace(-3, 3, 241); ys = np.linspace(-3.4, 3.4, 241)
    th, _ = L.phase_map("vdp", {"eps": eps}, X, T, xs, ys, tol=1e-6)
    K.savedata("ch04_vdp1", xs=xs, ys=ys, th=th, X=X, T=T)
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 5.2))
    ax = axs[0]
    pm = ax.pcolormesh(xs, ys, th, cmap=K.CYCLIC, shading="auto", rasterized=True)
    ax.plot(X[:, 0], X[:, 1], color="k", lw=2)
    ax.plot(0, 0, "o", mfc="white", mec="k", ms=7)
    cb = fig.colorbar(pm, ax=ax, fraction=0.045); cb.set_label("asymptotic phase $\\vartheta$")
    ax.set_title("asymptotic phase of every point"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax = axs[1]
    draw_isochrons(ax, xs, ys, th, nlev=20)
    ax.plot(X[:, 0], X[:, 1], color="k", lw=2)
    N = 20
    for j in range(N):
        k = j * (len(X) - 1) // N
        ax.plot(*X[k], "o", color=K.PHASE_CMAP(j / N), ms=5, zorder=6, mec="white", mew=0.5)
    ax.plot(0, 0, "o", mfc="white", mec="k", ms=7, zorder=6)
    ax.set_title("20 isochrons, equally spaced in phase"); ax.set_xlabel("$x$")
    for a in axs:
        a.set_xlim(-3, 3); a.set_ylim(-3.4, 3.4); a.set_aspect("equal")
    fig.tight_layout()
    K.save(fig, "ch04_vdp", wide=True)


if __name__ == "__main__":
    radial_and_spiral()
    definition_gif()
    cloud_gif()
    vdp_isochrons()
