"""Chapter 2 figures: a periodic orbit as a clock (phase on the cycle).
Run:  .venv/bin/python scripts/isochron/figs/ch02_figs.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import iso_lib as L
import figkit as K

EPS = 0.1


def vdp_cycle(eps=EPS):
    m = L.load("vdp", eps=eps)
    t, X, T = L.periodic_orbit(m, (2.0, 0.0), dt=0.0005)
    return m, t, X, T


def colored_line(ax, X, c, cmap, lw=4, vmin=0, vmax=1):
    seg = np.stack([X[:-1], X[1:]], axis=1)
    lc = LineCollection(seg, cmap=cmap, norm=plt.Normalize(vmin, vmax), lw=lw, capstyle="round")
    lc.set_array(c[:-1]); ax.add_collection(lc)
    return lc


def markers_gif():
    m, t, X, T = vdp_cycle()
    F = L.rhs_np("vdp", {"eps": EPS})
    K.savedata("ch02_vdp", t=t, X=X, T=T, eps=EPS)
    N = 24
    idx0 = (np.arange(N) * (len(X) - 1) // N)
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    K.streams(ax, F, (-2.6, 2.6), (-1.4, 1.4), n=34, color="#dde1e5", scale=(EPS, 1))
    v = np.linspace(-2.6, 2.6, 300)
    ax.plot(v, v ** 3 / 3 - v, color=K.NULL_X, lw=1.2, alpha=0.8, label="$\\dot x = 0$")
    ax.plot(X[:, 0], X[:, 1], color=K.CYCLE, lw=1.6)
    dots = ax.scatter(X[idx0, 0], X[idx0, 1], c=np.arange(N) / N, cmap=K.PHASE_CMAP, s=46, zorder=5, edgecolor="white", lw=0.6)
    ax.plot(*X[0], "*", color=K.ORANGE, ms=14, zorder=6)
    ax.annotate("$\\gamma_0$  (phase 0)", X[0], xytext=(10, 8), textcoords="offset points", fontsize=11)
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", fontsize=11, bbox=K.BOX, zorder=10)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-1.4, 1.4); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(f"24 points equally spaced in phase (van der Pol, $\\varepsilon = {EPS}$)")
    frames = 160
    step = (len(X) - 1) / frames

    def upd(i):
        idx = (idx0 + int(round(i * step))) % (len(X) - 1)
        dots.set_offsets(X[idx])
        txt.set_text(f"$t/T = {i / frames:.2f}$")
    K.save_gif(fig, upd, frames, "ch02_markers", fps=16, still_frame=0)


def speed_and_arclength():
    m, t, X, T = vdp_cycle()
    F = L.rhs_np("vdp", {"eps": EPS})
    th = t / T
    sp = np.hypot(*F(X.T))
    ds = np.r_[0, np.cumsum(np.hypot(*np.diff(X, axis=0).T))]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    ax = axs[0]
    lc = colored_line(ax, X, th, K.CYCLIC)
    ax.plot(*X[0], "*", color=K.ORANGE, ms=13, zorder=5)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-1.2, 1.2); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("the cycle coloured by phase $\\vartheta$")
    cb = fig.colorbar(lc, ax=ax, fraction=0.05); cb.set_label("$\\vartheta$")
    ax = axs[1]
    ax.plot(th, sp, color=K.INK, lw=1.8)
    ax.set_ylim(0, None)
    ax.set_xlabel("phase $\\vartheta = t/T$"); ax.set_ylabel("speed $|\\vec F|$")
    ax.set_title("speed along the cycle")
    ax = axs[2]
    ax.plot(th, ds / ds[-1], color=K.ACC2, lw=2, label="van der Pol")
    ax.plot([0, 1], [0, 1], "--", color=K.GREY, lw=1.4, label="uniform speed (circle)")
    ax.set_xlabel("phase $\\vartheta$"); ax.set_ylabel("arclength / total length")
    ax.set_title("arclength is not phase"); ax.legend(fontsize=9)
    fig.tight_layout()
    K.save(fig, "ch02_speed", wide=True)
    frac_fast = np.mean(sp > np.median(sp) * 10)
    print(f"T = {T:.4f}; fraction of time at >10x median speed: {frac_fast:.3f}")
    jumps = (np.abs(X[:, 0]) < 1.0)
    print(f"fraction of the period with |x|<1 (the jumps): {jumps.mean():.3f}; of the length: {np.sum(np.hypot(*np.diff(X, axis=0).T)[jumps[:-1]]) / ds[-1]:.3f}")


def circle_map():
    m, t, X, T = vdp_cycle(eps=1.0)
    th = t / T
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.2))
    ax = axs[0]
    colored_line(ax, X, th, K.CYCLIC, lw=6)
    for q in [0, 0.25, 0.5, 0.75]:
        k = int(q * (len(X) - 1))
        ax.plot(*X[k], "o", color=K.INK, ms=6)
        u = X[k] / np.linalg.norm(X[k])                  # label radially outward
        ax.annotate(f"$\\vartheta={q}$", X[k], xytext=(22 * u[0], 18 * u[1]), textcoords="offset points",
                    fontsize=11, ha="left" if u[0] > 0 else "right", va="center")
    ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_title("$\\Gamma$ (van der Pol, $\\varepsilon=1$)")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax = axs[1]
    a = 2 * np.pi * th
    colored_line(ax, np.column_stack([np.cos(a), np.sin(a)]), th, K.CYCLIC, lw=6)
    for q in [0, 0.25, 0.5, 0.75]:
        u = np.array([np.cos(2 * np.pi * q), np.sin(2 * np.pi * q)])
        ax.plot(*u, "o", color=K.INK, ms=6)
        ax.annotate(f"$\\vartheta={q}$", u, xytext=(18 * u[0], 16 * u[1]), textcoords="offset points",
                    fontsize=11, ha="left" if u[0] > 0.1 else ("right" if u[0] < -0.1 else "center"), va="center")
    ax.set_aspect("equal"); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.axis("off")
    ax.set_title("the phase circle $S^1$: $\\vartheta \\mapsto (\\cos 2\\pi\\vartheta, \\sin 2\\pi\\vartheta)$")
    fig.tight_layout()
    K.save(fig, "ch02_circle", wide=True)


if __name__ == "__main__":
    markers_gif()
    speed_and_arclength()
    circle_map()
