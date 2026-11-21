"""Chapter 13 figures: the global isochrons of the reduced Hodgkin-Huxley neuron (paper section 4).
Needs the phase maps from rhh_phasemaps.py.
Run:  .venv/bin/python scripts/isochron/figs/ch13_figs.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import iso_lib as L
import figkit as K

SC = np.array([150.0, 0.5])


def cyc():
    c = K.loaddata("rhh_cycle")
    return c["t"], c["X"], float(c["T"])


def vnull(ax, xlim, ylim, **kw):
    F = L.rhs_np("rhh", {})
    vg = np.linspace(xlim[0] - 0.00123, xlim[1] + 0.00177, 900); ng = np.linspace(ylim[0], ylim[1], 700)
    VV, NN = np.meshgrid(vg, ng)
    ax.contour(vg, ng, F(np.array([VV, NN]))[0], levels=[0], **kw)


def draw_isos(ax, d, levels, lw=0.8, alpha=1.0, color=None):
    curves = L.isochron_curves(d["xs"], d["ys"], d["th"], levels)
    for c in levels:
        for seg in curves[c]:
            ax.plot(seg[:, 0], seg[:, 1], color=color or K.PHASE_CMAP(c), lw=lw, alpha=alpha)
    return curves


def iso100():
    d = K.loaddata("rhh_phase_full")
    t, X, T = cyc()
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    levels = np.arange(100) / 100
    draw_isos(ax, d, levels, lw=0.6)
    ax.plot(X[:, 0], X[:, 1], color="k", lw=2.2)
    for j in range(100):
        k = int(j / 100 * (len(X) - 1))
        ax.plot(*X[k], "o", ms=2.6, color=K.PHASE_CMAP(j / 100), zorder=6)
    ax.plot(*X[0], "*", color=K.ORANGE, ms=13, zorder=7)
    ax.annotate("$\\gamma_0$", X[0], xytext=(6, 6), textcoords="offset points", fontsize=12)
    ax.plot(-59.6044, 0.4026, "o", mfc="white", mec="k", ms=6, zorder=7)
    ax.set_xlim(-90, 60); ax.set_ylim(0.3, 0.8); ax.set_xlabel("$V$"); ax.set_ylabel("$n$")
    ax.set_title("100 isochrons, equally spaced in phase (cf. paper Fig. 2)")
    sm = plt.cm.ScalarMappable(cmap=K.PHASE_CMAP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02); cb.set_label("phase $\\vartheta$")
    K.save(fig, "ch13_iso100", wide=True)
    # where on Gamma do the 100 isochrons start? fraction near the left branch
    left = np.mean([X[int(j / 100 * (len(X) - 1)), 0] < -50 for j in range(100)])
    print(f"fraction of the 100 isochron base points with V < -50: {left:.2f}")


def wide():
    d = K.loaddata("rhh_phase_wide")
    t, X, T = cyc()
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.pcolormesh(d["xs"], d["ys"], d["th"], cmap=K.CYCLIC, shading="auto", alpha=0.35, rasterized=True)
    draw_isos(ax, d, np.arange(40) / 40, lw=0.8)
    ax.plot(X[:, 0], X[:, 1], color="k", lw=2)
    ax.set_xlim(-200, 150); ax.set_ylim(0.0, 1.1); ax.set_xlabel("$V$"); ax.set_ylabel("$n$")
    ax.set_title("further out: outer branches run off almost horizontally")
    K.save(fig, "ch13_wide", wide=True)


def left_zoom():
    d = K.loaddata("rhh_phase_left")
    t, X, T = cyc()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.0))
    for ax, nlev, title in [(axs[0], 100, "zoom: 100 isochrons"), (axs[1], 400, "400 isochrons")]:
        draw_isos(ax, d, np.arange(nlev) / nlev, lw=0.45 if nlev > 100 else 0.7)
        ax.plot(X[:, 0], X[:, 1], color="k", lw=2)
        vnull(ax, (-80, -40), (0.36, 0.56), colors=[K.INK], linewidths=1.2, linestyles="dashed")
        ax.plot(-59.6044, 0.4026, "o", mfc="white", mec="k", ms=7, zorder=7)
        ax.set_xlim(-80, -40); ax.set_ylim(0.36, 0.56); ax.set_xlabel("$V$"); ax.set_ylabel("$n$"); ax.set_title(title)
    fig.tight_layout()
    K.save(fig, "ch13_left", wide=True)


def eq_zooms():
    t, X, T = cyc()
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.3))
    for ax, name, title in [(axs[0], "left", "window $[-80,-40]\\times[0.36,0.56]$"),
                            (axs[1], "eq", "zoom $\\times 8$"),
                            (axs[2], "eq2", "zoom $\\times 40$")]:
        d = K.loaddata(f"rhh_phase_{name}")
        ax.pcolormesh(d["xs"], d["ys"], d["th"], cmap=K.CYCLIC, shading="auto", rasterized=True)
        draw_isos(ax, d, np.arange(10) / 10, lw=0.6, color="white", alpha=0.8)
        ax.plot(-59.6044, 0.402577, "o", mfc="white", mec="k", ms=6, zorder=7)
        ax.set_xlim(d["xs"][0], d["xs"][-1]); ax.set_ylim(d["ys"][0], d["ys"][-1]); ax.set_title(title)
        ax.set_xlabel("$V$")
        ax.ticklabel_format(useOffset=False)
    axs[0].set_ylabel("$n$")
    fig.tight_layout()
    K.save(fig, "ch13_eqzoom", wide=True)


def arclength_iso():
    """The isochron of gamma_0 by continuation, coloured by arclength from gamma_0 (cf. paper Fig. 3)."""
    d = K.loaddata("ch09_bundle")
    t, X, T = cyc()
    F = L.rhs_np("rhh", {})
    w = d["w"][:, 0]
    out = []
    for side, arc in [(+1, 3.0), (-1, 1.5)]:
        P = L.isochron_bvp(F, t, X, T, w, eta=1e-4, arclen=arc, kmax=4, side=side, scale=tuple(SC),
                           hmax=0.006, tol=1e-5, max_nodes=600000)
        out.append(P)
        print("arclength isochron side", side, len(P), "points, end", P[-1])
    K.savedata("ch13_arciso", P1=out[0], P2=out[1])
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.0))
    for ax, lim in [(axs[0], (-90, 60, 0.3, 0.8)), (axs[1], (-75, -40, 0.37, 0.55))]:
        ax.plot(X[:, 0], X[:, 1], color="k", lw=1.6)
        for P in out:
            sarc = np.r_[0, np.cumsum(np.hypot(*(np.diff(P, axis=0) / SC).T))]
            seg = np.stack([P[:-1], P[1:]], axis=1)
            lc = LineCollection(seg, cmap="coolwarm_r", norm=plt.Normalize(0, 3.0), lw=2.2)
            lc.set_array(sarc[:-1]); ax.add_collection(lc)
        ax.plot(*X[0], "*", color=K.ORANGE, ms=13, zorder=7)
        ax.plot(-59.6044, 0.4026, "o", mfc="white", mec="k", ms=6, zorder=7)
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_xlabel("$V$"); ax.set_ylabel("$n$")
    axs[0].set_title("$I(\\gamma_0)$ coloured by arclength from $\\gamma_0$ (cf. paper Fig. 3)")
    axs[1].set_title("zoom: the boomerang turns")
    sm = plt.cm.ScalarMappable(cmap="coolwarm_r", norm=plt.Normalize(0, 3.0))
    cb = fig.colorbar(sm, ax=axs, fraction=0.025, pad=0.01); cb.set_label("arclength (scaled units)")
    K.save(fig, "ch13_arclength", wide=True)


def foliation_gif():
    """The foliation filling the plane: isochrons added one phase at a time; they never cross."""
    d = K.loaddata("rhh_phase_full")
    t, X, T = cyc()
    levels = np.arange(60) / 60
    curves = L.isochron_curves(d["xs"], d["ys"], d["th"], levels)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(X[:, 0], X[:, 1], color="k", lw=2)
    ax.set_xlim(-90, 60); ax.set_ylim(0.3, 0.8); ax.set_xlabel("$V$"); ax.set_ylabel("$n$")
    ax.set_title("the isochrons foliate the plane: they never cross")
    lab = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=11, bbox=K.BOX)
    drawn = []

    def upd(i):
        c = levels[i]
        for seg in curves[c]:
            drawn.append(ax.plot(seg[:, 0], seg[:, 1], color=K.PHASE_CMAP(c), lw=0.9)[0])
        k = int(c * (len(X) - 1))
        ax.plot(*X[k], "o", ms=3.5, color=K.PHASE_CMAP(c), zorder=6)
        lab.set_text(f"phases $0$ to ${c:.2f}$")
    K.save_gif(fig, upd, len(levels), "ch13_foliation", fps=10)


def basepoints():
    """Where do equally spaced isochrons start on Gamma? Phase against arclength along Gamma."""
    t, X, T = cyc()
    s = np.r_[0, np.cumsum(np.hypot(*(np.diff(X, axis=0) / SC).T))]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(s / s[-1], t / T, color=K.INK, lw=1.8)
    ax.plot([0, 1], [0, 1], ":", color=K.GREY)
    ax.set_xlabel("arclength along $\\Gamma$ from $\\gamma_0$ (fraction, scaled units)"); ax.set_ylabel("phase $\\vartheta$")
    ax.set_title("phase against arclength along the neuron's cycle")
    K.save(fig, "ch13_basepoints")


if __name__ == "__main__":
    which = sys.argv[1:] or ["iso100", "wide", "left", "eq", "arc", "gif", "base"]
    if "iso100" in which: iso100()
    if "wide" in which: wide()
    if "left" in which: left_zoom()
    if "eq" in which: eq_zooms()
    if "arc" in which: arclength_iso()
    if "gif" in which: foliation_gif()
    if "base" in which: basepoints()


def cover():
    """Cover art: 200 isochrons of the neuron, no axes."""
    d = K.loaddata("rhh_phase_full")
    t, X, T = cyc()
    fig = plt.figure(figsize=(10, 4.6))
    ax = fig.add_axes([0, 0, 1, 1])
    levels = np.arange(200) / 200
    draw_isos(ax, d, levels, lw=0.55)
    ax.plot(X[:, 0], X[:, 1], color="k", lw=2.4)
    ax.set_xlim(-90, 60); ax.set_ylim(0.32, 0.78); ax.axis("off")
    K.save(fig, "cover", wide=True, tight=False, labels=False)


if __name__ == "__main__" and "cover" in sys.argv:
    cover()
