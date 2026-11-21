"""Chapter 7 figures: phaseless sets.
Run:  .venv/bin/python scripts/isochron/figs/ch07_figs.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import iso_lib as L
import figkit as K

A, B, W = 0.5, 0.5, 1.0


def exact_sl2(XX, YY):
    """Asymptotic phase of r' = r(r^2-a^2)(1-r^2), th' = w - b r^2 (derived in the text)."""
    R = np.hypot(XX, YY); TH = np.arctan2(YY, XX)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = -(B / (2 * A ** 2)) * (np.log(1 - A ** 2 / R ** 2) - np.log(1 - A ** 2))
    th = ((TH + g) / (2 * np.pi)) % 1
    th[R <= A] = np.nan
    return th


def two_cycles():
    m = L.load("sl2", a=A, b=B, w=W)
    t, X, T = L.periodic_orbit(m, (0.8, 0.0), dt=0.002)
    print("sl2: T =", T, " exact", 2 * np.pi / (W - B))
    # XPP check on a coarse grid outside r = a
    xs = np.linspace(-1.6, 1.6, 61); ys = np.linspace(-1.6, 1.6, 61)
    XX, YY = np.meshgrid(xs, ys)
    keep = np.hypot(XX, YY) > A + 0.02
    pts = np.column_stack([XX[keep], YY[keep]])
    thx, _ = L.phase_points("sl2", {"a": A, "b": B, "w": W}, X, T, pts, tol=1e-6)
    ex = exact_sl2(XX, YY)[keep]
    err = np.abs(((thx - ex) + 0.5) % 1 - 0.5)
    print(f"sl2: XPP vs exact phase, max error {np.nanmax(err):.2e} over {np.sum(np.isfinite(thx))} points")
    # fine exact map for the picture
    n = 1400
    xs = np.linspace(-1.6, 1.6, n); ys = np.linspace(-1.6, 1.6, n)
    XX, YY = np.meshgrid(xs, ys)
    th = exact_sl2(XX, YY)
    K.savedata("ch07_sl2", a=A, b=B, w=W, T=T)
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 5.4))
    for ax, lim in [(axs[0], 1.6), (axs[1], 0.75)]:
        ax.pcolormesh(xs, ys, np.where(np.isnan(th), np.nan, th), cmap=K.CYCLIC, shading="auto", alpha=0.3, rasterized=True)
        nlev = 8
        for k in range(nlev):
            c = k / nlev
            s = np.sin(2 * np.pi * (th - c)); co = np.cos(2 * np.pi * (th - c))
            s = np.where(co > 0, s, np.nan)
            ax.contour(xs, ys, s, levels=[0], colors=[K.PHASE_CMAP(c)], linewidths=1.0)
        cc = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(cc), np.sin(cc), color="k", lw=2.2)
        ax.plot(A * np.cos(cc), A * np.sin(cc), "--", color=K.ACC2, lw=2)
        ax.fill(A * np.cos(cc), A * np.sin(cc), color="#eceff2", zorder=0)
        ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    axs[0].set_title("basin boundary $r=a$ (dashed): isochrons wind onto it")
    axs[1].set_title("zoom: every isochron comes close to every point of $r=a$")
    axs[0].text(0, 0, "not in $B(\\Gamma)$", ha="center", va="center", fontsize=10, color=K.INK)
    fig.tight_layout()
    K.save(fig, "ch07_twocycles", wide=True)


def vdp_zoom():
    """Isochrons near the unstable spiral point of van der Pol, at three magnifications."""
    eps = 1.0
    m = L.load("vdp", eps=eps)
    t, X, T = L.periodic_orbit(m, (2, 0), dt=0.002)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, R in zip(axs, [1.0, 0.1, 0.01]):
        xs = np.linspace(-R, R, 151); ys = np.linspace(-R, R, 151)
        th, nper = L.phase_map("vdp", {"eps": eps}, X, T, xs, ys, tol=1e-6, maxper=200)
        K.savedata(f"ch07_zoom_{R}", xs=xs, ys=ys, th=th)
        ax.pcolormesh(xs, ys, th, cmap=K.CYCLIC, shading="auto", rasterized=True)
        curves = L.isochron_curves(xs, ys, th, np.arange(8) / 8)
        for c, segs in curves.items():
            for seg in segs:
                ax.plot(seg[:, 0], seg[:, 1], color="white", lw=0.7, alpha=0.8)
        ax.plot(0, 0, "o", mfc="white", mec="k", ms=6)
        ax.set_aspect("equal"); ax.set_xlim(-R, R); ax.set_ylim(-R, R)
        ax.set_title(f"window $\\pm{R}$ around the fixed point")
        ax.set_xlabel("$x$")
        print("zoom", R, "max periods needed", nper.max())
    axs[0].set_ylabel("$y$")
    fig.tight_layout()
    K.save(fig, "ch07_vdpzoom", wide=True)


def noise_randomises():
    """Same small cloud of initial states: near the cycle the phases stay together; near the phaseless point they are scrambled."""
    eps = 1.0
    m = L.load("vdp", eps=eps)
    t, X, T = L.periodic_orbit(m, (2, 0), dt=0.002)
    rng = np.random.default_rng(1)
    sig = 0.08
    c_cyc = X[int(0.4 * (len(X) - 1))]
    clouds = {"near $\\Gamma$": c_cyc + sig * rng.standard_normal((400, 2)),
              "near the fixed point": np.array([0.0, 0.0]) + sig * rng.standard_normal((400, 2))}
    res = {}
    for k, pts in clouds.items():
        th, _ = L.phase_points("vdp", {"eps": eps}, X, T, pts, tol=1e-6, maxper=200)
        res[k] = th
    K.savedata("ch07_noise", **{("cyc" if "Gamma" in k else "fix"): v for k, v in res.items()}, X=X)
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axs[0]
    d = K.loaddata("ch04_vdp1")
    ax.pcolormesh(d["xs"], d["ys"], d["th"], cmap=K.CYCLIC, shading="auto", alpha=0.35, rasterized=True)
    ax.plot(X[:, 0], X[:, 1], color="k", lw=1.8)
    for (k, pts), col in zip(clouds.items(), [K.ACC, K.ACC2]):
        ax.plot(pts[:, 0], pts[:, 1], ".", ms=2.5, color=col, label=k)
    ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3.2, 3.2); ax.legend(loc="lower left", fontsize=9)
    ax.set_title("two clouds of 400 states (s.d. 0.08)"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax = axs[1]
    for (k, th), col in zip(res.items(), [K.ACC, K.ACC2]):
        ax.hist(th[np.isfinite(th)], bins=40, range=(0, 1), color=col, alpha=0.75, label=k)
    ax.set_xlabel("asymptotic phase $\\vartheta$"); ax.set_ylabel("number of states"); ax.legend(fontsize=9)
    ax.set_title("their asymptotic phases")
    fig.tight_layout()
    K.save(fig, "ch07_noise", wide=True)
    for k, th in res.items():
        z = np.nanmean(np.exp(2j * np.pi * th))
        print(f"{k}: phase coherence |<e^(2 pi i theta)>| = {abs(z):.3f}")


if __name__ == "__main__":
    two_cycles()
    vdp_zoom()
    noise_randomises()
