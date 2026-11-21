"""Chapter 'Four Oscillators, One Method': the same isochron workflow on van der Pol, FitzHugh-Nagumo,
Morris-Lecar and the reduced Hodgkin-Huxley neuron, with a comparison table.
Run:  .venv/bin/python scripts/isochron/figs/ch08b_figs.py"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import iso_lib as L
import figkit as K

CASES = [
    # key, model, pars, start, window (x, y), scale, labels, title
    ("vdp", "vdp", {"eps": 1.0}, (2.0, 0.0), ((-3, 3), (-3.4, 3.4)), (1, 1), ("x", "y"), "van der Pol ($\\varepsilon = 1$)"),
    ("fhn", "fhn", {"i": 0.5, "a": 0.7, "b": 0.8, "eps": 0.08}, (-1.0, -0.5), ((-2.5, 2.5), (-0.8, 2.0)), (1, 1), ("v", "w"), "FitzHugh–Nagumo"),
    ("ml", "ml", {}, (-0.3, 0.0), ((-0.6, 0.5), (-0.1, 0.6)), (1, 1), ("V", "w"), "Morris–Lecar (SNIC set)"),
    ("rhh", "rhh", {}, (-60, 0.45), ((-90, 60), (0.3, 0.8)), (150, 0.5), ("V", "n"), "reduced Hodgkin–Huxley"),
]


def analyse(case, grid=241):
    key, name, pars, x0, (xw, yw), sc, labs, title = case
    m = L.load(name, **pars)
    t, X, T = L.periodic_orbit(m, x0, dt=min(0.002, T_guess(name) / 5000))
    kap, nu, adj = L.floquet(m)
    tz = np.asarray(adj.t, float) / T
    Z = np.column_stack([adj[adj.names[1]], adj[adj.names[2]]]).astype(float) / T
    F = L.rhs_np(name, pars)
    div = np.array([np.trace(L.jac_fd(F, x)) for x in X])
    lnlam = float(np.trapezoid(div, t))
    # equilibrium inside the cycle: Newton from the cycle's centroid
    c = X.mean(axis=0)
    names = [v.lower() for v in m.vars[:2]]
    e = m.equilibrium({names[0]: c[0], names[1]: c[1]})
    exy = np.array([e.x[m.vars[0]], e.x[m.vars[1]]])
    # phase map (reuse the big ones where they exist)
    cache = {"vdp": "ch04_vdp1", "fhn": "ch14_fhn", "rhh": "rhh_phase_full"}
    if key in cache:
        d = K.loaddata(cache[key]); xs, ys, th = d["xs"], d["ys"], d["th"]
    else:
        xs = np.linspace(*xw, grid); ys = np.linspace(*yw, grid)
        th, _ = L.phase_map(name, pars, X, T, xs, ys, scale=sc, tol=1e-6, maxper=300, procs=7)
    # where do 100 equally spaced isochrons start? fraction of base points in the slowest 50% of arclength
    seg = np.hypot(*(np.diff(X, axis=0) / np.array(sc)).T)
    s = np.r_[0, np.cumsum(seg)]
    # time spent in the slowest half of the cycle (by arclength): fraction of the period
    speed = seg / np.diff(t)
    order = np.argsort(speed)
    half = order[np.cumsum(seg[order]) <= 0.5 * s[-1]]
    slow_time = float(np.sum(np.diff(t)[half]) / T)
    zx = Z[:, 0]
    res = dict(key=key, title=title, T=float(T), lnlam=lnlam, lam=float(np.exp(lnlam)), eq=exy.tolist(),
               eig=[complex(v) for v in e.eigenvalues], prc_min=float(zx.min()), prc_max=float(zx.max()),
               prc_neg_frac=float(np.mean(zx < -0.02 * np.abs(zx).max())), slow_time=slow_time)
    K.savedata(f"ch08b_{key}", t=t, X=X, T=T, tz=tz, Z=Z, xs=xs, ys=ys, th=th, eq=exy)
    m.close()
    return res


def T_guess(name):
    return {"vdp": 6.7, "fhn": 39.5, "ml": 16.5, "rhh": 11.8}[name]


def nullclines(ax, name, pars, xw, yw):
    F = L.rhs_np(name, pars)
    vg = np.linspace(xw[0] - 1e-4, xw[1] + 1.7e-4, 700); ng = np.linspace(yw[0], yw[1], 600)
    VV, NN = np.meshgrid(vg, ng)
    FF = F(np.array([VV, NN]))
    ax.contour(vg, ng, FF[0], levels=[0], colors=[K.NULL_X], linewidths=1.3, linestyles="dashed")
    ax.contour(vg, ng, FF[1], levels=[0], colors=[K.NULL_Y], linewidths=1.3, linestyles="dashed")


def four_panel(results):
    fig, axs = plt.subplots(2, 2, figsize=(12, 9.4))
    for ax, case in zip(axs.flat, CASES):
        key, name, pars, x0, (xw, yw), sc, labs, title = case
        d = K.loaddata(f"ch08b_{key}")
        curves = L.isochron_curves(d["xs"], d["ys"], d["th"], np.arange(30) / 30)
        for c, segs in curves.items():
            for sgm in segs:
                ax.plot(sgm[:, 0], sgm[:, 1], color=K.PHASE_CMAP(c), lw=0.8)
        nullclines(ax, name, pars, xw, yw)
        ax.plot(d["X"][:, 0], d["X"][:, 1], color="k", lw=2)
        ax.plot(*d["eq"], "o", mfc="white", mec="k", ms=6, zorder=6)
        ax.set_xlim(*xw); ax.set_ylim(*yw); ax.set_xlabel(f"${labs[0]}$"); ax.set_ylabel(f"${labs[1]}$"); ax.set_title(title)
    fig.tight_layout()
    K.save(fig, "ch08b_four", wide=True)


def timeseries_and_prc():
    fig, axs = plt.subplots(2, 4, figsize=(13.4, 5.6), sharex=True)
    for j, case in enumerate(CASES):
        key, name, pars, x0, win, sc, labs, title = case
        d = K.loaddata(f"ch08b_{key}")
        X, t, T = d["X"], d["t"], float(d["T"])
        axs[0, j].plot(t / T, X[:, 0], color=K.INK, lw=1.5)
        axs[0, j].set_title(title, fontsize=11); axs[0, j].set_ylabel(f"${labs[0]}$")
        zx = d["Z"][:, 0]
        axs[1, j].plot(d["tz"], zx / np.abs(zx).max(), color=K.ACC2, lw=1.6)
        axs[1, j].axhline(0, color=K.GREY, lw=0.8)
        axs[1, j].set_xlabel("phase $\\vartheta$"); axs[1, j].set_ylabel(f"$Z_{{{labs[0]}}}$ (normalised)")
        axs[1, j].set_ylim(-1.1, 1.1)
    fig.tight_layout()
    K.save(fig, "ch08b_prc", wide=True)


if __name__ == "__main__":
    results = [analyse(c) for c in CASES]
    out = K.DATA / "ch08b_table.json"
    out.write_text(json.dumps(results, indent=1, default=str))
    for r in results:
        print(f"{r['key']:4s} T={r['T']:9.4f} lnlam={r['lnlam']:9.3f} lam={r['lam']:.2e} eq={np.round(r['eq'], 4)} "
              f"eig={np.round(r['eig'], 4)} prc[{r['prc_min']:.4g},{r['prc_max']:.4g}] negfrac={r['prc_neg_frac']:.2f} slowtime={r['slow_time']:.2f}")
    four_panel(results)
    timeseries_and_prc()
