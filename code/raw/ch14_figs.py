"""Chapter 14 figures: what global isochrons are for, and what came next.
Run:  .venv/bin/python scripts/isochron/figs/ch14_figs.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import iso_lib as L
import figkit as K

SC = (150.0, 0.5)


def response_surface():
    """Phase shift of the neuron after an instantaneous voltage kick dV given at phase theta:
    read off the global isochrons (asymptotic phase of gamma_theta + (dV, 0))."""
    c = K.loaddata("rhh_cycle"); t, X, T = c["t"], c["X"], float(c["T"])
    phs = np.arange(120) / 120
    kicks = np.linspace(-40, 40, 61)
    pts = []
    for p in phs:
        k = int(round(p * (len(X) - 1)))
        for dv in kicks:
            pts.append(X[k] + np.array([dv, 0.0]))
    th, _ = L.phase_points("rhh", {}, X, T, np.array(pts), scale=SC, tol=1e-6, maxper=400, procs=7)
    th = th.reshape(len(phs), len(kicks))
    shift = ((th - phs[:, None]) + 0.5) % 1 - 0.5
    K.savedata("ch14_prs", phs=phs, kicks=kicks, shift=shift)
    d = K.loaddata("ch06_rhh_prc")
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    pm = ax.pcolormesh(phs, kicks, shift.T, cmap="RdBu_r", vmin=-0.5, vmax=0.5, shading="auto", rasterized=True)
    cb = fig.colorbar(pm, ax=ax, fraction=0.05); cb.set_label("phase shift $\\Delta\\vartheta$")
    ax.set_xlabel("phase of the kick $\\vartheta$"); ax.set_ylabel("voltage kick $\\Delta V$ (mV)")
    ax.set_title("finite kicks: read off the global isochrons")
    ax = axs[1]
    for dv, col in [(1.0, K.INK), (5.0, K.ACC), (15.0, K.BLUE), (30.0, K.ACC2)]:
        j = int(np.argmin(np.abs(kicks - dv)))
        ax.plot(phs, shift[:, j], color=col, lw=1.6, label=f"$\\Delta V = {kicks[j]:.0f}$ mV")
    ax.plot(d["tz"], d["zv"] * kicks[np.argmin(np.abs(kicks - 1.0))], "--", color=K.ORANGE, lw=1.4, label="iPRC $\\times$ 1 mV")
    ax.axhline(0, color=K.GREY, lw=0.8)
    ax.set_xlabel("phase of the kick $\\vartheta$"); ax.set_ylabel("$\\Delta\\vartheta$"); ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("small kicks follow the iPRC; big ones do not")
    fig.tight_layout()
    K.save(fig, "ch14_prs", wide=True)


def fourier_average():
    """Mauroy-Mezic: the isochrons are level sets of the argument of the Fourier average
    f*(x) = lim (1/t) int_0^t f(Phi_s(x)) exp(-i omega s) ds  (here f = x), a Koopman eigenfunction."""
    d = K.loaddata("ch04_vdp1"); X, T = d["X"], float(d["T"])
    m = L.load("vdp", eps=1.0)
    om = 2 * np.pi / T
    n = 71
    xs = np.linspace(-3, 3, n); ys = np.linspace(-3.4, 3.4, n)
    N = 25                                               # average over 25 periods
    dt = T / 200
    Fz = np.zeros((n, n), complex)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if abs(x) < 1e-9 and abs(y) < 1e-9:
                Fz[j, i] = np.nan; continue
            tt, P = L.traj(m, (x, y), N * T, dt=dt)
            Fz[j, i] = np.trapezoid(P[:, 0] * np.exp(-1j * om * tt), tt) / (N * T)
    # on the cycle x(t) = sum_k c_k exp(i k omega (t + theta T)), so the average is c_1 exp(2 pi i theta):
    # theta = (arg f* - arg c_1) / 2 pi, with arg c_1 = arg f*(gamma_0) because theta(gamma_0) = 0
    tt, P = L.traj(m, X[0], N * T, dt=dt)
    z0 = np.trapezoid(P[:, 0] * np.exp(-1j * om * tt), tt) / (N * T)
    th_f = ((np.angle(Fz) - np.angle(z0)) / (2 * np.pi)) % 1
    K.savedata("ch14_fourier", xs=xs, ys=ys, th=th_f)
    # compare with the direct phase map on the same grid points (interpolated)
    from scipy.interpolate import RegularGridInterpolator
    gi = RegularGridInterpolator((d["ys"], d["xs"]), np.exp(2j * np.pi * d["th"]))
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    ref = (np.angle(gi(np.stack([YY.ravel(), XX.ravel()], 1)).reshape(n, n)) / (2 * np.pi)) % 1
    R = np.hypot(XX, YY)
    err = np.abs(((th_f - ref) + 0.5) % 1 - 0.5)[R > 0.3]
    print(f"Fourier-average phase vs direct phase map: median |diff| = {np.nanmedian(err):.2e}, 95th percentile {np.nanpercentile(err, 95):.2e}")
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 5.0))
    for ax, th, title in [(axs[0], th_f, "Fourier averages (Koopman eigenfunction)"), (axs[1], None, "direct method (chapter 4)")]:
        if th is None:
            xs2, ys2, th2 = d["xs"], d["ys"], d["th"]
        else:
            xs2, ys2, th2 = xs, ys, th
        ax.pcolormesh(xs2, ys2, th2, cmap=K.CYCLIC, shading="auto", alpha=0.4, rasterized=True)
        curves = L.isochron_curves(xs2, ys2, np.nan_to_num(th2, nan=0.0), np.arange(16) / 16)
        for c, segs in curves.items():
            for seg in segs:
                ax.plot(seg[:, 0], seg[:, 1], color=K.PHASE_CMAP(c), lw=1.1)
        ax.plot(X[:, 0], X[:, 1], color="k", lw=1.8)
        ax.set_xlim(-3, 3); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_title(title)
    axs[0].set_ylabel("$y$")
    fig.tight_layout()
    K.save(fig, "ch14_fourier", wide=True)


def fhn_isochrons():
    """FitzHugh-Nagumo: the model of 'Winfree's puzzle' (Langfield, Krauskopf & Osinga 2014)."""
    pars = {"i": 0.5, "a": 0.7, "b": 0.8, "eps": 0.08}
    m = L.load("fhn", **pars)
    t, X, T = L.periodic_orbit(m, (-1.0, -0.5), dt=0.002)
    xs = np.linspace(-2.5, 2.5, 301); ys = np.linspace(-0.8, 2.0, 301)
    th, nper = L.phase_map("fhn", pars, X, T, xs, ys, scale=(1, 1), tol=1e-6, maxper=300, procs=7)
    K.savedata("ch14_fhn", xs=xs, ys=ys, th=th, X=X, T=T)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    curves = L.isochron_curves(xs, ys, th, np.arange(40) / 40)
    for c, segs in curves.items():
        for seg in segs:
            ax.plot(seg[:, 0], seg[:, 1], color=K.PHASE_CMAP(c), lw=0.8)
    v = np.linspace(-2.5, 2.5, 300)
    ax.plot(v, v - v ** 3 / 3 + pars["i"], "--", color=K.INK, lw=1)
    ax.plot(X[:, 0], X[:, 1], color="k", lw=2)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-0.8, 2.0); ax.set_xlabel("$v$"); ax.set_ylabel("$w$")
    ax.set_title("FitzHugh–Nagumo ($\\varepsilon = 0.08$): 40 isochrons")
    K.save(fig, "ch14_fhn")
    print("fhn: T", T, "NaN", np.isnan(th).sum())


if __name__ == "__main__":
    which = sys.argv[1:] or ["prs", "fourier", "fhn"]
    if "prs" in which: response_surface()
    if "fourier" in which: fourier_average()
    if "fhn" in which: fhn_isochrons()
