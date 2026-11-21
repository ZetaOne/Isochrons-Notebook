"""Chapter 12 figures: the reduced Hodgkin-Huxley model (paper eq. 4.1) and where it comes from.
Run:  .venv/bin/python scripts/isochron/figs/ch12_figs.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import pyxpp
import iso_lib as L
import figkit as K

HH4 = """# full Hodgkin-Huxley model (1952), same rate functions and parameters as the paper
par iapp=10,gna=120,gk=36,gl=0.3,vna=50,vk=-77,vl=-54.4,c=1
an(v)=0.01*(v+55)/(1-exp(-(v+55)/10))
bn(v)=0.125*exp(-(v+65)/80)
am(v)=0.1*(v+40)/(1-exp(-(v+40)/10))
bm(v)=4*exp(-(v+65)/18)
ah(v)=0.07*exp(-(v+65)/20)
bh(v)=1/(1+exp(-(v+35)/10))
v'=(iapp-gna*m^3*h*(v-vna)-gk*n^4*(v-vk)-gl*(v-vl))/c
m'=am(v)*(1-m)-bm(v)*m
h'=ah(v)*(1-h)-bh(v)*h
n'=an(v)*(1-n)-bn(v)*n
aux minf=am(v)/(am(v)+bm(v))
init v=-65,m=0.05,h=0.6,n=0.32
@ meth=cvode,tol=1e-10,atol=1e-10,dt=0.01,total=200,maxstor=200000,bound=10000
done
"""


def reduction():
    full = pyxpp.load(text=HH4)
    tr = full.run(total=200, dt=0.01)
    t = np.asarray(tr.t); V = tr["v"]; m = tr["m"]; h = tr["h"]; n = tr["n"]; mi = tr["minf"]
    k = t > 100
    red = L.load("rhh")
    t2, X2 = L.traj(red, (-65, 0.32), 200, dt=0.01)
    k2 = t2 > 100
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 3.9))
    ax = axs[0]
    ax.plot(t[k] - 100, V[k], color=K.INK, lw=1.5, label="full model (4 variables)")
    ax.plot(t2[k2] - 100, X2[k2, 0], "--", color=K.ACC2, lw=1.5, label="reduced model (2 variables)")
    ax.set_xlabel("$t$ (ms)"); ax.set_ylabel("$V$ (mV)"); ax.legend(fontsize=9, loc="lower right")
    ax.set_title("spike trains")
    ax = axs[1]
    ax.plot(V[k], m[k], color=K.BLUE, lw=1.4, label="$m(t)$, full model")
    vv = np.linspace(-80, 50, 300)
    am = 0.1 * (vv + 40) / (1 - np.exp(-(vv + 40) / 10)); bm = 4 * np.exp(-(vv + 65) / 18)
    ax.plot(vv, am / (am + bm), "--", color=K.ORANGE, lw=1.8, label="$m_\\infty(V)$")
    ax.set_xlabel("$V$"); ax.set_ylabel("$m$"); ax.legend(fontsize=9, loc="upper left")
    ax.set_title("$m$ is fast: it follows $m_\\infty(V)$")
    ax = axs[2]
    ax.plot(t[k] - 100, h[k] + n[k], color=K.ACC, lw=1.5, label="$h + n$, full model")
    ax.axhline(0.8, color=K.ORANGE, ls="--", lw=1.6, label="$0.8$ (reduction)")
    ax.set_xlabel("$t$ (ms)"); ax.set_ylim(0.5, 1.0); ax.legend(fontsize=9, loc="lower right")
    ax.set_title("$h + n$ stays roughly constant")
    fig.tight_layout()
    K.save(fig, "ch12_reduction", wide=True)
    print("full HH: h+n range over a cycle", (h[k] + n[k]).min(), (h[k] + n[k]).max())


def phase_plane():
    cyc = L.load("rhh")
    t, X, T = L.periodic_orbit(cyc, (-60, 0.45), dt=0.0005)
    e = cyc.equilibrium({"v": -59.6, "n": 0.40})
    print("equilibrium", e.x, "eigenvalues", e.eigenvalues)
    nc = cyc.nullclines(x="v", y="n", xlo=-100, xhi=60, ylo=0.0, yhi=1.0, mesh=400)
    F = L.rhs_np("rhh", {})
    K.savedata("ch12_pp", X=X, t=t, T=T, ncV=nc["V"], ncN=nc["N"], eq=np.array([e.x["V"], e.x["N"]]), eig=e.eigenvalues)
    fig = plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25])
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(np.r_[t, t + T, t + 2 * T] / T, np.r_[X[:, 0], X[:, 0], X[:, 0]], color=K.INK, lw=1.5)
    ax.set_ylabel("$V$ (mV)"); ax.set_xticklabels([]); ax.set_title(f"$T_\\Gamma = {T:.4f}$ ms")
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(np.r_[t, t + T, t + 2 * T] / T, np.r_[X[:, 1], X[:, 1], X[:, 1]], color=K.ACC, lw=1.5)
    ax.set_ylabel("$n$"); ax.set_xlabel("$t / T_\\Gamma$")
    ax = fig.add_subplot(gs[:, 1])
    K.streams(ax, F, (-90, 60), (0.3, 0.8), n=40, color="#dde1e5", scale=(1 / 150, 1 / 0.5))
    # nullclines as zero contours of the vector field on a fine grid (offset grid avoids the removable
    # singularities of the rate functions at V = -40 and V = -55)
    vg = np.linspace(-90.0013, 60.0017, 1200); ng = np.linspace(0.3, 0.8, 900)
    VV, NN = np.meshgrid(vg, ng)
    FV = F(np.array([VV, NN]))
    ax.contour(vg, ng, FV[0], levels=[0], colors=[K.NULL_X], linewidths=2)
    ax.contour(vg, ng, FV[1], levels=[0], colors=[K.NULL_Y], linewidths=2)
    ax.plot([], [], color=K.NULL_X, lw=2, label="$\\dot V = 0$")
    ax.plot([], [], color=K.NULL_Y, lw=2, label="$\\dot n = 0$")
    ax.plot(X[:, 0], X[:, 1], color="k", lw=2.4, label="$\\Gamma$")
    ax.plot(e.x["V"], e.x["N"], "o", mfc="white", mec="k", ms=8, zorder=6, label="$x^*$ (source)")
    ax.plot(*X[0], "*", color=K.ORANGE, ms=15, zorder=7, label="$\\gamma_0$ (max $V$)")
    ax.set_xlim(-90, 60); ax.set_ylim(0.3, 0.8); ax.set_xlabel("$V$ (mV)"); ax.set_ylabel("$n$")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0)); ax.set_title("phase plane (cf. paper Fig. 1)")
    fig.tight_layout()
    K.save(fig, "ch12_phaseplane", wide=True)


def auto_diagram():
    m = L.load("rhh", iapp=0.0)
    m.set(meth="cvode")
    m.run(total=300); m.run_last(total=50)
    eq = m.auto.run("iapp", start="steady", ds=0.05, dsmax=0.5, parmin=0, parmax=200, nmax=3000)
    hbs = eq.special("HB")
    print("Hopf points at iapp =", [round(h.x, 4) for h in hbs])
    deq = eq.to_pandas("v")
    branches = []
    for hb in hbs:
        try:
            po = m.auto.run(grab=hb.label, choice="periodic", ntst=120, dsmax=0.2, nmax=2000, parmin=0, parmax=200)
            branches.append(po.to_pandas("v"))
        except Exception as ex:
            print("periodic branch failed", ex)
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.3))
    for ax, xl in [(axs[0], (0, 200)), (axs[1], (0, 20))]:
        for st, ls in [(True, "-"), (False, "--")]:
            s = deq[deq.stable == st]
            ax.plot(s.par, s.ymax, ls, color=K.INK, lw=1.8, label=f"equilibrium, {'stable' if st else 'unstable'}")
        for b in branches:
            for st, col in [(True, K.ACC), (False, K.ACC2)]:
                s = b[b.stable == st]
                ax.plot(s.par, s.ymax, "o", ms=2, color=col, label=f"periodic orbits, {'stable' if st else 'unstable'} (max/min $V$)")
                ax.plot(s.par, s.ymin, "o", ms=2, color=col)
        for h in hbs:
            ax.plot(h.x, h.u0["v"], "s", color=K.ORANGE, ms=8, zorder=6)
        ax.axvline(10, color=K.BLUE, lw=1, ls=":")
        ax.set_xlim(*xl); ax.set_xlabel("$I_{\\rm app}$"); ax.set_ylabel("$V$")
    axs[0].set_title("AUTO: equilibria and periodic orbits"); axs[1].set_title("zoom: the paper's $I_{\\rm app} = 10$ (dotted)")
    h_, l_ = axs[0].get_legend_handles_labels(); uniq = dict(zip(l_, h_))
    axs[0].legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper right")
    fig.tight_layout()
    K.save(fig, "ch12_auto", wide=True)
    K.savedata("ch12_auto", eq=deq[["par", "ymax", "stable"]].to_numpy(float), hb=np.array([h.x for h in hbs]))


def timescales():
    """How fast is V compared with n along the cycle? Ratio of normalised speeds."""
    d = K.loaddata("ch12_pp")
    X, t, T = d["X"], d["t"], float(d["T"])
    F = L.rhs_np("rhh", {})
    Fv = F(X.T)
    rv = np.abs(Fv[0]) / 150.0; rn = np.abs(Fv[1]) / 0.5        # speeds relative to each variable's range
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.semilogy(t / T, rv, color=K.ACC2, lw=1.5, label="$|\\dot V|$ / 150 mV")
    ax.semilogy(t / T, rn, color=K.ACC, lw=1.5, label="$|\\dot n|$ / 0.5")
    ax.set_xlabel("phase $\\vartheta$"); ax.set_ylabel("relative speed (1/ms)"); ax.legend(fontsize=9)
    ax.set_title("relative speeds of $V$ and $n$ along $\\Gamma$")
    K.save(fig, "ch12_timescales")
    print("max ratio", np.max(rv / np.maximum(rn, 1e-12)))


if __name__ == "__main__":
    reduction()
    phase_plane()
    auto_diagram()
    timescales()
