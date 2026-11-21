"""Chapter 5 figures: stable manifolds, the time-T map and the invariant foliation.
Run:  .venv/bin/python scripts/isochron/figs/ch05_figs.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import pyxpp
import iso_lib as L
import figkit as K


def saddle_manifolds():
    """x' = x,  y' = -y + x^2: W^s is the y-axis, W^u is y = x^2/3 (exact)."""
    m = pyxpp.load(text="""# a saddle with a curved unstable manifold
x'=x
y'=-y+x^2
init x=0.01,y=0
@ total=5,dt=0.005,meth=cvode,tol=1e-10,atol=1e-10,bound=1000
done
""")
    F = lambda U: np.array([U[0], -U[1] + U[0] ** 2])
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.5))
    for ax, zoom in [(axs[0], 2.0), (axs[1], 0.25)]:
        K.streams(ax, F, (-zoom, zoom), (-zoom, zoom * 1.4 if zoom > 1 else zoom), n=30, color="#d4d9de")
        # unstable manifold: integrate forward from points just off the saddle along the unstable eigenvector (1, 0)
        for sgn in (1, -1):
            t, X = L.traj(m, (sgn * 1e-4, 0.0), 12, dt=0.005)
            ax.plot(X[:, 0], X[:, 1], color=K.ACC2, lw=2.4, label="$W^u$ (computed)" if sgn > 0 else None)
            # stable manifold: backward from points along (0, 1)
            t, X = L.traj(m, (0.0, sgn * 1e-4), 12, dt=0.005, back=True)
            ax.plot(X[:, 0], X[:, 1], color=K.ACC, lw=2.4, label="$W^s$ (computed)" if sgn > 0 else None)
        xs = np.linspace(-zoom, zoom, 200)
        ax.plot(xs, 0 * xs, "--", color=K.ACC2, lw=1.2, label="$E^u$: unstable eigenvector")
        ax.plot(0 * xs, xs, "--", color=K.ACC, lw=1.2, label="$E^s$: stable eigenvector")
        ax.plot(0, 0, "o", color=K.INK, ms=6)
        ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom * 1.4 if zoom > 1 else zoom); ax.set_aspect("equal")
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    axs[0].set_title("$\\dot x = x,\\ \\dot y = -y + x^2$: manifolds bend")
    axs[1].set_title("zoom: $W^u$ is tangent to $E^u$ at the saddle")
    axs[0].legend(fontsize=8.5, loc="upper center")
    fig.tight_layout()
    K.save(fig, "ch05_saddle", wide=True)


def timeT_iterates():
    """Points on the isochron I(gamma_0) jump along it towards gamma_0 under the time-T map.
    A weakly attracting twisted oscillator is used so that the jumps are visible:
    r' = a r (1 - r^2), theta' = w - b r^2 with a = 0.1: multiplier exp(-2 a T) = 0.17,
    isochrons theta - (b/a) ln r = const (exact)."""
    a, b, w = 0.1, 0.25, 1.0
    m = pyxpp.load(text=f"""# weakly attracting oscillator with twist
x'=a*x*(1-x^2-y^2)-y*(w-b*(x^2+y^2))
y'=a*y*(1-x^2-y^2)+x*(w-b*(x^2+y^2))
par a={a},b={b},w={w}
init x=1,y=0
@ total=10,dt=0.001,meth=cvode,tol=1e-11,atol=1e-11,bound=1000,maxstor=200000
done
""")
    T = 2 * np.pi / (w - b)
    lam = np.exp(-2 * a * T)
    g0 = np.array([1.0, 0.0])
    c = b / a
    rr = np.geomspace(0.02, 3.2, 3000)
    iso = np.column_stack([rr * np.cos(c * np.log(rr)), rr * np.sin(c * np.log(rr))])
    starts = [np.array([2.6 * np.cos(c * np.log(2.6)), 2.6 * np.sin(c * np.log(2.6))]),
              np.array([0.18 * np.cos(c * np.log(0.18)), 0.18 * np.sin(c * np.log(0.18))])]
    its = []
    for s0 in starts:
        pts = [s0]
        for k in range(6):
            t, P = L.traj(m, pts[-1], T, dt=T / 2000)
            pts.append(P[-1])
        its.append(np.array(pts))
    K.savedata("ch05_iterates", iso=iso, its=np.array(its), T=T, lam=lam, a=a, b=b, w=w)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(iso[:, 0], iso[:, 1], color=K.PHASE_CMAP(0.0), lw=2.2, label="$I(\\gamma_0)$: $\\theta = (b/a)\\ln r$")
    ax.plot(np.cos(th), np.sin(th), color="k", lw=1.6, label="$\\Gamma$")
    for P, col in zip(its, [K.ACC2, K.BLUE]):
        ax.plot(P[:, 0], P[:, 1], "o", color=col, ms=7, zorder=6, mec="white")
        for k in range(len(P) - 1):
            ax.annotate("", xy=P[k + 1], xytext=P[k], arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0,
                        connectionstyle="arc3,rad=0.3", shrinkA=4, shrinkB=4))
    ax.plot(*g0, "*", color=K.ORANGE, ms=16, zorder=7)
    ax.annotate("$\\gamma_0$", g0, xytext=(8, 4), textcoords="offset points", fontsize=12)
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"$\\Phi_T$ slides points along $I(\\gamma_0)$ to $\\gamma_0$ ($\\lambda = {lam:.2f}$)")
    K.save(fig, "ch05_iterates")
    for P in its:
        print("distances to gamma0 under Phi_T:", np.round(np.linalg.norm(P - g0, axis=1), 5),
              " max |phase error| along the way:", np.abs(np.angle((P[:, 0] + 1j * P[:, 1]) * np.exp(-1j * c * np.log(np.hypot(*P.T))))).max())


def bundle():
    """Stable Floquet directions along Gamma (the linear stable normal bundle) are the tangents of the isochrons."""
    eps = 1.0
    d = K.loaddata("ch04_vdp1")
    xs, ys, th, X, T = d["xs"], d["ys"], d["th"], d["X"], float(d["T"])
    F = L.rhs_np("vdp", {"eps": eps})
    M = L.monodromy(F, X[0], T, steps=20000)
    w, V = np.linalg.eig(M)
    v = np.real(V[:, np.argmin(np.abs(w))])
    # transport v along the orbit with the variational equation (RK4), renormalising
    N = 20000; h = T / N; u = X[0].copy(); vs = [v.copy()]; us = [u.copy()]
    for k in range(N):
        def G(z):
            return np.concatenate([F(z[:2]), L.jac_fd(F, z[:2]) @ z[2:]])
        z = np.concatenate([u, v])
        k1 = G(z); k2 = G(z + h / 2 * k1); k3 = G(z + h / 2 * k2); k4 = G(z + h * k3)
        z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        u, v = z[:2], z[2:] / np.linalg.norm(z[2:])
        us.append(u.copy()); vs.append(v.copy())
    us = np.array(us); vs = np.array(vs)
    K.savedata("ch05_bundle", us=us[::100], vs=vs[::100], T=T)
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 5.2))
    for ax, lim in [(axs[0], (-3, 3, -3.4, 3.4)), (axs[1], (1.0, 2.6, -0.6, 1.6))]:
        nlev = 20
        curves = L.isochron_curves(xs, ys, th, np.arange(nlev) / nlev)
        for c, segs in curves.items():
            for seg in segs:
                ax.plot(seg[:, 0], seg[:, 1], color=K.PHASE_CMAP(c), lw=1.1, alpha=0.8)
        ax.plot(X[:, 0], X[:, 1], color="k", lw=1.8)
        L_ = 0.25 if ax is axs[0] else 0.12
        for j in range(nlev):
            k = int(j / nlev * N)
            p, q = us[k], vs[k]
            ax.plot([p[0] - L_ * q[0], p[0] + L_ * q[0]], [p[1] - L_ * q[1], p[1] + L_ * q[1]], color=K.ORANGE, lw=3.2, solid_capstyle="round", zorder=5)
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    axs[0].set_title("orange: stable Floquet direction at 20 points of $\\Gamma$")
    axs[1].set_title("zoom: each one is tangent to its isochron")
    fig.tight_layout()
    K.save(fig, "ch05_bundle", wide=True)


if __name__ == "__main__":
    saddle_manifolds()
    timeT_iterates()
    bundle()
