"""Chapter 3 figures: stability of a cycle (Poincare map, time-T map, Floquet multipliers).
Run:  .venv/bin/python scripts/isochron/figs/ch03_figs.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import iso_lib as L
import figkit as K


def section_gif():
    """Successive crossings of the section {x = 0, y > 0} converge to the fixed point of the Poincare map."""
    eps = 1.0
    m = L.load("vdp", eps=eps)
    F = L.rhs_np("vdp", {"eps": eps})
    _, cyc, T = L.periodic_orbit(m, (2, 0))
    t, X = L.traj(m, (0.0, 0.25), 6 * T, dt=0.005)
    # crossings of x = 0 going left (x decreasing) with y > 0
    i = np.where((X[:-1, 0] < 0) & (X[1:, 0] >= 0) & (X[1:, 1] > 0))[0]   # x = 0 crossed rightwards
    cr = X[i] + (X[i + 1] - X[i]) * (X[i, 0] / (X[i, 0] - X[i + 1, 0]))[:, None]
    K.savedata("ch03_section", X=X, cyc=cyc, cr=cr, T=T)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    K.streams(ax, F, (-3, 3), (-3, 3), n=28, color="#dde1e5")
    ax.plot(cyc[:, 0], cyc[:, 1], color=K.CYCLE, lw=1.4, alpha=0.6)
    ax.plot([0, 0], [0, 3], color=K.ORANGE, lw=3, alpha=0.8)
    ax.text(0.08, 2.75, "section $\\Sigma$", color=K.ORANGE, fontsize=11)
    line, = ax.plot([], [], color=K.ACC, lw=1.4)
    pts = ax.scatter([], [], s=50, color=K.ACC2, zorder=5, edgecolor="white")
    lab = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", fontsize=11, bbox=K.BOX, zorder=10)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("Poincaré map: follow the orbit from $\\Sigma$ back to $\\Sigma$")
    frames = 150
    ks = np.linspace(1, len(X) - 1, frames).astype(int)

    def upd(f):
        k = ks[f]
        line.set_data(X[:k, 0], X[:k, 1])
        n = int(np.sum(i < k))
        pts.set_offsets(cr[:n] if n else np.empty((0, 2)))
        lab.set_text(f"crossings so far: {n}" + (f",  $y_{n} = {cr[n - 1, 1]:.4f}$" if n else ""))
    K.save_gif(fig, upd, frames, "ch03_section", fps=18)
    print("crossings y:", np.round(cr[:, 1], 6))


def return_map():
    """Return map of r' = a r (1 - r^2), theta' = 1 on the section theta = 0 (exact)."""
    a = 0.12
    T = 2 * np.pi
    P = lambda r: 1 / np.sqrt(1 + (1 / r ** 2 - 1) * np.exp(-2 * a * T))
    lam = np.exp(-2 * a * T)
    r = np.linspace(0.05, 1.8, 400)
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.4))
    ax = axs[0]
    ax.plot(r, P(r), color=K.INK, lw=2, label="$P(r)$")
    ax.plot(r, r, "--", color=K.GREY, lw=1.2, label="$r_{n+1} = r_n$")
    for r0, col in [(0.15, K.ACC2), (1.75, K.ACC)]:
        x = r0
        for _ in range(8):
            y = P(x)
            ax.plot([x, x], [x, y], color=col, lw=1); ax.plot([x, y], [y, y], color=col, lw=1)
            x = y
    ax.plot(1, 1, "o", color=K.INK, ms=7)
    ax.set_xlabel("$r_n$"); ax.set_ylabel("$r_{n+1}$"); ax.set_aspect("equal"); ax.set_xlim(0, 1.8); ax.set_ylim(0, 1.8)
    ax.set_title(f"cobweb: slope at $r=1$ is $e^{{-2aT}} = {lam:.3f}$"); ax.legend(fontsize=9, loc="upper left")
    ax = axs[1]
    rs = [1.6]
    for _ in range(10):
        rs.append(P(rs[-1]))
    rs = np.array(rs)
    ax.semilogy(np.abs(rs - 1), "o-", color=K.ACC, label="$|r_n - 1|$")
    ax.semilogy(np.abs(rs[0] - 1) * lam ** np.arange(len(rs)) * (abs(rs[1] - 1) / abs(rs[0] - 1)) / lam, "--", color=K.ACC2, label="$\\propto \\lambda^n$")
    ax.set_xlabel("return number $n$"); ax.set_title("distance to the cycle shrinks by $\\lambda$ per turn")
    ax.legend(fontsize=9)
    fig.tight_layout()
    K.save(fig, "ch03_returnmap", wide=True)


def monodromy_gif():
    """A small circle of perturbations around gamma_0, carried for one period: it collapses onto the
    tangent (eigenvalue 1) while the transverse direction shrinks by the Floquet multiplier."""
    eps = 1.0
    m = L.load("vdp", eps=eps)
    F = L.rhs_np("vdp", {"eps": eps})
    t, cyc, T = L.periodic_orbit(m, (2, 0), dt=0.001)
    g0 = cyc[0]
    M = L.monodromy(F, g0, T, steps=20000)
    w, V = np.linalg.eig(M)
    k = np.argmin(np.abs(w)); lam = float(np.real(w[k])); vs = np.real(V[:, k]); vt = F(g0) / np.linalg.norm(F(g0))
    # divergence integral (Liouville): ln lambda = int_0^T (1 - x^2)/eps dt
    div = (1 - cyc[:, 0] ** 2) / eps
    lnlam = np.trapezoid(div, t)
    kap, nu, _ = L.floquet(m) if False else (None, None, None)
    print(f"vdP eps=1: T={T:.6f}, multiplier (monodromy) {lam:.6e}, exp(int div) {np.exp(lnlam):.6e}")
    rad = 0.35
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    starts = g0 + rad * np.column_stack([np.cos(ang), np.sin(ang)])
    paths = np.array([L.traj(m, s, T, dt=T / 400)[1] for s in starts])
    K.savedata("ch03_monodromy", paths=paths, cyc=cyc, g0=g0, M=M, lam=lam, vs=vs, vt=vt, T=T)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.plot(cyc[:, 0], cyc[:, 1], color=K.CYCLE, lw=1.3, alpha=0.5)
    ring, = ax.fill([], [], color=K.ACC2, alpha=0.35, lw=0)
    ring_e, = ax.plot([], [], color=K.ACC2, lw=1.2)
    ax.plot(*g0, "*", color=K.ORANGE, ms=14, zorder=6)
    lab = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", fontsize=11, bbox=K.BOX, zorder=10)
    ax.set_xlim(-3, 3); ax.set_ylim(-3.2, 3.2); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("a circle of perturbations of $\\gamma_0$, carried by the flow")
    frames = paths.shape[1]

    def upd(f):
        P = paths[:, f, :]
        ring.set_xy(P); ring_e.set_data(np.r_[P[:, 0], P[0, 0]], np.r_[P[:, 1], P[0, 1]])
        lab.set_text(f"$t/T = {f / (frames - 1):.2f}$")

    def still():
        P = paths[:, 0, :]
        ax.plot(np.r_[P[:, 0], P[0, 0]], np.r_[P[:, 1], P[0, 1]], color=K.ACC2, lw=1, ls="--")
        Pf = paths[:, -1, :]
        ax.plot(Pf[:, 0], Pf[:, 1], color=K.ACC2, lw=5, solid_capstyle="round", label="after one period: a segment along $\\Gamma$")
        for vec, col, name in [(vt, K.ACC, "tangent $\\vec F(\\gamma_0)$: eigenvalue 1"), (vs, K.BLUE, f"stable direction: eigenvalue $\\lambda={lam:.1e}$")]:
            ax.plot(*np.array([g0 - 0.9 * vec, g0 + 0.9 * vec]).T, color=col, lw=2.2, label=name)
        ax.plot([], [], color=K.ACC2, lw=1, ls="--", label="start: circle of radius 0.35")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
        lab.set_text("")
    K.save_gif(fig, upd, frames, "ch03_monodromy", fps=24, still_draw=still)
    return lam, lnlam


def liouville_figure():
    """ln(multiplier) = integral of the divergence over one period, for vdP at several eps and for the
    reduced Hodgkin-Huxley model of the paper."""
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 3.9))
    for eps, col in [(1.0, K.ACC), (0.3, K.BLUE), (0.1, K.ACC2)]:
        m = L.load("vdp", eps=eps)
        t, X, T = L.periodic_orbit(m, (2, 0), dt=min(0.001, eps / 200))
        div = (1 - X[:, 0] ** 2) / eps
        ln = np.trapezoid(div, t)
        rows.append((f"van der Pol, eps = {eps}", T, ln))
        axs[0].plot(t / T, div, color=col, lw=1.6, label=f"$\\varepsilon={eps}$: $\\ln\\lambda = {ln:.2f}$")
        m.close()
    axs[0].axhline(0, color=K.GREY, lw=0.8)
    axs[0].set_xlabel("phase $\\vartheta$"); axs[0].set_ylabel("$\\nabla\\cdot\\vec F$ along $\\Gamma$")
    axs[0].set_title("van der Pol: divergence along the cycle"); axs[0].legend(fontsize=9)
    axs[0].set_ylim(-35, 12)
    m = L.load("rhh")
    t, X, T = L.periodic_orbit(m, (-60, 0.45), dt=0.0002)
    F = L.rhs_np("rhh", {})
    div = np.array([np.trace(L.jac_fd(F, x)) for x in X])
    ln = np.trapezoid(div, t)
    rows.append(("reduced Hodgkin-Huxley (paper)", T, ln))
    axs[1].plot(t / T, div, color=K.INK, lw=1.4)
    axs[1].axhline(0, color=K.GREY, lw=0.8)
    axs[1].set_xlabel("phase $\\vartheta$"); axs[1].set_ylabel("$\\nabla\\cdot\\vec F$")
    axs[1].set_title(f"reduced Hodgkin–Huxley: $\\ln\\lambda = {ln:.2f}$")
    fig.tight_layout()
    K.save(fig, "ch03_liouville", wide=True)
    K.savedata("ch03_liouville", rows=np.array(rows, dtype=object))
    for r in rows:
        print(f"{r[0]:35s} T = {r[1]:9.4f}   ln(lambda) = {r[2]:10.3f}   lambda = {np.exp(r[2]):.3e}")


if __name__ == "__main__":
    section_gif()
    return_map()
    monodromy_gif()
    liouville_figure()
