"""Chapter 10 figures: numerical continuation (natural vs pseudo-arclength), collocation meshes.
Run:  .venv/bin/python scripts/isochron/figs/ch10_figs.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import figkit as K

# a branch with two folds:  G(x, mu) = x^3 - 3x - mu = 0
G = lambda x, mu: x ** 3 - 3 * x - mu
Gx = lambda x, mu: 3 * x ** 2 - 3
Gm = lambda x, mu: -1.0


def natural(x0=-2.2, mu0=None, dmu=0.12, n=60):
    mu = G(x0, 0) if mu0 is None else mu0
    xs, mus, pred = [x0], [mu], []
    x = x0
    for k in range(n):
        mu_new = mus[-1] + dmu
        xg = xs[-1]                                     # predictor: previous solution
        pred.append((xg, mu_new))
        ok = False
        for it in range(30):                             # Newton in x at fixed mu
            dx = -G(xg, mu_new) / Gx(xg, mu_new)
            xg += dx
            if abs(dx) < 1e-12:
                ok = True; break
        if not ok or abs(xg - xs[-1]) > 0.8:
            return np.array(xs), np.array(mus), k, (xg, mu_new)
        xs.append(xg); mus.append(mu_new)
    return np.array(xs), np.array(mus), None, None


def arclength(x0=-2.2, h=0.25, n=60):
    mu0 = G(x0, 0) * 0 + (x0 ** 3 - 3 * x0)
    pts = [np.array([x0, mu0])]
    t = np.array([Gm(x0, mu0), -Gx(x0, mu0)]); t = t / np.linalg.norm(t)   # tangent of G = 0: (G_mu, -G_x)
    if t[1] < 0:
        t = -t
    preds = []
    for k in range(n):
        p = pts[-1]; q = p + h * t                                         # predictor along the tangent
        preds.append(q.copy())
        for it in range(30):                                               # corrector: G = 0 and t.(z - q) = 0
            J = np.array([[Gx(*q), Gm(*q)], [t[0], t[1]]])
            r = np.array([G(*q), np.dot(t, q - (p + h * t))])
            dz = np.linalg.solve(J, -r); q = q + dz
            if np.linalg.norm(dz) < 1e-12:
                break
        tn = np.array([Gm(*q), -Gx(*q)]); tn = tn / np.linalg.norm(tn)
        if np.dot(tn, t) < 0:
            tn = -tn
        pts.append(q); t = tn
        if q[0] > 2.3:
            break
    return np.array(pts), np.array(preds)


def continuation_figure():
    xx = np.linspace(-2.4, 2.4, 400)
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    for ax in axs:
        ax.plot(xx ** 3 - 3 * xx, xx, color=K.GREY, lw=5, alpha=0.45, label="solution curve $G(x,\\mu)=0$")
        ax.plot([2, -2], [-1, 1], "s", color=K.ORANGE, ms=8, label="folds")
        ax.set_xlabel("parameter $\\mu$"); ax.set_xlim(-6.5, 6.5)
    xs, mus, fail, last = natural()
    ax = axs[0]
    ax.plot(mus, xs, "o-", color=K.ACC2, ms=4, lw=1.2, label="natural continuation in $\\mu$")
    ax.annotate("stuck at the fold:\nno nearby solution\nfor larger $\\mu$", (mus[-1], xs[-1]), xytext=(-5.8, 1.2),
                arrowprops=dict(arrowstyle="-|>", color=K.INK, lw=1), fontsize=10)
    ax.set_ylabel("solution $x$"); ax.set_title("natural continuation fails at a fold"); ax.legend(fontsize=9, loc="lower right")
    pts, preds = arclength()
    ax = axs[1]
    for p0, q in zip(pts[:-1], preds):
        ax.plot([p0[1], q[1]], [p0[0], q[0]], color=K.BLUE, lw=0.8, alpha=0.7)
    ax.plot(pts[:, 1], pts[:, 0], "o-", color=K.ACC, ms=4, lw=1.2, label="pseudo-arclength continuation")
    ax.plot(preds[:, 1], preds[:, 0], ".", color=K.BLUE, ms=4, label="predictor steps")
    ax.set_title("pseudo-arclength goes round the folds"); ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    K.save(fig, "ch10_continuation", wide=True)
    K.savedata("ch10_cont", nat_x=xs, nat_mu=mus, arc=pts, preds=preds)


def predictor_corrector():
    """One pseudo-arclength step, magnified: tangent predictor, corrector on the perpendicular line."""
    xx = np.linspace(-2.4, 2.4, 400)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(xx ** 3 - 3 * xx, xx, color=K.GREY, lw=4, alpha=0.5)
    x0 = -1.62; p = np.array([x0, x0 ** 3 - 3 * x0])
    t = np.array([-1.0, -(3 * x0 ** 2 - 3)]); t /= np.linalg.norm(t)
    if t[1] < 0:
        t = -t
    h = 1.25
    q = p + h * t
    nvec = np.array([-t[1], t[0]])
    z = q.copy()
    for it in range(30):
        J = np.array([[3 * z[0] ** 2 - 3, -1.0], [t[0], t[1]]]); r = np.array([G(*z), np.dot(t, z - q)])
        z = z + np.linalg.solve(J, -r)
    P = lambda v: (v[1], v[0])
    ax.annotate("", xy=P(q), xytext=P(p), arrowprops=dict(arrowstyle="-|>", color=K.BLUE, lw=2))
    L2 = np.array([q - 1.2 * nvec, q + 1.2 * nvec])
    ax.plot(L2[:, 1], L2[:, 0], "--", color=K.ACC2, lw=1.4, label="corrector constraint $\\vec t\\cdot(\\vec z - \\vec z_{\\rm pred}) = 0$")
    ax.plot(*P(p), "o", color=K.INK, ms=8); ax.plot(*P(q), "o", color=K.BLUE, ms=7); ax.plot(*P(z), "o", color=K.ACC, ms=9)
    ax.annotate("$\\vec z_k$", P(p), xytext=(-24, -4), textcoords="offset points", fontsize=12)
    ax.annotate("predictor $\\vec z_k + h\\vec t$", P(q), xytext=(14, -22), textcoords="offset points", fontsize=11, color=K.BLUE)
    ax.annotate("corrected $\\vec z_{k+1}$ (Newton)", P(z), xytext=(-175, 10), textcoords="offset points", fontsize=11, color=K.ACC)
    ax.set_xlim(-0.6, 3.0); ax.set_ylim(-2.1, -0.1); ax.set_aspect("equal"); ax.set_xlabel("$\\mu$"); ax.set_ylabel("$x$")
    ax.legend(fontsize=9, loc="lower right"); ax.set_title("one predictor–corrector step")
    K.save(fig, "ch10_step")


def collocation_mesh():
    """The adaptive collocation mesh of the cycle+bundle BVP of chapter 9 (reduced Hodgkin-Huxley)."""
    d = K.loaddata("ch09_bundle")
    s, u = d["s"], d["u"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    axs[0].plot(s, u[0], color=K.INK, lw=1.4)
    axs[0].plot(s, u[0], "|", color=K.ACC2, ms=6, mew=0.6)
    axs[0].set_ylabel("$V$")
    axs[0].set_title(f"collocation mesh: {len(s)} nodes, refined by the solver where $V$ changes fastest (the spike)")
    h = np.diff(s)
    axs[1].semilogy(0.5 * (s[1:] + s[:-1]), h, color=K.ACC, lw=1.4)
    axs[1].set_ylabel("mesh interval"); axs[1].set_xlabel("rescaled time $s = t/T$")
    fig.tight_layout()
    K.save(fig, "ch10_mesh", wide=True)
    print("mesh: nodes", len(s), "smallest interval", h.min(), "largest", h.max())


if __name__ == "__main__":
    continuation_figure()
    predictor_corrector()
    collocation_mesh()
