"""Deck for Chapter 8: Slow-fast systems."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import manimgx as m
from deckkit import *  # noqa: F401,F403


def level_curves(xs, ys, th, c):
    """Isochron {theta = c} from a phase map (matplotlib contour of sin, keep cos > 0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = np.sin(2 * np.pi * (th - c)); co = np.cos(2 * np.pi * (th - c))
    s = np.where(co > 0, s, np.nan)
    fig, ax = plt.subplots()
    cs = ax.contour(xs, ys, s, levels=[0.0])
    segs = [p for p in cs.allsegs[0] if len(p) > 4]
    plt.close(fig)
    return segs


def clipped(ax, X, xr, yr, color=INK, width=2.0):
    """curve() split into the pieces that lie inside the window xr x yr."""
    X = np.asarray(X, float)
    ok = (X[:, 0] >= xr[0]) & (X[:, 0] <= xr[1]) & (X[:, 1] >= yr[0]) & (X[:, 1] <= yr[1]) & np.all(np.isfinite(X), axis=1)
    g = m.VGroup(); run = []
    for p, o in zip(X, ok):
        if o:
            run.append(p)
        else:
            if len(run) > 1:
                g.add(curve(ax, np.array(run), color=color, width=width))
            run = []
    if len(run) > 1:
        g.add(curve(ax, np.array(run), color=color, width=width))
    return g


def strike(mob, color=RED):
    return m.Line(mob.get_corner(m.DL), mob.get_corner(m.UR), color=color, stroke_width=5)


def phase_at(xs, ys, th, p):
    j = int(np.argmin(np.abs(xs - p[0]))); i = int(np.argmin(np.abs(ys - p[1])))
    return float(th[i, j])


def vdp_rk4(P, t_end, dt, eps, a=0.0):
    """Trajectories of eps x' = y - x^3/3 + x, y' = a - x (t_end < 0 integrates backwards)."""
    F = lambda U: np.column_stack([(U[:, 1] - U[:, 0] ** 3 / 3 + U[:, 0]) / eps, a - U[:, 0]])
    n = int(round(abs(t_end) / dt)); h = np.sign(t_end) * dt
    U = np.array(P, float); out = [U.copy()]
    for _ in range(n):
        k1 = F(U); k2 = F(U + h / 2 * k1); k3 = F(U + h / 2 * k2); k4 = F(U + h * k3)
        U = U + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        U = np.clip(U, -10, 10)
        out.append(U.copy())
    return np.array(out)


def cubic(ax, x0, x1, color, width=4, n=200):
    xx = np.linspace(x0, x1, n)
    return curve(ax, np.column_stack([xx, xx ** 3 / 3 - xx]), color=color, width=width)


class Deck08(Deck):
    chapter = "Chapter 8"
    title = "Slow-Fast Systems"
    subtitle = "Relaxation oscillations, slow manifolds and canards"

    def construct(self):
        self.title_card()
        self.two_scales()
        self.morph()
        self.singular_period()
        self.critical_manifold()
        self.anatomy()
        self.slow_manifolds()
        self.canards()
        self.isochrons()
        self.ridge()
        self.hard()
        self.recap(["$\\varepsilon\\dot x = f$, $\\dot y = g$: fast horizontal motion, slow creep along\nthe critical manifold $C_0 = \\{f = 0\\}$",
                    "$\\partial f/\\partial x < 0$ attracts, $\\partial f/\\partial x > 0$ repels; folds separate them,\nand relaxation oscillations jump at the folds",
                    "Slow manifolds $S^{a,r}_\\varepsilon$ lie $O(\\varepsilon)$ from $C_0$: attracting ones found\nforwards in time, repelling ones backwards",
                    "Canards follow the repelling slow manifold; the canard explosion\nhappens in an exponentially small range of the parameter",
                    "Isochrons crowd on the slow parts of $\\Gamma$ and pile up along the\nrepelling slow manifold"])

    # ------------------------------------------------------------------
    def two_scales(self):
        h = heading("Two time scales")
        e1 = tex(r"\varepsilon\,\dot x = f(x,y),\qquad \dot y = g(x,y),\qquad 0 < \varepsilon \ll 1", font_size=38).move_to(pt(0, 2.5))
        e2 = tex(r"\text{van der Pol: }\ \varepsilon\dot x = y - \frac{x^3}{3} + x,\qquad \dot y = -x", font_size=36, color=TEAL).next_to(e1, m.DOWN, buff=0.3)
        self.play(m.FadeIn(h), m.Write(e1))
        self.play(m.Write(e2))
        self.next_slide("form")
        ax = plane((-3, 3, 1), (-1.5, 1.5, 1), width=7.6, height=3.8, at=pt(-2.4, -1.65), ticks=False)
        c0 = cubic(ax, -2.35, 2.35, MUTED, width=6).set_stroke(opacity=0.6)
        self.play(m.Create(ax), m.Create(c0))
        rng = np.random.default_rng(1)
        P = np.column_stack([rng.uniform(-2.8, 2.8, 22), rng.uniform(-1.3, 1.3, 22)])
        traj = vdp_rk4(P, 2.5, 0.002, 0.05)
        k = m.ValueTracker(0)
        def cl():
            g = m.VGroup()
            i = int(k.get_value())
            for j in range(len(P)):
                p = traj[i, j]
                if abs(p[0]) < 3 and abs(p[1]) < 1.5:
                    g.add(m.Dot(ax.c2p(*p), radius=0.06, color=YELLOW))
            return g
        dots = m.always_redraw(cl)
        self.add(dots)
        n = prose("$\\varepsilon = 0.05$: states shoot\nhorizontally onto the grey\ncurve $f = 0$, then creep\nslowly along it", font_size=24).move_to(pt(4.6, -1.5))
        self.play(m.FadeIn(n))
        self.next_slide()
        self.play(k.animate.set_value(300), run_time=4, rate_func=m.linear)
        self.play(k.animate.set_value(len(traj) - 1), run_time=5, rate_func=m.linear)
        self.wait(BEAT)
        self.next_slide("track")
        self.clear()

    def morph(self):
        d = data("ch08_relax")
        h = heading("Relaxation oscillations: shrink epsilon")
        ax = plane((-2.6, 2.6, 1), (-2.4, 2.4, 1), width=6.6, height=6.1, at=pt(-3.0, -0.45), ticks=False)
        c0 = cubic(ax, -2.25, 2.25, MUTED, width=6).set_stroke(opacity=0.5)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(c0), run_time=1.0)
        keys = ["X1.0", "X0.3", "X0.1", "X0.02"]
        cols = [BLUE, TEAL, ORANGE, MAGENTA]
        cyc = curve(ax, d[keys[0]], color=cols[0], width=4, step=4)
        lab = m.MathTex(r"\varepsilon = 1", font_size=44, color=cols[0]).move_to(pt(3.6, 2.2))
        self.play(m.Create(cyc), m.FadeIn(lab))
        n = prose("grey: the cubic $y = x^3/3 - x$", font_size=24, color=MUTED).move_to(pt(3.6, 1.3))
        self.play(m.FadeIn(n))
        self.next_slide()
        for k in range(1, 4):
            c2 = curve(ax, d[keys[k]], color=cols[k], width=4, step=4)
            l2 = m.MathTex(r"\varepsilon = %s" % keys[k][1:], font_size=44, color=cols[k]).move_to(lab)
            self.play(m.Transform(cyc, c2), m.Transform(lab, l2), run_time=1.8)
            self.wait(0.6)
        self.next_slide("eps shrinks")
        Y = 2 / 3
        xx = np.linspace(-2, -1, 60)
        sing = m.VGroup(
            curve(ax, np.column_stack([xx, xx ** 3 / 3 - xx]), color=YELLOW, width=3),
            curve(ax, np.array([[-1, Y], [2, Y]]), color=YELLOW, width=3),
            curve(ax, np.column_stack([-xx, -(xx ** 3 / 3 - xx)]), color=YELLOW, width=3),
            curve(ax, np.array([[1, -Y], [-2, -Y]]), color=YELLOW, width=3))
        self.play(m.Create(sing), run_time=2)
        n2 = prose("$\\varepsilon \\to 0$: a singular cycle (yellow)\nof two slow pieces on the outer\nbranches and two horizontal jumps", font_size=24, color=YELLOW).move_to(pt(3.6, -0.3))
        self.play(m.FadeIn(n2))
        n3 = prose("$T = 6.66$ at $\\varepsilon = 1$, $2.87$ at $0.1$, $2.07$ at $0.02$", font_size=22, color=MUTED).next_to(n2, m.DOWN, buff=0.4)
        self.play(m.FadeIn(n3))
        self.wait(HOLD)
        self.next_slide("singular cycle")
        self.clear()

    def singular_period(self):
        h = heading("The period of the singular cycle")
        self.play(m.FadeIn(h))
        n = prose("on the slow branch the state stays on the cubic, so differentiate $y = x^3/3 - x$ in time:", font_size=25, color=MUTED, width=12.8).move_to(pt(0, 2.55))
        e1 = tex(r"\dot y = (x^2 - 1)\,\dot x", r"= -x", r"\quad\Rightarrow\quad", r"\dot x = \frac{-x}{x^2 - 1}", font_size=40).next_to(n, m.DOWN, buff=0.3)
        self.play(m.FadeIn(n), m.Write(e1[0]))
        self.next_slide()
        b = m.Brace(e1[1], m.DOWN, color=TEAL)
        bl = prose("$\\dot y = -x$", font_size=22, color=TEAL).next_to(b, m.DOWN, buff=0.08)
        self.play(m.Write(e1[1]), m.GrowFromCenter(b), m.FadeIn(bl))
        self.play(m.Write(e1[2:]))
        self.next_slide()
        n2 = prose("time to creep from $x = 2$ down to the fold at $x = 1$ ($dt = dx/\\dot x$):", font_size=25, color=MUTED).next_to(bl, m.DOWN, buff=0.4)
        e2 = tex(r"\int_1^2 \frac{x^2 - 1}{x}\,dx = \Big[\frac{x^2}{2} - \ln x\Big]_1^2 = \frac{3}{2} - \ln 2", font_size=40).next_to(n2, m.DOWN, buff=0.3)
        self.play(m.FadeIn(n2), m.Write(e2))
        self.next_slide()
        e3 = tex(r"T_0 = 2\Big(\frac{3}{2} - \ln 2\Big) = 3 - 2\ln 2 \approx 1.61", font_size=44, color=YELLOW).next_to(e2, m.DOWN, buff=0.45)
        n3 = prose("two slow branches; the jumps take no time when $\\varepsilon = 0$", font_size=23, color=YELLOW).next_to(e3, m.DOWN, buff=0.2)
        self.play(m.Write(e3), m.FadeIn(n3))
        self.wait(HOLD)
        self.next_slide("period")
        self.clear()

    def critical_manifold(self):
        h = heading("The critical manifold and its stability")
        ax = plane((-2.6, 2.6, 1), (-1.6, 1.6, 1), width=7.0, height=4.6, at=pt(-2.8, -0.9), ticks=False)
        self.play(m.FadeIn(h), m.Create(ax), run_time=0.8)
        e = tex(r"C_0 = \{f = 0\}:\ \ y = \frac{x^3}{3} - x", font_size=36).move_to(pt(3.8, 2.4))
        c0 = cubic(ax, -2.3, 2.3, MUTED, width=5)
        self.play(m.Write(e), m.Create(c0))
        n = prose("freeze $y$: the fast equation $\\varepsilon\\dot x = f(x, y)$\nis a flow on a line", font_size=24).next_to(e, m.DOWN, buff=0.3)
        self.play(m.FadeIn(n))
        self.next_slide()
        Y = m.ValueTracker(0.3)

        def line_flow():
            y0 = Y.get_value()
            g = m.VGroup(m.Line(ax.c2p(-2.6, y0), ax.c2p(2.6, y0), color=BLUE, stroke_width=2))
            roots = np.roots([1 / 3, 0, -1, -y0])
            roots = np.sort(roots[np.abs(roots.imag) < 1e-7].real)
            for r in roots:
                if abs(r) <= 2.6:
                    st = abs(r) > 1
                    g.add(m.Dot(ax.c2p(r, y0), radius=0.1, color=TEAL if st else MAGENTA))
            for x in np.linspace(-2.4, 2.4, 13):
                f = y0 - x ** 3 / 3 + x
                if abs(f) < 0.06:
                    continue
                s = np.sign(f) * 0.17
                g.add(m.Arrow(ax.c2p(x - s, y0), ax.c2p(x + s, y0), buff=0, color=YELLOW, stroke_width=4, max_tip_length_to_length_ratio=0.5))
            return g
        lf = m.always_redraw(line_flow)
        self.add(lf)
        self.play(Y.animate.set_value(-0.5), run_time=3)
        self.play(Y.animate.set_value(0.4), run_time=3)
        self.next_slide()
        att = m.VGroup(cubic(ax, -2.3, -1, TEAL, width=7), cubic(ax, 1, 2.3, TEAL, width=7))
        rep = cubic(ax, -1, 1, MAGENTA, width=7)
        folds = m.VGroup(dot(ax, (-1, 2 / 3), color=YELLOW, r=0.12), dot(ax, (1, -2 / 3), color=YELLOW, r=0.12))
        self.play(m.Create(att), m.Create(rep), m.FadeIn(folds))
        e2 = tex(r"\frac{\partial f}{\partial x} = 1 - x^2", font_size=38).next_to(n, m.DOWN, buff=0.35)
        rows = m.VGroup(prose("$\\lvert x \\rvert > 1$: $\\partial f/\\partial x < 0$, attracting", font_size=24, color=TEAL),
                        prose("$\\lvert x \\rvert < 1$: $\\partial f/\\partial x > 0$, repelling", font_size=24, color=MAGENTA),
                        prose("folds $(\\pm 1, \\mp\\frac23)$: $\\partial f/\\partial x = 0$", font_size=24, color=YELLOW)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.2).next_to(e2, m.DOWN, buff=0.3)
        self.play(m.Write(e2))
        for r in rows:
            self.play(m.FadeIn(r, shift=m.RIGHT * 0.2), run_time=0.6)
        self.wait(HOLD)
        self.next_slide("stability")
        self.clear()

    def anatomy(self):
        d = data("ch08_relax")
        X = d["X0.02"]
        h = heading("Anatomy of a relaxation oscillation")
        ax = plane((-2.6, 2.6, 1), (-1.2, 1.2, 1), width=7.6, height=3.9, at=pt(-2.6, -0.2), ticks=False)
        att = m.VGroup(cubic(ax, -2.3, -1, TEAL, width=6), cubic(ax, 1, 2.3, TEAL, width=6)).set_stroke(opacity=0.6)
        rep = cubic(ax, -1, 1, MAGENTA, width=6).set_stroke(opacity=0.6)
        cyc = curve(ax, X, color=INK, width=2.5, step=5)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(att), m.Create(rep), m.Create(cyc), run_time=1.2)
        k = m.ValueTracker(0)
        st = m.always_redraw(lambda: dot(ax, X[int(k.get_value())], color=YELLOW, r=0.13))
        self.add(st)
        n = prose("$\\varepsilon = 0.02$. The dot moves\nin real time: equal time steps", font_size=24).move_to(pt(4.7, 2.0))
        self.play(m.FadeIn(n))
        self.next_slide()
        N = len(X) - 1
        self.play(k.animate.set_value(N), run_time=8, rate_func=m.linear)
        steps = m.VGroup(prose("1. creep along an attracting branch", font_size=24, color=TEAL),
                         prose("2. reach the fold: no branch to follow", font_size=24, color=YELLOW),
                         prose("3. jump horizontally, in no time", font_size=24),
                         prose("4. creep along the other branch...", font_size=24, color=TEAL)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.22)
        steps.move_to(pt(0, -3.0))
        steps = m.VGroup(m.VGroup(steps[0], steps[1]).arrange(m.RIGHT, buff=0.8), m.VGroup(steps[2], steps[3]).arrange(m.RIGHT, buff=0.8)).arrange(m.DOWN, buff=0.2).move_to(pt(0, -3.05))
        self.play(m.FadeIn(steps))
        self.play(k.animate.set_value(0), run_time=0.01)
        self.play(k.animate.set_value(N), run_time=8, rate_func=m.linear)
        self.wait(BEAT)
        self.next_slide("anatomy")
        self.clear()

    def slow_manifolds(self):
        h = heading("Slow manifolds: what survives for epsilon > 0")
        self.play(m.FadeIn(h))
        thm = prose("Fenichel: away from the folds, each branch of $C_0$ perturbs to an invariant\nslow manifold $S_\\varepsilon$ within $O(\\varepsilon)$, with the same stability", font_size=25, width=12.8).move_to(pt(0, 2.5))
        self.play(m.FadeIn(thm))
        self.next_slide()
        eps = 0.1
        ax = plane((-2.6, 2.6, 1), (-1.4, 1.4, 1), width=7.4, height=4.4, at=pt(-2.6, -1.0), ticks=False)
        c0 = m.DashedVMobject(cubic(ax, -2.3, 2.3, MUTED, width=3), num_dashes=50)
        self.play(m.Create(ax), m.Create(c0))
        rng = np.random.default_rng(4)
        # forwards: states near the left branch converge onto S^a
        Pa = np.column_stack([rng.uniform(-2.5, -1.2, 14), rng.uniform(-1.0, 0.0, 14)])
        Ta = vdp_rk4(Pa, 0.6, 0.002, eps)
        # backwards: states near the middle branch converge onto S^r
        xm = rng.uniform(-0.7, 0.7, 14)
        Pr = np.column_stack([xm + rng.uniform(-0.35, 0.35, 14), xm ** 3 / 3 - xm])
        Tr = vdp_rk4(Pr, -0.5, 0.002, eps)
        k = m.ValueTracker(0)
        def cl(T, col):
            def f():
                i = min(int(k.get_value()), len(T) - 1)
                return m.VGroup(*[m.Dot(ax.c2p(*p), radius=0.06, color=col) for p in T[i] if abs(p[0]) < 2.6 and abs(p[1]) < 1.4])
            return f
        A = m.always_redraw(cl(Ta, TEAL))
        self.add(A)
        n1 = prose("forwards in time:\nstates collapse onto $S^a_\\varepsilon$", font_size=24, color=TEAL).move_to(pt(4.6, 0.3))
        self.play(m.FadeIn(n1))
        self.play(k.animate.set_value(len(Ta) - 1), run_time=4, rate_func=m.linear)
        Sa = curve(ax, Ta[-1][np.argsort(Ta[-1][:, 1])], color=TEAL, width=4)
        self.next_slide()
        k.set_value(0)
        B = m.always_redraw(cl(Tr, MAGENTA))
        self.add(B)
        n2 = prose("backwards in time the repelling\nbranch attracts: states collapse\nonto $S^r_\\varepsilon$", font_size=24, color=MAGENTA).move_to(pt(4.6, -1.6))
        self.play(m.FadeIn(n2))
        self.play(k.animate.set_value(len(Tr) - 1), run_time=4, rate_func=m.linear)
        n3 = prose("the trick behind every isochron computation:\nbackward time for repelling objects", font_size=24, color=YELLOW, width=4.8).move_to(pt(4.5, -3.1))
        self.play(m.FadeIn(n3))
        self.wait(HOLD)
        self.next_slide("slow manifolds")
        self.clear()

    def canards(self):
        d = data("ch08_canards")
        par, amp, hb = d["par"], d["amp"], float(d["hb"])
        cyc = [np.array(c, float) for c in d["cyc"]]
        h = heading("Canards: following the repelling branch")
        e = tex(r"\varepsilon\dot x = y - \frac{x^3}{3} + x,\qquad \dot y = a - x,\qquad \varepsilon = 0.05", font_size=34).move_to(pt(0, 2.65))
        self.play(m.FadeIn(h), m.Write(e))
        n = prose("fixed point at $x = a$: stable for $a > 1$, Hopf bifurcation at the fold $a = 1$", font_size=24, color=MUTED).next_to(e, m.DOWN, buff=0.2)
        self.play(m.FadeIn(n))
        self.next_slide()
        x0, x1 = 0.9915, 1.0005
        bd = plane((x0, x1, 0.002), (0, 4.4, 1), width=5.2, height=4.2, at=pt(-3.7, -1.3), xlabel="a", ylabel="", ticks=False)
        bl = prose("cycle amplitude", font_size=22, color=MUTED).next_to(bd, m.UP, buff=0.05).align_to(bd, m.LEFT)
        keep = (par >= x0) & (par <= x1)
        br = curve(bd, np.column_stack([par[keep], amp[keep]]), color=INK, width=3)
        hl = m.DashedLine(bd.c2p(hb, 0), bd.c2p(hb, 4.4), color=ORANGE, stroke_width=2)
        hlab = prose("Hopf", font_size=22, color=ORANGE).next_to(bd.c2p(hb, 4.4), m.DOWN, buff=0.1).shift(m.LEFT * 0.4)
        sc = m.VGroup(m.MathTex(r"\Delta a = 0.002", font_size=26, color=MUTED))
        bar = m.Line(bd.c2p(0.993, 0.3), bd.c2p(0.995, 0.3), color=MUTED, stroke_width=3)
        sc.next_to(bar, m.DOWN, buff=0.12)
        pp = plane((-2.4, 2.4, 1), (-1.0, 1.3, 1), width=6.0, height=4.0, at=pt(3.4, -1.3), ticks=False)
        c0 = cubic(pp, -2.25, 2.25, MUTED, width=5).set_stroke(opacity=0.5)
        self.play(m.Create(bd), m.FadeIn(bl), m.Create(hl), m.FadeIn(hlab), m.Create(bar), m.FadeIn(sc), m.Create(pp), m.Create(c0), run_time=1.2)
        self.play(m.Create(br), run_time=2)
        self.next_slide("branch")
        cols = [phase_color(k / 6) for k in range(6)]
        idx = []
        for c in cyc:
            A = c[:, 0].max() - c[:, 0].min()
            j = int(np.argmin(np.abs(amp - A) + 10 * (~keep)))
            idx.append(j)
        mk = dot(bd, (par[idx[0]], amp[idx[0]]), color=cols[0], r=0.1)
        cc = curve(pp, cyc[0], color=cols[0], width=4, step=4)
        self.play(m.FadeIn(mk), m.Create(cc))
        ghosts = m.VGroup()
        for k in range(1, 6):
            mk2 = dot(bd, (par[idx[k]], amp[idx[k]]), color=cols[k], r=0.1)
            c2 = curve(pp, cyc[k], color=cols[k], width=4, step=4)
            ghosts.add(cc.copy().set_stroke(width=2, opacity=0.5), mk.copy())
            self.add(ghosts)
            self.play(m.Transform(cc, c2), m.Transform(mk, mk2), run_time=1.4)
            self.wait(0.4)
        self.next_slide("explosion")
        n2 = prose("within $\\Delta a \\approx 10^{-4}$ (width $\\sim e^{-c/\\varepsilon}$) the cycle grows from tiny to full size:\nheadless ducks, then ducks with heads, then the relaxation oscillation", font_size=23, color=YELLOW, width=12.8)
        n2.to_edge(m.DOWN, buff=0.1)
        self.play(m.FadeOut(n), m.FadeIn(n2))
        self.wait(HOLD)
        self.next_slide("canard lesson")
        self.clear()

    def isochrons(self):
        d = data("ch08_vdp01")
        xs, ys, th, X = d["xs"], d["ys"], d["th"], d["X"]
        h = heading("Isochrons of a slow-fast oscillator")
        ax = plane((-2.6, 2.6, 1), (-1.4, 1.4, 1), width=8.6, height=4.65, at=pt(-2.15, -0.75), ticks=False)
        cyc = curve(ax, X, color=INK, width=4, step=5)
        c0 = m.DashedVMobject(cubic(ax, -1.75, 1.75, MUTED, width=3), num_dashes=40)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), m.Create(c0), run_time=1.2)
        n = prose("van der Pol, $\\varepsilon = 0.1$,\n30 isochrons (XPPAUT)", font_size=24).move_to(pt(4.85, 2.2))
        self.play(m.FadeIn(n))
        self.next_slide()
        N = 30
        groups = [m.VGroup(*[curve(ax, s, color=phase_color(j / N), width=2) for s in level_curves(xs, ys, th, j / N)]) for j in range(N)]
        self.play(m.LaggedStart(*[m.Create(g) for g in groups], lag_ratio=0.1), run_time=5)
        n2 = prose("equally spaced in time, so\nalmost all start on the slow\nbranches; the jumps get few", font_size=23, color=TEAL).next_to(n, m.DOWN, buff=0.4)
        self.play(m.FadeIn(n2))
        self.next_slide("slow branches")
        n3 = prose("inside: squeezed against the\nrepelling middle branch, with\nsharp hairpin turns where they\nleave it (first boomerangs)", font_size=23, color=MAGENTA).next_to(n2, m.DOWN, buff=0.4)
        xr, yr = (-1.2, 1.2), (-0.65, 0.65)
        az = plane((xr[0], xr[1], 0.5), (yr[0], yr[1], 0.5), width=8.6, height=4.65, at=pt(-2.15, -0.75), ticks=False)
        _xx = np.linspace(-1.2, 1.2, 200)
        zc0 = m.VGroup(*[m.DashedVMobject(pc, num_dashes=40) for pc in clipped(az, np.column_stack([_xx, _xx ** 3 / 3 - _xx]), xr, yr, color=MUTED, width=3)])
        zis = m.VGroup(*[clipped(az, s, xr, yr, color=phase_color(j / N), width=2.2) for j in range(N) for s in level_curves(xs, ys, th, j / N)])
        self.play(m.FadeOut(m.VGroup(ax, cyc, c0, *groups)), run_time=0.6)
        self.play(m.FadeIn(az), m.Create(zc0), run_time=0.6)
        self.play(m.Create(zis), m.FadeIn(n3), run_time=3)
        self.wait(HOLD)
        self.next_slide("zoom")
        self.clear()

    def ridge(self):
        d = data("ch08_vdp01")
        xs, ys, th, X = d["xs"], d["ys"], d["th"], d["X"]
        h = heading("The repelling slow manifold is a ridge")
        ax = plane((-2.6, 2.6, 1), (-1.4, 1.4, 1), width=8.6, height=4.65, at=pt(-2.15, -0.75), ticks=False)
        cyc = curve(ax, X, color=INK, width=3, step=5)
        rep = cubic(ax, -1, 1, MAGENTA, width=4).set_stroke(opacity=0.7)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), m.Create(rep), run_time=1.0)
        y0 = 0.25
        r = np.roots([1 / 3, 0, -1, -y0]); xm = float(r[np.argmin(np.abs(r.real))].real)
        # locate the true ridge on this row: where the phase jumps most
        row = np.argmin(np.abs(ys - y0))
        e = np.exp(2j * np.pi * th[row]); jump = np.abs(np.diff(e)); jump[np.abs(xs[:-1] - xm) > 0.4] = 0
        xr = 0.5 * (xs[np.argmax(jump)] + xs[np.argmax(jump) + 1])
        offs = np.linspace(-0.1, 0.1, 11)
        P = np.column_stack([xr + offs, np.full_like(offs, y0)])
        phs = [phase_at(xs, ys, th, p) for p in P]
        dots0 = m.VGroup(*[m.Dot(ax.c2p(*p), radius=0.07, color=phase_color(c)) for p, c in zip(P, phs)])
        self.play(m.FadeIn(dots0))
        n = prose("11 states within $\\pm 0.1$ of the\nridge, coloured by their phase", font_size=23).move_to(pt(4.85, 2.2))
        self.play(m.FadeIn(n))
        self.next_slide()
        T = vdp_rk4(P, 2.0, 0.001, 0.1)
        k = m.ValueTracker(0)
        dots = m.always_redraw(lambda: m.VGroup(*[m.Dot(ax.c2p(*T[int(k.get_value()), j]), radius=0.07, color=phase_color(phs[j])) for j in range(len(P))]))
        trails = m.always_redraw(lambda: m.VGroup(*[curve(ax, T[: max(2, int(k.get_value()) + 1), j], color=phase_color(phs[j]), width=1.5).set_stroke(opacity=0.6) for j in range(len(P))]))
        self.remove(dots0)
        self.add(trails, dots)
        self.play(k.animate.set_value(len(T) - 1), run_time=7, rate_func=m.linear)
        n2 = prose("left of the ridge: fall left;\nright: fall right, half a cycle\nlater. The closest ones creep\nalong it first: any phase at all", font_size=23, color=YELLOW).next_to(n, m.DOWN, buff=0.4)
        self.play(m.FadeIn(n2))
        self.next_slide()
        n3 = prose("every phase within an exponentially\nthin layer: almost a phaseless set,\nbut inside the basin", font_size=23, color=MAGENTA).next_to(n2, m.DOWN, buff=0.3).align_to(n2, m.LEFT)
        self.play(m.FadeIn(n3))
        self.wait(HOLD)
        self.next_slide("ridge")
        self.clear()

    def hard(self):
        self.bullets("Why slow-fast isochrons are hard to compute", [
            "Tiny Floquet multiplier ($\\ln\\lambda \\sim -1/\\varepsilon$): backward integration\nmagnifies errors by $1/\\lambda$ per period",
            "Phase changes enormously across thin layers: a grid needs extreme\nresolution there and wastes effort elsewhere",
            "Near $S^r_\\varepsilon$ nearby states have wildly different futures",
            "Chapter 10: why standard methods fail; Chapter 12: continuation\nof orbit segments gets round them",
        ], font_size=27)
