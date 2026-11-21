"""Deck for Chapter 11: Numerical continuation in a nutshell (predictor-corrector, pseudo-arclength, collocation)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import manimgx as m
from deckkit import *  # noqa: F401,F403


class EdgeAxes(m.Axes):
    """Axes whose lines sit on the left and bottom edges of the window, whatever the ranges."""
    _ox, _oy = 0.0, 0.0

    def coords_to_point(self, *coords):
        c = list(coords)
        if len(c) > 0:
            c[0] = np.asarray(c[0], float) - self._ox
        if len(c) > 1:
            c[1] = np.asarray(c[1], float) - self._oy
        return super().coords_to_point(*c)

    c2p = coords_to_point


def eplane(xr, yr, width=7.0, height=5.0, at=pt(0, -0.4), xlabel="x", ylabel="y", ticks=True, fs=24):
    """Like deckkit.plane, but axes on the window edges, ticks at round values, readable numbers, labels clear of them."""
    ax = EdgeAxes(x_range=(0, xr[1] - xr[0], xr[1] - xr[0]), y_range=(0, yr[1] - yr[0], yr[1] - yr[0]), x_length=width, y_length=height,
                  tips=False, axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": False, "include_numbers": False})
    ax._ox, ax._oy = xr[0], yr[0]
    ax.move_to(at)
    if ticks:
        def fmt(v, step):
            dec = max(0, -int(np.floor(np.log10(step) + 1e-9)))
            s = f"{v:.{dec}f}"
            return "0" if float(s) == 0 else s

        def vals(r):
            k0 = int(np.ceil(r[0] / r[2] - 1e-6)); k1 = int(np.floor(r[1] / r[2] + 1e-6))
            return [k * r[2] for k in range(k0, k1 + 1)]
        for v in vals(xr):
            p = ax.c2p(v, yr[0])
            ax.add(m.Line(p, p + m.DOWN * 0.1, color=MUTED, stroke_width=2))
            ax.add(m.MathTex(fmt(v, xr[2]), font_size=fs, color=MUTED).next_to(p, m.DOWN, buff=0.17))
        for v in vals(yr):
            p = ax.c2p(xr[0], v)
            ax.add(m.Line(p, p + m.LEFT * 0.1, color=MUTED, stroke_width=2))
            ax.add(m.MathTex(fmt(v, yr[2]), font_size=fs, color=MUTED).next_to(p, m.LEFT, buff=0.17))
    lx = m.MathTex(xlabel, font_size=36, color=MUTED).next_to(ax.c2p(xr[1], yr[0]), m.RIGHT, buff=0.2)
    ly = m.MathTex(ylabel, font_size=36, color=MUTED).next_to(ax.c2p(xr[0], yr[1]), m.UP, buff=0.2)
    ax.add(lx, ly)
    return ax


def s_plane(at=pt(-3.1, -0.5), scale=0.7):
    """(mu, x) plane for x^3 - 3x - mu = 0, equal units on both axes so perpendiculars look perpendicular."""
    return eplane((-4.5, 4.5, 1), (-2.5, 2.5, 1), width=9 * scale, height=5 * scale, at=at, xlabel=r"\mu", ylabel="x")


def s_curve(ax, color=MUTED, width=3, xr=(-2.25, 2.25)):
    x = np.linspace(xr[0], xr[1], 400)
    return curve(ax, np.column_stack([x ** 3 - 3 * x, x]), color=color, width=width)


def mx(x):
    """point (mu, x) on the S-curve"""
    return np.array([x ** 3 - 3 * x, x])


class Deck10(Deck):
    chapter = "Chapter 11"
    title = "Numerical Continuation in a Nutshell"
    subtitle = "Following solutions as something changes: Newton, pseudo-arclength and collocation"

    def construct(self):
        self.title_card()
        self.curves_of_solutions()
        self.natural()
        self.one_step()
        self.arclength_run()
        self.why_it_works()
        self.collocation()
        self.whole_segment()
        self.mesh()
        self.bvp_continuation()
        self.recap(["Continuation: a curve of solutions point by point, predictor along the tangent, corrector by Newton",
                    "Natural continuation in $\\mu$ fails at folds; pseudo-arclength goes round them",
                    "Its points come out equally spaced in arclength: exactly what isochrons need",
                    "Collocation: piecewise polynomials, ODE enforced at Gauss points, whole segment at once",
                    "Adaptive mesh: dense where the solution is fast (the spike)",
                    "Continuing BVP solutions is AUTO's core skill and the engine of the paper"])

    # ------------------------------------------------------------------
    def curves_of_solutions(self):
        h = heading("Solutions come in curves")
        self.play(m.FadeIn(h))
        e = tex(r"G(x,\mu) = x^3 - 3x - \mu = 0", font_size=40).move_to(pt(3.8, 2.3))
        cnt = prose("1 equation, 2 unknowns $(x, \\mu)$:\none unknown too many, so the\nsolutions form a curve", font_size=24, width=5.8).move_to(pt(3.8, 1.0))
        ax = s_plane()
        sc = s_curve(ax, color=TEAL, width=4)
        self.play(m.Write(e), m.Create(ax), run_time=1.2)
        self.play(m.FadeIn(cnt))
        self.play(m.Create(sc), run_time=2.5)
        self.next_slide("S-curve")
        k = m.ValueTracker(-2.2)
        dd = m.always_redraw(lambda: dot(ax, mx(k.get_value()), color=YELLOW, r=0.1))
        self.add(dd)
        self.play(k.animate.set_value(2.2), run_time=4, rate_func=m.linear)
        ex = prose("other curves of solutions:\n• equilibria as a parameter changes\n• periodic orbits (canard explosion)\n• orbit segments tracing an isochron (Ch. 12)", font_size=22, color=MUTED, width=5.8).move_to(pt(3.8, -1.5))
        self.play(m.FadeIn(ex))
        cont = prose("continuation: compute the curve point by point,\neach solution the starting guess for the next", font_size=24, color=YELLOW, width=12.6).to_edge(m.DOWN, buff=0.2)
        self.play(m.FadeIn(cont))
        self.wait(HOLD)
        self.next_slide("continuation")
        self.clear()

    def natural(self):
        d = data("ch10_cont"); nx, nmu = d["nat_x"], d["nat_mu"]
        h = heading("Natural continuation: step in μ, then Newton in x")
        ax = s_plane()
        sc = s_curve(ax)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(sc), run_time=1)
        txt = prose("increase $\\mu$ a little;\nsolve $G(x,\\mu) = 0$ for $x$ by Newton,\nstarting from the previous $x$", font_size=24, width=5.8).move_to(pt(4.0, 2.0))
        self.play(m.FadeIn(txt))
        self.next_slide()
        k = m.ValueTracker(0)
        n = len(nx)
        dots = m.always_redraw(lambda: m.VGroup(*[dot(ax, (nmu[i], nx[i]), color=MAGENTA, r=0.06) for i in range(int(k.get_value()) + 1)]))
        guide = m.always_redraw(lambda: m.DashedLine(ax.c2p(nmu[int(k.get_value())], -2.5), ax.c2p(nmu[int(k.get_value())], 2.5), color=MAGENTA, stroke_width=1.5))
        self.add(guide, dots)
        self.play(k.animate.set_value(n - 1), run_time=5, rate_func=m.linear)
        self.next_slide("climb")
        # the next step overshoots the fold
        mu2 = nmu[-1] + (nmu[-1] - nmu[-2])
        r = np.roots([1, 0, -3, -mu2]); xr = float(np.real(r[np.argmax(np.real(r))]))
        self.remove(guide)
        g2 = m.DashedLine(ax.c2p(mu2, -2.5), ax.c2p(mu2, 2.5), color=RED, stroke_width=2.5)
        jump = m.Arrow(ax.c2p(nmu[-1], nx[-1]), ax.c2p(mu2, xr), buff=0.1, color=RED, stroke_width=4)
        far = dot(ax, (mu2, xr), color=RED, r=0.09)
        self.play(m.Create(g2))
        self.play(m.GrowArrow(jump), m.FadeIn(far))
        msg = prose("past the fold at $\\mu = 2$ there is no\nsolution nearby: Newton jumps to a\nfar-away branch (or fails)", font_size=24, color=RED, width=5.8).move_to(pt(4.0, 0.2))
        self.play(m.FadeIn(msg))
        self.next_slide("jump")
        tan = m.Line(ax.c2p(2, -1.8), ax.c2p(2, -0.2), color=YELLOW, stroke_width=4)
        e = tex(r"G_x = 3x^2 - 3 = 0\ \text{ at } x = \pm 1", font_size=36, color=YELLOW).move_to(pt(4.0, -1.6))
        why = prose("the curve is vertical there as a function of $\\mu$:\n$\\mu$ is a bad parameter, and Newton's\ndivision by $G_x$ breaks down", font_size=22, color=MUTED).next_to(e, m.DOWN, buff=0.25)
        self.play(m.Create(tan), m.Write(e), m.FadeIn(why))
        self.wait(HOLD)
        self.next_slide("fold")
        self.clear()

    def one_step(self):
        h = heading("Pseudo-arclength: one predictor–corrector step")
        sc_ = 4.0
        ax = eplane((1.0, 2.4, 0.2), (-1.6, -0.3, 0.2), width=1.4 * sc_, height=1.3 * sc_, at=pt(-3.2, -0.45), xlabel=r"\mu", ylabel="x")
        sc = s_curve(ax, xr=(-1.5, -0.35))
        zl = prose("zoom on the fold at $\\mu = 2$", font_size=22, color=MUTED).next_to(ax, m.UP, buff=0.05).shift(m.RIGHT * 0.8)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(sc), m.FadeIn(zl), run_time=1)
        x0 = -1.2
        Z0 = mx(x0)
        t = np.array([3 * x0 ** 2 - 3, 1.0]); t /= np.linalg.norm(t)
        hstep = 0.25
        Zp = Z0 + hstep * t
        xs = np.linspace(-1.5, -0.35, 20001)
        f = (xs ** 3 - 3 * xs - Zp[0]) * t[0] + (xs - Zp[1]) * t[1]
        idx = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
        best = min(idx, key=lambda i: np.linalg.norm(mx(xs[i]) - Zp))
        Z1 = mx(xs[best])
        e0 = tex(r"\vec Z = (x, \mu)", font_size=38).move_to(pt(4.0, 2.4))
        l0 = prose("treat $x$ and $\\mu$ alike", font_size=22, color=MUTED).next_to(e0, m.DOWN, buff=0.15)
        d0 = dot(ax, Z0, color=TEAL, r=0.1)
        lz = tex(r"\vec Z_k", font_size=36, color=TEAL).next_to(d0, m.DR, buff=0.08)
        self.play(m.Write(e0), m.FadeIn(l0), m.FadeIn(d0), m.FadeIn(lz))
        self.next_slide()
        arr = m.Arrow(ax.c2p(*Z0), ax.c2p(*Zp), buff=0, color=BLUE, stroke_width=5)
        dp = dot(ax, Zp, color=BLUE, r=0.09)
        e1 = tex(r"\vec Z_{\text{pred}} = \vec Z_k + h\,\vec t_k", font_size=38, color=BLUE).move_to(pt(4.0, 1.1))
        l1 = prose("predictor: step $h$ along the tangent", font_size=22, color=MUTED).next_to(e1, m.DOWN, buff=0.15)
        self.play(m.GrowArrow(arr), m.FadeIn(dp), m.Write(e1), m.FadeIn(l1))
        self.next_slide("predictor")
        nrm = np.array([-t[1], t[0]])
        a = ax.c2p(*(Zp - 0.3 * nrm)); b = ax.c2p(*(Zp + 0.3 * nrm))
        perp = m.DashedLine(a, b, color=ORANGE, stroke_width=3)
        e2 = tex(r"\vec t_k\cdot(\vec Z - \vec Z_{\text{pred}}) = 0", font_size=38, color=ORANGE).move_to(pt(4.0, -0.3))
        l2 = prose("corrector: Newton on $G = 0$ plus this line (10.1),\nperpendicular to the tangent", font_size=22, color=MUTED, width=5.4).next_to(e2, m.DOWN, buff=0.15)
        self.play(m.Create(perp), m.Write(e2), m.FadeIn(l2))
        self.next_slide()
        d1 = dot(ax, Z1, color=TEAL, r=0.1)
        lz1 = tex(r"\vec Z_{k+1}", font_size=36, color=TEAL).next_to(d1, m.RIGHT, buff=0.15)
        mv = dp.copy().set_color(YELLOW)
        self.play(mv.animate.move_to(ax.c2p(*Z1)), run_time=1.2)
        self.play(m.FadeIn(d1), m.FadeIn(lz1), m.FadeOut(mv))
        why = prose("the new point lies about $h$ further along the\ncurve, even though it has turned back in $\\mu$", font_size=24, color=YELLOW, width=5.8).move_to(pt(4.0, -2.4))
        self.play(m.FadeIn(why))
        self.wait(HOLD)
        self.next_slide("corrector")
        self.clear()

    def arclength_run(self):
        d = data("ch10_cont"); arc, preds = d["arc"], d["preds"]
        h = heading("Pseudo-arclength goes round both folds")
        ax = s_plane()
        sc = s_curve(ax)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(sc), run_time=1)
        leg = m.VGroup(prose("blue: predictor along the tangent", font_size=22, color=BLUE),
                       prose("teal: corrected point on the curve", font_size=22, color=TEAL)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.2).move_to(pt(4.0, 2.2))
        self.play(m.FadeIn(leg))
        self.next_slide()
        k = m.ValueTracker(0)
        n = len(preds)

        def draw():
            j = int(k.get_value())
            g = m.VGroup()
            for i in range(j + 1):
                g.add(dot(ax, arc[i][::-1], color=TEAL, r=0.055))
            if j < n:
                g.add(m.Line(ax.c2p(*arc[j][::-1]), ax.c2p(*preds[j][::-1]), color=BLUE, stroke_width=3))
                g.add(dot(ax, preds[j][::-1], color=BLUE, r=0.05))
            return g
        run = m.always_redraw(draw)
        self.add(run)
        self.play(k.animate.set_value(n), run_time=10, rate_func=m.linear)
        t1 = prose("nothing special happens at the folds;\nthe points come out roughly equally\nspaced in arclength", font_size=24, width=5.8).move_to(pt(4.0, 0.3))
        self.play(m.FadeIn(t1))
        self.next_slide("round the folds")
        q = prose("for isochrons: continuation \"automatically provides\nan arclength parameterization\" — points where the\ncurve is, not crowded as in backward integration", font_size=22, color=YELLOW, width=5.8).move_to(pt(4.0, -1.9))
        self.play(m.FadeIn(q))
        self.wait(HOLD)
        self.next_slide("arclength")
        self.clear()

    def why_it_works(self):
        h = heading("Why the extra equation saves the day at a fold")
        self.play(m.FadeIn(h))
        lt = m.Text("natural: Newton in x alone", font_size=28, weight=m.BOLD, color=MAGENTA).move_to(pt(-3.5, 2.3))
        l1 = tex(r"x \leftarrow x - \frac{G}{G_x}", font_size=42).move_to(pt(-3.5, 1.1))
        l2 = tex(r"\text{fold } (x=-1):\ G_x = 0", font_size=38, color=MAGENTA).move_to(pt(-3.5, -0.1))
        l3 = prose("division by zero: the\nJacobian is singular", font_size=24, color=MUTED).move_to(pt(-3.5, -1.3))
        self.play(m.FadeIn(lt), m.Write(l1))
        self.play(m.Write(l2), m.FadeIn(l3))
        self.next_slide("natural singular")
        rt = m.Text("pseudo-arclength: Newton in (x, μ)", font_size=28, weight=m.BOLD, color=TEAL).move_to(pt(3.5, 2.3))
        r1 = tex(r"J = \begin{pmatrix} G_x & G_\mu \\ t_x & t_\mu \end{pmatrix}", font_size=40).move_to(pt(3.5, 1.0))
        r2 = tex(r"\text{fold: } J = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix},\quad \det J = 1", font_size=38, color=TEAL).move_to(pt(3.5, -0.4))
        r3 = prose("$G_\\mu = -1$ never vanishes, and the tangent\n$\\vec t = (1, 0)$ is perfectly well defined there", font_size=22, color=MUTED, width=5.8).move_to(pt(3.5, -1.6))
        self.play(m.FadeIn(rt), m.Write(r1))
        self.play(m.Write(r2), m.FadeIn(r3))
        self.next_slide("augmented invertible")
        ad = prose("bonus: adapt $h$ (shrink when Newton struggles, grow when it converges fast).\nAUTO does this automatically.", font_size=24, color=YELLOW, width=12.6).to_edge(m.DOWN, buff=0.45)
        self.play(m.FadeIn(ad))
        self.wait(HOLD)
        self.next_slide("adaptive step")
        self.clear()

    def collocation(self):
        h = heading("Collocation: a differential equation becomes algebra")
        ax = eplane((0, 1, 0.2), (-1.6, 1.6, 0.5), width=7.0, height=4.4, at=pt(-3.0, -0.3), xlabel="s", ylabel="u", ticks=False)
        f = lambda s: np.sin(2 * np.pi * s) + 0.45 * np.sin(5 * np.pi * s + 0.4)
        fp = lambda s: 2 * np.pi * np.cos(2 * np.pi * s) + 0.45 * 5 * np.pi * np.cos(5 * np.pi * s + 0.4)
        self.play(m.FadeIn(h), m.Create(ax), run_time=1)
        ntst = 5
        edges = np.linspace(0, 1, ntst + 1)
        ticks = m.VGroup(*[m.Line(ax.c2p(e, -1.6), ax.c2p(e, 1.6), color=MUTED, stroke_width=1.5) for e in edges])
        st1 = prose("1. split $[0,1]$ into NTST mesh intervals", font_size=24, width=5.8).move_to(pt(4.0, 2.4))
        self.play(m.Create(ticks), m.FadeIn(st1))
        self.next_slide()
        cols = [TEAL, BLUE]
        pieces = m.VGroup()
        for i in range(ntst):
            s = np.linspace(edges[i], edges[i + 1], 40)
            pieces.add(curve(ax, np.column_stack([s, f(s)]), color=cols[i % 2], width=5))
        st2 = prose("2. on each, a polynomial\nof degree NCOL = 4", font_size=24, width=5.8).next_to(st1, m.DOWN, aligned_edge=m.LEFT, buff=0.35)
        self.play(m.LaggedStart(*[m.Create(p) for p in pieces], lag_ratio=0.3), m.FadeIn(st2), run_time=2)
        self.next_slide()
        g = (np.polynomial.legendre.leggauss(4)[0] + 1) / 2
        cps = m.VGroup()
        for i in range(ntst):
            for gg in g:
                s = edges[i] + gg * (edges[i + 1] - edges[i])
                p = ax.c2p(s, f(s))
                v = ax.c2p(s + 0.03, f(s) + 0.03 * fp(s)) - p
                v = v / np.linalg.norm(v) * 0.28
                cps.add(m.VGroup(m.Line(p - v, p + v, color=YELLOW, stroke_width=3), m.Dot(p, radius=0.05, color=YELLOW)))
        st3 = prose("3. demand $\\vec u\\,' = kT\\vec F(\\vec u)$ exactly at\n4 Gauss points per interval (yellow)", font_size=24, width=5.8).next_to(st2, m.DOWN, aligned_edge=m.LEFT, buff=0.35)
        self.play(m.LaggedStart(*[m.FadeIn(c) for c in cps], lag_ratio=0.05), m.FadeIn(st3), run_time=2)
        self.next_slide()
        joins = m.VGroup(*[m.Dot(ax.c2p(e, f(e)), radius=0.08, color=INK) for e in edges])
        st4 = prose("4. pieces join continuously;\nimpose the boundary conditions", font_size=24, width=5.8).next_to(st3, m.DOWN, aligned_edge=m.LEFT, buff=0.35)
        self.play(m.FadeIn(joins), m.FadeIn(st4))
        self.next_slide()
        res = prose("result: one large algebraic system for all\ncoefficients, solved by Newton. NTST = 100,\nNCOL = 4, 2 equations: about\n$100 \\times 5 \\times 2 = 1000$ unknowns", font_size=24, color=YELLOW, width=5.8).next_to(st4, m.DOWN, aligned_edge=m.LEFT, buff=0.4)
        self.play(m.FadeIn(res))
        self.wait(HOLD)
        self.next_slide("collocation")
        self.clear()

    def whole_segment(self):
        h = heading("The whole segment is solved at once")
        ax = eplane((0, 1, 0.2), (-1.6, 1.6, 0.5), width=7.0, height=4.4, at=pt(-3.0, -0.4), xlabel="s", ylabel="u", ticks=False)
        f = lambda s: np.sin(2 * np.pi * s) + 0.45 * np.sin(5 * np.pi * s + 0.4)
        guess = lambda s: f(0) + (f(1) - f(0)) * s + 0.9 * np.sin(np.pi * s)
        s = np.linspace(0, 1, 200)
        sm = np.linspace(0, 1, 21)
        self.play(m.FadeIn(h), m.Create(ax), run_time=1)
        tru = curve(ax, np.column_stack([s, f(s)]), color=MUTED, width=2)
        tru = m.DashedVMobject(tru, num_dashes=40)

        def state(a, col):
            u = (1 - a) * guess(s) + a * f(s)
            um = (1 - a) * guess(sm) + a * f(sm)
            return m.VGroup(curve(ax, np.column_stack([s, u]), color=col, width=4),
                            m.VGroup(*[dot(ax, (x, y), color=INK, r=0.05) for x, y in zip(sm, um)]))
        cur = state(0.0, ORANGE)
        lab = prose("start: a rough guess\n(the previous solution)", font_size=24, color=ORANGE, width=5.8).move_to(pt(4.0, 1.8))
        self.play(m.Create(tru), m.Create(cur), m.FadeIn(lab), run_time=1.5)
        self.next_slide()
        info = prose("each Newton step moves every mesh\npoint together, using the equations\non both sides of it", font_size=24, width=5.8).move_to(pt(4.0, 0.0))
        self.play(m.FadeIn(info))
        for i, a in enumerate([0.7, 0.96, 1.0]):
            nxt = state(a, TEAL if a == 1.0 else ORANGE)
            self.play(m.Transform(cur, nxt), run_time=1.2)
            self.wait(0.4)
        why = prose("no information has to travel step by step\nfrom one end: the error at one point is\ncontrolled from both sides.  That is what\nbackward integration lacked.", font_size=24, color=YELLOW, width=5.8).move_to(pt(4.0, -2.2))
        self.play(m.FadeIn(why))
        sch = prose("(schematic)", font_size=22, color=MUTED).next_to(ax, m.DOWN, buff=0.1)
        self.play(m.FadeIn(sch))
        self.wait(HOLD)
        self.next_slide("whole segment")
        self.clear()

    def mesh(self):
        d = data("ch09_bundle"); s, u = d["s"], d["u"]
        h = heading("The mesh adapts: dense at the spike")
        ax = eplane((0, 1, 0.2), (-90, 60, 30), width=8.0, height=4.2, at=pt(-2.2, 0.0), xlabel="s", ylabel="V")
        V = curve(ax, np.column_stack([s, u[0]]), color=TEAL, width=3)
        self.play(m.FadeIn(h), m.Create(ax), run_time=1)
        self.play(m.Create(V), run_time=2)
        lab = prose("$V(s)$ on the neuron's cycle\n(the Chapter 10 BVP, scipy's mesh)", font_size=22, color=TEAL).move_to(pt(4.6, 2.0))
        self.play(m.FadeIn(lab))
        self.next_slide()
        sub = s[::6]
        y0 = ax.c2p(0, -90)[1] - 0.55
        rug = m.VGroup(*[m.Line(pt(ax.c2p(x, 0)[0], y0), pt(ax.c2p(x, 0)[0], y0 + 0.35), color=ORANGE, stroke_width=1.2) for x in sub])
        rl = prose("mesh nodes (every 6th)", font_size=22, color=ORANGE).next_to(rug, m.DOWN, buff=0.15)
        self.play(m.LaggedStart(*[m.Create(r) for r in rug], lag_ratio=0.003), run_time=2.5)
        self.play(m.FadeIn(rl))
        ds = np.diff(s)
        mid = 0.5 * (s[:-1] + s[1:])
        spike = ds[(mid < 0.02) | (mid > 0.98)].mean(); slow = ds[(mid > 0.3) & (mid < 0.7)].mean()
        t1 = prose(f"intervals about {slow / spike:.0f}× smaller\nduring the spike than in recovery", font_size=24).move_to(pt(4.6, 0.4))
        t2 = prose("the solver adds intervals where\nthe solution changes quickly:\n$V$ moves 120 mV in a fraction of a ms", font_size=22, color=MUTED).move_to(pt(4.6, -1.2))
        self.play(m.FadeIn(t1), m.FadeIn(t2))
        self.wait(HOLD)
        self.next_slide("mesh")
        self.clear()

    def bvp_continuation(self):
        h = heading("Continuation of boundary-value problems")
        ax = s_plane(at=pt(-3.6, 0.3), scale=0.6)
        sc = s_curve(ax, color=MUTED)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(sc), run_time=1)
        ax2 = eplane((0, 1, 0.2), (-2.5, 2.5, 1), width=5.0, height=3.2, at=pt(3.4, 0.5), xlabel="s", ylabel=r"\vec u", ticks=False)
        self.play(m.Create(ax2))
        k = m.ValueTracker(-2.0)
        dd = m.always_redraw(lambda: dot(ax, mx(k.get_value()), color=YELLOW, r=0.1))
        ss = np.linspace(0, 1, 120)

        def seg():
            x = k.get_value()
            y = x * np.sin(np.pi * ss) + 0.5 * np.sin(3 * np.pi * ss + x)
            return curve(ax2, np.column_stack([ss, y]), color=TEAL, width=4)
        sg = m.always_redraw(seg)
        l1 = prose("a point on the branch ...", font_size=22, color=YELLOW).next_to(ax, m.DOWN, buff=0.2)
        l2 = prose("... is a whole orbit segment", font_size=22, color=TEAL).next_to(ax2, m.DOWN, buff=0.2)
        self.add(dd, sg)
        self.play(m.FadeIn(l1), m.FadeIn(l2))
        self.play(k.animate.set_value(2.0), run_time=5, rate_func=m.there_and_back)
        self.next_slide("family")
        self.play(m.FadeOut(l1), m.FadeOut(l2), run_time=0.4)
        rows = m.VGroup(
            tex(r"\vec Z = (\text{all polynomial coefficients},\ \text{free parameters})", font_size=34),
            tex(r"G(\vec Z) = 0:\ \text{collocation equations} + \text{boundary conditions}", font_size=34),
            tex(r"\vec t_k\cdot(\vec Z - \vec Z_{\text{pred}}) = 0", font_size=34, color=ORANGE),
        ).arrange(m.DOWN, buff=0.25).to_edge(m.DOWN, buff=0.3)
        for r in rows:
            self.play(m.FadeIn(r, shift=m.UP * 0.2), run_time=0.7)
        self.wait(HOLD)
        self.next_slide("AUTO")
        self.clear()
