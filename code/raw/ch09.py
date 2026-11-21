"""Deck for Chapter 10: Computing isochrons, the problem (paper Section 2 and start of Section 3)."""
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


def rhh_F(u):
    """Reduced Hodgkin-Huxley right-hand side (paper eq. 4.1)."""
    v, n = u
    an = 0.01 * (v + 55) / (1 - np.exp(-(v + 55) / 10)); bn = 0.125 * np.exp(-(v + 65) / 80)
    am = 0.1 * (v + 40) / (1 - np.exp(-(v + 40) / 10)); bm = 4 * np.exp(-(v + 65) / 18)
    mi = am / (am + bm)
    dv = 10 - 120 * mi ** 3 * (0.8 - n) * (v - 50) - 36 * n ** 4 * (v + 77) - 0.3 * (v + 54.4)
    return np.array([dv, an * (1 - n) - bn * n])


def backward_runs(X, T, idx):
    """Integrate backwards from points ON the cycle (double precision); stop when leaving the window."""
    from scipy.integrate import solve_ivp
    runs = []
    for i in idx:
        def ev(t, u):
            return 1.0 if (-90 < u[0] < 60 and 0.3 < u[1] < 0.8) else -1.0
        ev.terminal = True
        s = solve_ivp(lambda t, u: -rhh_F(u), (0, T), X[i], method="LSODA", rtol=1e-10, atol=1e-12,
                      events=ev, max_step=0.01)
        Y = s.y.T
        ok = (Y[:, 0] > -90) & (Y[:, 0] < 60) & (Y[:, 1] > 0.3) & (Y[:, 1] < 0.8)
        runs.append((s.t[ok], Y[ok]))
    return runs


def screen_dir(ax, p, w, length=0.55):
    """Unit screen direction of the phase-space vector w at p (the 'scaled picture')."""
    w = np.asarray(w, float) / (np.linalg.norm(w) + 1e-300)   # |w| varies over many orders along the cycle
    a = ax.c2p(*p); b = ax.c2p(p[0] + 1e-2 * w[0], p[1] + 1e-2 * w[1])
    d = b - a
    return d / (np.linalg.norm(d) + 1e-12) * length


class Deck09(Deck):
    chapter = "Chapter 10"
    title = "Computing Isochrons: The Problem"
    subtitle = "Backward integration, local approximations, and the cycle as a boundary-value problem"

    def construct(self):
        self.title_card()
        self.three_ways()
        self.cycle_as_bvp()
        self.shrinking_vector()
        self.log_formulation()
        self.counting()
        self.bundle_picture()
        self.show_figure("ch09_bundle", "The stable bundle of the neuron's cycle (paper §2.2)",
                         "(a) $\\vec w$ at 40 points equally spaced in phase: they crowd on the slow left part of $\\Gamma$.\n"
                         "(b) The angle of $\\vec w$: horizontal near the spike, far from horizontal during recovery.",
                         height=4.6)
        self.backward_neuron()
        self.sensitivity()
        self.show_figure("ch09_backward", "Backward integration with XPPAUT",
                         "(a) van der Pol: 41 equally spaced points of $L_\\eta$ land on the isochron, but very unevenly.\n"
                         "(b) Neuron: runs started on $\\Gamma$ fall off it within one period.", height=4.9)
        self.the_cure()
        self.recap(["Grids are robust but blind to thin layers; backward integration is ill-conditioned when $1/\\lambda \\gg 1$",
                    "Neuron: $1/\\mu \\approx 1.5\\times 10^{20}$ per period, beyond double precision",
                    "Cycle and stable direction together as one BVP, with $\\vec n = e^{\\kappa t}\\vec w$",
                    "We reproduce $T = 11.8463$ and $\\ln\\mu = -46.466$",
                    "The cure: boundary-value problems for whole orbit segments, then continuation"])

    # ------------------------------------------------------------------
    def three_ways(self):
        h = heading("Three ways to compute an isochron")
        self.play(m.FadeIn(h))
        cards = []
        specs = [
            ("1. Brute force on a grid", "integrate each grid point\nforwards, read its phase,\ndraw level curves",
             "robust and parallel,\nbut blind to thin layers", TEAL),
            ("2. Backward integration", "flow the local piece $L_\\eta$\nbackwards:\n$I(\\vec\\gamma) \\approx \\bigcup_k \\Phi_{-kT}(L_\\eta)$",
             "classic for stable\nmanifolds", MAGENTA),
            ("3. Higher-order pieces", "Taylor series or the\nparameterisation method:\nan accurate local piece",
             "still has to be\nextended by method 2", ORANGE),
        ]
        for title, body, note, col in specs:
            t = m.Text(title, font_size=24, weight=m.BOLD, color=col)
            b = prose(body, font_size=22)
            n = prose(note, font_size=22, color=MUTED)
            g = m.VGroup(t, b, n).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.3)
            if g.width > 3.9:
                g.scale_to_fit_width(3.9)
            box = m.RoundedRectangle(width=4.3, height=3.6, corner_radius=0.15, color=col, stroke_width=2)
            g.move_to(box)
            cards.append(m.VGroup(box, g))
        row = m.VGroup(*cards).arrange(m.RIGHT, buff=0.3).move_to(pt(0, 0.45))
        # little pictures: grid / backward arrow / curved piece
        icons = m.VGroup(
            m.VGroup(*[m.Dot(pt(x, y), radius=0.035, color=TEAL) for x in np.linspace(-0.6, 0.6, 5) for y in np.linspace(-0.25, 0.25, 3)]),
            m.Arrow(pt(0.5, 0), pt(-0.5, 0), buff=0, color=MAGENTA, stroke_width=4),
            m.Arc(radius=0.8, start_angle=1.2, angle=0.8, color=ORANGE, stroke_width=4),
        )
        for ic, c in zip(icons, cards):
            ic.next_to(c, m.DOWN, buff=0.25)
        for c, ic in zip(cards, icons):
            self.play(m.FadeIn(c, shift=m.UP * 0.2), m.FadeIn(ic), run_time=0.8)
            self.wait(BEAT)
            self.next_slide()
        need = prose("All three first need the cycle $\\Gamma$ and the stable direction $\\vec v_s$ at each of its points",
                     font_size=26, color=YELLOW, width=12.6).to_edge(m.DOWN, buff=0.35)
        self.play(m.FadeIn(need))
        self.wait(HOLD)
        self.next_slide("two ingredients")
        self.clear()

    def cycle_as_bvp(self):
        d = data("rhh_cycle"); X, T = d["X"], float(d["T"])
        h = heading("A periodic orbit is a boundary-value problem")
        ax = eplane((-90, 60, 30), (0.3, 0.8, 0.1), width=6.2, height=4.6, at=pt(-3.2, -0.6), xlabel="V", ylabel="n")
        cyc = curve(ax, X, color=MUTED, width=3, step=40)
        self.play(m.FadeIn(h), m.Create(ax), run_time=1)
        q = prose("Where does it start?\nHow long does it last?", font_size=26, color=MUTED).move_to(pt(3.6, 2.2))
        self.play(m.FadeIn(q))
        k = m.ValueTracker(0)
        N = len(X)
        dd = m.always_redraw(lambda: dot(ax, X[min(int(k.get_value()), N - 1)], color=TEAL, r=0.1))
        tr = m.always_redraw(lambda: curve(ax, X[: max(2, int(k.get_value()) + 1)], color=TEAL, width=4, step=40))
        self.add(tr, dd)
        self.play(m.Create(cyc), k.animate.set_value(N - 1), run_time=4, rate_func=m.linear)
        e1 = tex(r"s = \frac{t}{T}\in[0,1]", font_size=36).move_to(pt(3.6, 1.0))
        e2 = tex(r"\vec u\,'(s) = T\,\vec F(\vec u),\qquad \vec u(1) = \vec u(0)", font_size=36).move_to(pt(3.6, 0.1))
        why = prose("$T$ becomes an unknown number;\nthe interval is now fixed", font_size=22, color=MUTED).next_to(e2, m.DOWN, buff=0.25)
        self.play(m.Write(e1), m.Write(e2), m.FadeIn(why), run_time=1.6)
        self.wait(BEAT)
        self.next_slide("bvp")
        # time shifts: every start point works
        self.play(m.FadeOut(tr), m.FadeOut(why))
        sh = prose("but any time shift is again a solution:\nthe start point can slide anywhere", font_size=24, color=ORANGE).move_to(pt(3.6, -1.1))
        self.play(m.FadeIn(sh))
        self.play(k.animate.set_value(N // 2), run_time=2.0)
        self.play(k.animate.set_value(0), run_time=2.0)
        star = m.Star(n=5, outer_radius=0.2, color=YELLOW, fill_opacity=1).move_to(ax.c2p(*X[0]))
        pc = tex(r"f_1(\vec u(0)) = 0", font_size=38, color=YELLOW).move_to(pt(3.6, -2.4))
        pcl = prose("phase condition: start where $\\dot V = 0$,\nat the spike peak $\\vec\\gamma_0$", font_size=22, color=MUTED).next_to(pc, m.DOWN, buff=0.2)
        self.play(m.FadeIn(star, scale=0.5), m.Write(pc), m.FadeIn(pcl))
        self.wait(HOLD)
        self.next_slide("phase condition")
        self.clear()

    def shrinking_vector(self):
        d = data("ch09_bundle"); s, u, w, T, kap = d["s"], d["u"], d["w"], float(d["T"]), float(d["kappa"])
        h = heading("The stable vector shrinks by a factor 10⁻²⁰ every period")
        ax = eplane((-90, 60, 30), (0.3, 0.8, 0.1), width=6.4, height=4.8, at=pt(-3.1, -0.6), xlabel="V", ylabel="n")
        cyc = curve(ax, u.T, color=MUTED, width=3, step=2)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), run_time=1)
        e = tex(r"\vec n^s(T) = \mu\,\vec n^s(0),\qquad \mu = e^{\kappa T}", font_size=38).move_to(pt(3.6, 2.3))
        self.play(m.Write(e))
        ph = m.ValueTracker(0.0)

        def at(p):
            p = min(max(p, 0.0), 1.0)
            return np.array([np.interp(p, s, u[0]), np.interp(p, s, u[1])]), np.array([np.interp(p, s, w[0]), np.interp(p, s, w[1])])

        def arrow_n():
            p, ww = at(ph.get_value())
            L = 1.1 * np.exp(kap * T * ph.get_value())
            a = ax.c2p(*p)
            if L < 0.03:
                return m.Dot(a, radius=0.06, color=MAGENTA)
            return m.Arrow(a, a + screen_dir(ax, p, ww, L), buff=0, color=MAGENTA, stroke_width=5)
        arr = m.always_redraw(arrow_n)
        pd = m.always_redraw(lambda: m.Dot(ax.c2p(*at(ph.get_value())[0]), radius=0.08, color=INK))
        lab = prose("$\\log_{10}\\lvert\\vec n^s\\rvert$ =", font_size=26, color=MAGENTA).move_to(pt(2.6, 1.0))
        num = m.DecimalNumber(0.0, num_decimal_places=1, font_size=40, color=MAGENTA).next_to(lab, m.RIGHT, buff=0.25)
        num.add_updater(lambda mob: mob.set_value(kap * T * ph.get_value() / np.log(10)))
        self.play(m.FadeIn(arr), m.FadeIn(pd), m.FadeIn(lab), m.FadeIn(num))
        self.next_slide("vector")
        self.play(ph.animate.set_value(1.0), run_time=6, rate_func=m.linear)
        num.clear_updaters()
        msg = prose("after one period: $\\mu = 6.6\\times 10^{-21}$\n20 orders of magnitude below the cycle:\nno computer can carry that vector", font_size=24, color=YELLOW).move_to(pt(3.6, -0.6))
        self.play(m.FadeIn(msg))
        self.wait(BEAT)
        self.next_slide("shrunk")
        # the remedy: the normalised direction w, carried round, stays size one
        self.play(m.FadeOut(msg), m.FadeOut(arr), m.FadeOut(lab), m.FadeOut(num))
        ph.set_value(0.0)

        def arrow_w():
            p, ww = at(ph.get_value())
            a = ax.c2p(*p)
            return m.Arrow(a, a + screen_dir(ax, p, ww, 1.1), buff=0, color=ORANGE, stroke_width=5)
        arw = m.always_redraw(arrow_w)
        rem = tex(r"\vec n^s(t) = e^{\kappa t}\,\vec w(t)", font_size=42, color=ORANGE).move_to(pt(3.6, 1.0))
        rl = prose("factor out the shrinking:\n$\\vec w$ is periodic and keeps length 1", font_size=24).next_to(rem, m.DOWN, buff=0.3)
        self.play(m.Write(rem), m.FadeIn(rl), m.FadeIn(arw))
        self.next_slide("remedy")
        self.play(ph.animate.set_value(1.0), run_time=6, rate_func=m.linear)
        self.wait(BEAT)
        self.next_slide("w is periodic")
        self.clear()

    def log_formulation(self):
        h = heading("The logarithmic formulation, step by step")
        self.play(m.FadeIn(h))
        e0 = tex(r"\vec n\,' = T\,D\vec F(\vec u)\,\vec n", font_size=40).move_to(pt(0, 2.3))
        l0 = prose("variational equation (3.1) in rescaled time $s$", font_size=22, color=MUTED).next_to(e0, m.DOWN, buff=0.15)
        self.play(m.Write(e0), m.FadeIn(l0))
        self.next_slide()
        e1 = tex(r"\vec n = e^{\kappa T s}\,\vec w", font_size=40, color=ORANGE).move_to(pt(0, 1.1))
        self.play(m.Write(e1))
        self.next_slide()
        e2 = tex(r"\kappa T\,", r"e^{\kappa T s}", r"\vec w + ", r"e^{\kappa T s}", r"\vec w\,' = T\,D\vec F\,", r"e^{\kappa T s}", r"\vec w", font_size=40).move_to(pt(0, -0.1))
        l2 = prose("product rule on the left, substitution on the right", font_size=22, color=MUTED).next_to(e2, m.DOWN, buff=0.15)
        self.play(m.Write(e2), m.FadeIn(l2), run_time=1.6)
        self.next_slide()
        strikes = m.VGroup(*[m.Line(e2[i].get_corner(m.DL), e2[i].get_corner(m.UR), color=RED, stroke_width=5) for i in (1, 3, 5)])
        why = prose("the same non-zero factor on every term: divide it out", font_size=22, color=RED)
        why.move_to(l2)
        self.play(m.Create(strikes), m.FadeOut(l2), m.FadeIn(why))
        self.next_slide("cancel")
        e3 = tex(r"\vec w\,' = T\big(D\vec F(\vec u)\,\vec w - \kappa\,\vec w\big)", font_size=44, color=YELLOW).move_to(pt(0, -1.6))
        self.play(m.Write(e3))
        e4 = prose("together with $\\vec u\\,' = T\\vec F(\\vec u)$, $\\vec u(1) = \\vec u(0)$, $\\vec w(1) = \\vec w(0)$, phase condition, $\\lvert\\vec w(0)\\rvert = 1$",
                   font_size=24, width=12.6).move_to(pt(0, -2.6))
        e5 = prose("unknowns: the functions $\\vec u, \\vec w$ and the numbers $T, \\kappa$;  then $\\mu = e^{\\kappa T}$", font_size=24, color=MUTED, width=12.6).next_to(e4, m.DOWN, buff=0.2)
        self.play(m.FadeIn(e4), m.FadeIn(e5))
        self.wait(HOLD)
        self.next_slide("log formulation")
        self.clear()

    def counting(self):
        h = heading("Is it well posed? Count unknowns and conditions")
        self.play(m.FadeIn(h))
        left = m.VGroup(
            m.Text("unknowns", font_size=28, weight=m.BOLD, color=TEAL),
            m.VGroup(tex(r"\vec u(s),\ \vec w(s)", font_size=36), prose("4 first-order ODEs", font_size=22, color=MUTED)).arrange(m.RIGHT, buff=0.3),
            m.VGroup(tex(r"T,\ \kappa", font_size=36), prose("2 unknown numbers", font_size=22, color=MUTED)).arrange(m.RIGHT, buff=0.3),
            tex(r"4 + 2 = 6\ \text{conditions needed}", font_size=36, color=TEAL),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.45).move_to(pt(-3.5, 0.3))
        right_rows = [
            (r"\vec u(1) = \vec u(0)", "2"),
            (r"\vec w(1) = \vec w(0)", "2"),
            (r"f_1(\vec u(0)) = 0", "1"),
            (r"\lvert\vec w(0)\rvert = 1", "1"),
        ]
        right = m.VGroup(m.Text("conditions", font_size=28, weight=m.BOLD, color=ORANGE))
        for f, c in right_rows:
            right.add(m.VGroup(tex(f, font_size=36), tex(r"\to " + c, font_size=36, color=ORANGE)).arrange(m.RIGHT, buff=0.3))
        right.add(tex(r"2+2+1+1 = 6", font_size=36, color=ORANGE))
        right.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.35).move_to(pt(3.4, 0.3))
        for r in left:
            self.play(m.FadeIn(r, shift=m.RIGHT * 0.2), run_time=0.6)
        self.next_slide()
        for r in right:
            self.play(m.FadeIn(r, shift=m.LEFT * 0.2), run_time=0.6)
        ok = prose("equal: the problem is well posed  (the normalisation also rules out $\\vec w \\equiv 0$)", font_size=24, color=YELLOW, width=12.6).to_edge(m.DOWN, buff=0.45)
        self.play(m.FadeIn(ok))
        self.wait(BEAT)
        self.next_slide("count")
        self.clear()
        # results
        h = heading("Solved by collocation: the paper's numbers")
        self.play(m.FadeIn(h))
        rows = [
            (r"T_\Gamma = 11.846275", "XPP periodic orbit: 11.846275; paper: 11.8463"),
            (r"\ln\mu = \kappa T = -46.46603", "Liouville, integral of div F: $-46.46606$"),
            (r"\mu = 6.6\times 10^{-21}", "XPP's Floquet routine: \"numerically zero\""),
            (r"\vec w(0) = (-1,\ 1.37\times 10^{-4})", "eigenvector of the monodromy matrix: the same"),
        ]
        g = m.VGroup(*[m.VGroup(tex(a, font_size=36), prose(b, font_size=22, color=MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12) for a, b in rows])
        g.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.4).move_to(pt(0, -0.3))
        for r in g:
            self.play(m.FadeIn(r, shift=m.RIGHT * 0.2), run_time=0.6)
        self.wait(HOLD)
        self.next_slide("results")
        self.clear()

    def bundle_picture(self):
        d = data("ch09_bundle"); s, u, w = d["s"], d["u"], d["w"]
        h = heading("40 stable directions, equally spaced in phase")
        ax = eplane((-90, 60, 30), (0.3, 0.8, 0.1), width=7.4, height=5.4, at=pt(-2.5, -0.5), xlabel="V", ylabel="n")
        cyc = curve(ax, u.T, color=MUTED, width=3, step=2)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), run_time=1)
        segs = m.VGroup()
        for j in range(40):
            p = np.array([np.interp(j / 40, s, u[0]), np.interp(j / 40, s, u[1])])
            ww = np.array([np.interp(j / 40, s, w[0]), np.interp(j / 40, s, w[1])])
            dv = screen_dir(ax, p, ww, 0.3)
            a = ax.c2p(*p)
            segs.add(m.VGroup(m.Line(a - dv, a + dv, color=ORANGE, stroke_width=4), m.Dot(a, radius=0.04, color=INK)))
        self.play(m.LaggedStart(*[m.FadeIn(sg) for sg in segs], lag_ratio=0.08), run_time=3.5)
        t1 = prose("they crowd on the slow left branch:\nequal phase steps, small distances", font_size=24).move_to(pt(4.4, 1.2))
        t2 = prose("horizontal at the spike,\nfar from horizontal in recovery", font_size=24, color=MUTED).move_to(pt(4.4, -0.6))
        t3 = prose("$L_\\eta(\\vec\\gamma) = \\vec\\gamma + \\delta\\vec w$, $\\lvert\\delta\\rvert \\le \\eta$:\nthe local piece to be extended", font_size=24, color=ORANGE).move_to(pt(4.4, -2.3))
        self.play(m.FadeIn(t1)); self.play(m.FadeIn(t2)); self.play(m.FadeIn(t3))
        self.wait(HOLD)
        self.next_slide("bundle")
        self.clear()

    def backward_neuron(self):
        d = data("rhh_cycle"); X, T = d["X"], float(d["T"])
        runs = backward_runs(X, T, [0, len(X) // 3, 2 * len(X) // 3])
        cols = [MAGENTA, BLUE, TEAL]
        h = heading("Backward integration cannot even stay on the cycle")
        ax = eplane((-90, 60, 30), (0.3, 0.8, 0.1), width=7.2, height=5.4, at=pt(-2.6, -0.5), xlabel="V", ylabel="n")
        cyc = curve(ax, X, color=INK, width=3, step=40)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), run_time=1)
        starts = m.VGroup(*[dot(ax, Y[0], color=c, r=0.11) for (tt, Y), c in zip(runs, cols)])
        txt = prose("three runs started exactly on $\\Gamma$\n(as exactly as double precision allows),\nintegrated backwards in time", font_size=24).move_to(pt(4.2, 1.9))
        self.play(m.FadeIn(starts), m.FadeIn(txt))
        self.next_slide("start on cycle")
        tau = m.ValueTracker(0.0)

        def path(j):
            tt, Y = runs[j]
            n = max(2, int(np.searchsorted(tt, tau.get_value())))
            return curve(ax, Y[:n], color=cols[j], width=3)

        def heads():
            g = m.VGroup()
            for j, (tt, Y) in enumerate(runs):
                n = min(len(tt) - 1, int(np.searchsorted(tt, tau.get_value())))
                g.add(dot(ax, Y[n], color=cols[j], r=0.08))
            return g
        paths = [m.always_redraw(lambda j=j: path(j)) for j in range(3)]
        hd = m.always_redraw(heads)
        tl = prose("backward time", font_size=24, color=MUTED).move_to(pt(3.4, 0.3))
        tn = m.DecimalNumber(0.0, num_decimal_places=2, font_size=36).next_to(tl, m.RIGHT, buff=0.3)
        tn.add_updater(lambda mob: mob.set_value(tau.get_value()))
        self.add(*paths, hd)
        self.play(m.FadeIn(tl), m.FadeIn(tn))
        self.play(tau.animate.set_value(T), run_time=8, rate_func=m.linear)
        tn.clear_updaters()
        res = tex(r"10^{-16}\times 1.5\times 10^{20} \approx 10^{4}", font_size=38, color=YELLOW).move_to(pt(3.9, -1.0))
        why = prose("backwards, the attracting cycle repels\nwith factor $1/\\mu$ per period:\nrounding error outgrows the whole cycle", font_size=24, color=MUTED).next_to(res, m.DOWN, buff=0.3)
        self.play(m.Write(res), m.FadeIn(why))
        self.wait(HOLD)
        self.next_slide("falls off")
        self.clear()

    def sensitivity(self):
        rows = data("ch09_sensitivity")["rows"]
        h = heading("Separations explode by the factor 1/λ")
        ax = eplane((-16, -2, 2), (-14, 22, 4), width=6.6, height=5.4, at=pt(-2.7, -0.55), xlabel=r"\log_{10}\delta", ylabel=r"\log_{10}\text{sep}")
        self.play(m.FadeIn(h), m.Create(ax), run_time=1)
        expl = prose("two starts $\\delta$ apart, one backward period:\nseparation $\\approx \\delta/\\lambda$", font_size=24).move_to(pt(4.0, 2.4))
        self.play(m.FadeIn(expl))
        one = m.DashedLine(ax.c2p(-16, 0), ax.c2p(-2, 0), color=MUTED, stroke_width=2)
        onel = prose("size of the cycle", font_size=22, color=MUTED).next_to(ax.c2p(-2, 0), m.UP, buff=0.1).shift(m.LEFT * 1.2)
        self.play(m.Create(one), m.FadeIn(onel))
        self.next_slide()
        specs = [(rows[0], TEAL, "van der Pol $\\varepsilon = 1$: $1/\\lambda \\approx 10^{3}$"),
                 (rows[1], ORANGE, "van der Pol $\\varepsilon = 0.3$: $1/\\lambda \\approx 10^{7}$")]
        labels = m.VGroup()
        for (r, col, txt) in specs:
            dl, sp = np.asarray(r[2], float), np.asarray(r[3], float)
            ok = np.isfinite(sp) & (sp > 0)
            pts = np.column_stack([np.log10(dl[ok]), np.log10(sp[ok])])
            c = curve(ax, pts, color=col, width=4)
            dots = m.VGroup(*[dot(ax, p, color=col, r=0.05) for p in pts])
            lab = prose(txt, font_size=22, color=col)
            labels.add(lab)
            self.play(m.Create(c), m.FadeIn(dots), run_time=2)
            self.add(lab.move_to(pt(4.0, 1.2 - 0.55 * (len(labels) - 1))))
            self.play(m.FadeIn(lab))
            self.next_slide()
        lam = 6.6e-21
        pr = m.DashedLine(ax.c2p(-16, -16 - np.log10(lam)), ax.c2p(-2, -2 - np.log10(lam)), color=MAGENTA, stroke_width=4)
        labr = prose("neuron: $\\delta/\\lambda$ with $1/\\lambda \\approx 1.5\\times 10^{20}$\n(every real run blows up first)", font_size=22, color=MAGENTA).move_to(pt(4.0, 0.0))
        self.play(m.Create(pr), m.FadeIn(labr), run_time=1.5)
        fl = m.DashedLine(ax.c2p(-16, -14), ax.c2p(-16, 22), color=YELLOW, stroke_width=3)
        why = prose("slope 1: the map is linear for tiny $\\delta$;\nshift up by $\\log_{10}(1/\\lambda)$.\nEven $\\delta = 10^{-16}$, the best double\nprecision can do, lands far above the cycle", font_size=22, color=YELLOW).move_to(pt(4.0, -2.0))
        self.play(m.Create(fl), m.FadeIn(why))
        self.wait(HOLD)
        self.next_slide("sensitivity")
        self.clear()

    def the_cure(self):
        h = heading("The cure: solve for whole orbit segments")
        self.play(m.FadeIn(h))
        # left: initial-value problem, a fan of trajectories diverging from one point
        lt = m.Text("initial-value problem", font_size=28, weight=m.BOLD, color=MAGENTA).move_to(pt(-3.6, 2.4))
        o = pt(-6.0, -0.2)
        fan = m.VGroup(*[m.ParametricFunction(lambda t, c=c: o + np.array([t, c * 0.009 * (np.exp(1.1 * t) - 1), 0]),
                                              t_range=[0, 4.6], color=MAGENTA, stroke_width=3) for c in np.linspace(-1, 1, 7)])
        src = m.Dot(o, radius=0.1, color=MAGENTA)
        ll = prose("everything hangs on one start point;\nits rounding error is magnified", font_size=22, color=MUTED).move_to(pt(-3.6, -2.4))
        self.play(m.FadeIn(lt), m.FadeIn(src))
        self.play(m.LaggedStart(*[m.Create(f) for f in fan], lag_ratio=0.1), run_time=2)
        self.play(m.FadeIn(ll))
        self.next_slide("ivp")
        # right: boundary-value problem, a whole segment with conditions at both ends
        rt = m.Text("boundary-value problem", font_size=28, weight=m.BOLD, color=TEAL).move_to(pt(3.6, 2.4))
        seg = m.ParametricFunction(lambda t: pt(1.2 + t, -0.2 + 0.9 * np.sin(1.3 * t) * np.exp(-0.2 * t)), t_range=[0, 4.8], color=TEAL, stroke_width=4)
        mesh = m.VGroup(*[m.Dot(seg.point_from_proportion(a), radius=0.05, color=INK) for a in np.linspace(0, 1, 17)])
        ends = m.VGroup(m.Square(0.3, color=ORANGE, stroke_width=3).move_to(seg.get_start()),
                        m.Square(0.3, color=ORANGE, stroke_width=3).move_to(seg.get_end()))
        rl = prose("conditions at both ends; the solver\nadjusts every point of the segment at once", font_size=22, color=MUTED).move_to(pt(3.6, -2.4))
        self.play(m.FadeIn(rt), m.Create(seg), m.FadeIn(ends), run_time=1.5)
        self.play(m.LaggedStart(*[m.FadeIn(p) for p in mesh], lag_ratio=0.05), m.FadeIn(rl))
        self.next_slide("bvp")
        q = prose("\"The accuracy of the method uses information of the flow along entire orbit segments.\"",
                  font_size=24, color=YELLOW, width=12.6).to_edge(m.DOWN, buff=0.35)
        self.play(m.FadeIn(q))
        self.wait(HOLD)
        self.next_slide("quote")
        self.clear()
        self.bullets("The plan for Chapters 11 and 11", [
            "an isochron = a one-parameter family of orbit segments that each end on $L_\\eta$",
            "each segment solved as a boundary-value problem (collocation)",
            "move along the family by continuation: points spaced by arclength,\nwhich fixes the unevenness of backward integration",
        ], font_size=28)
