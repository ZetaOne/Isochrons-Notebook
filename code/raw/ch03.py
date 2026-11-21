"""Deck for Chapter 3: Stability of a limit cycle (Poincare map, time-T map, Floquet multiplier)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import manimgx as m
from deckkit import *  # noqa: F401,F403


def strike(mob, color=RED):
    return m.Line(mob.get_corner(m.DL), mob.get_corner(m.UR), color=color, stroke_width=5)


def radial_r(r0, t, a=1.0):
    """Exact solution of r' = a r (1 - r^2)."""
    return 1.0 / np.sqrt(1.0 + (1.0 / r0 ** 2 - 1.0) * np.exp(-2 * a * t))


def unit(ax):
    return ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0]


class Deck03(Deck):
    chapter = "Chapter 3"
    title = "Stability of a Limit Cycle"
    subtitle = "Poincare map, time-T map and the Floquet multiplier"

    def construct(self):
        self.title_card()
        self.two_kinds()
        self.section()
        self.linear_return()
        self.solvable()
        self.cobweb()
        self.time_T()
        self.monodromy()
        self.eigen_one()
        self.same_everywhere()
        self.liouville()
        self.check_radial()
        self.divergence()
        self.table()
        self.left_eig()
        self.recap(["Along $\\Gamma$: perturbations persist as phase shifts. Across: they decay",
                    "Poincare map $P$ on a section: $\\delta_{n+1} = \\lambda\\,\\delta_n$",
                    "$\\Phi_T$ fixes every point of $\\Gamma$; $M = D\\Phi_T(\\vec\\gamma)$ has eigenvalues $1$ and $\\lambda$",
                    "Eigenvector $\\vec F(\\vec\\gamma)$ for 1, transverse $\\vec v_s$ for $\\lambda$",
                    "$\\lambda = \\exp\\int_0^T \\nabla\\cdot\\vec F\\,dt$: about $6.6\\times 10^{-21}$ for the neuron"])

    # ------------------------------------------------------------------
    def two_kinds(self):
        h = heading("Two kinds of perturbation")
        ax = plane((-1.8, 1.8, 1), (-1.8, 1.8, 1), width=5.8, height=5.8, at=pt(-3.4, -0.5), ticks=False)
        circ = m.Circle(radius=unit(ax), color=MUTED, stroke_width=3).move_to(ax.c2p(0, 0))
        eq = tex(r"\dot r = a\,r(1-r^2),\qquad \dot\theta = 1", font_size=38).move_to(pt(3.5, 2.4))
        self.play(m.FadeIn(h), m.Create(ax), m.Create(circ), m.Write(eq), run_time=1.2)
        a = 0.25
        k = m.ValueTracker(0.0)

        def P(r0, th0):
            t = k.get_value()
            r = radial_r(r0, t, a)
            return ax.c2p(r * np.cos(th0 + t), r * np.sin(th0 + t))

        ref = m.always_redraw(lambda: m.Dot(P(1.0, 0.0), radius=0.11, color=INK))
        tan = m.always_redraw(lambda: m.Dot(P(1.0, 0.55), radius=0.11, color=TEAL))
        tra = m.always_redraw(lambda: m.Dot(P(1.6, 0.0), radius=0.11, color=MAGENTA))
        trail = m.always_redraw(lambda: curve(ax, np.array([[radial_r(1.6, s, a) * np.cos(s), radial_r(1.6, s, a) * np.sin(s)]
                                                            for s in np.linspace(0, k.get_value(), 150)]), color=MAGENTA, width=2))
        leg = m.VGroup(
            prose("white: a state on $\\Gamma$", font_size=24),
            prose("teal: nudged along $\\Gamma$", font_size=24, color=TEAL),
            prose("magenta: nudged across $\\Gamma$", font_size=24, color=MAGENTA),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.2).move_to(pt(3.5, 1.0))
        self.play(m.FadeIn(ref), m.FadeIn(tan), m.FadeIn(tra), m.FadeIn(leg))
        self.add(trail)
        self.next_slide("nudges")
        self.play(k.animate.set_value(5 * np.pi), run_time=10, rate_func=m.linear)
        r1 = prose("along: nothing pulls it back, it stays\nahead forever, a neutral phase shift", font_size=24, color=TEAL).move_to(pt(3.5, -0.8))
        r2 = prose("across: it decays back to $\\Gamma$", font_size=24, color=MAGENTA).next_to(r1, m.DOWN, buff=0.3).align_to(r1, m.LEFT)
        r3 = prose("stability of a cycle can only mean\nstability across it: orbital stability", font_size=24, color=YELLOW).next_to(r2, m.DOWN, buff=0.3).align_to(r1, m.LEFT)
        self.play(m.FadeIn(r1))
        self.play(m.FadeIn(r2))
        self.play(m.FadeIn(r3))
        self.wait(HOLD)
        self.next_slide("orbital stability")
        self.clear()

    def section(self):
        d = data("ch03_section")
        X, cyc, cr = d["X"], d["cyc"], d["cr"]
        h = heading("The Poincare map: strobe the cycle")
        ax = plane((-3, 3, 1), (-3, 3, 1), width=6.0, height=6.0, at=pt(-3.3, -0.45), ticks=False)
        sig = m.Line(ax.c2p(0, 0.05), ax.c2p(0, 2.9), color=ORANGE, stroke_width=6)
        sl = tex(r"\Sigma", font_size=40, color=ORANGE).next_to(sig, m.LEFT, buff=0.1).shift(m.UP * 1.0)
        self.play(m.FadeIn(h), m.Create(ax), run_time=0.8)
        self.play(m.Create(sig), m.FadeIn(sl))
        t1 = prose("van der Pol ($\\varepsilon = 1$), section $\\Sigma = \\{x = 0,\\ y>0\\}$.\nStart on $\\Sigma$, follow the flow until it\nhits $\\Sigma$ again: that is $P(y)$.", font_size=22).move_to(pt(3.6, 2.2))
        self.play(m.FadeIn(t1))
        self.next_slide("section")
        k = m.ValueTracker(1)
        tr = m.always_redraw(lambda: curve(ax, X[: max(2, int(k.get_value()))], color=MAGENTA, width=2.5, step=2))
        hd = m.always_redraw(lambda: dot(ax, X[max(0, int(k.get_value()) - 1)], color=MAGENTA, r=0.09))
        self.add(tr, hd)
        # crossing indices
        idx = [int(np.argmin(np.sum((X - c) ** 2, axis=1))) for c in cr]
        labels = ["2.0828", "2.17262", "2.172707", "2.172713", "2.172708"]
        rows = m.VGroup()
        last = 0
        for n, (i, s) in enumerate(zip(idx[:5], labels)):
            if i > last:
                self.play(k.animate.set_value(i + 1), run_time=1.6 if n else 1.0, rate_func=m.linear)
            last = i
            p = m.Dot(ax.c2p(*cr[n]), radius=0.1, color=YELLOW)
            r = tex(r"y_{%d} = %s" % (n, s), font_size=34, color=YELLOW)
            rows.add(r)
            rows.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.15).move_to(pt(2.0, 0.9), aligned_edge=m.UP + m.LEFT)
            self.play(m.FadeIn(p, scale=2), m.FadeIn(r), run_time=0.5)
        res = prose("converging to $y^* \\approx 2.17271$,\nwhere $\\Gamma$ crosses $\\Sigma$:\na fixed point of $P$", font_size=24, color=YELLOW).move_to(pt(3.6, -2.9))
        self.play(m.FadeIn(res))
        self.wait(HOLD)
        self.next_slide("crossings")
        self.clear()

    def linear_return(self):
        h = heading("Linearise the return map")
        w = prose("a state slightly off $\\Gamma$ crosses $\\Sigma$ at $y^* + \\delta_0$, then $y^* + \\delta_1$, ...", font_size=26).move_to(pt(0, 2.3))
        self.play(m.FadeIn(h), m.FadeIn(w))
        e1 = tex(r"y^*", r"+ \delta_{n+1}", r"=", r"P(y^* + \delta_n)", r"\approx", r"P(y^*)", r"+ P'(y^*)\,\delta_n", font_size=44).move_to(pt(0, 1.1))
        self.play(m.Write(e1), run_time=1.4)
        self.next_slide()
        s1 = strike(e1[0])
        s2 = strike(e1[5])
        why = prose("$P(y^*) = y^*$: the cycle comes back to itself, so $y^*$ cancels", font_size=24, color=RED).next_to(e1, m.DOWN, buff=0.35)
        self.play(m.Create(s1), m.Create(s2), m.FadeIn(why))
        self.next_slide("y* cancels")
        e2 = tex(r"\delta_{n+1} = P'(y^*)\,\delta_n\ \ \Longrightarrow\ \ \delta_n = \lambda^n\,\delta_0,\qquad \lambda = P'(y^*)", font_size=42, color=YELLOW).move_to(pt(0, -0.9))
        self.play(m.Write(e2))
        self.next_slide()
        e3 = prose("$\\lvert\\lambda\\rvert < 1$: the distance to the cycle shrinks by $\\lambda$ every turn.\n$\\lambda$ is the Floquet multiplier.", font_size=26).move_to(pt(0, -2.4))
        self.play(m.FadeIn(e3))
        self.wait(HOLD)
        self.next_slide("floquet multiplier")
        self.clear()

    def solvable(self):
        h = heading("An exactly solvable return map")
        e0 = tex(r"\dot r = a\,r(1-r^2),\quad \dot\theta = 1,\quad \Sigma = \{\theta = 0\},\quad T = 2\pi", font_size=38).move_to(pt(0, 2.45))
        self.play(m.FadeIn(h), m.Write(e0))
        self.next_slide()
        e1 = tex(r"u = \frac{1}{r^2}:\qquad \dot u", r"=", r"-\frac{2\dot r}{r^3}", r"=", r"-2a\,\frac{1-r^2}{r^2}", r"=", r"-2a(u-1)", font_size=40).move_to(pt(0, 1.3))
        self.play(m.Write(e1[:3]))
        self.next_slide()
        self.play(m.Write(e1[3:5]))
        self.play(m.Write(e1[5:]))
        w1 = prose("a linear equation: $u-1$ decays like $e^{-2at}$", font_size=24, color=MUTED).next_to(e1, m.DOWN, buff=0.25)
        self.play(m.FadeIn(w1))
        self.next_slide()
        e2 = tex(r"\frac{1}{r(t)^2} - 1 = \Big(\frac{1}{r_0^2} - 1\Big)e^{-2at}\ \ \Longrightarrow\ \ P(r) = \Big(1 + (r^{-2}-1)\,e^{-2aT}\Big)^{-1/2}", font_size=38, color=YELLOW).move_to(pt(0, -0.35))
        self.play(m.Write(e2), run_time=1.6)
        self.next_slide()
        w2 = prose("quicker, near $r = 1$: put $r = 1+\\rho$", font_size=24, color=MUTED).move_to(pt(0, -1.45))
        e3 = tex(r"\dot\rho", r"=", r"a(1+\rho)\big(-2\rho", r"-\rho^2", r"\big)", r"=", r"-2a\rho", r"-2a\rho^2", r"-a\rho^3", font_size=40).move_to(pt(0, -2.3))
        self.play(m.FadeOut(w1), m.FadeIn(w2), m.Write(e3))
        self.next_slide()
        ss = [strike(e3[i]) for i in (3, 7, 8)]
        zero = tex("0", font_size=36, color=RED).next_to(m.VGroup(e3[7], e3[8]), m.DOWN, buff=0.15)
        self.play(*[m.Create(s) for s in ss], m.FadeIn(zero))
        w3 = prose("$\\rho^2$ and $\\rho^3$ are negligible for small $\\rho$: $\\rho \\to e^{-2aT}\\rho$, so $P'(1) = e^{-2aT}$", font_size=24, color=YELLOW).to_edge(m.DOWN, buff=0.15)
        self.play(m.FadeIn(w3))
        self.wait(HOLD)
        self.next_slide("P'(1)")
        self.clear()

    def cobweb(self):
        a, T = 0.12, 2 * np.pi
        lam = np.exp(-2 * a * T)
        P = lambda r: (1 + (r ** -2 - 1) * lam) ** -0.5
        h = heading("Cobwebs: iterate the return map")
        ax = plane((0, 2, 0.5), (0, 2, 0.5), width=4.8, height=4.8, at=pt(-3.2, -0.55), xlabel=r"r_n", ylabel=r"r_{n+1}", ticks=False)
        ax.add(tex("1", font_size=32, color=MUTED).next_to(ax.c2p(1, 0), m.DOWN, buff=0.12), tex("1", font_size=32, color=MUTED).next_to(ax.c2p(0, 1), m.LEFT, buff=0.12))
        diag = m.DashedLine(ax.c2p(0, 0), ax.c2p(2, 2), color=MUTED, stroke_width=2)
        g = ax.plot(P, x_range=(0.05, 2), color=TEAL, stroke_width=4)
        axR = plane((-2, 2, 1), (-2, 2, 1), width=4.8, height=4.8, at=pt(3.4, -0.55), ticks=False)
        sig = m.Line(axR.c2p(0, 0), axR.c2p(2, 0), color=ORANGE, stroke_width=5)
        circ = m.Circle(radius=unit(axR), color=MUTED, stroke_width=2).move_to(axR.c2p(0, 0))
        eq = tex(r"a = 0.12:\ \ \lambda = e^{-2aT} \approx 0.22", font_size=36).move_to(pt(0, 2.5))
        self.play(m.FadeIn(h), m.Create(ax), m.Create(diag), m.Create(axR), m.Create(circ), m.FadeIn(eq), run_time=1.2)
        self.play(m.Create(g), m.Create(sig))
        self.next_slide("return map")
        for r0, col in ((0.25, YELLOW), (1.95, MAGENTA)):
            r = r0
            st = m.Dot(ax.c2p(r, 0), radius=0.07, color=col)
            self.play(m.FadeIn(st), m.FadeIn(m.Dot(axR.c2p(r, 0), radius=0.08, color=col)), run_time=0.3)
            x0 = r
            for n in range(4):
                r1 = P(r)
                web = m.VMobject(color=col, stroke_width=3)
                web.set_points_as_corners([ax.c2p(r, r if n else 0), ax.c2p(r, r1), ax.c2p(r1, r1)])
                ts = np.linspace(0, T, 160)
                rr = radial_r(r, ts, a)
                sp = curve(axR, np.column_stack([rr * np.cos(ts), rr * np.sin(ts)]), color=col, width=2.5)
                self.play(m.Create(web), m.Create(sp), run_time=1.2, rate_func=m.linear)
                self.add(m.Dot(axR.c2p(r1, 0), radius=0.08, color=col))
                r = r1
            self.next_slide()
        w = prose("from inside and outside the crossings converge to the fixed point $r = 1$,\nthe distance shrinking by the slope $\\lambda$ there on every turn", font_size=22, color=MUTED, width=6.4).to_corner(m.DR, buff=0.15)
        self.play(m.FadeIn(w))
        self.wait(HOLD)
        self.next_slide("cobweb")
        self.clear()

    def time_T(self):
        h = heading("The time-T map: every point of the cycle is fixed")
        ax = plane((-2, 2, 1), (-2, 2, 1), width=5.8, height=5.8, at=pt(-3.4, -0.5), ticks=False)
        circ = m.Circle(radius=unit(ax), color=MUTED, stroke_width=3).move_to(ax.c2p(0, 0))
        self.play(m.FadeIn(h), m.Create(ax), m.Create(circ), run_time=1)
        e = tex(r"\Phi_T:\ \mathbb{R}^2 \to \mathbb{R}^2", font_size=40).move_to(pt(3.5, 2.4))
        w = prose("snapshot the plane, flow for exactly one\nperiod, snapshot again, compare", font_size=24).next_to(e, m.DOWN, buff=0.25)
        self.play(m.Write(e), m.FadeIn(w))
        a = 0.08
        angs = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        pts = [(1.0, th, YELLOW) for th in angs] + [(r0, th, MAGENTA) for th in angs[::2] for r0 in (0.35, 1.7)]
        ghosts = m.VGroup(*[m.Dot(ax.c2p(r0 * np.cos(th), r0 * np.sin(th)), radius=0.06, color=c, fill_opacity=0.35) for r0, th, c in pts])
        k = m.ValueTracker(0.0)
        moving = m.always_redraw(lambda: m.VGroup(*[m.Dot(ax.c2p(radial_r(r0, k.get_value(), a) * np.cos(th + k.get_value()),
                                                                  radial_r(r0, k.get_value(), a) * np.sin(th + k.get_value())),
                                                          radius=0.09, color=c) for r0, th, c in pts]))
        clk = m.always_redraw(lambda: tex(r"t = %.2f\,T" % (k.get_value() / (2 * np.pi)), font_size=38).move_to(pt(3.5, 0.6)))
        self.add(ghosts, moving, clk)
        self.next_slide()
        self.play(k.animate.set_value(2 * np.pi), run_time=7, rate_func=m.linear)
        e2 = tex(r"\Phi_T(\vec\gamma) = \vec\gamma\quad\text{for every } \vec\gamma\in\Gamma", font_size=38, color=YELLOW).move_to(pt(3.5, -0.5))
        w2 = prose("yellow: back exactly where they started.\nmagenta: closer to $\\Gamma$", font_size=24).next_to(e2, m.DOWN, buff=0.25)
        self.play(m.Write(e2), m.FadeIn(w2))
        self.next_slide()
        e3 = tex(r"\Phi_T(\vec\gamma + \vec\xi) \approx \vec\gamma + M\vec\xi,\quad M = D\Phi_T(\vec\gamma)", font_size=36, color=TEAL).move_to(pt(3.5, -2.3))
        w3 = prose("the monodromy matrix", font_size=22, color=TEAL).next_to(e3, m.DOWN, buff=0.15)
        self.play(m.Write(e3), m.FadeIn(w3))
        self.wait(HOLD)
        self.next_slide("time-T map")
        self.clear()

    def monodromy(self):
        d = data("ch03_monodromy")
        paths, cyc, g0, vs, vt, lam = d["paths"], d["cyc"], d["g0"], d["vs"], d["vt"], float(d["lam"])
        nP, nT = paths.shape[0], paths.shape[1]
        h = heading("What M does: a circle flattened in one period")
        ax = plane((-3, 3, 1), (-3, 3, 1), width=6.0, height=6.0, at=pt(-3.3, -0.45), ticks=False)
        cy = curve(ax, cyc, color=MUTED, width=2.5, step=20)
        star = m.Star(n=5, outer_radius=0.15, color=ORANGE, fill_opacity=1).move_to(ax.c2p(*g0))
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cy), m.FadeIn(star), run_time=1.2)
        k = m.ValueTracker(0)
        blob = m.always_redraw(lambda: m.Polygon(*[ax.c2p(*paths[j, int(k.get_value())]) for j in range(nP)],
                                                 color=MAGENTA, fill_color=MAGENTA, fill_opacity=0.5, stroke_width=2))
        clk = m.always_redraw(lambda: tex(r"t = %.2f\,T" % (k.get_value() / (nT - 1)), font_size=38, color=MAGENTA).move_to(pt(3.6, 2.3)))
        t1 = prose("48 states on a circle around $\\vec\\gamma_0$,\ncarried by the flow for one period", font_size=24).move_to(pt(3.6, 1.2))
        self.play(m.FadeIn(blob), m.FadeIn(clk), m.FadeIn(t1))
        self.next_slide("circle")
        self.play(k.animate.set_value(nT - 1), run_time=9, rate_func=m.linear)
        self.next_slide("one period")
        # zoomed comparison
        self.play(m.FadeOut(t1))
        zx = plane((g0[0] - 0.5, g0[0] + 0.5, 0.5), (g0[1] - 0.5, g0[1] + 0.5, 0.5), width=3.6, height=3.6, at=pt(3.6, -0.6), ticks=False, xlabel="", ylabel="")
        frame = m.SurroundingRectangle(zx, color=MUTED, buff=0.05, stroke_width=2)
        zc = curve(zx, cyc[np.all(np.abs(cyc - g0) < 0.5, axis=1)], color=MUTED, width=2)
        before = m.Polygon(*[zx.c2p(*paths[j, 0]) for j in range(nP)], color=ORANGE, stroke_width=3)
        after = m.VGroup(*[m.Dot(zx.c2p(*paths[j, -1]), radius=0.04, color=MAGENTA) for j in range(nP)])
        zl = prose("zoom near $\\vec\\gamma_0$", font_size=22, color=MUTED).next_to(frame, m.UP, buff=0.1)
        self.play(m.FadeIn(frame), m.FadeIn(zc), m.Create(before), m.FadeIn(zl))
        self.play(m.TransformFromCopy(before, after), run_time=1.5)
        c0 = zx.c2p(*g0)
        at = m.Arrow(c0, c0 + 1.3 * np.array([vt[0], vt[1], 0]), buff=0, color=TEAL, stroke_width=5)
        as_ = m.Arrow(c0, c0 + 1.3 * np.array([vs[0], vs[1], 0]), buff=0, color=BLUE, stroke_width=5)
        self.play(m.GrowArrow(at), m.GrowArrow(as_))
        w = prose("teal: tangent $\\vec F$, eigenvalue $1$, length kept\nblue: $\\vec v_s$, eigenvalue $\\lambda = 8.6\\times 10^{-4}$, crushed", font_size=22).to_edge(m.DOWN, buff=0.12).shift(m.RIGHT * 2.6)
        self.play(m.FadeIn(w))
        self.wait(HOLD)
        self.next_slide("eigen-directions")
        self.clear()

    def eigen_one(self):
        h = heading("Why one eigenvalue is exactly 1")
        w0 = prose("move $\\vec\\gamma$ a little along $\\Gamma$: $\\Phi_s(\\vec\\gamma)$ is on $\\Gamma$ too, so it is also fixed", font_size=26).move_to(pt(0, 2.4))
        self.play(m.FadeIn(h), m.FadeIn(w0))
        e1 = tex(r"\Phi_T\big(\Phi_s(\vec\gamma)\big) = \Phi_s(\vec\gamma)\qquad\text{for all } s", font_size=44).move_to(pt(0, 1.3))
        self.play(m.Write(e1))
        self.next_slide()
        w1 = prose("differentiate in $s$ at $s = 0$, using $\\frac{d}{ds}\\Phi_s(\\vec\\gamma)\\big\\rvert_{s=0} = \\vec F(\\vec\\gamma)$ and the chain rule", font_size=24, color=MUTED).move_to(pt(0, 0.25))
        self.play(m.FadeIn(w1))
        e2 = tex(r"D\Phi_T(\vec\gamma)\,\vec F(\vec\gamma) = \vec F(\vec\gamma)", font_size=48, color=YELLOW).move_to(pt(0, -0.8))
        box = m.SurroundingRectangle(e2, color=YELLOW, buff=0.2)
        self.play(m.Write(e2), m.Create(box))
        self.next_slide("eigenvalue 1")
        w2 = prose("the tangent $\\vec F(\\vec\\gamma)$ is an eigenvector of $M$ with eigenvalue $1$:\nthe neutral phase-shift direction, now in symbols", font_size=24, color=TEAL).move_to(pt(0, -2.1))
        self.play(m.FadeIn(w2))
        self.next_slide()
        w3 = prose("the other eigenvalue $\\lambda$, along a transverse $\\vec v_s$, is the Floquet multiplier ($0<\\lambda<1$)", font_size=24, color=BLUE).to_edge(m.DOWN, buff=0.25)
        self.play(m.FadeIn(w3))
        self.wait(HOLD)
        self.next_slide("other eigenvalue")
        self.clear()

    def same_everywhere(self):
        d = data("ch03_monodromy")
        cyc = d["cyc"]
        h = heading("The same multiplier at every point of the cycle")
        e1 = tex(r"\vec\gamma' = \Phi_s(\vec\gamma),\qquad \Phi_T\circ\Phi_s = \Phi_s\circ\Phi_T", font_size=40).move_to(pt(0, 2.4))
        self.play(m.FadeIn(h), m.Write(e1))
        self.next_slide()
        e2 = tex(r"D\Phi_T(\vec\gamma') = D\Phi_s(\vec\gamma)\, D\Phi_T(\vec\gamma)\, D\Phi_s(\vec\gamma)^{-1}", font_size=40, color=YELLOW).move_to(pt(0, 1.3))
        w2 = prose("chain rule: the two matrices are similar, so they share eigenvalues $1$ and $\\lambda$", font_size=24, color=MUTED).next_to(e2, m.DOWN, buff=0.25)
        self.play(m.Write(e2), m.FadeIn(w2))
        self.next_slide()
        ax = plane((-3, 3, 1), (-3, 3, 1), width=3.9, height=3.9, at=pt(-3.6, -1.85), ticks=False, xlabel="", ylabel="")
        cy = curve(ax, cyc, color=MUTED, width=2.5, step=20)
        self.play(m.Create(ax), m.Create(cy))
        N = len(cyc)
        k = m.ValueTracker(0.0)

        def bundle():
            g = m.VGroup()
            for j in range(16):
                i = int(((j / 16 + k.get_value()) % 1.0) * (N - 1))
                p, q = cyc[i], cyc[(i + 5) % N] - cyc[i]
                tdir = q / np.hypot(*q)
                nrm = np.array([-tdir[1], tdir[0]])
                g.add(m.Line(ax.c2p(*(p - 0.35 * nrm)), ax.c2p(*(p + 0.35 * nrm)), color=BLUE, stroke_width=3))
            return g

        b = m.always_redraw(bundle)
        w3 = prose("but the eigenvectors move with the point:\nthe stable directions all round $\\Gamma$ form the\nlinear stable normal bundle, the tangent lines\nof the isochrons on $\\Gamma$ (Chapters 5 and 9).\n(sketch: lines drawn across $\\Gamma$)", font_size=24).move_to(pt(2.6, -1.8))
        self.play(m.FadeIn(b), m.FadeIn(w3))
        self.play(k.animate.set_value(1.0), run_time=5, rate_func=m.linear)
        self.wait(BEAT)
        self.next_slide("normal bundle")
        self.clear()

    def liouville(self):
        h = heading("Computing the multiplier: Liouville's formula")
        e1 = tex(r"\dot Y = D\vec F\big(\vec\gamma(t)\big)\,Y,\qquad Y(0) = I,\qquad M = Y(T)", font_size=40).move_to(pt(0, 2.4))
        w1 = prose("first variational equation: differentiate the flow in its initial condition", font_size=24, color=MUTED).next_to(e1, m.DOWN, buff=0.2)
        self.play(m.FadeIn(h), m.Write(e1), m.FadeIn(w1))
        self.next_slide()
        e2 = tex(r"\frac{d}{dt}\det Y = \text{tr}\big(D\vec F\big)\det Y\ \ \Longrightarrow\ \ \det Y(T) = \exp\int_0^T \text{tr}\,D\vec F\,dt", font_size=38).move_to(pt(0, 0.9))
        self.play(m.Write(e2), run_time=1.6)
        self.next_slide()
        e3 = tex(r"\det M = 1\cdot\lambda,\qquad \text{tr}\,D\vec F = \nabla\cdot\vec F", font_size=40).move_to(pt(0, -0.3))
        w3 = prose("determinant = product of the eigenvalues; the trace is the divergence", font_size=24, color=MUTED).next_to(e3, m.DOWN, buff=0.2)
        self.play(m.Write(e3), m.FadeIn(w3))
        self.next_slide()
        e4 = tex(r"\lambda = \exp\int_0^T \nabla\cdot\vec F\big(\vec\gamma(t)\big)\,dt", font_size=48, color=YELLOW).move_to(pt(0, -2.0))
        box = m.SurroundingRectangle(e4, color=YELLOW, buff=0.2)
        self.play(m.Write(e4), m.Create(box))
        w4 = prose("one period maps a tiny disc to a sliver of the same length and $\\lambda$ times the width:\nthe area is multiplied by $\\lambda$", font_size=22, color=MUTED).to_edge(m.DOWN, buff=0.12)
        self.play(m.FadeIn(w4))
        self.wait(HOLD)
        self.next_slide("liouville")
        self.clear()

    def check_radial(self):
        h = heading("Check on the radial isochron clock")
        e0 = tex(r"\vec F = \big(x(1-x^2-y^2) - y,\ \ y(1-x^2-y^2) + x\big)", font_size=40).move_to(pt(0, 2.4))
        self.play(m.FadeIn(h), m.Write(e0))
        self.next_slide()
        e1 = tex(r"\nabla\cdot\vec F", r"=", r"(1 - 3x^2 - y^2)", r"+", r"(1 - x^2 - 3y^2)", r"=", r"2 - 4r^2", font_size=42).move_to(pt(0, 1.1))
        self.play(m.Write(e1[:5]), run_time=1.4)
        w1 = prose("$\\partial/\\partial x$ of the first component, $\\partial/\\partial y$ of the second", font_size=24, color=MUTED).next_to(e1, m.DOWN, buff=0.25)
        self.play(m.FadeIn(w1))
        self.next_slide()
        self.play(m.Write(e1[5:]))
        self.next_slide()
        e2 = tex(r"r = 1:\quad \nabla\cdot\vec F = -2\ \ \Longrightarrow\ \ \lambda = e^{-2T}", font_size=44, color=YELLOW).move_to(pt(0, -0.8))
        self.play(m.Write(e2))
        w2 = prose("the same as the Poincare map with $a = 1$: $P'(1) = e^{-2aT}$", font_size=24, color=TEAL).next_to(e2, m.DOWN, buff=0.35)
        self.play(m.FadeIn(w2))
        w3 = prose("van der Pol, $\\varepsilon = 1$: integrating the divergence gives $\\lambda = 8.596957\\times 10^{-4}$,\nthe variational equation gives $8.596952\\times 10^{-4}$", font_size=22, color=MUTED).to_edge(m.DOWN, buff=0.25)
        self.play(m.FadeIn(w3))
        self.wait(HOLD)
        self.next_slide("check")
        self.clear()

    def divergence(self):
        d = data("ch03_monodromy")
        cyc, T = d["cyc"], float(d["T"])
        N = len(cyc)
        div = 1 - cyc[:, 0] ** 2
        t = np.linspace(0, T, N)
        I = np.concatenate([[0], np.cumsum(0.5 * (div[1:] + div[:-1]) * np.diff(t))])
        h = heading("Divergence along the cycle adds up to ln(lambda)")
        ax = plane((-3, 3, 1), (-3, 3, 1), width=5.0, height=5.0, at=pt(-4.2, -0.2), ticks=False)
        seg = m.VGroup()
        step = 40
        for a in range(0, N - 1, step):
            col = ORANGE if div[a] > 0 else BLUE
            seg.add(curve(ax, cyc[a:a + step + 1], color=col, width=5))
        gx = plane((0, 7, 1), (-4, 1, 1), width=6.4, height=4.0, at=pt(2.9, 0.1), xlabel="t", ylabel=r"\nabla\cdot\vec F", ticks=False)
        gx.add(tex("0", font_size=32, color=MUTED).next_to(gx.c2p(0, 0), m.LEFT, buff=0.12),
               tex("-3", font_size=32, color=MUTED).next_to(gx.c2p(0, -3), m.LEFT, buff=0.12))
        zero = m.DashedLine(gx.c2p(0, 0), gx.c2p(7, 0), color=MUTED, stroke_width=2)
        g = curve(gx, np.column_stack([t, div]), color=INK, width=3, step=10)
        eq = tex(r"\nabla\cdot\vec F = 1 - x^2", font_size=36).next_to(gx, m.DOWN, buff=0.35)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(seg), m.FadeIn(eq), run_time=1.4)
        self.play(m.Create(gx), m.Create(zero), m.Create(g), run_time=1.4)
        key = prose("orange: $\\lvert x\\rvert<1$, area expands. blue: $\\lvert x\\rvert>1$, area shrinks", font_size=22, color=MUTED).next_to(ax, m.DOWN, buff=0.1).align_to(ax, m.LEFT)
        self.play(m.FadeIn(key))
        self.next_slide("divergence")
        k = m.ValueTracker(0)

        def area():
            i = max(2, int(k.get_value()))
            pts = [gx.c2p(t[0], 0)] + [gx.c2p(t[j], div[j]) for j in range(0, i, 10)] + [gx.c2p(t[i - 1], div[i - 1]), gx.c2p(t[i - 1], 0)]
            return m.Polygon(*pts, stroke_width=0, fill_color=TEAL, fill_opacity=0.4)

        ar = m.always_redraw(area)
        d1 = m.always_redraw(lambda: dot(ax, cyc[min(int(k.get_value()), N - 1)], color=YELLOW, r=0.1))
        d2 = m.always_redraw(lambda: dot(gx, (t[min(int(k.get_value()), N - 1)], div[min(int(k.get_value()), N - 1)]), color=YELLOW, r=0.08))
        cnt = m.always_redraw(lambda: tex(r"\int_0^t \nabla\cdot\vec F\,dt = %.2f" % I[min(int(k.get_value()), N - 1)], font_size=38, color=TEAL).move_to(pt(2.9, -2.6)))
        self.add(ar, d1, d2, cnt)
        self.play(k.animate.set_value(N - 1), run_time=10, rate_func=m.linear)
        res = tex(r"\lambda = e^{-7.06} = 8.6\times 10^{-4}", font_size=40, color=YELLOW).move_to(pt(2.9, -3.45))
        self.play(m.Write(res))
        self.wait(HOLD)
        self.next_slide("integral")
        self.clear()
        self.show_figure("ch03_liouville", "Smaller epsilon, far stronger attraction",
                         "The slow branches ($\\lvert x\\rvert>1$) dominate the time, so $\\ln\\lambda$ grows roughly like $1/\\varepsilon$. Right: the paper's neuron, $\\ln\\lambda = -46.47$.")

    def table(self):
        d = data("ch03_liouville")
        rows = d["rows"]
        h = heading("How strongly do cycles attract?")
        self.play(m.FadeIn(h))
        names = [r"\text{van der Pol}, \varepsilon = 1", r"\text{van der Pol}, \varepsilon = 0.3", r"\text{van der Pol}, \varepsilon = 0.1", r"\text{reduced Hodgkin-Huxley}"]
        lams = [r"8.6\times 10^{-4}", r"1.3\times 10^{-7}", r"2.7\times 10^{-17}", r"6.6\times 10^{-21}"]
        cells = [[tex(r"\text{system}", font_size=34, color=MUTED), tex("T", font_size=34, color=MUTED),
                  tex(r"\ln\lambda", font_size=34, color=MUTED), tex(r"\lambda", font_size=34, color=MUTED)]]
        for (nm, Tv, ll), lab, lv in zip(rows, names, lams):
            cells.append([tex(lab, font_size=34), tex("%.3f" % float(Tv), font_size=34), tex("%.2f" % float(ll), font_size=34),
                          tex(lv, font_size=34, color=YELLOW)])
        flat = m.VGroup(*[c for row in cells for c in row])
        flat.arrange_in_grid(rows=len(cells), cols=4, buff=(0.7, 0.35), col_alignments="lrrr").move_to(pt(0, 0.6))
        self.play(m.FadeIn(m.VGroup(*cells[0])))
        line = m.Line(pt(-6, 0), pt(6, 0), color=MUTED, stroke_width=2).next_to(m.VGroup(*cells[0]), m.DOWN, buff=0.15)
        line.set_x(flat.get_x())
        self.play(m.Create(line))
        for row in cells[1:]:
            self.play(m.FadeIn(m.VGroup(*row), shift=m.RIGHT * 0.2), run_time=0.7)
            self.next_slide()
        w = prose("the neuron squeezes transverse perturbations by about $10^{20}$ in one period:\nin double precision $\\lambda$ is numerically zero. A superb clock, but backward time\nstretches by $10^{20}$ per period. That is why naive isochron computations fail (Chapter 10).", font_size=22, color=MUTED).to_edge(m.DOWN, buff=0.2)
        self.play(m.FadeIn(w))
        self.wait(HOLD)
        self.next_slide("table")
        self.clear()

    def left_eig(self):
        d = data("ch03_monodromy")
        cyc, g0, vs, vt = d["cyc"], d["g0"], d["vs"], d["vt"]
        h = heading("Preview: the left eigenvector")
        ax = plane((0.4, 3.4, 1), (-1.6, 2.4, 1), width=4.5, height=6.0, at=pt(-3.8, -0.6), ticks=False)
        N = len(cyc)
        sel = np.vstack([cyc[-int(0.06 * N):], cyc[:int(0.08 * N)]])
        cy = curve(ax, sel, color=MUTED, width=3)
        cyl = tex(r"\Gamma", font_size=36, color=MUTED).next_to(ax.c2p(*sel[0]), m.UP, buff=0.1)
        c0 = ax.c2p(*g0)
        star = m.Star(n=5, outer_radius=0.14, color=ORANGE, fill_opacity=1).move_to(c0)
        L = 2.0
        aF = m.Arrow(c0, c0 + L * np.array([vt[0], vt[1], 0]), buff=0, color=TEAL, stroke_width=5)
        aS = m.Arrow(c0, c0 + L * np.array([vs[0], vs[1], 0]), buff=0, color=BLUE, stroke_width=5)
        wv = np.array([-vs[1], vs[0]])
        wv = wv if wv @ vt > 0 else -wv
        aW = m.Arrow(c0, c0 + L * np.array([wv[0], wv[1], 0]), buff=0, color=YELLOW, stroke_width=5)
        lF = tex(r"\vec F", font_size=36, color=TEAL).next_to(aF.get_end(), m.LEFT, buff=0.15)
        lS = tex(r"\vec v_s", font_size=36, color=BLUE).next_to(aS.get_end(), m.UP + m.RIGHT, buff=0.4)
        lW = tex(r"\vec w", font_size=36, color=YELLOW).next_to(aW.get_end(), m.RIGHT, buff=0.15)
        self.play(m.FadeIn(h), m.Create(cy), m.FadeIn(cyl), m.FadeIn(star), run_time=1)
        self.play(m.GrowArrow(aF), m.GrowArrow(aS), m.FadeIn(lF), m.FadeIn(lS))
        e1 = tex(r"M\vec v = \lambda\vec v\quad\text{(right)},\qquad \vec w^{T} M = \lambda\vec w^{T}\quad\text{(left)}", font_size=34).move_to(pt(2.6, 2.2))
        self.play(m.Write(e1))
        self.next_slide()
        self.play(m.GrowArrow(aW), m.FadeIn(lW))
        r = m.RightAngle(m.Line(c0, aS.get_end()), m.Line(c0, aW.get_end()), length=0.25, color=MUTED)
        self.play(m.Create(r))
        w = prose("the left eigenvector $\\vec w$ for eigenvalue $1$\nis perpendicular to $\\vec v_s$.\nChapter 6: suitably scaled, it is $\\nabla\\vartheta$,\nthe phase response curve (the adjoint method)", font_size=24).move_to(pt(3.0, -0.4))
        self.play(m.FadeIn(w))
        self.wait(HOLD)
        self.next_slide("left eigenvector")
        self.clear()
