"""Deck for Chapter 4: Asymptotic phase and isochrons."""
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


class Deck04(Deck):
    chapter = "Chapter 4"
    title = "Asymptotic Phase and Isochrons"
    subtitle = "Extending the clock to the whole basin"

    def construct(self):
        self.title_card()
        self.thought_experiment()
        self.definition()
        self.radial_clock()
        self.twist()
        self.cloud()
        self.properties()
        self.snapshots()
        self.show_figure("ch04_vdp", "A real oscillator: van der Pol, computed with XPPAUT",
                         "Isochrons crowd where the flow is slow, curve where states are sheared, and all meet at the fixed point.")
        self.recap(["$\\vartheta(\\vec x_0)$ = phase of the on-cycle state that $\\vec x_0$ converges to",
                    "Isochron $I(\\vec\\gamma)$ = all states that end in lockstep with $\\vec\\gamma$",
                    "Rays without twist, logarithmic spirals with twist",
                    "$\\Phi_t$ maps isochrons to isochrons; $\\nabla\\vartheta\\cdot\\vec F = 1/T$",
                    "Isochrons meet where there is no phase"])

    # ------------------------------------------------------------------
    def thought_experiment(self):
        d = data("ch04_def")
        P, g, g1, X = d["P"], d["g"], d["g1"], d["X"]
        h = heading("Start off the cycle. Whom do you end up with?")
        ax = plane((-3, 3, 1), (-3.2, 3.2, 1), width=6.2, height=6.4, at=pt(-2.6, -0.45), ticks=False)
        cyc = curve(ax, X, color=MUTED, width=3, step=10)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), run_time=1.2)
        k = m.ValueTracker(0)
        a = m.always_redraw(lambda: dot(ax, P[int(k.get_value())], color=MAGENTA, r=0.11))
        b = m.always_redraw(lambda: dot(ax, g[int(k.get_value())], color=TEAL, r=0.09))
        c = m.always_redraw(lambda: dot(ax, g1[int(k.get_value())], color=MUTED, r=0.09))
        trail = m.always_redraw(lambda: curve(ax, P[: max(2, int(k.get_value()) + 1)], color=MAGENTA, width=2))
        leg = m.VGroup(
            prose("magenta: a state $\\vec x_0$ off the cycle", font_size=24, color=MAGENTA),
            prose("teal: the on-cycle state with the same phase", font_size=24, color=TEAL),
            prose("grey: an on-cycle state with another phase", font_size=24, color=MUTED),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.25).move_to(pt(3.6, 1.2))
        self.play(m.FadeIn(a), m.FadeIn(b), m.FadeIn(c), m.FadeIn(trail), m.FadeIn(leg), run_time=0.8)
        self.next_slide("three states")
        self.play(k.animate.set_value(len(P) - 1), run_time=9, rate_func=m.linear)
        msg = prose("magenta and teal end up on top of each other:\nthey have the same asymptotic phase", font_size=24, color=YELLOW).move_to(pt(3.6, -1.6))
        self.play(m.FadeIn(msg))
        self.wait(HOLD)
        self.next_slide("converge")
        self.clear()

    def definition(self):
        h = heading("Asymptotic phase (paper, Definition 2.1)")
        e = tex(r"\lim_{t\to\infty}\big|\,", r"\Phi(t,\vec x_0)", r"-", r"\Phi\big(t+\vartheta(\vec x_0)\,T,\ \vec\gamma_0\big)", r"\,\big| = 0", font_size=46)
        e.move_to(pt(0, 1.2))
        self.play(m.FadeIn(h), m.Write(e), run_time=1.6)
        self.wait(BEAT)
        self.next_slide("definition")
        b1 = m.Brace(e[1], m.DOWN, color=MAGENTA)
        l1 = prose("our state at time $t$", font_size=26, color=MAGENTA).next_to(b1, m.DOWN, buff=0.2)
        self.play(m.GrowFromCenter(b1), m.FadeIn(l1))
        self.next_slide()
        b2 = m.Brace(e[3], m.UP, color=TEAL)
        l2 = prose("the on-cycle state that started at phase $\\vartheta$", font_size=26, color=TEAL).next_to(b2, m.UP, buff=0.2)
        self.play(m.GrowFromCenter(b2), m.FadeIn(l2))
        self.next_slide()
        iso = tex(r"I(\vec\gamma) = \{\vec x_0 \in B(\Gamma) : \vartheta(\vec x_0) = \vartheta(\vec\gamma)\}", font_size=44, color=YELLOW).move_to(pt(0, -2.2))
        lab = prose("isochron: a level set of the asymptotic phase (Definition 2.2)", font_size=26, color=MUTED).next_to(iso, m.DOWN, buff=0.3)
        self.play(m.Write(iso), m.FadeIn(lab), run_time=1.4)
        self.wait(HOLD)
        self.next_slide("isochron definition")
        self.clear()

    def radial_clock(self):
        h = heading("No twist: the radial isochron clock")
        eq = tex(r"\dot r = r(1-r^2),\qquad \dot\theta = 1", font_size=40).move_to(pt(3.5, 2.2))
        ax = plane((-2, 2, 1), (-2, 2, 1), width=5.6, height=5.6, at=pt(-3.0, -0.5), ticks=False)
        circ = m.Circle(radius=ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0], color=INK, stroke_width=4).move_to(ax.c2p(0, 0))
        self.play(m.FadeIn(h), m.Create(ax), m.Write(eq), m.Create(circ), run_time=1.2)
        why = prose("$\\dot\\theta$ does not depend on $r$:\nevery state turns at the same rate,\nso states on a ray stay on a ray", font_size=26).move_to(pt(3.5, 0.4))
        self.play(m.FadeIn(why))
        self.next_slide("radial clock")
        rays = m.VGroup(*[m.Line(ax.c2p(0, 0), ax.c2p(2.4 * np.cos(2 * np.pi * k / 12), 2.4 * np.sin(2 * np.pi * k / 12)),
                                 color=phase_color(k / 12), stroke_width=3) for k in range(12)])
        self.play(m.LaggedStart(*[m.Create(r) for r in rays], lag_ratio=0.08), run_time=2)
        res = tex(r"\vartheta(r,\theta) = \frac{\theta}{2\pi}", font_size=44, color=YELLOW).move_to(pt(3.5, -1.6))
        self.play(m.Write(res))
        self.next_slide("rays")
        # states on one ray rotate together
        ang = m.ValueTracker(0.0)
        rs = np.array([0.15, 0.4, 0.7, 1.3, 1.7])
        dots = m.always_redraw(lambda: m.VGroup(*[m.Dot(ax.c2p(r * np.cos(ang.get_value()), r * np.sin(ang.get_value())),
                                                        radius=0.08, color=YELLOW) for r in rs]))
        self.add(dots)
        self.play(ang.animate.set_value(2 * np.pi), run_time=5, rate_func=m.linear)
        self.wait(BEAT)
        self.next_slide("rotate together")
        self.clear()

    def twist(self):
        h = heading("Twist makes spirals")
        e0 = tex(r"\dot r = r(1-r^2),\qquad \dot\theta = \omega - b\,r^2", font_size=40).move_to(pt(0, 2.3))
        self.play(m.FadeIn(h), m.Write(e0))
        txt = prose("inside the cycle states turn faster, outside slower:\nstates on a ray are sheared apart", font_size=26, color=MUTED).next_to(e0, m.DOWN, buff=0.35)
        self.play(m.FadeIn(txt))
        self.next_slide("twist")
        e1 = tex(r"\psi = \theta - b\ln r", font_size=42, color=YELLOW).move_to(pt(0, 0.3))
        self.play(m.Write(e1))
        self.next_slide()
        e2 = tex(r"\dot\psi", r"=", r"\dot\theta - b\,\frac{\dot r}{r}", r"=", r"\omega", r"- b r^2", r"- b", r"+ b r^2", font_size=42).move_to(pt(0, -0.9))
        self.play(m.Write(e2), run_time=1.6)
        why = prose("using $\\dot\\theta = \\omega - br^2$ and $\\dot r / r = 1 - r^2$", font_size=24, color=MUTED).next_to(e2, m.DOWN, buff=0.25)
        self.play(m.FadeIn(why))
        self.next_slide()
        s1 = m.Line(e2[5].get_corner(m.DL), e2[5].get_corner(m.UR), color=RED, stroke_width=5)
        s2 = m.Line(e2[7].get_corner(m.DL), e2[7].get_corner(m.UR), color=RED, stroke_width=5)
        self.play(m.Create(s1), m.Create(s2))
        zero = tex(r"0", font_size=38, color=RED).next_to(m.VGroup(e2[5], e2[7]), m.UP, buff=0.25)
        self.play(m.FadeIn(zero))
        self.next_slide("the r-terms cancel")
        self.play(m.FadeOut(why))
        e3 = tex(r"\dot\psi = \omega - b \quad\text{for every trajectory}", font_size=42, color=YELLOW).move_to(pt(0, -2.2))
        self.play(m.Write(e3))
        self.next_slide()
        e4 = tex(r"\vartheta(r,\theta) = \frac{\theta - b\ln r}{2\pi}\quad\Rightarrow\quad\text{isochrons } \theta = b\ln r + c", font_size=38, color=TEAL).move_to(pt(0, -3.2))
        self.play(m.Write(e4))
        self.wait(HOLD)
        self.next_slide("log spirals")
        self.clear()

    def cloud(self):
        d = data("ch04_cloud")
        paths, b = d["paths"], float(d["b"])
        h = heading("States on one isochron move as one")
        ax = plane((-2.4, 2.4, 1), (-2.4, 2.4, 1), width=6.0, height=6.0, at=pt(-2.6, -0.45), ticks=False)
        circ = m.Circle(radius=ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0], color=INK, stroke_width=4).move_to(ax.c2p(0, 0))
        rr = np.geomspace(0.03, 2.35, 300)
        spirals = m.VGroup(*[curve(ax, np.column_stack([rr * np.cos(b * np.log(rr) + 2 * np.pi * k / 12),
                                                         rr * np.sin(b * np.log(rr) + 2 * np.pi * k / 12)]),
                                   color=phase_color(k / 12), width=2.5) for k in range(12)])
        self.play(m.FadeIn(h), m.Create(ax), m.Create(circ), run_time=1)
        self.play(m.LaggedStart(*[m.Create(s) for s in spirals], lag_ratio=0.07), run_time=2.4)
        note = prose("$b = 0.6$: twelve isochrons,\nlogarithmic spirals", font_size=26).move_to(pt(3.7, 1.6))
        self.play(m.FadeIn(note))
        self.next_slide("spirals")
        k = m.ValueTracker(0)
        cl = m.always_redraw(lambda: m.VGroup(*[m.Dot(ax.c2p(*paths[j, int(k.get_value())]), radius=0.06, color=YELLOW)
                                                for j in range(paths.shape[0])]))
        self.play(spirals.animate.set_stroke(opacity=0.35), m.FadeIn(cl))
        msg = prose("26 states placed on one spiral:\nat every moment they lie on a common isochron\nand they land on $\\Gamma$ together", font_size=24, color=YELLOW, width=5.6).move_to(pt(3.9, -1.9))
        self.play(m.FadeIn(msg))
        self.next_slide()
        self.play(k.animate.set_value(paths.shape[1] - 1), run_time=7, rate_func=m.linear)
        self.wait(HOLD)
        self.next_slide("cloud")
        self.clear()

    def properties(self):
        h = heading("Four properties")
        rows = [
            (r"I\big(\Phi_t(\vec\gamma)\big) = \Phi_t\big(I(\vec\gamma)\big)", "the flow carries isochrons to isochrons"),
            (r"\Phi_T\big(I(\vec\gamma)\big) = I(\vec\gamma)", "one period maps each isochron into itself"),
            (r"\text{one isochron through every point of } B(\Gamma)", "they foliate the basin"),
            (r"\nabla\vartheta\cdot\vec F = \tfrac1T", "phase advances uniformly everywhere"),
        ]
        self.play(m.FadeIn(h))
        g = m.VGroup()
        for i, (f, w) in enumerate(rows):
            r = m.VGroup(tex(f, font_size=36), prose(w, font_size=24, color=MUTED)).arrange(m.RIGHT, buff=0.6)
            g.add(r)
        g.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.55).move_to(pt(0, -0.3))
        for r in g:
            self.play(m.FadeIn(r, shift=m.RIGHT * 0.2))
            self.wait(BEAT)
            self.next_slide()
        self.clear()

    def snapshots(self):
        d = data("ch04_vdp1")
        xs, ys, th, X = d["xs"], d["ys"], d["th"], d["X"]
        h = heading("Equally spaced isochrons are snapshots of one curve")
        ax = plane((-3, 3, 1), (-3.4, 3.4, 1), width=5.8, height=6.4, at=pt(-2.8, -0.45), ticks=False)
        cyc = curve(ax, X, color=INK, width=3.5, step=10)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), run_time=1)
        txt = prose("each isochron is the previous one\ncarried by the flow for $T/20$", font_size=26).move_to(pt(3.6, 1.8))
        self.play(m.FadeIn(txt))
        self.next_slide()
        for j in range(20):
            segs = level_curves(xs, ys, th, j / 20)
            g = m.VGroup(*[curve(ax, sgm, color=phase_color(j / 20), width=2.4) for sgm in segs])
            self.play(m.Create(g), run_time=0.35)
        pde = tex(r"\nabla\vartheta\cdot\vec F = \frac1T", font_size=44, color=YELLOW).move_to(pt(3.6, -0.6))
        why = prose("fast flow: isochrons far apart\nslow flow: isochrons crowded", font_size=24, color=MUTED).next_to(pde, m.DOWN, buff=0.4)
        self.play(m.Write(pde), m.FadeIn(why))
        self.wait(HOLD)
        self.next_slide("snapshots")
        self.clear()
