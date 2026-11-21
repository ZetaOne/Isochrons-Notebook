"""Deck for Chapter 2: The limit cycle as a clock."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import manimgx as m
from deckkit import *  # noqa: F401,F403


def phase_cycle(ax, X, n=72, width=5.0):
    """The closed curve X (sampled uniformly in time from phase 0) coloured by phase."""
    N = len(X)
    g = m.VGroup()
    for j in range(n):
        a, b = j * N // n, min((j + 1) * N // n + 1, N)
        seg = X[a:b] if b < N else np.vstack([X[a:], X[:1]])
        g.add(curve(ax, seg, color=phase_color(j / n), width=width))
    return g


def vdp1(p):
    x, y = p
    return np.array([y - x ** 3 / 3 + x, -x])


def rk4(f, p, dt, n):
    out = [np.array(p, float)]
    for _ in range(n):
        q = out[-1]
        k1 = f(q); k2 = f(q + dt / 2 * k1); k3 = f(q + dt / 2 * k2); k4 = f(q + dt * k3)
        out.append(q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    return np.array(out)


class Deck02(Deck):
    chapter = "Chapter 2"
    title = "The Limit Cycle as a Clock"
    subtitle = "Phase measures time, not distance"

    def construct(self):
        self.title_card()
        self.loop_in_time()
        self.phase_circle()
        self.zero()
        self.uniform_rate()
        self.markers()
        self.show_figure("ch02_speed", "Speed along the relaxation cycle",
                         "Left: colour by phase. Middle: speed $\\lvert \\vec F\\rvert$ varies by a factor of about 8. Right: arclength against phase; steep parts are the jumps.")
        self.arclength()
        self.two_clocks()
        self.kick()
        self.recap(["On $\\Gamma$: $\\vartheta(\\vec\\gamma_t) = t/T$, time since $\\vec\\gamma_0$ over the period",
                    "Zero phase: the maximum of the first coordinate (the paper, XPPAUT)",
                    "$d\\vartheta/dt = 1/T$ everywhere on $\\Gamma$, however fast the state moves",
                    "Equal phase steps crowd where the flow is slow: $ds/d\\vartheta = T\\lvert\\vec F\\rvert$",
                    "Phase differences on $\\Gamma$ never change. Off $\\Gamma$: which phase? (Chapter 4)"])

    # ------------------------------------------------------------------
    def loop_in_time(self):
        d = data("ch03_monodromy")
        cyc, T = d["cyc"], float(d["T"])
        N = len(cyc)
        h = heading("The cycle is a loop in time")
        ax = plane((-3, 3, 1), (-3, 3, 1), width=6.0, height=6.0, at=pt(-3.2, -0.45), ticks=False)
        cy = curve(ax, cyc, color=MUTED, width=3, step=10)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cy), run_time=1.2)
        g0 = m.Star(n=5, outer_radius=0.17, color=ORANGE, fill_opacity=1).move_to(ax.c2p(*cyc[0]))
        l0 = tex(r"\vec\gamma_0", font_size=38, color=ORANGE).next_to(g0, m.RIGHT, buff=0.1)
        sub = prose("van der Pol, $\\varepsilon = 1$, period $T = 6.663$", font_size=22, color=MUTED).next_to(ax, m.DOWN, buff=0.05)
        self.play(m.FadeIn(g0), m.FadeIn(l0), m.FadeIn(sub))
        e1 = tex(r"\vec\gamma_t := \Phi_t(\vec\gamma_0)", font_size=42).move_to(pt(3.6, 2.2))
        w1 = prose("label each point of $\\Gamma$ by the time\nit takes to get there from $\\vec\\gamma_0$", font_size=24).next_to(e1, m.DOWN, buff=0.3)
        self.play(m.Write(e1), m.FadeIn(w1))
        self.next_slide("label by time")
        k = m.ValueTracker(0)
        dd = m.always_redraw(lambda: dot(ax, cyc[min(int(k.get_value()), N - 1)], color=TEAL, r=0.11))
        trail = m.always_redraw(lambda: curve(ax, cyc[: max(2, min(int(k.get_value()), N - 1) + 1)], color=TEAL, width=5, step=5))
        clock = m.always_redraw(lambda: tex(r"t = %.2f" % (min(k.get_value(), N - 1) / (N - 1) * T), font_size=42, color=TEAL).move_to(pt(3.6, -0.2)))
        self.add(trail, dd, clock)
        self.play(k.animate.set_value(N - 1), run_time=7, rate_func=m.linear)
        e2 = tex(r"\Phi_T(\vec\gamma_0) = \vec\gamma_0", font_size=42, color=YELLOW).move_to(pt(3.6, -1.4))
        e3 = tex(r"\Gamma = \{\vec\gamma_t : 0 \le t < T\}", font_size=40).move_to(pt(3.6, -2.5))
        w3 = prose("$[0,T)$ wrapped once round, $t=T$ glued to $t=0$", font_size=22, color=MUTED).next_to(e3, m.DOWN, buff=0.2)
        self.play(m.Write(e2))
        self.play(m.Write(e3), m.FadeIn(w3))
        self.wait(HOLD)
        self.next_slide("loop in time")
        self.clear()

    def phase_circle(self):
        d = data("ch03_monodromy")
        cyc = d["cyc"]
        N = len(cyc)
        h = heading("Phase: time as a fraction of the period")
        e = tex(r"\vartheta(\vec\gamma_t) = \frac{t}{T} \in [0,1)", font_size=44, color=YELLOW).move_to(pt(0, 2.45))
        self.play(m.FadeIn(h), m.Write(e))
        ax = plane((-3, 3, 1), (-3, 3, 1), width=5.0, height=5.0, at=pt(-3.4, -1.0), ticks=False)
        pcy = phase_cycle(ax, cyc)
        cen = pt(3.4, -1.0)
        Rc = 2.0
        circ = m.VGroup(*[m.Arc(radius=Rc, start_angle=np.pi / 2 - 2 * np.pi * j / 72, angle=-2 * np.pi / 72, color=phase_color(j / 72), stroke_width=6, arc_center=cen) for j in range(72)])
        self.play(m.Create(ax), m.Create(pcy), m.Create(circ), run_time=1.6)
        ticks = m.VGroup()
        for v, s, dirn in ((0, "0", m.UP), (0.25, "0.25", m.RIGHT), (0.5, "0.5", m.DOWN), (0.75, "0.75", m.LEFT)):
            a = np.pi / 2 - 2 * np.pi * v
            p = cen + Rc * np.array([np.cos(a), np.sin(a), 0])
            ticks.add(tex(s, font_size=34, color=MUTED).next_to(p, dirn, buff=0.15))
        self.play(m.FadeIn(ticks))
        arrow = m.Arrow(pt(-0.65, 0.4), pt(0.2, 0.4), buff=0, color=MUTED)
        self.play(m.GrowArrow(arrow))
        k = m.ValueTracker(0.0)

        def dots():
            c = k.get_value() % 1.0
            p = cyc[int(c * (N - 1))]
            a = np.pi / 2 - 2 * np.pi * c
            return m.VGroup(dot(ax, p, color=INK, r=0.11),
                            m.Dot(cen + Rc * np.array([np.cos(a), np.sin(a), 0]), radius=0.11, color=INK))

        dd = m.always_redraw(dots)
        val = m.always_redraw(lambda: tex(r"\vartheta = %.2f" % (k.get_value() % 1.0), font_size=38).move_to(cen))
        self.add(dd, val)
        self.play(k.animate.set_value(1.99), run_time=9, rate_func=m.linear)
        w = prose("the colours match: one point of $\\Gamma$ for each point of the circle", font_size=24, color=MUTED).to_edge(m.DOWN, buff=0.15)
        self.play(m.FadeIn(w))
        self.wait(BEAT)
        self.next_slide("phase circle")
        self.clear()

    def zero(self):
        d = data("ch03_monodromy")
        cyc = d["cyc"]
        h = heading("Where is zero?")
        ax = plane((-3, 3, 1), (-3, 3, 1), width=5.6, height=5.6, at=pt(-3.4, -0.5), ticks=False)
        pcy = phase_cycle(ax, cyc)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(pcy), run_time=1.2)
        t1 = prose("The choice of $\\vec\\gamma_0$ is arbitrary. Moving it\nshifts every phase by the same constant,\nlike Indian Standard Time versus Greenwich.", font_size=24).move_to(pt(3.3, 1.9))
        self.play(m.FadeIn(t1))
        self.next_slide()
        i = int(np.argmax(cyc[:, 0]))
        vline = m.DashedLine(ax.c2p(cyc[i, 0], -3), ax.c2p(cyc[i, 0], 3), color=ORANGE, stroke_width=3)
        g0 = m.Star(n=5, outer_radius=0.18, color=ORANGE, fill_opacity=1).move_to(ax.c2p(*cyc[i]))
        t2 = prose("Osinga and Moehlis: $\\vartheta = 0$ at the\nglobal maximum of the first coordinate.\nFor a neuron: the peak of the spike.", font_size=24, color=ORANGE).move_to(pt(3.3, -0.3))
        self.play(m.Create(vline), m.FadeIn(g0, scale=2), m.FadeIn(t2))
        t3 = prose("XPPAUT's Initialconds > Periodic uses\nthe same convention, so does this book.", font_size=22, color=MUTED).move_to(pt(3.3, -2.2))
        self.play(m.FadeIn(t3))
        self.wait(HOLD)
        self.next_slide("zero phase")
        self.clear()

    def uniform_rate(self):
        h = heading("Phase always advances at the same rate")
        w0 = prose("follow the point $\\vec\\gamma_t$ for a further time $s$:", font_size=26).move_to(pt(0, 2.4))
        self.play(m.FadeIn(h), m.FadeIn(w0))
        e1 = tex(r"\Phi_s(\vec\gamma_t)", r"=", r"\Phi_s\big(\Phi_t(\vec\gamma_0)\big)", r"=", r"\Phi_{t+s}(\vec\gamma_0)", r"=", r"\vec\gamma_{t+s}", font_size=44).move_to(pt(0, 1.3))
        self.play(m.Write(e1[:3]))
        b1 = prose("definition of $\\vec\\gamma_t$", font_size=22, color=MUTED).next_to(e1[2], m.DOWN, buff=0.25)
        self.play(m.FadeIn(b1))
        self.next_slide()
        self.play(m.Write(e1[3:5]))
        b2 = prose("$\\Phi_s\\circ\\Phi_t = \\Phi_{t+s}$", font_size=22, color=MUTED).next_to(e1[4], m.DOWN, buff=0.25)
        self.play(m.FadeIn(b2))
        self.next_slide()
        self.play(m.Write(e1[5:]))
        self.next_slide()
        e2 = tex(r"\vartheta:\ \frac{t}{T}\ \longmapsto\ \frac{t+s}{T}", font_size=44).move_to(pt(0, -0.4))
        self.play(m.Write(e2))
        self.next_slide()
        e3 = tex(r"\frac{d\vartheta}{dt} = \frac{1}{T}\quad\text{everywhere on } \Gamma", font_size=46, color=YELLOW).move_to(pt(0, -1.9))
        box = m.SurroundingRectangle(e3, color=YELLOW, buff=0.2)
        w = prose("the state races through some parts and crawls through others;\nthe phase does not care: phase measures time, so it ticks uniformly", font_size=24, color=MUTED).to_edge(m.DOWN, buff=0.2)
        self.play(m.Write(e3), m.Create(box))
        self.play(m.FadeIn(w))
        self.wait(HOLD)
        self.next_slide("uniform phase")
        self.clear()

    def markers(self):
        d = data("ch02_vdp")
        X, T = d["X"], float(d["T"])
        N = len(X)
        h = heading("24 marks, equally spaced in phase")
        ax = plane((-2.5, 2.5, 1), (-1.1, 1.1, 0.5), width=7.6, height=5.4, at=pt(-2.6, -0.55), ticks=False, ylabel="")
        nc = ax.plot(lambda x: x ** 3 / 3 - x, x_range=(-2.12, 2.12), color=GREEN, stroke_width=3)
        cy = curve(ax, X, color=MUTED, width=2.5, step=5)
        eq = tex(r"\dot x = \frac{y - x^3/3 + x}{\varepsilon},\quad \dot y = -x", font_size=34).move_to(pt(4.2, 2.4))
        sub = prose("$\\varepsilon = 0.1$: a relaxation oscillator", font_size=22, color=MUTED).next_to(eq, m.DOWN, buff=0.2)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(nc), m.Create(cy), m.Write(eq), m.FadeIn(sub), run_time=1.4)
        k = m.ValueTracker(0.0)

        def marks():
            c = k.get_value()
            return m.VGroup(*[dot(ax, X[int(((j / 24 + c) % 1.0) * (N - 1))], color=phase_color(j / 24), r=0.09) for j in range(24)])

        mk = m.always_redraw(marks)
        self.play(m.FadeIn(mk))
        t1 = prose("crowded on the slow branches\nthat hug the cubic nullcline (green),\nfar apart on the fast jumps", font_size=24).move_to(pt(4.2, 0.4))
        self.play(m.FadeIn(t1))
        self.next_slide("markers")
        t2 = prose("they bunch and spread as they go round,\nbut each stays exactly $1/24$ of a period\nahead of its neighbour", font_size=24, color=YELLOW, width=5.0).move_to(pt(4.3, -1.7))
        self.play(m.FadeIn(t2))
        self.play(k.animate.set_value(2.0), run_time=12, rate_func=m.linear)
        w = prose("equal steps in time, not in distance: the 1-D version of an isochron", font_size=22, color=MUTED).to_edge(m.DOWN, buff=0.12)
        self.play(m.FadeIn(w))
        self.wait(BEAT)
        self.next_slide("markers move")
        self.clear()

    def arclength(self):
        d = data("ch02_vdp")
        X, T = d["X"], float(d["T"])
        h = heading("Phase is not arclength")
        e1 = tex(r"\frac{ds}{d\vartheta}", r"=", r"\frac{ds/dt}{d\vartheta/dt}", r"=", r"T\,\lvert\vec F(\vec\gamma_{\vartheta T})\rvert", font_size=42).move_to(pt(-3.3, 1.6))
        self.play(m.FadeIn(h), m.Write(e1[:3]))
        w1 = prose("chain rule", font_size=22, color=MUTED).next_to(e1, m.DOWN, buff=0.3)
        self.play(m.FadeIn(w1))
        self.next_slide()
        self.play(m.Write(e1[3:]))
        w2 = prose("speed $ds/dt = \\lvert\\vec F\\rvert$, phase rate $d\\vartheta/dt = 1/T$", font_size=22, color=MUTED).move_to(w1)
        self.play(m.FadeOut(w1), m.FadeIn(w2))
        w3 = prose("large speed: a small phase step\ncovers a large distance", font_size=24, color=YELLOW).move_to(pt(-3.3, -1.0))
        self.play(m.FadeIn(w3))
        self.next_slide("ds/dtheta")
        seg = np.hypot(*np.diff(X, axis=0).T)
        s = np.concatenate([[0], np.cumsum(seg)])
        s = s / s[-1]
        th = np.linspace(0, 1, len(X))
        ax = plane((0, 1, 0.5), (0, 1, 0.5), width=4.6, height=4.6, at=pt(3.7, -0.6), xlabel=r"\vartheta", ylabel=r"s/L", ticks=False)
        diag = m.DashedLine(ax.c2p(0, 0), ax.c2p(1, 1), color=MUTED, stroke_width=2)
        self.play(m.Create(ax), m.Create(diag))
        g = m.VGroup(*[curve(ax, np.column_stack([th[a:a + 120], s[a:a + 120]]), color=phase_color(a / len(X)), width=4)
                      for a in range(0, len(X) - 1, 119)])
        self.play(m.Create(g), run_time=3, rate_func=m.linear)
        w4 = prose("dashed: uniform speed. Steep parts are the jumps:\nabout 40 per cent of the length in 20 per cent of the time", font_size=22, color=MUTED).to_edge(m.DOWN, buff=0.15)
        self.play(m.FadeIn(w4))
        self.wait(HOLD)
        self.next_slide("arclength vs phase")
        self.clear()

    def two_clocks(self):
        d = data("ch02_vdp")
        X = d["X"]
        N = len(X)
        h = heading("Two clocks keep their phase difference")
        ax = plane((-2.5, 2.5, 1), (-1.1, 1.1, 0.5), width=7.6, height=5.4, at=pt(-2.6, -0.55), ticks=False, ylabel="")
        cy = curve(ax, X, color=MUTED, width=2.5, step=5)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cy), run_time=1)
        k = m.ValueTracker(0.1)
        dphi = 0.3

        def pos(c):
            return X[int((c % 1.0) * (N - 1))]

        a = m.always_redraw(lambda: dot(ax, pos(k.get_value()), color=TEAL, r=0.12))
        b = m.always_redraw(lambda: dot(ax, pos(k.get_value() + dphi), color=MAGENTA, r=0.12))
        link = m.always_redraw(lambda: m.DashedLine(ax.c2p(*pos(k.get_value())), ax.c2p(*pos(k.get_value() + dphi)), color=YELLOW, stroke_width=2))
        dist = m.always_redraw(lambda: tex(r"\text{distance} = %.2f" % np.hypot(*(pos(k.get_value() + dphi) - pos(k.get_value()))), font_size=36, color=YELLOW).move_to(pt(4.3, 0.9)))
        ph = tex(r"\vartheta_2 - \vartheta_1 = 0.30", font_size=40, color=INK).move_to(pt(4.3, 2.1))
        self.play(m.FadeIn(a), m.FadeIn(b), m.FadeIn(link), m.FadeIn(dist), m.FadeIn(ph))
        self.play(k.animate.set_value(2.1), run_time=12, rate_func=m.linear)
        w = prose("both phases grow at rate $1/T$, so\ntheir difference never changes, while\nthe distance between them does", font_size=24).move_to(pt(4.3, -1.0))
        w2 = prose("synchronisation is a question about phase", font_size=22, color=MUTED).move_to(pt(4.3, -2.6))
        self.play(m.FadeIn(w), m.FadeIn(w2))
        self.wait(HOLD)
        self.next_slide("two clocks")
        self.clear()

    def kick(self):
        d = data("ch03_monodromy")
        cyc, T = d["cyc"], float(d["T"])
        N = len(cyc)
        dt = T / (N - 1)
        h = heading("The question that leads to isochrons")
        ax = plane((-3, 3, 1), (-3, 3, 1), width=6.0, height=6.0, at=pt(-3.2, -0.45), ticks=False)
        cy = curve(ax, cyc, color=MUTED, width=3, step=10)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cy), run_time=1)
        # kick at phase 0.3, push to the right by 1.2
        i0 = int(0.3 * (N - 1))
        stride = 20
        kicked = rk4(vdp1, cyc[i0] + np.array([0.0, 1.3]), dt * stride, int(2.0 * (N - 1) / stride))
        ref = np.array([cyc[(i0 + j * stride) % (N - 1)] for j in range(len(kicked))])
        k = m.ValueTracker(0)
        pre = m.ValueTracker(0)
        dA = m.always_redraw(lambda: dot(ax, cyc[int(pre.get_value())], color=TEAL, r=0.11))
        self.play(m.FadeIn(dA))
        self.play(pre.animate.set_value(i0), run_time=2, rate_func=m.linear)
        bolt = m.Arrow(ax.c2p(*cyc[i0]), ax.c2p(*(cyc[i0] + np.array([0.0, 1.3]))), buff=0, color=ORANGE, stroke_width=5)
        lab = prose("a kick", font_size=24, color=ORANGE).next_to(bolt, m.UP, buff=0.1)
        self.play(m.GrowArrow(bolt), m.FadeIn(lab))
        self.remove(dA)
        ghost = m.always_redraw(lambda: dot(ax, ref[int(k.get_value())], color=MUTED, r=0.1))
        kd = m.always_redraw(lambda: dot(ax, kicked[int(k.get_value())], color=MAGENTA, r=0.11))
        ktr = m.always_redraw(lambda: curve(ax, kicked[: max(2, int(k.get_value()) + 1)], color=MAGENTA, width=2.5))
        self.add(ktr, ghost, kd)
        t1 = prose("grey: where the clock would be without the kick\nmagenta: the kicked state", font_size=22).move_to(pt(3.6, 2.2))
        self.play(m.FadeIn(t1))
        self.next_slide("kick")
        self.play(m.FadeOut(bolt), m.FadeOut(lab))
        self.play(k.animate.set_value(len(kicked) - 1), run_time=8, rate_func=m.linear)
        q = prose("it returns to $\\Gamma$ and is a clock again,\nbut shifted against the grey one.\nCan we tell the new phase\nat the moment of the kick?", font_size=26, color=YELLOW).move_to(pt(3.6, 0.0))
        self.play(m.FadeIn(q))
        a = prose("Chapter 3: how fast states return.\nChapter 4: which phase. The answer: isochrons.", font_size=22, color=MUTED).move_to(pt(3.6, -2.2))
        self.play(m.FadeIn(a))
        self.wait(HOLD)
        self.next_slide("question")
        self.clear()
