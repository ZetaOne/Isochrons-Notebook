"""Deck for Chapter 6: Phase response curves."""
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


class Chart(m.VGroup):
    """A framed graph: data box with tick labels outside it, so curves never cross the labels."""

    def __init__(self, xr, yr, width, height, at, xticks=(), yticks=(), xlabel="", ylabel="", fs=26, zero=True):
        super().__init__()
        self.xr, self.yr, self.w, self.h = xr, yr, width, height
        self.o = np.array(at, float) - np.array([width / 2, height / 2, 0])
        self.frame = m.Rectangle(width=width, height=height, color=MUTED, stroke_width=2).move_to(at)
        self.add(self.frame)
        if zero and yr[0] < 0 < yr[1]:
            self.add(m.DashedLine(self.c2p(xr[0], 0), self.c2p(xr[1], 0), color=MUTED, stroke_width=1.5))
        for v in xticks:
            self.add(m.Line(self.c2p(v, yr[0]), self.c2p(v, yr[0]) + 0.1 * m.UP, color=MUTED, stroke_width=2))
            self.add(m.MathTex(f"{v:g}", font_size=fs, color=MUTED).next_to(self.c2p(v, yr[0]), m.DOWN, buff=0.12))
        for v in yticks:
            self.add(m.Line(self.c2p(xr[0], v), self.c2p(xr[0], v) + 0.1 * m.RIGHT, color=MUTED, stroke_width=2))
            self.add(m.MathTex(f"{v:g}", font_size=fs, color=MUTED).next_to(self.c2p(xr[0], v), m.LEFT, buff=0.12))
        if xlabel:
            self.add(m.MathTex(xlabel, font_size=34, color=MUTED).next_to(self.frame, m.DOWN, buff=0.12).align_to(self.frame, m.RIGHT))
        if ylabel:
            self.add(m.MathTex(ylabel, font_size=34, color=MUTED).next_to(self.frame, m.LEFT, buff=0.12).align_to(self.frame, m.UP).shift(0.5 * m.UP))

    def c2p(self, x, y):
        return self.o + np.array([(x - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w,
                                  (y - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h, 0.0])


def strike(mob, color=RED):
    return m.Line(mob.get_corner(m.DL), mob.get_corner(m.UR), color=color, stroke_width=5)


def phase_gradient(xs, ys, th):
    """grad(theta) on the grid, computed from exp(2 pi i theta) so the jump 1 -> 0 does no harm."""
    e = np.exp(2j * np.pi * th)
    ey, ex = np.gradient(e, ys, xs)
    gx = np.imag(np.conj(e) * ex) / (2 * np.pi)
    gy = np.imag(np.conj(e) * ey) / (2 * np.pi)
    return gx, gy


def at(xs, ys, A, p):
    j = int(np.argmin(np.abs(xs - p[0]))); i = int(np.argmin(np.abs(ys - p[1])))
    return A[i, j]


def isochron_group(ax, xs, ys, th, n, width=2.0, opacity=0.6):
    g = m.VGroup()
    for j in range(n):
        for s in level_curves(xs, ys, th, j / n):
            g.add(curve(ax, s, color=phase_color(j / n), width=width).set_stroke(opacity=opacity))
    return g


class Deck06(Deck):
    chapter = "Chapter 6"
    title = "Phase Response Curves"
    subtitle = "The isochrons seen from the cycle"

    def construct(self):
        self.title_card()
        self.kick()
        self.prc_definition()
        self.small_kicks()
        self.gradient_picture()
        self.adjoint()
        self.left_eigenvector()
        self.checking()
        self.types()
        self.neuron()
        self.recap(["A kick moves the state to another isochron:\n$\\Delta\\vartheta = \\vartheta(\\vec\\gamma + \\Delta\\vec x) - \\vartheta(\\vec\\gamma)$, decided at the instant of the kick",
                    "Small kicks: $\\Delta\\vartheta \\approx \\nabla\\vartheta\\cdot\\Delta\\vec x$; the iPRC $\\vec Z = \\nabla\\vartheta$ is\nperpendicular to the isochrons, with $\\vec Z\\cdot\\vec F = 1/T$",
                    "$\\vec Z$ solves the adjoint equation and is the left eigenvector of $M$,\nperpendicular to $\\vec v_s$",
                    "Large kicks need global isochrons; topology (type 0 versus type 1)\nforces phaseless sets"])

    # ------------------------------------------------------------------
    def kick(self):
        v = data("ch04_vdp1")
        xs, ys, th = v["xs"], v["ys"], v["th"]
        d = data("ch06_kick")
        P, ghost, p0, p1, X, shift = d["P"], d["ghost"], d["p0"], d["p1"], d["X"], float(d["shift"])
        h = heading("A kick is a jump between isochrons")
        ax = plane((-3, 3, 1), (-3.4, 3.4, 1), width=5.6, height=6.35, at=pt(-3.6, -0.45), ticks=False, xlabel="", ylabel="")
        cyc = curve(ax, X, color=INK, width=3.5, step=10)
        isos = isochron_group(ax, xs, ys, th, 20, width=2.0, opacity=0.65)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), run_time=1.0)
        self.play(m.Create(isos), run_time=2.0)
        t0 = prose("van der Pol ($\\varepsilon = 1$), 20 isochrons.\nThe state sits on $\\Gamma$ at phase $0.30$", font_size=25).move_to(pt(3.2, 2.3))
        a = dot(ax, p0, color=MAGENTA, r=0.11)
        self.play(m.FadeIn(t0), m.FadeIn(a))
        self.next_slide("setup")
        arr = m.Arrow(ax.c2p(*p0), ax.c2p(*p1), buff=0, color=ORANGE, stroke_width=6)
        self.play(m.GrowArrow(arr), a.animate.move_to(ax.c2p(*p1)), run_time=1.0)
        t1 = prose("kick $\\Delta x = 0.9$: it lands on the isochron\nof an earlier phase", font_size=25, color=ORANGE).next_to(t0, m.DOWN, buff=0.35)
        self.play(m.FadeIn(t1))
        self.next_slide("kick")
        k = m.ValueTracker(0)
        self.remove(a)
        A = m.always_redraw(lambda: dot(ax, P[int(k.get_value())], color=MAGENTA, r=0.11))
        G = m.always_redraw(lambda: dot(ax, ghost[int(k.get_value())], color=MUTED, r=0.1))
        tr = m.always_redraw(lambda: curve(ax, P[: max(2, int(k.get_value()) + 1)], color=MAGENTA, width=2))
        self.add(tr, A, G)
        t2 = prose("magenta: kicked state; grey: unkicked copy", font_size=23).next_to(t1, m.DOWN, buff=0.35)
        self.play(m.FadeIn(t2))
        self.play(k.animate.set_value(len(P) - 1), run_time=9, rate_func=m.linear)
        e = tex(r"\Delta\vartheta = \vartheta(\vec\gamma + \Delta\vec x) - \vartheta(\vec\gamma) = " + f"{shift:.3f}", font_size=36, color=YELLOW).move_to(pt(3.2, -1.2))
        t3 = prose("permanently behind. No dynamics in the formula:\nthe shift is decided at the instant of the kick", font_size=23, color=MUTED).next_to(e, m.DOWN, buff=0.3)
        self.play(m.Write(e), m.FadeIn(t3))
        self.wait(HOLD)
        self.next_slide("relax")
        self.clear()

    def prc_definition(self):
        d = data("ch06_prc_vdp")
        phs, amps, shifts = d["phs"], d["amps"], d["shifts"]
        v = data("ch04_vdp1"); X = v["X"]
        h = heading("The phase response curve: kick at every phase")
        e = tex(r"\mathrm{PRC}(\vartheta) = \vartheta\big(\vec\gamma_{\vartheta T} + \Delta\vec x\big) - \vartheta", font_size=38).move_to(pt(0, 2.45))
        self.play(m.FadeIn(h), m.Write(e))
        n = prose("taken in $(-\\frac12, \\frac12]$: positive = advance (next spike early), negative = delay", font_size=24, color=MUTED).next_to(e, m.DOWN, buff=0.2)
        self.play(m.FadeIn(n))
        self.next_slide("definition")
        sm = plane((-3, 3, 1), (-3.2, 3.2, 1), width=4.0, height=4.2, at=pt(-4.6, -1.4), ticks=False, xlabel="", ylabel="")
        cyc = curve(sm, X, color=INK, width=3, step=10)
        gr = Chart((0, 1), (-0.12, 0.14), 6.4, 3.6, pt(2.9, -1.3), xticks=(0, 0.5, 1), yticks=(-0.1, 0, 0.1), xlabel=r"\vartheta")
        yl = prose("$\\Delta\\vartheta$", font_size=26, color=MUTED).next_to(gr.frame, m.LEFT, buff=0.15).align_to(gr.frame, m.UP).shift(0.55 * m.UP)
        self.play(m.Create(sm), m.Create(cyc), m.Create(gr), m.FadeIn(yl), run_time=1.0)
        lab = prose("direct method: kick $\\Delta x = 0.9$, wait, compare", font_size=23, color=ORANGE).next_to(gr.frame, m.UP, buff=0.12).align_to(gr.frame, m.RIGHT)
        self.play(m.FadeIn(lab))
        self.next_slide()
        j = 2; amp = float(amps[j])
        pts = m.VGroup()
        for i, ph in enumerate(phs):
            p = X[int(ph * (len(X) - 1))]
            arr = m.Arrow(sm.c2p(*p), sm.c2p(p[0] + amp, p[1]), buff=0, color=ORANGE, stroke_width=4, max_tip_length_to_length_ratio=0.35)
            d0 = dot(sm, p, color=MAGENTA, r=0.08)
            q = m.Dot(gr.c2p(ph, shifts[j, i]), radius=0.055, color=ORANGE)
            pts.add(q)
            self.add(arr, d0, q)
            self.wait(0.12)
            self.remove(arr, d0)
        self.wait(HOLD)
        self.next_slide("direct PRC")
        self.clear()

    def small_kicks(self):
        h = heading("Small kicks: the PRC is a gradient")
        self.play(m.FadeIn(h))
        e1 = tex(r"\Delta\vartheta", r"=", r"\vartheta(\vec\gamma + \Delta\vec x)", r"-", r"\vartheta(\vec\gamma)", font_size=42).move_to(pt(0, 1.9))
        self.play(m.Write(e1))
        n1 = prose("Taylor-expand the first term for a small $\\Delta\\vec x$:", font_size=26, color=MUTED).next_to(e1, m.DOWN, buff=0.3)
        self.play(m.FadeIn(n1))
        self.next_slide()
        e2 = tex(r"\Delta\vartheta", r"=", r"\vartheta(\vec\gamma)", r"+ \nabla\vartheta(\vec\gamma)\cdot\Delta\vec x + \dots", r"-", r"\vartheta(\vec\gamma)", font_size=42).next_to(n1, m.DOWN, buff=0.35)
        self.play(m.Write(e2))
        self.next_slide()
        s1, s2 = strike(e2[2]), strike(e2[5])
        self.play(m.Create(s1), m.Create(s2))
        z = tex(r"0", font_size=36, color=RED).next_to(m.VGroup(e2[2], e2[5]), m.DOWN, buff=0.25)
        self.play(m.FadeIn(z))
        self.next_slide("cancel")
        e3 = tex(r"\Delta\vartheta \approx \nabla\vartheta(\vec\gamma)\cdot\Delta\vec x", font_size=44, color=YELLOW).move_to(pt(0, -1.6))
        self.play(m.Write(e3))
        e4 = tex(r"\vec Z(\vartheta) := \nabla\vartheta\big(\vec\gamma_{\vartheta T}\big)", font_size=40, color=TEAL).next_to(e3, m.DOWN, buff=0.4)
        n4 = prose("the infinitesimal PRC (phase sensitivity function)", font_size=24, color=TEAL).next_to(e4, m.DOWN, buff=0.2)
        self.play(m.Write(e4), m.FadeIn(n4))
        self.wait(HOLD)
        self.next_slide("iPRC")
        self.clear()

    def gradient_picture(self):
        v = data("ch04_vdp1")
        xs, ys, th, X = v["xs"], v["ys"], v["th"], v["X"]
        T = float(v["T"])
        gx, gy = phase_gradient(xs, ys, th)
        h = heading("A gradient is perpendicular to the level curves")
        ax = plane((-3, 3, 1), (-3.4, 3.4, 1), width=5.6, height=6.35, at=pt(-3.6, -0.45), ticks=False, xlabel="", ylabel="")
        cyc = curve(ax, X, color=INK, width=3.5, step=10)
        isos = isochron_group(ax, xs, ys, th, 40, width=1.6, opacity=0.6)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), m.Create(isos), run_time=2.0)
        t0 = prose("van der Pol, 40 isochrons", font_size=25).move_to(pt(3.2, 2.5))
        self.play(m.FadeIn(t0))
        self.next_slide()
        arrs = m.VGroup()
        for k in range(24):
            p = X[int(k / 24 * (len(X) - 1))]
            g = np.array([at(xs, ys, gx, p), at(xs, ys, gy, p)])
            arrs.add(m.Arrow(ax.c2p(*p), ax.c2p(*(p + 9.0 * g)), buff=0, color=ORANGE, stroke_width=4, max_tip_length_to_length_ratio=0.25))
        self.play(m.LaggedStart(*[m.GrowArrow(a) for a in arrs], lag_ratio=0.08), run_time=2.4)
        t1 = prose("orange: $\\nabla\\vartheta$ on $\\Gamma$ (from the phase map).\nLong where isochrons crowd, short where sparse", font_size=24, color=ORANGE).next_to(t0, m.DOWN, buff=0.3)
        self.play(m.FadeIn(t1))
        self.next_slide("gradient")
        rows = m.VGroup(
            prose("• kick along the isochron: no phase shift at all", font_size=24),
            prose("• kick across the isochron: the biggest shift", font_size=24),
            prose("• crowded isochrons: very sensitive to kicks", font_size=24),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.22).next_to(t1, m.DOWN, buff=0.4)
        for r in rows:
            self.play(m.FadeIn(r, shift=m.RIGHT * 0.2), run_time=0.6)
        self.next_slide()
        e = tex(r"\nabla\vartheta\cdot\vec F = \frac{1}{T}", font_size=40, color=YELLOW).next_to(rows, m.DOWN, buff=0.4)
        n = prose("the part along the flow is always $1/(T\\lvert\\vec F\\rvert)$;\nthe information is in the part across", font_size=22, color=MUTED).next_to(e, m.DOWN, buff=0.2)
        self.play(m.Write(e), m.FadeIn(n))
        self.wait(HOLD)
        self.next_slide("Z.F")
        self.clear()

    def adjoint(self):
        h = heading("The adjoint method: the gradient without isochrons")
        self.play(m.FadeIn(h))
        n0 = prose("two nearby trajectories $\\vec\\gamma(t)$ and $\\vec\\gamma(t) + \\vec\\xi(t)$: both phases advance at rate $1/T$,\nso their phase difference $\\vec Z\\cdot\\vec\\xi$ never changes", font_size=25, width=12.8).move_to(pt(0, 2.5))
        self.play(m.FadeIn(n0))
        e1 = tex(r"0 = \frac{d}{dt}\big(\vec Z\cdot\vec\xi\big)", r"= \dot{\vec Z}\cdot\vec\xi + \vec Z\cdot\dot{\vec\xi}", font_size=38).next_to(n0, m.DOWN, buff=0.2)
        self.play(m.Write(e1))
        self.next_slide()
        n1 = prose("variational equation $\\dot{\\vec\\xi} = A\\vec\\xi$, $A = D\\vec F(\\vec\\gamma(t))$; and $\\vec Z\\cdot A\\vec\\xi = (A^{T}\\vec Z)\\cdot\\vec\\xi$", font_size=24, color=MUTED).next_to(e1, m.DOWN, buff=0.2)
        e2 = tex(r"0 = \dot{\vec Z}\cdot\vec\xi + \vec Z\cdot A\vec\xi = \big(\dot{\vec Z} + A^{T}\vec Z\big)\cdot\vec\xi", font_size=38).next_to(n1, m.DOWN, buff=0.2)
        self.play(m.FadeIn(n1), m.Write(e2))
        self.next_slide()
        n2 = prose("true for every $\\vec\\xi$, so the bracket is zero:", font_size=24, color=MUTED).next_to(e2, m.DOWN, buff=0.2)
        e3 = tex(r"\dot{\vec Z} = -D\vec F\big(\vec\gamma(t)\big)^{T}\vec Z,\qquad \vec Z(t+T) = \vec Z(t),\qquad \vec Z\cdot\vec F = \frac{1}{T}", font_size=36, color=YELLOW).next_to(n2, m.DOWN, buff=0.3)
        fr = m.SurroundingRectangle(e3, color=YELLOW, buff=0.15)
        self.play(m.FadeIn(n2), m.Write(e3), m.Create(fr))
        self.next_slide("adjoint equation")
        n3 = prose("other solutions grow by $1/\\lambda$ per period, so integrate backwards in time:\nlike the power method, it converges to the periodic one (this is what XPPAUT does)", font_size=23, color=TEAL, width=12.8).next_to(fr, m.DOWN, buff=0.25)
        self.play(m.FadeIn(n3))
        self.wait(HOLD)
        self.next_slide("backward")
        self.clear()

    def left_eigenvector(self):
        v = data("ch04_vdp1")
        xs, ys, th, X = v["xs"], v["ys"], v["th"], v["X"]
        b = data("ch05_bundle")
        us, vs = b["us"], b["vs"]
        gx, gy = phase_gradient(xs, ys, th)
        h = heading("Algebra and geometry agree")
        ax = plane((-3, 3, 1), (-3.4, 3.4, 1), width=5.6, height=6.35, at=pt(-3.6, -0.45), ticks=False, xlabel="", ylabel="")
        cyc = curve(ax, X, color=INK, width=3.5, step=10)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), run_time=1.0)
        e1 = tex(r"\vec Z(0)^{T} M = \vec Z(0)^{T}", font_size=40, color=YELLOW).move_to(pt(3.2, 2.3))
        n1 = prose("$\\vec Z$ is the left eigenvector of the\nmonodromy matrix $M$ for eigenvalue 1", font_size=24).next_to(e1, m.DOWN, buff=0.25)
        self.play(m.Write(e1), m.FadeIn(n1))
        self.next_slide()
        e2 = tex(r"\vec Z \perp \vec v_s", font_size=44, color=YELLOW).next_to(n1, m.DOWN, buff=0.45)
        n2 = prose("left and right eigenvectors of different\neigenvalues are perpendicular (Appendix A)", font_size=24).next_to(e2, m.DOWN, buff=0.25)
        self.play(m.Write(e2), m.FadeIn(n2))
        self.next_slide()
        idx = np.linspace(0, len(us) - 1, 13).astype(int)[:-1]
        segs, arrs, marks = m.VGroup(), m.VGroup(), m.VGroup()
        for i in idx:
            p, w = us[i], vs[i]
            segs.add(m.Line(ax.c2p(*(p - 0.45 * w)), ax.c2p(*(p + 0.45 * w)), color=TEAL, stroke_width=5))
            g = np.array([at(xs, ys, gx, p), at(xs, ys, gy, p)])
            arrs.add(m.Arrow(ax.c2p(*p), ax.c2p(*(p + 0.7 * g / np.linalg.norm(g))), buff=0, color=ORANGE, stroke_width=4, max_tip_length_to_length_ratio=0.3))
        self.play(m.LaggedStart(*[m.GrowFromCenter(s) for s in segs], lag_ratio=0.08), run_time=1.6)
        self.play(m.LaggedStart(*[m.GrowArrow(a) for a in arrs], lag_ratio=0.08), run_time=1.6)
        n3 = prose("teal: $\\vec v_s$ (tangent to the isochron)\norange: direction of $\\nabla\\vartheta$, always at a right angle", font_size=23, color=MUTED).next_to(n2, m.DOWN, buff=0.4)
        self.play(m.FadeIn(n3))
        self.wait(HOLD)
        self.next_slide("perpendicular")
        self.clear()

    def checking(self):
        h = heading("Checking it: direct kicks against the adjoint")
        self.play(m.FadeIn(h))
        n = prose("twisted oscillator, exact phase $\\vartheta = (\\theta - b\\ln r)/2\\pi$; on $r = 1$:", font_size=25, color=MUTED).move_to(pt(0, 2.55))
        e = tex(r"\frac{\partial\vartheta}{\partial x}", r"= \frac{1}{2\pi}\Big(", r"\frac{\partial\theta}{\partial x}", r"- b\,", r"\frac{\partial \ln r}{\partial x}", r"\Big)", r"= \frac{-\sin\theta - b\cos\theta}{2\pi}", font_size=36).next_to(n, m.DOWN, buff=0.2)
        self.play(m.FadeIn(n), m.Write(e))
        e[2].set_color(TEAL); e[4].set_color(TEAL)
        l1 = prose("using $\\frac{\\partial\\theta}{\\partial x} = -\\frac{y}{r^2} = -\\sin\\theta$ and $\\frac{\\partial \\ln r}{\\partial x} = \\frac{x}{r^2} = \\cos\\theta$ on $r = 1$", font_size=24, color=TEAL).next_to(e, m.DOWN, buff=0.25)
        self.play(m.FadeIn(l1))
        self.next_slide("exact gradient")
        self.play(m.FadeOut(l1))
        cols = [TEAL, ORANGE, MAGENTA]
        panels = []
        for k, (name, ttl, yr) in enumerate([("ch06_prc_sl", "(a) twisted oscillator, $b = 0.6$", (-0.25, 0.25, 0.2)),
                                             ("ch06_prc_vdp", "(b) van der Pol, $\\varepsilon = 1$", (-0.15, 0.15, 0.1))]):
            d = data(name)
            ax = Chart((0, 1), yr[:2], 5.4, 3.0, pt(-3.2 + 6.9 * k, -1.55), xticks=(0, 0.5, 1), yticks=(-yr[2], 0, yr[2]), xlabel=r"\vartheta")
            tl = prose(ttl, font_size=22).next_to(ax, m.UP, buff=0.1)
            zc = curve(ax, np.column_stack([d["tz"], d["z"]]), color=INK, width=3)
            panels.append((ax, d, tl, zc))
            self.play(m.Create(ax), m.FadeIn(tl), run_time=0.8)
        for ax, d, tl, zc in panels:
            self.play(m.Create(zc), run_time=1.2)
        ax, d = panels[0][0], panels[0][1]
        tt = np.linspace(0, 1, 300)
        ex = (-np.sin(2 * np.pi * tt) - 0.6 * np.cos(2 * np.pi * tt)) / (2 * np.pi)
        exc = m.DashedVMobject(curve(ax, np.column_stack([tt, ex]), color=YELLOW, width=3), num_dashes=40)
        self.play(m.Create(exc))
        leg = prose("white: adjoint $Z_x$;  yellow dashed: exact;  dots: direct kicks / kick size", font_size=22, color=MUTED).to_edge(m.DOWN, buff=0.12)
        self.play(m.FadeIn(leg))
        self.next_slide("curves")
        for j in range(3):
            anims = []
            for ax, d, tl, zc in panels:
                amp = float(d["amps"][j])
                g = m.VGroup(*[m.Dot(ax.c2p(p, s / amp), radius=0.045, color=cols[j]) for p, s in zip(d["phs"], d["shifts"][j])])
                anims.append(m.FadeIn(g))
            self.play(*anims, run_time=0.8)
            self.wait(0.6)
        msg = prose("small kicks (teal) sit on the adjoint curve; large kicks (magenta) deviate:\nfor finite kicks you need the global isochrons", font_size=24, color=YELLOW, width=12.8).move_to(pt(0, 2.0))
        self.play(m.FadeOut(m.VGroup(n, e)), m.FadeIn(msg))
        self.wait(HOLD)
        self.next_slide("deviation")
        self.clear()

    def types(self):
        h = heading("Big kicks: Winfree's type 1 and type 0 resetting")
        ax = plane((-1.6, 2.9, 1), (-1.7, 1.7, 1), width=5.4, height=4.1, at=pt(-3.6, 0.2), ticks=False, xlabel="", ylabel="")
        r1 = ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0]
        rays = m.VGroup(*[m.Line(ax.c2p(0, 0), ax.c2p(1.65 * np.cos(2 * np.pi * k / 12), 1.65 * np.sin(2 * np.pi * k / 12)),
                                 color=phase_color(k / 12), stroke_width=2.5).set_stroke(opacity=0.7) for k in range(12)])
        circ = m.Circle(radius=r1, color=INK, stroke_width=3).move_to(ax.c2p(0, 0))
        o = dot(ax, (0, 0), color=RED, r=0.08)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(rays), m.Create(circ), m.FadeIn(o), run_time=1.2)
        gr = Chart((0, 1), (0, 1), 3.4, 3.4, pt(3.9, 0.3), xticks=(0, 0.5, 1), yticks=(0, 0.5, 1), xlabel=r"\vartheta", ylabel=r"\vartheta'")
        self.play(m.Create(gr))
        A = m.ValueTracker(0.3)
        kicked = m.always_redraw(lambda: m.DashedVMobject(m.Circle(radius=r1, color=ORANGE, stroke_width=4).move_to(ax.c2p(A.get_value(), 0)), num_dashes=40))

        def newphase():
            a = A.get_value()
            t = np.linspace(0, 1, 600)
            np_ = (np.arctan2(np.sin(2 * np.pi * t), np.cos(2 * np.pi * t) + a) / (2 * np.pi)) % 1
            g = m.VGroup()
            run = [0]
            for i in range(1, len(t)):
                if abs(np_[i] - np_[i - 1]) > 0.5:
                    if len(run) > 1:
                        g.add(curve(gr, np.column_stack([t[run], np_[run]]), color=ORANGE, width=3.5))
                    run = []
                run.append(i)
            if len(run) > 1:
                g.add(curve(gr, np.column_stack([t[run], np_[run]]), color=ORANGE, width=3.5))
            return g
        pc = m.always_redraw(newphase)
        lab = m.always_redraw(lambda: m.MathTex(r"A = %.2f" % A.get_value(), font_size=38, color=ORANGE).move_to(pt(4.4, 2.6)))
        self.add(kicked, pc, lab)
        n = prose("radial clock, every cycle point kicked by $A$ in $x$:\nthe new phase is the angle seen from the origin", font_size=24).move_to(pt(0, -2.6))
        self.play(m.FadeIn(n))
        self.next_slide("setup")
        self.play(A.animate.set_value(0.8), run_time=3, rate_func=m.linear)
        n1 = prose("$A < 1$: the kicked circle encloses the origin; new phase goes once round: degree 1 (type 1)", font_size=23, color=TEAL).move_to(pt(0, -2.4))
        self.play(m.FadeOut(n), m.FadeIn(n1))
        self.next_slide("type 1")
        self.play(A.animate.set_value(1.0), run_time=2, rate_func=m.linear)
        n2 = prose("$A = 1$: the circle passes through the phaseless point; at $\\vartheta = \\frac12$ the new phase is undefined", font_size=23, color=RED).next_to(n1, m.DOWN, buff=0.2)
        self.play(m.FadeIn(n2))
        self.next_slide("A=1")
        self.play(A.animate.set_value(1.6), run_time=3, rate_func=m.linear)
        n3 = prose("$A > 1$: the origin is outside; new phase wobbles but never goes round: degree 0 (type 0)", font_size=23, color=MAGENTA).next_to(n2, m.DOWN, buff=0.2)
        self.play(m.FadeIn(n3))
        self.next_slide("type 0")
        self.play(m.FadeOut(m.VGroup(n1, n2, n3)))
        msg = prose("A degree is an integer and cannot change continuously, so some kick must\nsend the state into a set with no phase: Winfree's phaseless sets", font_size=24, color=YELLOW, width=12.8).move_to(pt(0, -2.6))
        self.play(m.FadeIn(msg))
        self.wait(HOLD)
        self.next_slide("phaseless")
        self.clear()

    def neuron(self):
        d = data("ch06_rhh_prc")
        tz, zv, X, t, T = d["tz"], d["zv"], d["X"], d["t"], float(d["T"])
        h = heading("A neuron's PRC (reduced Hodgkin-Huxley)")
        a1 = Chart((0, 1), (-80, 50), 7.2, 2.3, pt(-1.9, 1.45), yticks=(-60, 0, 40), zero=False)
        a2 = Chart((0, 1), (-0.02, 0.05), 7.2, 2.5, pt(-1.9, -1.75), xticks=(0, 0.2, 0.4, 0.6, 0.8, 1), yticks=(0, 0.04), xlabel=r"\vartheta")
        vl = m.MathTex(r"V", font_size=34, color=MUTED).next_to(a1.frame, m.LEFT, buff=0.9)
        yl = m.MathTex(r"\frac{\partial\vartheta}{\partial V}", font_size=32, color=MUTED).next_to(a2.frame, m.LEFT, buff=0.95)
        vc = curve(a1, np.column_stack([t / T, X[:, 0]]), color=BLUE, width=3, step=4)
        zc = curve(a2, np.column_stack([tz, zv]), color=ORANGE, width=3.5)
        self.play(m.FadeIn(h), m.Create(a1), m.Create(a2), m.FadeIn(yl), m.FadeIn(vl), run_time=1.0)
        self.play(m.Create(vc), run_time=1.4)
        self.play(m.Create(zc), run_time=1.4)
        self.next_slide("curves")
        s = m.ValueTracker(0.0)
        def cursor():
            x = s.get_value()
            ln = m.DashedLine(a1.c2p(x, 50), a2.c2p(x, -0.02), color=YELLOW, stroke_width=2)
            i = min(int(x * (len(X) - 1)), len(X) - 1); j = min(int(x * (len(tz) - 1)), len(tz) - 1)
            return m.VGroup(ln, m.Dot(a1.c2p(x, X[i, 0]), radius=0.08, color=YELLOW), m.Dot(a2.c2p(x, zv[j]), radius=0.08, color=YELLOW))
        cur = m.always_redraw(cursor)
        self.add(cur)
        notes = [(0.4, "phase 0 to 0.4: spike and recovery.\n$\\partial\\vartheta/\\partial V \\approx 0$: kicks are absorbed\n(the refractory period)", MUTED),
                 (0.62, "near phase 0.6: slightly negative,\na depolarising kick delays the spike", TEAL),
                 (0.86, "peak $0.048$ per mV at phase $0.86$:\njust before the spike a small push\nfires it early", ORANGE)]
        prev = None
        for x, txt, col in notes:
            self.play(s.animate.set_value(x), run_time=2.5, rate_func=m.linear)
            p = prose(txt, font_size=23, color=col).move_to(pt(4.6, 0.0))
            self.play(m.FadeOut(prev) if prev else m.Wait(0.01), m.FadeIn(p))
            prev = p
            self.next_slide()
        self.play(s.animate.set_value(1.0), run_time=1.0, rate_func=m.linear)
        self.play(m.FadeOut(prev))
        fin = prose("sharp peak = crowded isochrons;\nlong flat stretch = sparse isochrons\n(the paper's Figure 2)", font_size=23, color=YELLOW).move_to(pt(4.6, 0.0))
        self.play(m.FadeIn(fin))
        self.wait(HOLD)
        self.next_slide("neuron PRC")
        self.clear()
