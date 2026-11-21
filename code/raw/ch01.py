"""Deck for Chapter 1: Planar dynamics, a recap."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import manimgx as m
from deckkit import *  # noqa: F401,F403


def vdp_field(x, y, eps=1.0):
    return np.array([(y - x ** 3 / 3 + x) / eps, -x])


def arrow_field(ax, F, xs, ys, length=0.32, color=MUTED, width=2.0):
    """Unit-length arrows of the vector field F on a grid (direction only)."""
    g = m.VGroup()
    for x in xs:
        for y in ys:
            v = np.asarray(F(x, y), float)
            n = np.hypot(*v)
            if n < 1e-9:
                continue
            p = ax.c2p(x, y)
            d = np.array([v[0], v[1], 0.0]) / n * length
            g.add(m.Arrow(p - d / 2, p + d / 2, buff=0, stroke_width=width, color=color,
                          max_tip_length_to_length_ratio=0.35))
    return g


def strike(mob, color=RED):
    return m.Line(mob.get_corner(m.DL), mob.get_corner(m.UR), color=color, stroke_width=5)


def radial_r(r0, t, a=1.0):
    """Exact solution of r' = a r (1 - r^2)."""
    return 1.0 / np.sqrt(1.0 + (1.0 / r0 ** 2 - 1.0) * np.exp(-2 * a * t))


def hopf_r(r0, t, mu):
    """Exact solution of r' = mu r - r^3."""
    if abs(mu) < 1e-6:
        q = 2 * t
    else:
        q = -np.expm1(-2 * mu * t) / mu
    inv = np.exp(-2 * mu * t) / r0 ** 2 + q
    return 1.0 / np.sqrt(np.maximum(inv, 1e-12))


class Deck01(Deck):
    chapter = "Chapter 1"
    title = "Planar Dynamics: a Recap"
    subtitle = "Vector fields, flows, fixed points and limit cycles"

    def construct(self):
        self.title_card()
        self.vector_field()
        self.two_facts()
        self.flow_blob()
        self.flow_rules()
        self.linearise()
        self.trace_det()
        self.saddle_spiral()
        self.show_figure("ch01_linear", "The six basic linear phase portraits",
                         "Teal lines: eigenvectors with $\\mathrm{Re}\\,\\lambda<0$ (in), magenta: $\\mathrm{Re}\\,\\lambda>0$ (out). Hartman-Grobman: hyperbolic points look like this.")
        self.nullclines()
        self.radial_clock()
        self.cartesian()
        self.trapping()
        self.hopf()
        self.recap(["$\\dot{\\vec x} = \\vec F(\\vec x)$ is a vector field; its flow $\\Phi_t$ moves the whole plane",
                    "$\\Phi_0 = \\mathrm{id}$, $\\Phi_{t+s} = \\Phi_t\\circ\\Phi_s$, $\\Phi_t^{-1} = \\Phi_{-t}$",
                    "Near a hyperbolic fixed point the flow is linear: eigenvectors are the special directions",
                    "A stable limit cycle $\\Gamma$ is an isolated attracting closed orbit with period $T$",
                    "Hopf bifurcations create cycles; the twist $b$ will shape the isochrons"])

    # ------------------------------------------------------------------
    def vector_field(self):
        h = heading("A differential equation is a field of arrows")
        ax = plane((-3, 3, 1), (-3, 3, 1), width=6.0, height=6.0, at=pt(-3.1, -0.45), ticks=False)
        arrows = arrow_field(ax, vdp_field, np.linspace(-2.7, 2.7, 10), np.linspace(-2.7, 2.7, 10))
        e1 = tex(r"\dot x = f(x,y),\qquad \dot y = g(x,y)", font_size=38).move_to(pt(3.6, 2.3))
        self.play(m.FadeIn(h), m.Create(ax), m.Write(e1), run_time=1.2)
        self.play(m.LaggedStart(*[m.GrowArrow(a) for a in arrows], lag_ratio=0.01), run_time=2)
        e2 = tex(r"\dot{\vec x} = \vec F(\vec x)", font_size=46, color=YELLOW).move_to(pt(3.6, 1.2))
        why = prose("the arrow at $\\vec x$ is the velocity\nof a state sitting at $\\vec x$ right now", font_size=24).move_to(pt(3.6, 0.0))
        self.play(m.Write(e2), m.FadeIn(why))
        self.next_slide("vector field")
        P = data("ch01_blob")["paths"][7]
        k = m.ValueTracker(0)
        d = m.always_redraw(lambda: dot(ax, P[int(k.get_value())], color=MAGENTA, r=0.1))
        tr = m.always_redraw(lambda: curve(ax, P[: max(2, int(k.get_value()) + 1)], color=MAGENTA, width=3))
        msg = prose("a solution is a curve tangent to the\narrows everywhere, travelled at the\nspeed they prescribe: a phase portrait", font_size=24, color=MAGENTA).move_to(pt(3.6, -1.9))
        self.add(tr, d)
        self.play(m.FadeIn(msg), arrows.animate.set_opacity(0.55))
        self.play(k.animate.set_value(len(P) - 1), run_time=7, rate_func=m.linear)
        self.wait(BEAT)
        self.next_slide("solution follows arrows")
        self.clear()

    def two_facts(self):
        h = heading("Two facts we use constantly")
        self.play(m.FadeIn(h))
        ax = plane((-2, 2, 1), (-2, 2, 1), width=4.6, height=4.6, at=pt(-4.2, -0.6), ticks=False)
        f1 = lambda s: 0.5 * np.sin(1.4 * s) + 0.8
        f2 = lambda s: 0.5 * np.sin(1.4 * s + 0.6) - 0.7
        c1 = m.ParametricFunction(lambda s: ax.c2p(s, f1(s)), t_range=(-1.9, 1.9), color=TEAL, stroke_width=4)
        c2 = m.ParametricFunction(lambda s: ax.c2p(s, f2(s)), t_range=(-1.9, 1.9), color=ORANGE, stroke_width=4)
        bad = m.ParametricFunction(lambda s: ax.c2p(s, 0.8 * s), t_range=(-1.9, 1.9), color=RED, stroke_width=3)
        ss = np.linspace(-1.9, 1.9, 2000)
        sx = ss[np.argmin(np.abs(f1(ss) - 0.8 * ss))]
        cross = m.Cross(scale_factor=0.25, stroke_color=RED).move_to(ax.c2p(sx, 0.8 * sx))
        t1 = prose("1. Uniqueness: through every point passes exactly\none solution, so trajectories never cross.", font_size=24).move_to(pt(2.4, 1.6))
        self.play(m.Create(ax), m.Create(c1), m.Create(c2), m.FadeIn(t1), run_time=1.4)
        self.play(m.Create(bad), run_time=1.0)
        self.play(m.FadeIn(cross))
        why = prose("if two met, that point would start two solutions", font_size=22, color=RED).next_to(t1, m.DOWN, buff=0.25).align_to(t1, m.LEFT)
        self.play(m.FadeIn(why))
        self.next_slide("uniqueness")
        self.play(m.FadeOut(bad), m.FadeOut(cross))
        t2 = prose("2. Autonomy: if $\\vec x(t)$ is a solution, so is $\\vec x(t+s)$.\nThe future depends only on where you are,\nnot on when you arrived.", font_size=24).move_to(pt(2.4, -1.4)).align_to(t1, m.LEFT)
        k = m.ValueTracker(-1.9)
        a = m.always_redraw(lambda: m.Dot(ax.c2p(k.get_value(), f1(k.get_value())), radius=0.09, color=YELLOW))
        b = m.always_redraw(lambda: m.Dot(ax.c2p(max(-1.9, k.get_value() - 1.2), f1(max(-1.9, k.get_value() - 1.2))), radius=0.09, color=MAGENTA))
        self.play(m.FadeIn(t2), m.FadeIn(a), m.FadeIn(b))
        self.play(k.animate.set_value(1.9), run_time=4, rate_func=m.linear)
        lab = prose("magenta = yellow, started later", font_size=22, color=MAGENTA).next_to(ax, m.DOWN, buff=0.15)
        self.play(m.FadeIn(lab))
        self.next_slide("autonomy")
        self.clear()

    def flow_blob(self):
        d = data("ch01_blob")
        paths, cyc, dt = d["paths"], d["cyc"], float(d["dt"])
        h = heading("The flow moves the whole plane at once")
        ax = plane((-3, 3, 1), (-3, 3, 1), width=6.2, height=6.2, at=pt(-3.0, -0.45), ticks=False)
        arrows = arrow_field(ax, vdp_field, np.linspace(-2.7, 2.7, 10), np.linspace(-2.7, 2.7, 10)).set_opacity(0.35)
        e = tex(r"\Phi_t(\vec x_0) = \vec x(t)\ \text{ where }\ \vec x(0)=\vec x_0", font_size=36).move_to(pt(3.6, 2.4))
        self.play(m.FadeIn(h), m.Create(ax), m.FadeIn(arrows), m.Write(e), run_time=1.2)
        txt = prose("take every starting point and let it\nmove for time $t$: a map of the plane\nto itself, like a river carrying leaves", font_size=24).move_to(pt(3.6, 1.1))
        k = m.ValueTracker(0)
        blob = m.always_redraw(lambda: m.Polygon(*[ax.c2p(*paths[j, int(k.get_value())]) for j in range(paths.shape[0])],
                                                 color=MAGENTA, fill_color=MAGENTA, fill_opacity=0.55, stroke_width=2))
        clock = m.always_redraw(lambda: tex(r"t = %.1f" % (k.get_value() * dt), font_size=40, color=MAGENTA).move_to(pt(3.6, -0.3)))
        self.play(m.FadeIn(txt), m.FadeIn(blob), m.FadeIn(clock))
        self.next_slide("flow")
        self.play(k.animate.set_value(paths.shape[1] - 1), run_time=10, rate_func=m.linear)
        cy = curve(ax, cyc, color=INK, width=3.5, step=10)
        self.play(m.Create(cy), run_time=1.2)
        msg = prose("stretched along the motion, squeezed across it:\nthe disc ends up lying along the closed curve,\nthe limit cycle", font_size=24, color=YELLOW).move_to(pt(3.6, -2.0))
        self.play(m.FadeIn(msg))
        self.wait(HOLD)
        self.next_slide("blob on cycle")
        self.clear()

    def flow_rules(self):
        d = data("ch01_blob")
        cyc = d["cyc"]
        P = d["paths"][20]
        h = heading("Three rules of the flow")
        self.play(m.FadeIn(h))
        r1 = tex(r"\Phi_0 = \text{id}", font_size=40).move_to(pt(3.4, 2.3))
        r1l = prose("in zero time nothing moves", font_size=22, color=MUTED).next_to(r1, m.DOWN, buff=0.15)
        self.play(m.Write(r1), m.FadeIn(r1l))
        self.next_slide()
        ax = plane((-3, 3, 1), (-3, 3, 1), width=5.8, height=5.8, at=pt(-3.4, -0.5), ticks=False)
        tr = curve(ax, P, color=MUTED, width=2.5)
        i0, i1, i2 = 0, 60, 200
        x0 = dot(ax, P[i0], color=TEAL, r=0.1)
        self.play(m.Create(tr), m.FadeIn(x0))
        l0 = tex(r"\vec x_0", font_size=36, color=TEAL).move_to(ax.c2p(-0.45, 0.95))
        self.play(m.FadeIn(l0))
        seg1 = curve(ax, P[i0:i1 + 1], color=ORANGE, width=5)
        y = dot(ax, P[i1], color=ORANGE, r=0.1)
        ly = tex(r"\vec y = \Phi_s(\vec x_0)", font_size=36, color=ORANGE).move_to(ax.c2p(0.25, -0.45))
        self.play(m.Create(seg1), run_time=1.2)
        self.play(m.FadeIn(y), m.FadeIn(ly))
        seg2 = curve(ax, P[i1:i2 + 1], color=MAGENTA, width=5)
        z = dot(ax, P[i2], color=MAGENTA, r=0.1)
        lz = tex(r"\Phi_t(\vec y)", font_size=36, color=MAGENTA).move_to(ax.c2p(-0.85, -1.0))
        self.play(m.Create(seg2), run_time=1.2)
        self.play(m.FadeIn(z), m.FadeIn(lz))
        r2 = tex(r"\Phi_{t+s} = \Phi_t\circ\Phi_s", font_size=40, color=YELLOW).move_to(pt(3.4, 0.9))
        r2l = prose("by autonomy, the solution from $\\vec y$ is the old\nsolution shifted in time: after $t$ more it is\nwhere the old one is at time $t+s$", font_size=22, color=MUTED).next_to(r2, m.DOWN, buff=0.2)
        self.play(m.Write(r2), m.FadeIn(r2l))
        self.next_slide("composition")
        r3 = tex(r"\Phi_t^{-1} = \Phi_{-t}", font_size=40, color=TEAL).move_to(pt(3.4, -1.7))
        r3l = prose("run time backwards: where was this point\n$t$ time units ago?", font_size=22, color=MUTED).next_to(r3, m.DOWN, buff=0.2)
        back = curve(ax, P[i0:i2 + 1][::-1], color=TEAL, width=3)
        self.play(m.Write(r3), m.FadeIn(r3l))
        self.play(m.MoveAlongPath(z.copy().set_color(TEAL), back), run_time=2)
        self.wait(BEAT)
        self.next_slide("inverse")
        self.clear()

    def linearise(self):
        h = heading("Near a fixed point the flow is linear")
        e0 = tex(r"\vec F(\vec x^*) = \vec 0,\qquad \vec x = \vec x^* + \vec\xi", font_size=40).move_to(pt(0, 2.4))
        self.play(m.FadeIn(h), m.Write(e0))
        self.next_slide()
        e1 = tex(r"\dot{\vec\xi}", r"=", r"\vec F(\vec x^*+\vec\xi)", r"=", r"\vec F(\vec x^*)", r"+", r"D\vec F(\vec x^*)\,\vec\xi", r"+", r"O(\lvert\vec\xi\rvert^2)", font_size=40).move_to(pt(0, 1.2))
        why = prose("Taylor expansion of $\\vec F$ around the fixed point", font_size=24, color=MUTED).next_to(e1, m.DOWN, buff=0.3)
        self.play(m.Write(e1), m.FadeIn(why), run_time=1.6)
        self.next_slide()
        s = strike(e1[4])
        z = tex(r"\vec 0", font_size=36, color=RED).next_to(e1[4], m.UP, buff=0.2)
        w2 = prose("the first term vanishes: $\\vec x^*$ is a fixed point", font_size=24, color=RED).move_to(why)
        self.play(m.Create(s), m.FadeIn(z), m.FadeOut(why), m.FadeIn(w2))
        self.next_slide("F(x*) = 0")
        e2 = tex(r"\dot{\vec\xi} = A\vec\xi,\qquad A = D\vec F(\vec x^*) = \begin{pmatrix} \partial f/\partial x & \partial f/\partial y \\ \partial g/\partial x & \partial g/\partial y \end{pmatrix}", font_size=38, color=YELLOW).move_to(pt(0, -0.9))
        w3 = prose("drop the tiny quadratic terms: the Jacobian matrix $A$ is all that is left", font_size=24, color=MUTED).next_to(e2, m.DOWN, buff=0.25)
        self.play(m.Write(e2), m.FadeIn(w3), run_time=1.6)
        self.next_slide()
        e3 = tex(r"A\vec v = \lambda\vec v\ \ \Rightarrow\ \ \vec\xi(t) = e^{\lambda t}\vec v", font_size=40, color=TEAL).move_to(pt(0, -2.75))
        w4 = prose("both sides equal $\\lambda e^{\\lambda t}\\vec v$", font_size=24, color=MUTED).next_to(e3, m.DOWN, buff=0.2)
        self.play(m.Write(e3), m.FadeIn(w4))
        self.wait(HOLD)
        self.next_slide("eigen solutions")
        self.clear()

    def trace_det(self):
        h = heading("Trace and determinant decide everything")
        e = tex(r"\lambda_{1,2} = \frac{\tau \pm \sqrt{\tau^2-4\Delta}}{2},\qquad \tau = \text{tr}\,A,\ \ \Delta = \det A", font_size=38).move_to(pt(0, 2.45))
        self.play(m.FadeIn(h), m.Write(e))
        ax = plane((-4, 4, 2), (-2, 4.5, 2), width=7.4, height=4.6, at=pt(-2.5, -1.3), xlabel=r"\tau", ylabel=r"\Delta", ticks=False)
        par = ax.plot(lambda t: t * t / 4, x_range=(-4, 4), color=YELLOW, stroke_width=3)
        cen = m.Line(ax.c2p(0, 0), ax.c2p(0, 4.5), color=ORANGE, stroke_width=4)
        self.play(m.Create(ax), run_time=1)
        self.play(m.Create(par), m.Create(cen), run_time=1.4)
        lab = [
            ("saddle", (2.2, -1.1), MAGENTA), ("stable\nnode", (-3.1, 0.75), TEAL), ("unstable\nnode", (3.1, 0.75), MAGENTA),
            ("stable\nspiral", (-1.3, 3.3), TEAL), ("unstable\nspiral", (1.3, 3.3), MAGENTA),
        ]
        g = m.VGroup(*[prose(s, font_size=22, color=c).move_to(ax.c2p(*p)) for s, p, c in lab])
        self.play(m.LaggedStart(*[m.FadeIn(x) for x in g], lag_ratio=0.2), run_time=1.5)
        side = m.VGroup(
            prose("$\\Delta<0$: real eigenvalues of opposite sign", font_size=22),
            prose("$\\tau^2>4\\Delta>0$: same sign, a node", font_size=22),
            prose("$\\tau^2<4\\Delta$: complex $\\tau/2 \\pm i\\omega$, a spiral", font_size=22),
            prose("$\\tau = 0$: centre, the Hopf borderline", font_size=22, color=ORANGE),
            prose("yellow parabola: $\\tau^2 = 4\\Delta$", font_size=22, color=YELLOW),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.3).move_to(pt(4.2, -1.1))
        self.play(m.FadeIn(side))
        self.wait(HOLD)
        self.next_slide("trace-determinant plane")
        self.clear()

    def saddle_spiral(self):
        h = heading("Watch the eigenvectors at work")
        axL = plane((-2.4, 2.4, 1), (-2.2, 2.2, 1), width=5.2, height=4.5, at=pt(-3.5, -0.55), ticks=False)
        axR = plane((-2.4, 2.4, 1), (-2.2, 2.2, 1), width=5.2, height=4.5, at=pt(3.5, -0.55), ticks=False)
        tL = tex(r"\text{saddle } A=\begin{pmatrix}1&0\\0&-1\end{pmatrix}", font_size=34).next_to(axL, m.UP, buff=0.15)
        tR = tex(r"\text{spiral } A=\begin{pmatrix}-0.3&-1\\1&-0.3\end{pmatrix}", font_size=34).next_to(axR, m.UP, buff=0.15)
        self.play(m.FadeIn(h), m.Create(axL), m.Create(axR), m.FadeIn(tL), m.FadeIn(tR), run_time=1.2)
        unst = m.Line(axL.c2p(-2, 0), axL.c2p(2, 0), color=MAGENTA, stroke_width=4)
        stab = m.Line(axL.c2p(0, -2), axL.c2p(0, 2), color=TEAL, stroke_width=4)
        self.play(m.Create(unst), m.Create(stab))
        starts = [(sx * e, sy * 1.9) for e in (0.05, 0.15, 0.4) for sx in (-1, 1) for sy in (-1, 1)]
        spst = [(1.8 * np.cos(a), 1.8 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
        k = m.ValueTracker(0.0)

        def sad(p, t):
            return np.array([p[0] * np.exp(t), p[1] * np.exp(-t)])

        def spi(p, t):
            c, s = np.cos(t), np.sin(t)
            return np.exp(-0.3 * t) * np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])

        def clip(X):
            X = np.array(X)
            ok = np.all(np.abs(X) <= 2.0, axis=1)
            return X[: np.argmin(ok) if not ok.all() else len(X)]

        def trails():
            t = k.get_value()
            ts = np.linspace(0, t, 120) if t > 0 else np.array([0.0, 0.0])
            gL = m.VGroup()
            for p in starts:
                X = clip([sad(p, s) for s in ts * 0.55])
                if len(X) > 1:
                    gL.add(curve(axL, X, color=YELLOW, width=2.5))
                    if np.all(np.abs(X[-1]) < 1.97):
                        gL.add(dot(axL, X[-1], color=YELLOW, r=0.06))
            gR = m.VGroup()
            for p in spst:
                X = np.array([spi(p, s) for s in ts])
                gR.add(curve(axR, X, color=YELLOW, width=2.5))
                gR.add(dot(axR, X[-1], color=YELLOW, r=0.06))
            return m.VGroup(gL, gR)

        tr = m.always_redraw(trails)
        self.add(tr)
        self.play(k.animate.set_value(12.0), run_time=7, rate_func=m.linear)
        msg = prose("saddle: in along the teal stable direction, out along the magenta unstable one\nspiral: complex eigenvalues turn, $\\tau<0$ shrinks", font_size=22, color=MUTED).to_edge(m.DOWN, buff=0.15)
        self.play(m.FadeIn(msg))
        self.wait(BEAT)
        self.next_slide("saddle and spiral")
        msg2 = prose("the teal and magenta lines start the stable and unstable manifolds $W^s$, $W^u$ (Chapter 5)", font_size=22, color=TEAL).to_edge(m.DOWN, buff=0.15)
        self.play(m.FadeOut(msg), m.FadeIn(msg2))
        self.wait(BEAT)
        self.next_slide("manifolds")
        self.clear()

    def nullclines(self):
        d = data("ch01_fhn")
        cyc = d["cyc"]
        I, a, b = 0.5, 0.7, 0.8
        h = heading("Nullclines: FitzHugh-Nagumo, a toy neuron")
        ax = plane((-2.5, 2.5, 1), (-1, 2, 1), width=7.4, height=4.8, at=pt(-2.7, -0.7), xlabel="v", ylabel="w", ticks=False)
        eq = tex(r"\dot v = v - \frac{v^3}{3} - w + I", font_size=36).move_to(pt(4.2, 2.4))
        eq2 = tex(r"\dot w = 0.08\,(v + 0.7 - 0.8\,w)", font_size=36).next_to(eq, m.DOWN, buff=0.25)
        self.play(m.FadeIn(h), m.Create(ax), m.Write(eq), m.Write(eq2), run_time=1.2)
        F = lambda v, w: np.array([v - v ** 3 / 3 - w + I, 0.08 * (v + a - b * w)])
        arrows = arrow_field(ax, F, np.linspace(-2.3, 2.3, 11), np.linspace(-0.85, 1.85, 7), length=0.3).set_opacity(0.4)
        self.play(m.FadeIn(arrows))
        vn = ax.plot(lambda v: v - v ** 3 / 3 + I, x_range=(-2.2, 2.2), color=GREEN, stroke_width=4)
        wn = ax.plot(lambda v: (v + a) / b, x_range=(-1.5, 0.9), color=RED, stroke_width=4)
        l1 = prose("green: $\\dot v = 0$, flow purely vertical", font_size=22, color=GREEN).move_to(pt(4.2, 0.9))
        self.play(m.Create(vn), m.FadeIn(l1))
        self.next_slide()
        l2 = prose("red: $\\dot w = 0$, flow purely horizontal", font_size=22, color=RED).next_to(l1, m.DOWN, buff=0.2).align_to(l1, m.LEFT)
        self.play(m.Create(wn), m.FadeIn(l2))
        # fixed point
        vs = np.roots([-1 / 3, 0, 1 - 1 / b, I - a / b])
        v0 = float(np.real(vs[np.argmin(np.abs(np.imag(vs)))]))
        fp = m.Dot(ax.c2p(v0, (v0 + a) / b), radius=0.1, color=YELLOW)
        l3 = prose("they cross at the fixed point\n(unstable here)", font_size=22, color=YELLOW).next_to(l2, m.DOWN, buff=0.2).align_to(l1, m.LEFT)
        self.play(m.FadeIn(fp, scale=2), m.FadeIn(l3))
        self.next_slide("nullclines")
        cy = curve(ax, cyc, color=INK, width=3.5, step=5)
        self.play(m.Create(cy), run_time=2)
        k = m.ValueTracker(0)
        dd = m.always_redraw(lambda: dot(ax, cyc[int(k.get_value()) % len(cyc)], color=MAGENTA, r=0.11))
        l4 = prose("every other trajectory ends on this\nclosed curve: the neuron fires forever", font_size=22, color=MAGENTA).next_to(l3, m.DOWN, buff=0.3).align_to(l1, m.LEFT)
        self.play(m.FadeIn(dd), m.FadeIn(l4))
        self.play(k.animate.set_value(2 * len(cyc) - 1), run_time=8, rate_func=m.linear)
        self.next_slide("fhn cycle")
        self.clear()

    def radial_clock(self):
        h = heading("A limit cycle: the radial isochron clock")
        eq = tex(r"\dot r = r(1-r^2),\qquad \dot\theta = 1", font_size=42).move_to(pt(3.4, 2.4))
        ax = plane((-2, 2, 1), (-2, 2, 1), width=5.6, height=5.6, at=pt(-3.4, -0.5), ticks=False)
        R = ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0]
        self.play(m.FadeIn(h), m.Create(ax), m.Write(eq), run_time=1.2)
        # phase line of r
        rl = m.NumberLine(x_range=(0, 2, 0.5), length=5.0, include_numbers=False, color=MUTED).move_to(pt(3.4, 0.9))
        rl.add(*[tex(s, font_size=34, color=MUTED).next_to(rl.n2p(v), m.DOWN, buff=0.15) for v, s in ((0, "0"), (1, "1"), (2, "2"))])
        rlab = tex("r", font_size=36, color=MUTED).next_to(rl, m.RIGHT, buff=0.2)
        o = m.Circle(radius=0.09, color=INK, stroke_width=3).move_to(rl.n2p(0))
        one = m.Dot(rl.n2p(1), radius=0.1, color=INK)
        arr = m.VGroup(m.Arrow(rl.n2p(0.25), rl.n2p(0.75), buff=0, color=TEAL, stroke_width=4),
                       m.Arrow(rl.n2p(1.85), rl.n2p(1.2), buff=0, color=TEAL, stroke_width=4)).shift(m.UP * 0.35)
        self.play(m.Create(rl), m.FadeIn(rlab), m.FadeIn(o), m.FadeIn(one), m.GrowArrow(arr[0]), m.GrowArrow(arr[1]))
        why = prose("$\\dot r>0$ for $0<r<1$, $\\dot r<0$ for $r>1$:\nevery $r \\neq 0$ is driven to $r=1$,\nwhile $\\dot\\theta = 1$ turns everything at unit speed", font_size=24).move_to(pt(3.4, -0.9))
        self.play(m.FadeIn(why))
        self.next_slide("radial equation")
        starts = [(0.08, 0.0), (0.25, 2.1), (1.9, 1.0), (1.6, 4.0), (0.5, 4.5)]
        k = m.ValueTracker(0.0)

        def pic():
            t = k.get_value()
            ts = np.linspace(0, t, 200)
            g = m.VGroup()
            for r0, a0 in starts:
                r = radial_r(r0, ts)
                X = np.column_stack([r * np.cos(a0 + ts), r * np.sin(a0 + ts)])
                g.add(curve(ax, X, color=YELLOW, width=2.2))
                g.add(dot(ax, X[-1], color=YELLOW, r=0.07))
                g.add(m.Dot(rl.n2p(r[-1]), radius=0.07, color=YELLOW))
            return g

        pc = m.always_redraw(pic)
        self.add(pc)
        self.play(k.animate.set_value(12.0), run_time=8, rate_func=m.linear)
        circ = m.Circle(radius=R, color=INK, stroke_width=5).move_to(ax.c2p(0, 0))
        res = prose("$r=1$: a stable limit cycle with period $T = 2\\pi$\n(isolated: neighbours are not closed, they spiral in)", font_size=24, color=YELLOW).move_to(pt(3.4, -2.6))
        self.play(m.Create(circ), m.FadeIn(res))
        self.wait(HOLD)
        self.next_slide("limit cycle")
        self.clear()

    def cartesian(self):
        h = heading("The same clock in Cartesian coordinates")
        e0 = tex(r"x = r\cos\theta,\qquad y = r\sin\theta", font_size=40).move_to(pt(0, 2.3))
        self.play(m.FadeIn(h), m.Write(e0))
        self.next_slide()
        e1 = tex(r"\dot x", r"=", r"\dot r\cos\theta - r\sin\theta\,\dot\theta", font_size=42).move_to(pt(0, 1.1))
        w1 = prose("product and chain rule", font_size=24, color=MUTED).next_to(e1, m.DOWN, buff=0.2)
        self.play(m.Write(e1), m.FadeIn(w1))
        self.next_slide()
        e2 = tex(r"=", r"r(1-r^2)\cos\theta", r"-", r"r\sin\theta", r"\cdot 1", font_size=42).move_to(pt(0, -0.3))
        w2 = prose("insert $\\dot r = r(1-r^2)$ and $\\dot\\theta = 1$", font_size=24, color=MUTED).next_to(e2, m.DOWN, buff=0.2)
        self.play(m.FadeOut(w1), m.Write(e2), m.FadeIn(w2))
        self.next_slide()
        e3 = tex(r"\dot x = x(1-x^2-y^2) - y,\qquad \dot y = y(1-x^2-y^2) + x", font_size=40, color=YELLOW).move_to(pt(0, -1.9))
        w3 = prose("since $r\\cos\\theta = x$, $r\\sin\\theta = y$, $r^2 = x^2+y^2$; similarly for $\\dot y$", font_size=24, color=MUTED).next_to(e3, m.DOWN, buff=0.25)
        self.play(m.FadeOut(w2), m.Write(e3), m.FadeIn(w3))
        self.wait(HOLD)
        self.next_slide("cartesian form")
        self.clear()

    def trapping(self):
        h = heading("Poincare-Bendixson: trap a trajectory, get a cycle")
        ax = plane((-2.4, 2.4, 1), (-2.4, 2.4, 1), width=5.8, height=5.8, at=pt(-3.3, -0.5), ticks=False)
        u = ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0]
        ann = m.Annulus(inner_radius=0.5 * u, outer_radius=2 * u, fill_color=TEAL, fill_opacity=0.18, stroke_width=0).move_to(ax.c2p(0, 0))
        cin = m.Circle(radius=0.5 * u, color=TEAL, stroke_width=3).move_to(ax.c2p(0, 0))
        cout = m.Circle(radius=2 * u, color=TEAL, stroke_width=3).move_to(ax.c2p(0, 0))
        self.play(m.FadeIn(h), m.Create(ax), run_time=0.8)
        self.play(m.FadeIn(ann), m.Create(cin), m.Create(cout))
        thm = prose("A trajectory that stays forever in a closed,\nbounded region with no fixed points\napproaches a closed orbit.", font_size=24).move_to(pt(3.5, 2.0))
        self.play(m.FadeIn(thm))
        self.next_slide()
        ins = m.VGroup()
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            c, s = np.cos(a), np.sin(a)
            ins.add(m.Arrow(ax.c2p(0.25 * c, 0.25 * s), ax.c2p(0.75 * c, 0.75 * s), buff=0, color=ORANGE, stroke_width=4, max_tip_length_to_length_ratio=0.4))
            ins.add(m.Arrow(ax.c2p(2.35 * c, 2.35 * s), ax.c2p(1.7 * c, 1.7 * s), buff=0, color=ORANGE, stroke_width=4, max_tip_length_to_length_ratio=0.35))
        self.play(m.LaggedStart(*[m.GrowArrow(x) for x in ins], lag_ratio=0.05), run_time=1.5)
        e1 = tex(r"r = \frac12:\quad \dot r = \frac12\Big(1-\frac14\Big) = \frac38 > 0", font_size=36).move_to(pt(3.5, 0.1))
        e2 = tex(r"r = 2:\quad \dot r = 2(1-4) = -6 < 0", font_size=36).move_to(pt(3.5, -1.0))
        self.play(m.Write(e1))
        self.play(m.Write(e2))
        self.next_slide()
        concl = prose("the flow only enters the annulus and it\nholds no fixed point, so a closed orbit\nmust lie inside: here $r = 1$", font_size=24, color=YELLOW).move_to(pt(3.5, -2.5))
        circ = m.Circle(radius=u, color=INK, stroke_width=4).move_to(ax.c2p(0, 0))
        self.play(m.FadeIn(concl), m.Create(circ))
        self.wait(HOLD)
        self.next_slide("trapping region")
        self.clear()

    def hopf(self):
        d = data("ch01_hopfnf")
        eq, po = d["eq"], d["po"]
        h = heading("The Hopf bifurcation: how clocks are born")
        e = tex(r"\dot r = \mu r - r^3,\qquad \dot\theta = \omega + b r^2", font_size=40).move_to(pt(0, 2.55))
        self.play(m.FadeIn(h), m.Write(e))
        axB = plane((-0.5, 1.0, 0.5), (-1.1, 1.1, 0.5), width=5.6, height=4.0, at=pt(-3.3, -0.75), xlabel=r"\mu", ylabel="x", ticks=False)
        axB.add(*[tex(s_, font_size=30, color=MUTED).next_to(axB.c2p(v, 0), m.DOWN, buff=0.12) for v, s_ in ((0.5, "0.5"),)])
        axB.add(*[tex(s_, font_size=30, color=MUTED).next_to(axB.c2p(0, v), m.LEFT, buff=0.12) for v, s_ in ((1.0, "1"), (-1.0, "-1"))])
        axP = plane((-1.2, 1.2, 1), (-1.2, 1.2, 1), width=4.4, height=4.4, at=pt(3.6, -1.0), ticks=False)
        self.play(m.Create(axB), m.Create(axP), run_time=1)
        st = eq[eq[:, 2] > 0.5]
        un = eq[eq[:, 2] < 0.5]
        eqs = curve(axB, st[:, :2], color=TEAL, width=4)
        equ = m.DashedVMobject(curve(axB, un[:, :2], color=TEAL, width=3), num_dashes=20)
        pmax = curve(axB, po[:, [0, 1]], color=MAGENTA, width=4)
        pmin = curve(axB, po[:, [0, 2]], color=MAGENTA, width=4)
        hb = m.Square(side_length=0.18, color=ORANGE, fill_opacity=1).move_to(axB.c2p(0, 0))
        self.play(m.Create(eqs), m.Create(equ), run_time=1)
        self.play(m.Create(pmax), m.Create(pmin), m.FadeIn(hb), run_time=1.4)
        lab = prose("teal: rest state (dashed = unstable)\nmagenta: $\\pm\\sqrt{\\mu}$, the cycle (AUTO)", font_size=22, color=MUTED).next_to(axB, m.DOWN, buff=0.15)
        self.play(m.FadeIn(lab))
        self.next_slide("bifurcation diagram")
        mu = m.ValueTracker(-0.35)
        marker = m.always_redraw(lambda: m.Line(axB.c2p(mu.get_value(), -1.1), axB.c2p(mu.get_value(), 1.1), color=YELLOW, stroke_width=3))
        ts = np.linspace(0, 40, 700)

        def traj():
            u = mu.get_value()
            g = m.VGroup()
            for r0, a0, col in ((0.12, 0.0, YELLOW), (1.15, np.pi, ORANGE)):
                r = hopf_r(r0, ts, u)
                g.add(curve(axP, np.column_stack([r * np.cos(a0 + ts), r * np.sin(a0 + ts)]), color=col, width=2))
            return g

        tr = m.always_redraw(traj)
        mul = m.always_redraw(lambda: tex(r"\mu = %.2f" % mu.get_value(), font_size=36, color=YELLOW).next_to(axP, m.UP, buff=0.1))
        self.play(m.FadeOut(lab), m.FadeIn(marker), m.FadeIn(tr), m.FadeIn(mul))
        self.next_slide()
        self.play(mu.animate.set_value(0.0), run_time=4, rate_func=m.linear)
        self.play(mu.animate.set_value(0.85), run_time=6, rate_func=m.linear)
        self.next_slide("sweep mu")
        self.play(m.FadeOut(axB), m.FadeOut(eqs), m.FadeOut(equ), m.FadeOut(pmax), m.FadeOut(pmin), m.FadeOut(hb), m.FadeOut(marker))
        notes = m.VGroup(
            prose("$\\mu<0$: stable spiral at the origin", font_size=24),
            prose("$\\mu>0$: $\\dot r = 0 \\Rightarrow r = \\sqrt{\\mu}$, a stable cycle", font_size=24),
            prose("(divide $\\mu r - r^3 = 0$ by $r$)", font_size=22, color=MUTED),
            prose("its radius grows like $\\sqrt{\\mu}$", font_size=24),
            prose("twist $b$: speed depends on amplitude;\nno effect on the cycle, but it will\nreshape the isochrons (Chapter 4)", font_size=24, color=ORANGE),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.3).move_to(pt(-3.2, -0.9))
        self.play(m.LaggedStart(*[m.FadeIn(x) for x in notes], lag_ratio=0.3), run_time=2)
        self.wait(HOLD)
        self.next_slide("hopf summary")
        self.clear()
