"""Deck for Chapter 5: Isochrons are stable manifolds."""
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


def strike(mob, color=RED):
    return m.Line(mob.get_corner(m.DL), mob.get_corner(m.UR), color=color, stroke_width=5)


def phase_at(xs, ys, th, p):
    j = int(np.argmin(np.abs(xs - p[0]))); i = int(np.argmin(np.abs(ys - p[1])))
    return float(th[i, j])


# ---- the weakly attracting twisted oscillator of Figure 5.2 (exact flow) ----
A5, B5, W5 = 0.1, 0.25, 1.0


def sl_flow(x, y, t):
    """Exact flow of r' = a r (1 - r^2), th' = w - b r^2 (NaN where it blows up)."""
    r0 = np.hypot(x, y); th0 = np.arctan2(y, x)
    C = 1.0 / r0 ** 2 - 1.0
    q = 1.0 + C * np.exp(-2 * A5 * t)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(q > 0, 1.0 / q, np.nan)
        th = th0 + W5 * t - B5 * (t + np.log(q / (1.0 + C)) / (2 * A5))
    r = np.sqrt(r2)
    return r * np.cos(th), r * np.sin(th)


class Deck05(Deck):
    chapter = "Chapter 5"
    title = "Isochrons Are Stable Manifolds"
    subtitle = "The theorem behind the whole paper"

    def construct(self):
        self.title_card()
        self.claim()
        self.saddle()
        self.local_to_global_saddle()
        self.maps()
        self.three_facts()
        self.iterates()
        self.foliation()
        self.strong()
        self.bundle()
        self.backward_growth()
        self.why_hard()
        self.recap(["A stable manifold = all points that converge to a fixed point;\nsmooth, tangent to $E^s$, grown from a local piece by backward flow",
                    "$I(\\vec\\gamma) = W^{ss}(\\vec\\gamma)$ = stable manifold of $\\vec\\gamma$ for the time-$T$ map $\\Phi_T$",
                    "Isochrons exist and are smooth: Hirsch-Pugh-Shub invariant foliation",
                    "The isochron leaves $\\Gamma$ along $\\vec v_s$: $L_\\eta = \\vec\\gamma + \\delta\\,\\vec v_s$",
                    "Global isochron = backward images $\\Phi_{-kT}(L_\\eta)$; naive version is ill-conditioned"])

    # ------------------------------------------------------------------
    def claim(self):
        h = heading("The one sentence of this chapter")
        box = prose("An isochron is the stable manifold\nof a fixed point of the time-$T$ map.", font_size=38, color=YELLOW)
        frame = m.SurroundingRectangle(box, color=TEAL, buff=0.35, corner_radius=0.12)
        m.VGroup(box, frame).move_to(pt(0, 1.3))
        self.play(m.FadeIn(h), m.Write(box), m.Create(frame), run_time=1.6)
        self.next_slide("claim")
        items = ["why isochrons exist", "why they are smooth curves",
                 "which direction they leave the cycle in", "how to compute them: grow locally, flow backwards"]
        rows = m.VGroup(*[prose("• it explains " + s, font_size=28) for s in items]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.28)
        rows.move_to(pt(0, -1.6))
        for r in rows:
            self.play(m.FadeIn(r, shift=m.RIGHT * 0.2), run_time=0.6)
        self.wait(HOLD)
        self.next_slide("consequences")
        self.clear()

    def saddle(self):
        h = heading("The simplest stable manifold: a saddle")
        ax = plane((-2, 2, 1), (-2, 2, 1), width=5.8, height=5.8, at=pt(-3.4, -0.45), ticks=False)
        eq = tex(r"\dot x = x,\qquad \dot y = -y + x^2", font_size=40).move_to(pt(3.4, 2.3))
        self.play(m.FadeIn(h), m.Create(ax), m.Write(eq), run_time=1.2)
        # a grid of states moved by the exact flow
        g = np.linspace(-1.8, 1.8, 7)
        P0 = np.array([(x, y) for x in g for y in g])
        t = m.ValueTracker(0.0)

        def flow(p, s):
            x0, y0 = p
            x = x0 * np.exp(s)
            y = (y0 - x0 ** 2 / 3) * np.exp(-s) + x0 ** 2 / 3 * np.exp(2 * s)
            return x, y

        def cloud():
            pts = []
            for p in P0:
                x, y = flow(p, t.get_value())
                if abs(x) < 2.05 and abs(y) < 2.05:
                    pts.append(m.Dot(ax.c2p(x, y), radius=0.05, color=YELLOW))
            return m.VGroup(*pts)
        dots = m.always_redraw(cloud)
        txt = prose("states slide in along one direction\nand are flung out along the other", font_size=26).move_to(pt(3.4, 1.1))
        self.add(dots)
        self.play(m.FadeIn(txt))
        self.next_slide("flow")
        self.play(t.animate.set_value(2.2), run_time=4, rate_func=m.linear)
        self.next_slide()
        self.remove(dots)
        ws = m.Line(ax.c2p(0, -2), ax.c2p(0, 2), color=TEAL, stroke_width=5)
        xx = np.linspace(-2, 2, 120)
        wu = curve(ax, np.column_stack([xx, xx ** 2 / 3]), color=MAGENTA, width=5)
        eu = m.DashedLine(ax.c2p(-2, 0), ax.c2p(2, 0), color=MAGENTA, stroke_width=2.5)
        lab_s = m.MathTex(r"W^s", font_size=36, color=TEAL).next_to(ax.c2p(0, 1.8), m.RIGHT, buff=0.15)
        lab_u = m.MathTex(r"W^u", font_size=36, color=MAGENTA).move_to(ax.c2p(1.0, 1.25))
        lab_e = m.MathTex(r"E^u", font_size=32, color=MAGENTA).next_to(ax.c2p(1.75, 0), m.DOWN, buff=0.15)
        sad = dot(ax, (0, 0), color=INK, r=0.09)
        self.play(m.Create(ws), m.Create(wu), m.Create(eu), m.FadeIn(sad), m.FadeIn(lab_s), m.FadeIn(lab_u), m.FadeIn(lab_e), run_time=1.6)
        self.play(m.FadeOut(txt))
        why = prose("on $x = 0$: $\\dot x = 0$ and $\\dot y = -y$,\nso the $y$-axis is $W^s$ exactly", font_size=26, color=TEAL).move_to(pt(3.4, 1.0))
        self.play(m.FadeIn(why))
        self.next_slide("Ws")
        c1 = tex(r"y = \frac{x^2}{3}", r"\ \Rightarrow\ ", r"\dot y = \frac{2}{3}x\,\dot x = \frac{2}{3}x^2", font_size=36).move_to(pt(3.4, -0.5))
        c2 = tex(r"-y + x^2 = -\frac{x^2}{3} + x^2 = \frac{2}{3}x^2", font_size=36).next_to(c1, m.DOWN, buff=0.35)
        self.play(m.Write(c1))
        self.next_slide()
        self.play(m.Write(c2))
        ok = prose("both sides agree: the parabola is invariant,\nand it is tangent to $E^u$ at the saddle", font_size=24, color=MAGENTA).next_to(c2, m.DOWN, buff=0.35)
        self.play(m.FadeIn(ok))
        self.wait(HOLD)
        self.next_slide("Wu parabola")
        self.clear()

    def local_to_global_saddle(self):
        h = heading("Global from local: let the flow grow a tiny piece")
        ax = plane((-2, 2, 1), (-0.6, 2, 1), width=6.4, height=4.2, at=pt(-3.2, -0.2), ticks=False)
        xx = np.linspace(-2, 2, 120)
        wu = curve(ax, np.column_stack([xx, xx ** 2 / 3]), color=MAGENTA, width=2).set_stroke(opacity=0.35)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(wu), run_time=1.0)
        s = np.linspace(-0.15, 0.15, 121)
        t = m.ValueTracker(0.0)

        def seg():
            tt = t.get_value()
            X = np.column_stack([s * np.exp(tt), s ** 2 / 3 * (np.exp(2 * tt) - np.exp(-tt))])
            X = X[(np.abs(X[:, 0]) < 2.0) & (X[:, 1] < 2.0)]
            return curve(ax, X, color=YELLOW, width=5)
        piece = m.always_redraw(seg)
        txt = prose("start: a short straight piece of $E^u$\n(length $0.3$)", font_size=26).move_to(pt(3.5, 1.9))
        self.add(piece)
        self.play(m.FadeIn(txt))
        self.next_slide("tiny piece")
        self.play(t.animate.set_value(2.6), run_time=5, rate_func=m.linear)
        txt2 = prose("the flow stretches it and bends it\nonto the true curve $W^u$", font_size=26, color=YELLOW).next_to(txt, m.DOWN, buff=0.4)
        self.play(m.FadeIn(txt2))
        self.next_slide("grown")
        e = tex(r"W^s(\vec x^*) = \bigcup_{t\ge 0}\Phi_{-t}\big(W^s_{\text{loc}}(\vec x^*)\big)", font_size=38, color=TEAL).move_to(pt(3.5, -1.0))
        n = prose("for $W^s$: the same, with time reversed\n(every algorithm starts here, the paper too)", font_size=24, color=MUTED).next_to(e, m.DOWN, buff=0.35)
        self.play(m.Write(e), m.FadeIn(n))
        self.wait(HOLD)
        self.next_slide("5.1")
        self.clear()

    def maps(self):
        h = heading("Maps have stable manifolds too")
        ax = plane((-2.2, 2.2, 1), (-0.6, 3.4, 1), width=6.0, height=5.4, at=pt(-3.3, -0.55), ticks=False)
        eq = tex(r"G(x,y) = \Big(\frac{x}{2},\ 2y - \frac{3}{2}x^2\Big)", font_size=36).move_to(pt(3.5, 2.3))
        xx = np.linspace(-2.1, 2.1, 120)
        ws = curve(ax, np.column_stack([xx, 6 / 7 * xx ** 2]), color=TEAL, width=4)
        p = dot(ax, (0, 0), color=INK, r=0.09)
        pl = m.MathTex(r"\vec p", font_size=34).next_to(p, m.DL, buff=0.08)
        self.play(m.FadeIn(h), m.Create(ax), m.Write(eq), m.FadeIn(p), m.FadeIn(pl), run_time=1.2)
        self.play(m.Create(ws))
        txt = prose("teal: the parabola $y = \\frac{6}{7}x^2$ is $W^s(\\vec p)$.\ncheck: $G(x, \\frac{6}{7}x^2) = (\\frac{x}{2}, \\frac{6}{7}\\frac{x^2}{4})$,\nback on the same parabola", font_size=24).move_to(pt(3.5, 1.0))
        self.play(m.FadeIn(txt))
        self.next_slide("map")
        def G(q):
            return np.array([q[0] / 2, 2 * q[1] - 1.5 * q[0] ** 2])
        q = np.array([1.9, 6 / 7 * 1.9 ** 2])
        d = dot(ax, q, color=YELLOW, r=0.1)
        self.play(m.FadeIn(d))
        for k in range(5):
            q2 = G(q)
            arc = m.ArcBetweenPoints(ax.c2p(*q), ax.c2p(*q2), angle=-0.8, color=YELLOW, stroke_width=2)
            self.play(m.Create(arc), d.animate.move_to(ax.c2p(*q2)), run_time=0.8)
            q = q2
        t2 = prose("each jump: distance to $\\vec p$ shrinks by $\\approx\\lambda = \\frac12$", font_size=24, color=YELLOW).move_to(pt(3.5, 0.0))
        self.play(m.FadeIn(t2))
        self.next_slide("hop")
        q = np.array([0.4, 6 / 7 * 0.16 + 0.12])
        d2 = dot(ax, q, color=ORANGE, r=0.1)
        self.play(m.FadeIn(d2))
        for k in range(4):
            q2 = G(q)
            if q2[1] > 3.3:
                break
            arc = m.ArcBetweenPoints(ax.c2p(*q), ax.c2p(*q2), angle=0.8, color=ORANGE, stroke_width=2)
            self.play(m.Create(arc), d2.animate.move_to(ax.c2p(*q2)), run_time=0.8)
            q = q2
        t3 = prose("a point slightly off $W^s$ is pushed away", font_size=24, color=ORANGE).next_to(t2, m.DOWN, buff=0.3)
        e = tex(r"W^s(\vec p) = \bigcup_{n\ge 0} G^{-n}\big(W^s_{\text{loc}}(\vec p)\big)", font_size=38, color=TEAL).move_to(pt(3.5, -1.9))
        self.play(m.FadeIn(t3), m.Write(e))
        self.wait(HOLD)
        self.next_slide("5.2")
        self.clear()

    def three_facts(self):
        h = heading("Put the pieces together: take G = time-T map")
        self.play(m.FadeIn(h))
        facts = [
            "1. $\\vec\\gamma \\in \\Gamma$ is a fixed point of $\\Phi_T$ (Chapter 3)",
            "2. $D\\Phi_T(\\vec\\gamma)$: eigenvalue $\\lambda$, $\\lvert\\lambda\\rvert < 1$, along $\\vec v_s$; eigenvalue $1$ along $\\vec F$",
            "3. same phase as $\\vec\\gamma$ $\\Leftrightarrow$ $\\lvert \\Phi_t(\\vec x) - \\Phi_t(\\vec\\gamma) \\rvert \\to 0$ (Chapter 4)",
        ]
        rows = m.VGroup(*[prose(f, font_size=27, width=12.6) for f in facts]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.32)
        rows.next_to(h, m.DOWN, buff=0.5).to_edge(m.LEFT, buff=0.7)
        for r in rows:
            self.play(m.FadeIn(r, shift=m.RIGHT * 0.2), run_time=0.7)
        self.next_slide("facts")
        s = prose("sample fact 3 at the times $t = nT$:", font_size=26, color=MUTED).next_to(rows, m.DOWN, buff=0.45).to_edge(m.LEFT, buff=0.7)
        e = tex(r"\Phi_{nT}(\vec x)", r"=", r"\Phi_T^{\,n}(\vec x)", r"\ \longrightarrow\ ", r"\Phi_T^{\,n}(\vec\gamma)", r"=", r"\vec\gamma", font_size=42).next_to(s, m.DOWN, buff=0.3)
        e.set_x(0)
        self.play(m.FadeIn(s), m.Write(e[:4]))
        self.next_slide()
        self.play(m.Write(e[4:]))
        b = m.Brace(e[6], m.DOWN, color=TEAL)
        bl = prose("$\\vec\\gamma$ is fixed by $\\Phi_T$", font_size=22, color=TEAL).next_to(b, m.DOWN, buff=0.1)
        self.play(m.GrowFromCenter(b), m.FadeIn(bl))
        self.next_slide()
        res = tex(r"I(\vec\gamma) = W^s\big(\vec\gamma;\ \Phi_T\big)", font_size=48, color=YELLOW).move_to(pt(0, -3.1))
        fr = m.SurroundingRectangle(res, color=YELLOW, buff=0.2)
        self.play(m.Write(res), m.Create(fr))
        self.wait(HOLD)
        self.next_slide("boxed")
        self.clear()

    def iterates(self):
        d = data("ch05_iterates")
        iso, its, lam = d["iso"], d["its"], float(d["lam"])
        T = float(d["T"])
        h = heading("Watch it: points on an isochron under the time-T map")
        R = 2.7
        ax = plane((-R, R, 1), (-R, R, 1), width=6.2, height=6.2, at=pt(-3.0, -0.45), ticks=False, xlabel="", ylabel="")
        circ = m.Circle(radius=ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0], color=INK, stroke_width=4).move_to(ax.c2p(0, 0))
        rr = np.geomspace(0.02, R, 400)
        faint = m.VGroup(*[curve(ax, np.column_stack([rr * np.cos(2.5 * np.log(rr) + 2 * np.pi * k / 10),
                                                       rr * np.sin(2.5 * np.log(rr) + 2 * np.pi * k / 10)]),
                                 color=phase_color(k / 10), width=1.5).set_stroke(opacity=0.35) for k in range(1, 10)])
        isoc = curve(ax, iso[(np.abs(iso[:, 0]) < R) & (np.abs(iso[:, 1]) < R)], color=TEAL, width=4)
        g0 = dot(ax, (1, 0), color=INK, r=0.1)
        eq = tex(r"\dot r = 0.1\,r(1-r^2),\quad \dot\theta = 1 - 0.25\,r^2", font_size=34).move_to(pt(3.6, 2.4))
        self.play(m.FadeIn(h), m.Create(ax), m.Create(circ), m.Write(eq), run_time=1.2)
        self.play(m.FadeIn(faint), m.Create(isoc), m.FadeIn(g0), run_time=1.4)
        txt = prose("white dot: $\\vec\\gamma_0 = (1, 0)$; teal: its\nisochron, the spiral $\\theta = 2.5\\ln r$", font_size=26, color=TEAL).move_to(pt(3.6, 1.3))
        self.play(m.FadeIn(txt))
        self.next_slide("setup")
        cols = [YELLOW, ORANGE]
        dots = [dot(ax, its[j, 0], color=cols[j], r=0.1) for j in range(2)]
        self.play(*[m.FadeIn(x) for x in dots])
        msg = prose("run each point for one period $T$:\nit loops once round and lands\nback on the same spiral", font_size=24, color=YELLOW).move_to(pt(3.6, -0.15))
        self.play(m.FadeIn(msg))
        self.next_slide()
        for k in range(4):
            anims, trails = [], []
            for j in range(2):
                ts = np.linspace(0, T, 160)
                x, y = sl_flow(its[j, k, 0] * np.ones_like(ts), its[j, k, 1] * np.ones_like(ts), ts)
                tr = curve(ax, np.column_stack([x, y]), color=cols[j], width=1.6).set_stroke(opacity=0.6)
                trails.append(tr)
                anims.append(m.MoveAlongPath(dots[j], tr))
            self.play(*[m.Create(tr) for tr in trails], *anims, run_time=1.8, rate_func=m.linear)
            self.play(*[m.FadeOut(tr) for tr in trails], run_time=0.3)
            for j in range(2):
                self.add(dot(ax, its[j, k + 1], color=cols[j], r=0.06))
        ratio = tex(r"\frac{\lvert \vec x_{n+1} - \vec\gamma_0 \rvert}{\lvert \vec x_n - \vec\gamma_0 \rvert}\approx \lambda = " + f"{lam:.2f}",
                    font_size=38, color=YELLOW).move_to(pt(3.6, -2.0))
        self.play(m.Write(ratio))
        note = prose("exactly how a stable manifold of a map behaves", font_size=22, color=MUTED).next_to(ratio, m.DOWN, buff=0.3)
        self.play(m.FadeIn(note))
        self.wait(HOLD)
        self.next_slide("iterates")
        self.clear()

    def foliation(self):
        d = data("ch04_vdp1")
        xs, ys, th, X = d["xs"], d["ys"], d["th"], d["X"]
        h = heading("A whole curve of fixed points: the invariant foliation")
        ax = plane((-3, 3, 1), (-3.4, 3.4, 1), width=5.4, height=6.1, at=pt(-3.9, -0.45), ticks=False, xlabel="", ylabel="")
        cyc = curve(ax, X, color=INK, width=3.5, step=10)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), run_time=1.0)
        N = 12
        leaves = [m.VGroup(*[curve(ax, s, color=phase_color(j / N), width=2.2) for s in level_curves(xs, ys, th, j / N)]) for j in range(N)]
        self.play(m.LaggedStart(*[m.Create(l) for l in leaves], lag_ratio=0.1), run_time=2.4)
        thm = prose("Hirsch-Pugh-Shub: the basin $B(\\Gamma)$ is filled\nby leaves $W^{ss}(\\vec\\gamma)$, one per $\\vec\\gamma \\in \\Gamma$", font_size=25).move_to(pt(2.9, 2.2))
        self.play(m.FadeIn(thm))
        self.next_slide("foliation")
        bl = ["each leaf is a curve through $\\vec\\gamma$, as smooth as $\\vec F$",
              "tangent at $\\vec\\gamma$ to the stable direction $\\vec v_s(\\vec\\gamma)$",
              "invariant: $\\Phi_t(W^{ss}(\\vec\\gamma)) = W^{ss}(\\Phi_t(\\vec\\gamma))$",
              "different leaves are disjoint and fill the basin"]
        rows = m.VGroup(*[prose("• " + b, font_size=23) for b in bl]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.22)
        rows.next_to(thm, m.DOWN, buff=0.4).align_to(thm, m.LEFT)
        for r in rows:
            self.play(m.FadeIn(r, shift=m.RIGHT * 0.2), run_time=0.6)
        self.next_slide()
        # invariance: one highlighted leaf travels with its cycle point
        for l in leaves:
            l.set_stroke(opacity=0.25)
        k = m.ValueTracker(0)
        def hl():
            j = int(k.get_value()) % N
            g = leaves[j].copy().set_stroke(color=YELLOW, width=5, opacity=1)
            g.add(dot(ax, X[int(j / N * (len(X) - 1))], color=YELLOW, r=0.11))
            return g
        lit = m.always_redraw(hl)
        self.add(lit)
        msg = prose("the flow carries a leaf to the leaf\nof the point it carries $\\vec\\gamma$ to", font_size=24, color=YELLOW).next_to(rows, m.DOWN, buff=0.4).align_to(thm, m.LEFT)
        self.play(m.FadeIn(msg))
        self.play(k.animate.set_value(2 * N - 0.01), run_time=6, rate_func=m.linear)
        self.wait(BEAT)
        self.next_slide("invariance")
        self.clear()

    def strong(self):
        h = heading("Why the word strong?")
        self.play(m.FadeIn(h))
        a = tex(r"W^s(\Gamma)", r"= \{\vec x : \Phi_t(\vec x) \to \Gamma \text{ as a set}\} = B(\Gamma)", font_size=40).move_to(pt(0, 1.6))
        b = tex(r"W^{ss}(\vec\gamma)", r"= \{\vec x : \lvert \Phi_t(\vec x) - \Phi_t(\vec\gamma) \rvert \to 0\}", font_size=40).move_to(pt(0, 0.2))
        a[0].set_color(MUTED); b[0].set_color(TEAL)
        self.play(m.Write(a))
        la = prose("every point of the basin converges to the cycle...", font_size=26, color=MUTED).next_to(a, m.DOWN, buff=0.25)
        self.play(m.FadeIn(la))
        self.next_slide()
        self.play(m.Write(b))
        lb = prose("...but only these converge in step with $\\vec\\gamma$: a stronger demand", font_size=26, color=TEAL).next_to(b, m.DOWN, buff=0.25)
        self.play(m.FadeIn(lb))
        self.next_slide()
        c = tex(r"I(\vec\gamma) = W^{ss}(\vec\gamma)", font_size=50, color=YELLOW).move_to(pt(0, -1.9))
        n = prose("\"as smooth as $\\vec F$\": isochrons have no corners, even when they look sharp", font_size=24, color=MUTED).next_to(c, m.DOWN, buff=0.35)
        self.play(m.Write(c), m.FadeIn(n))
        self.wait(HOLD)
        self.next_slide("strong stable")
        self.clear()

    def bundle(self):
        d = data("ch04_vdp1")
        xs, ys, th, X = d["xs"], d["ys"], d["th"], d["X"]
        b = data("ch05_bundle")
        us, vs = b["us"], b["vs"]
        h = heading("Which way does the isochron leave the cycle?")
        ax = plane((-3, 3, 1), (-3.4, 3.4, 1), width=5.4, height=6.1, at=pt(-3.9, -0.45), ticks=False, xlabel="", ylabel="")
        cyc = curve(ax, X, color=INK, width=3.5, step=10)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), run_time=1.0)
        e = tex(r"L_\eta(\vec\gamma) = \{\vec\gamma + \delta\,\vec v_s(\vec\gamma) : \lvert\delta\rvert \le \eta\}", font_size=36, color=ORANGE).move_to(pt(2.9, 2.3))
        t1 = prose("orange: the stable Floquet direction $\\vec v_s$\nat 20 cycle points (van der Pol, $\\varepsilon = 1$)", font_size=24, color=ORANGE).next_to(e, m.DOWN, buff=0.3)
        self.play(m.Write(e), m.FadeIn(t1))
        idx = np.linspace(0, len(us) - 1, 21).astype(int)[:-1]
        segs = m.VGroup()
        for i in idx:
            p, v = us[i], vs[i]
            segs.add(m.Line(ax.c2p(*(p - 0.45 * v)), ax.c2p(*(p + 0.45 * v)), color=ORANGE, stroke_width=5))
        self.play(m.LaggedStart(*[m.GrowFromCenter(s) for s in segs], lag_ratio=0.1), run_time=2.4)
        self.next_slide("bundle")
        isos = m.VGroup()
        for i in idx:
            c = phase_at(xs, ys, th, us[i])
            for sg in level_curves(xs, ys, th, c):
                isos.add(curve(ax, sg, color=phase_color(c), width=2))
        self.play(m.Create(isos), run_time=2.4)
        self.bring_to_front(segs)
        t2 = prose("each orange segment is tangent to the isochron\nof its point (computed independently)", font_size=24).next_to(t1, m.DOWN, buff=0.35)
        self.play(m.FadeIn(t2))
        self.next_slide("tangent")
        # not perpendicular: tangent F at a few points
        F = lambda u: np.array([u[1] - u[0] ** 3 / 3 + u[0], -u[0]])
        arrs = m.VGroup()
        for i in idx[::4]:
            p = us[i]; f = F(p); f = f / np.linalg.norm(f)
            arrs.add(m.Arrow(ax.c2p(*p), ax.c2p(*(p + 0.8 * f)), buff=0, color=BLUE, stroke_width=4, max_tip_length_to_length_ratio=0.3))
        self.play(m.LaggedStart(*[m.GrowArrow(a) for a in arrs], lag_ratio=0.15))
        t3 = prose("blue: the flow direction $\\vec F$. The angle between\n$\\vec v_s$ and $\\vec F$ varies: isochrons are usually\nnot perpendicular to the cycle", font_size=24, color=BLUE).next_to(t2, m.DOWN, buff=0.35)
        self.play(m.FadeIn(t3))
        self.next_slide()
        t4 = tex(r"\vec v_s(\Phi_s\vec\gamma)\ \parallel\ D\Phi_s(\vec\gamma)\,\vec v_s(\vec\gamma)", font_size=34).next_to(t3, m.DOWN, buff=0.35)
        t5 = prose("one eigenvector, carried round by the variational equation", font_size=21, color=MUTED).next_to(t4, m.DOWN, buff=0.15)
        self.play(m.Write(t4), m.FadeIn(t5))
        self.wait(HOLD)
        self.next_slide("transport")
        self.clear()

    def backward_growth(self):
        h = heading("From local to global: flow the segment backwards")
        ax = plane((-2.1, 2.1, 1), (-2.1, 2.1, 1), width=6.2, height=6.2, at=pt(-3.0, -0.45), ticks=False, xlabel="", ylabel="")
        circ = m.Circle(radius=ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0], color=INK, stroke_width=4).move_to(ax.c2p(0, 0))
        rr = np.geomspace(0.02, 2.1, 400)
        spiral = curve(ax, np.column_stack([rr * np.cos(2.5 * np.log(rr)), rr * np.sin(2.5 * np.log(rr))]), color=TEAL, width=2).set_stroke(opacity=0.4)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(circ), m.Create(spiral), run_time=1.2)
        e = tex(r"I(\vec\gamma) \approx \bigcup_{k\ge 0}\Phi_{-kT}\big(L_\eta(\vec\gamma)\big)", font_size=38, color=YELLOW).move_to(pt(3.6, 2.3))
        self.play(m.Write(e))
        v = np.array([1.0, 2.5]) / np.hypot(1, 2.5)
        dl = np.linspace(-0.004, 0.004, 4001)
        P = np.array([1.0, 0.0]) + dl[:, None] * v
        T = 2 * np.pi / (W5 - B5)
        t = m.ValueTracker(0.0)

        def img():
            x, y = sl_flow(P[:, 0], P[:, 1], -t.get_value())
            ok = np.isfinite(x) & (np.abs(x) < 2.1) & (np.abs(y) < 2.1)
            Q = np.column_stack([x, y])
            Q = np.where(ok[:, None], Q, np.nan)
            # split at gaps
            g = m.VGroup()
            run = []
            for q, o in zip(Q, ok):
                if o:
                    run.append(q)
                elif len(run) > 1:
                    g.add(curve(ax, np.array(run), color=ORANGE, width=5)); run = []
                else:
                    run = []
            if len(run) > 1:
                g.add(curve(ax, np.array(run), color=ORANGE, width=5))
            if len(g) == 0:
                g.add(dot(ax, (1, 0), color=ORANGE, r=0.06))
            return g
        seg = m.always_redraw(img)
        ring = m.Circle(radius=0.22, color=ORANGE, stroke_width=3).move_to(ax.c2p(1, 0))
        self.add(seg)
        self.play(m.Create(ring))
        t1 = prose("orange: $L_\\eta$, a straight piece of length $0.008$", font_size=24, color=ORANGE).move_to(pt(3.6, 1.2))
        self.play(m.FadeIn(t1))
        self.next_slide("L_eta")
        kl = m.always_redraw(lambda: m.MathTex(r"t = -%.2f\,T" % (t.get_value() / T), font_size=36, color=ORANGE).move_to(pt(3.6, 0.4)))
        self.add(kl)
        self.play(m.FadeOut(ring), run_time=0.4)
        self.play(t.animate.set_value(3.6 * T), run_time=8, rate_func=m.linear)
        t2 = prose("each backward period stretches it by $1/\\lambda \\approx 5.3$\nand bends it onto the spiral", font_size=24).move_to(pt(3.6, -0.7))
        t3 = prose("van der Pol ($\\varepsilon = 1$): $1/\\lambda \\approx 1200$, so $10^{-4} \\to 0.1 \\to 100$", font_size=22, color=MUTED).next_to(t2, m.DOWN, buff=0.35)
        self.play(m.FadeIn(t2))
        self.play(m.FadeIn(t3))
        self.wait(HOLD)
        self.next_slide("growth")
        self.clear()

    def why_hard(self):
        self.bullets("Looks like the end of the story. It is not.", [
            "Uneven stretching: in slow-fast systems almost all the growth\nhappens in a tiny part of $L_\\eta$, so the curve is badly resolved\nexactly where it is most interesting",
            "Sensitivity: for the neuron model $1/\\lambda \\approx 10^{20} > 10^{16}$, the limit of\ndouble precision: indistinguishable starts end far apart",
            "The paper keeps the backward-flow idea but solves for whole orbit\nsegments that end on $L_\\eta$ (a boundary-value problem, Chapter 12)",
        ], font_size=27)
