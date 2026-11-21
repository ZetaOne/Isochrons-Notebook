"""Deck for Chapter 7: Phaseless sets."""
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


def vdp_rk4(P, t_end, dt=0.01, eps=1.0):
    """Trajectories of van der Pol (Lienard form) for an array of starts; returns (n_steps+1, n, 2)."""
    F = lambda U: np.column_stack([(U[:, 1] - U[:, 0] ** 3 / 3 + U[:, 0]) / eps, -U[:, 0]])
    U = np.array(P, float); out = [U.copy()]
    for _ in range(int(round(t_end / dt))):
        k1 = F(U); k2 = F(U + dt / 2 * k1); k3 = F(U + dt / 2 * k2); k4 = F(U + dt * k3)
        U = U + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(U.copy())
    return np.array(out)


# system (7.1)
A7, B7, W7 = 0.5, 0.5, 1.0


def sl2_isochron(c, r):
    """Isochron of phase c of r' = r(r^2-a^2)(1-r^2), th' = w - b r^2, as a function of r > a."""
    th = 2 * np.pi * c + (B7 / (2 * A7 ** 2)) * (np.log(1 - A7 ** 2 / r ** 2) - np.log(1 - A7 ** 2))
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


class Deck07(Deck):
    chapter = "Chapter 7"
    title = "Phaseless Sets"
    subtitle = "Where the clock stops, and why every isochron goes there"

    def construct(self):
        self.title_card()
        self.undefined()
        self.winding()
        self.zoom()
        self.self_similar()
        self.noise()
        self.desync()
        self.two_cycles_line()
        self.two_cycles_phase()
        self.two_cycles_picture()
        self.conjecture()
        self.recap(["Phaseless set = complement of the basin; in the plane there is\nalways one inside the cycle (usually an unstable fixed point)",
                    "Winding number: every isochron accumulates on the phaseless set;\nnear a spiral source the picture is self-similar",
                    "Small noise near the phaseless set randomises the phase completely",
                    "Unstable cycle as basin boundary: every isochron spirals onto it\n(Winfree's Conjecture C, proved by Guckenheimer)",
                    "3-D: intricate limits on the boundary; the paper finds similar\nintricacy inside the basin of a 2-D slow-fast system"])

    # ------------------------------------------------------------------
    def undefined(self):
        v = data("ch04_vdp1"); X = v["X"]
        h = heading("Where phase is undefined")
        ax = plane((-3, 3, 1), (-3.4, 3.4, 1), width=5.6, height=6.35, at=pt(-3.6, -0.45), ticks=False)
        cyc = curve(ax, X, color=INK, width=3.5, step=10)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), run_time=1.0)
        e = tex(r"\text{phaseless set} = \mathbb{R}^n \setminus B(\Gamma)", font_size=40, color=YELLOW).move_to(pt(3.2, 2.4))
        n = prose("states that never join the rhythm\n(Winfree: its boundary $\\partial B(\\Gamma)$)", font_size=24).next_to(e, m.DOWN, buff=0.25)
        self.play(m.Write(e), m.FadeIn(n))
        self.next_slide("definition")
        ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        P = np.column_stack([0.15 * np.cos(ang), 0.15 * np.sin(ang)])
        traj = vdp_rk4(P, 12.0, dt=0.01)
        k = m.ValueTracker(0)
        cl = m.always_redraw(lambda: m.VGroup(*[m.Dot(ax.c2p(*traj[int(k.get_value()), j]), radius=0.06, color=TEAL) for j in range(len(P))]))
        fp = dot(ax, (0, 0), color=RED, r=0.1)
        self.add(cl, fp)
        n2 = prose("states near the inside fixed point spiral out\nto $\\Gamma$; the fixed point itself never moves", font_size=24, color=TEAL).next_to(n, m.DOWN, buff=0.4)
        self.play(m.FadeIn(n2))
        self.play(k.animate.set_value(len(traj) - 1), run_time=6, rate_func=m.linear)
        self.next_slide()
        n3 = prose("Poincaré–Bendixson: the region inside a cycle\ncontains a fixed point, which cannot converge to $\\Gamma$.\nEvery planar oscillator has a phaseless set inside.", font_size=24, color=YELLOW, width=6.2).next_to(n2, m.DOWN, buff=0.4).align_to(n2, m.LEFT)
        self.play(m.FadeIn(n3))
        self.wait(HOLD)
        self.next_slide("PB")
        self.clear()

    def winding(self):
        v = data("ch04_vdp1")
        xs, ys, th, X = v["xs"], v["ys"], v["th"], v["X"]
        h = heading("Every isochron runs into the fixed point")
        ax = plane((-3, 3, 1), (-3.4, 3.4, 1), width=5.6, height=6.35, at=pt(-3.6, -0.45), ticks=False, xlabel="", ylabel="")
        cyc = curve(ax, X, color=INK, width=3.0, step=10)
        isos = m.VGroup(*[curve(ax, s, color=phase_color(j / 12), width=2.0).set_stroke(opacity=0.7)
                          for j in range(12) for s in level_curves(xs, ys, th, j / 12)])
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), m.FadeIn(isos), run_time=1.2)
        R = m.ValueTracker(2.3)
        ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)

        def ring():
            r = R.get_value()
            g = m.VGroup()
            for a in ang:
                p = (r * np.cos(a), r * np.sin(a))
                g.add(m.Dot(ax.c2p(*p), radius=min(0.065, 0.9 * r / len(ang) * 2 * np.pi), color=phase_color(phase_at(xs, ys, th, p))))
            return g
        rg = m.always_redraw(ring)
        self.add(rg)
        n = prose("a loop $C$ around the fixed point, each state\ncoloured by its asymptotic phase", font_size=24).move_to(pt(3.3, 2.3))
        self.play(m.FadeIn(n))
        self.next_slide("loop")
        n2 = prose("winding number: how often the phase goes round\nthe phase circle as we go once round $C$.\nAn integer, so it cannot change as $C$ shrinks", font_size=24, color=MUTED).next_to(n, m.DOWN, buff=0.35)
        self.play(m.FadeIn(n2))
        self.play(R.animate.set_value(0.35), run_time=6, rate_func=m.smooth)
        n3 = prose("near $\\Gamma$ it is 1, so every tiny loop still shows\nall phases: every isochron reaches every\nneighbourhood of the fixed point", font_size=24, color=YELLOW).next_to(n2, m.DOWN, buff=0.35)
        self.play(m.FadeIn(n3))
        self.wait(HOLD)
        self.next_slide("winding")
        self.clear()

    def zoom(self):
        h = heading("Zoom in by 10, and again by 10")
        self.play(m.FadeIn(h))
        panels = m.VGroup()
        for k, w in enumerate(["1.0", "0.1", "0.01"]):
            d = data(f"ch07_zoom_{w}")
            xs, ys, th = d["xs"], d["ys"], d["th"]
            W = float(w)
            ax = plane((-W, W, W), (-W, W, W), width=3.7, height=3.7, at=pt(-4.5 + 4.5 * k, -0.2), ticks=False, xlabel="", ylabel="")
            box = m.Square(side_length=3.7, color=MUTED, stroke_width=2).move_to(ax.c2p(0, 0))
            g = m.VGroup(*[curve(ax, s, color=phase_color(j / 8), width=3) for j in range(8) for s in level_curves(xs, ys, th, j / 8)])
            lab = m.MathTex(r"\text{half-width } %s" % w, font_size=32).next_to(box, m.DOWN, buff=0.25)
            panels.add(m.VGroup(box, g, lab))
        arrows = m.VGroup(*[m.Arrow(panels[k][0].get_right(), panels[k + 1][0].get_left(), buff=0.1, color=YELLOW, stroke_width=4)
                            for k in range(2)])
        xl = m.VGroup(*[m.MathTex(r"\times 10", font_size=30, color=YELLOW).next_to(a, m.UP, buff=0.1) for a in arrows])
        self.play(m.Create(panels[0][0]), m.Create(panels[0][1]), m.FadeIn(panels[0][2]), run_time=1.6)
        self.next_slide()
        for k in range(2):
            self.play(m.GrowArrow(arrows[k]), m.FadeIn(xl[k]))
            self.play(m.TransformFromCopy(panels[k][0], panels[k + 1][0]), m.Create(panels[k + 1][1]), m.FadeIn(panels[k + 1][2]), run_time=1.6)
            self.next_slide()
        n = prose("van der Pol ($\\varepsilon = 1$) around the unstable fixed point, eight isochrons each,\ncomputed afresh: the same picture at every scale, turned by a fixed angle", font_size=24, color=MUTED, width=12.8).to_edge(m.DOWN, buff=0.3)
        self.play(m.FadeIn(n))
        self.wait(HOLD)
        self.next_slide("self-similar")
        self.clear()

    def self_similar(self):
        h = heading("Why the picture repeats: a linear spiral source")
        self.play(m.FadeIn(h))
        e1 = tex(r"\dot\rho = \sigma\rho,\qquad \dot\phi = \omega", r"\qquad(\sigma = 0.5,\ \omega \approx 0.87)", font_size=40).move_to(pt(0, 2.2))
        e1[1].set_color(MUTED)
        n1 = prose("near $\\vec x^*$ the flow is linear and looks the same at every scale", font_size=25, color=MUTED).next_to(e1, m.DOWN, buff=0.25)
        self.play(m.Write(e1), m.FadeIn(n1))
        self.next_slide()
        e2 = tex(r"\rho(t) = \rho_0\, e^{\sigma t}", r"\ \Rightarrow\ ", r"\text{escaping from } \frac{\rho_0}{10} \text{ costs } \Delta t = \frac{\ln 10}{\sigma} \text{ extra}", font_size=38).move_to(pt(0, 0.6))
        self.play(m.Write(e2))
        self.next_slide()
        e3 = tex(r"\Delta\phi = \omega\,\Delta t = \frac{\omega}{\sigma}\ln 10 = \sqrt{3}\,\ln 10 \approx 4.0 \text{ rad}", font_size=40, color=YELLOW).move_to(pt(0, -0.8))
        n3 = prose("the extra turn while escaping shows up as the rotation between the panels", font_size=24, color=YELLOW).next_to(e3, m.DOWN, buff=0.25)
        self.play(m.Write(e3), m.FadeIn(n3))
        self.next_slide()
        n4 = prose("near a spiral source the isochrons are, to leading order, logarithmic spirals\n(the paper finds the same near the neuron's equilibrium, Chapter 14)", font_size=24, color=TEAL, width=12.8).move_to(pt(0, -2.6))
        self.play(m.FadeIn(n4))
        self.wait(HOLD)
        self.next_slide("rotation")
        self.clear()

    def noise(self):
        d = data("ch07_noise")
        cyc_ph, fix_ph, X = d["cyc"], d["fix"], d["X"]
        h = heading("Phase randomisation: two equally small clouds")
        ax = plane((-3, 3, 1), (-3.4, 3.4, 1), width=5.2, height=5.9, at=pt(-4.0, -0.55), ticks=False)
        cyc = curve(ax, X, color=INK, width=3.0, step=10)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), run_time=1.0)
        rng = np.random.default_rng(3)
        n = 150
        c0 = X[int(np.median(cyc_ph) * (len(X) - 1))]
        A = c0 + 0.08 * rng.standard_normal((n, 2))
        B = 0.08 * rng.standard_normal((n, 2))
        da = m.VGroup(*[m.Dot(ax.c2p(*p), radius=0.04, color=TEAL) for p in A])
        db = m.VGroup(*[m.Dot(ax.c2p(*p), radius=0.04, color=MAGENTA) for p in B])
        self.play(m.FadeIn(da), m.FadeIn(db))
        n0 = prose("Gaussian clouds, standard deviation 0.08:\nteal near $\\Gamma$, magenta at the fixed point", font_size=24).move_to(pt(3.3, 2.5))
        self.play(m.FadeIn(n0))
        self.next_slide("clouds")
        # phase dial
        C = pt(3.3, -0.85); R = 1.6
        dial = m.Circle(radius=R, color=MUTED, stroke_width=3).move_to(C)
        ticks = m.VGroup(*[m.Line(C + R * np.array([np.cos(a), np.sin(a), 0]) * 0.93, C + R * np.array([np.cos(a), np.sin(a), 0]) * 1.07,
                                  color=MUTED, stroke_width=2) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)])
        dl = prose("asymptotic phase", font_size=22, color=MUTED).next_to(dial, m.DOWN, buff=0.2)
        self.play(m.Create(dial), m.FadeIn(ticks), m.FadeIn(dl))
        self.next_slide()
        def onring(ph, rr):
            a = 2 * np.pi * ph
            return C + rr * np.array([np.cos(a), np.sin(a), 0])
        self.play(*[da[i].animate.move_to(onring(cyc_ph[i], R + 0.12 * ((i % 3) - 1))) for i in range(n)], run_time=2.0)
        self.play(*[db[i].animate.move_to(onring(fix_ph[i], R - 0.25 - 0.1 * (i % 3))) for i in range(n)], run_time=2.5)
        z1 = abs(np.mean(np.exp(2j * np.pi * cyc_ph))); z2 = abs(np.mean(np.exp(2j * np.pi * fix_ph)))
        e = tex(r"\big\lvert\langle e^{2\pi i\vartheta}\rangle\big\rvert:\ " + f"{z1:.3f}" + r"\ \text{vs}\ " + f"{z2:.3f}", font_size=34, color=YELLOW).move_to(pt(3.3, 1.45))
        self.play(m.Write(e))
        self.next_slide()
        n1 = prose("near $\\Gamma$: a sharp phase. Near the fixed point:\nscrambled over the whole cycle", font_size=23, color=YELLOW).to_edge(m.DOWN, buff=0.15).set_x(3.3)
        self.play(m.FadeOut(dl), m.FadeIn(n1))
        self.wait(HOLD)
        self.next_slide("randomised")
        self.clear()

    def desync(self):
        self.bullets("Why anyone cares: desynchronising neurons", [
            "Winfree: near a phaseless set, small noise randomises the phase",
            "Drive synchronised neurons to the phaseless set, let noise act:\nthey come back with random phases (pathological synchrony\nis implicated in Parkinsonian tremor)",
            "The paper's new suggestion: in a slow-fast neuron, aim at the\nrepelling slow manifold instead, because isochrons crowd there too",
        ], font_size=27)

    def two_cycles_line(self):
        h = heading("When the phaseless set is a curve")
        e = tex(r"\dot r = r\,(r^2 - a^2)(1 - r^2),\qquad \dot\theta = \omega - b r^2", font_size=40).move_to(pt(0, 2.3))
        self.play(m.FadeIn(h), m.Write(e))
        n = prose("read the radial equation as a flow on a line (Strogatz, chapter 2)", font_size=25, color=MUTED).next_to(e, m.DOWN, buff=0.3)
        self.play(m.FadeIn(n))
        self.next_slide()
        ax = m.NumberLine(x_range=[0, 1.5, 0.5], length=10, color=MUTED, include_numbers=False).move_to(pt(0, -0.3))
        x0, xa, x1 = ax.n2p(0), ax.n2p(0.5), ax.n2p(1)
        lab = m.VGroup(m.MathTex("0", font_size=34).next_to(x0, m.DOWN, buff=0.25),
                       m.MathTex("a", font_size=34).next_to(xa, m.DOWN, buff=0.25),
                       m.MathTex("1", font_size=34).next_to(x1, m.DOWN, buff=0.25),
                       m.MathTex("r", font_size=34, color=MUTED).next_to(ax, m.RIGHT, buff=0.2))
        self.play(m.Create(ax), m.FadeIn(lab))
        dots = m.VGroup(m.Dot(x0, radius=0.12, color=TEAL),
                        m.Circle(radius=0.12, color=MAGENTA, stroke_width=4).move_to(xa),
                        m.Dot(x1, radius=0.12, color=TEAL))
        ar = m.VGroup(m.Arrow(ax.n2p(0.38), ax.n2p(0.12), buff=0, color=YELLOW),
                      m.Arrow(ax.n2p(0.62), ax.n2p(0.88), buff=0, color=YELLOW),
                      m.Arrow(ax.n2p(1.42), ax.n2p(1.12), buff=0, color=YELLOW))
        self.play(m.FadeIn(dots), m.LaggedStart(*[m.GrowArrow(a) for a in ar], lag_ratio=0.3))
        n2 = m.VGroup(prose("origin: stable", font_size=24, color=TEAL).next_to(x0, m.UP, buff=0.4),
                      prose("$r = a$: unstable cycle", font_size=24, color=MAGENTA).next_to(xa, m.UP, buff=0.4),
                      prose("$r = 1$: our $\\Gamma$", font_size=24, color=TEAL).next_to(x1, m.UP, buff=0.4).shift(m.RIGHT * 0.6))
        self.play(m.FadeIn(n2))
        self.next_slide()
        n3 = prose("basin of $\\Gamma$ = $\\{r > a\\}$; its boundary, the unstable cycle $r = a$,\nis the phaseless set that touches the basin", font_size=25, color=YELLOW, width=12.8).move_to(pt(0, -2.4))
        self.play(m.FadeIn(n3))
        self.wait(HOLD)
        self.next_slide("phase line")
        self.clear()

    def two_cycles_phase(self):
        h = heading("Its asymptotic phase, exactly as in Chapter 4")
        self.play(m.FadeIn(h))
        n = prose("look for $\\psi = \\theta + g(r)$ that advances at a constant rate:", font_size=25, color=MUTED).move_to(pt(0, 2.55))
        e1 = tex(r"\dot\psi = \omega - br^2 + g'(r)\,r(r^2-a^2)(1-r^2) = \omega - b", font_size=38).next_to(n, m.DOWN, buff=0.25)
        self.play(m.FadeIn(n), m.Write(e1))
        self.next_slide()
        e2 = tex(r"g'(r) = \frac{-b", r"(1-r^2)", r"}{r(r^2-a^2)", r"(1-r^2)", r"}", r"= \frac{-b}{r(r^2-a^2)}", font_size=40).next_to(e1, m.DOWN, buff=0.4)
        w = prose("($\\omega - br^2 - (\\omega - b) = -b(r^2 - 1)$)", font_size=22, color=MUTED).next_to(e2, m.DOWN, buff=0.15)
        self.play(m.Write(e2[:5]), m.FadeIn(w))
        self.next_slide()
        s1, s2 = strike(e2[1]), strike(e2[3])
        self.play(m.Create(s1), m.Create(s2))
        self.play(m.Write(e2[5]))
        self.next_slide("cancel")
        n2 = prose("and $\\frac{d}{dr}\\ln(1 - a^2/r^2) = \\frac{2a^2}{r(r^2-a^2)}$, so", font_size=25, color=MUTED).next_to(w, m.DOWN, buff=0.3)
        e3 = tex(r"\vartheta(r,\theta) = \frac{1}{2\pi}\Big[\theta - \frac{b}{2a^2}\Big(\ln\Big(1 - \frac{a^2}{r^2}\Big) - \ln(1-a^2)\Big)\Big]", font_size=38, color=YELLOW).next_to(n2, m.DOWN, buff=0.25)
        self.play(m.FadeIn(n2), m.Write(e3))
        self.next_slide()
        n3 = prose("as $r \\to a$ the logarithm goes to $-\\infty$: the correction to $\\theta$ grows without bound", font_size=24, color=MAGENTA, width=12.8).next_to(e3, m.DOWN, buff=0.3)
        self.play(m.FadeIn(n3))
        self.wait(HOLD)
        self.next_slide("phase formula")
        self.clear()

    def two_cycles_picture(self):
        h = heading("Isochrons wind onto the unstable cycle")
        ax = plane((-1.6, 1.6, 1), (-1.6, 1.6, 1), width=6.2, height=6.2, at=pt(-3.2, -0.45), ticks=False)
        u = ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0]
        gam = m.Circle(radius=u, color=INK, stroke_width=4).move_to(ax.c2p(0, 0))
        inner = m.Circle(radius=A7 * u, color=MAGENTA, stroke_width=0, fill_color=m.GREY_E, fill_opacity=1).move_to(ax.c2p(0, 0))
        dash = m.DashedVMobject(m.Circle(radius=A7 * u, color=MAGENTA, stroke_width=4).move_to(ax.c2p(0, 0)), num_dashes=30)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(gam), m.FadeIn(inner), m.Create(dash), run_time=1.2)
        n = prose("$a = 0.5$, $b = 0.5$, $\\omega = 1$\nwhite: $\\Gamma$;  dashed: $r = a$\ngrey: goes to the origin, no phase", font_size=24).move_to(pt(3.6, 2.0))
        self.play(m.FadeIn(n))
        self.next_slide()
        r = A7 + np.concatenate([np.linspace(1.6 * 1.414 - A7, 0.02, 200), np.geomspace(0.02, 1e-9, 600)])
        isos = m.VGroup()
        for j in range(8):
            Q = sl2_isochron(j / 8, r)
            Q = Q[(np.abs(Q[:, 0]) <= 1.6) & (np.abs(Q[:, 1]) <= 1.6)]
            isos.add(curve(ax, Q, color=phase_color(j / 8), width=2.5))
        self.play(m.LaggedStart(*[m.Create(c) for c in isos], lag_ratio=0.12), run_time=6, rate_func=m.linear)
        n2 = prose("drawn from outside in: each isochron winds\nround $r = a$ infinitely often, like a spring\ncompressed onto a circle", font_size=24, color=YELLOW).next_to(n, m.DOWN, buff=0.4)
        self.play(m.FadeIn(n2))
        self.next_slide("winding")
        L = 0.75
        az = plane((-L, L, 0.25), (-L, L, 0.25), width=6.2, height=6.2, at=pt(-3.2, -0.45), ticks=False)
        uz = az.c2p(1, 0)[0] - az.c2p(0, 0)[0]
        cc = np.linspace(0, 2 * np.pi, 200)
        zin = m.Circle(radius=A7 * uz, stroke_width=0, fill_color=m.GREY_E, fill_opacity=1).move_to(az.c2p(0, 0))
        zd = m.DashedVMobject(m.Circle(radius=A7 * uz, color=MAGENTA, stroke_width=4).move_to(az.c2p(0, 0)), num_dashes=40)
        rz = A7 + np.geomspace(L * 1.42 - A7, 1e-12, 1500)
        ziso = m.VGroup(*[clipped(az, sl2_isochron(j / 8, rz), (-L, L), (-L, L), color=phase_color(j / 8), width=2.5) for j in range(8)])
        zl = prose("zoom: half-width $0.75$", font_size=22, color=MUTED).next_to(az, m.DOWN, buff=0.05)
        self.play(m.FadeOut(m.VGroup(ax, gam, inner, dash, isos)), run_time=0.6)
        self.play(m.FadeIn(az), m.FadeIn(zin), m.Create(zd), m.FadeIn(zl), run_time=0.8)
        self.play(m.Create(ziso), run_time=3)
        n3 = prose("zoom: every isochron passes arbitrarily\nclose to every point of the dashed circle", font_size=24, color=MAGENTA).next_to(n2, m.DOWN, buff=0.4)
        self.play(m.FadeIn(n3))
        self.wait(HOLD)
        self.next_slide("zoom")
        self.clear()

    def conjecture(self):
        h = heading("Winfree's conjecture and the paper's question")
        self.play(m.FadeIn(h))
        box = prose("Conjecture C (Winfree 1974, proved by Guckenheimer 1975):\ngenerically, every isochron comes arbitrarily close\nto every point of the basin boundary $\\partial B(\\Gamma)$", font_size=28, color=YELLOW)
        fr = m.SurroundingRectangle(box, color=YELLOW, buff=0.25, corner_radius=0.1)
        m.VGroup(box, fr).move_to(pt(0, 1.7))
        self.play(m.FadeIn(box), m.Create(fr))
        self.next_slide()
        rows = m.VGroup(
            prose("• one fixed point as boundary: every isochron accumulates on it", font_size=25),
            prose("• in 3-D: isochrons are surfaces that must fold back and forth\n   infinitely often near a boundary surface", font_size=25),
            prose("• in the plane the boundary is just a point or a curve, so that\n   layered complexity cannot happen on it...", font_size=25),
            prose("• ...the paper finds it inside the basin of a planar slow-fast system,\n   near the repelling slow manifold (Chapter 8)", font_size=25, color=TEAL),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.25).next_to(fr, m.DOWN, buff=0.45)
        for r in rows:
            self.play(m.FadeIn(r, shift=m.RIGHT * 0.2), run_time=0.6)
            self.next_slide()
        self.clear()
