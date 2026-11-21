"""Deck for Chapter 12: The Osinga-Moehlis method, isochrons as families of orbit segments (paper Section 3)."""
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

# weakly attracting twisted oscillator  r' = a r (1 - r^2),  theta' = 1 - b r^2
A, B = 0.1, 0.25
TT = 2 * np.pi / (1 - B)                 # period
C = B / A                                # isochron of gamma=(1,0): theta = C ln r
NS = np.array([1.0, C]) / np.hypot(1.0, C)   # stable direction at gamma (tangent of the isochron)
GAM = np.array([1.0, 0.0])
ETA = 0.02
LAM = np.exp(-2 * A * TT)              # Floquet multiplier, about 0.19


def flow(p, t):
    """Exact flow of the twisted oscillator (psi = theta - C ln r advances at rate 1 - b). Returns nan past blow-up."""
    x, y = p
    r0 = np.hypot(x, y); th0 = np.arctan2(y, x)
    t = np.asarray(t, float)
    den = 1 + (1 / r0 ** 2 - 1) * np.exp(-2 * A * t)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, 1 / np.sqrt(np.abs(den)), np.nan)
        th = th0 - C * np.log(r0) + (1 - B) * t + C * np.log(r)
    return np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)


def start_point(k, d):
    return flow(GAM + d * NS, -k * TT)


def segment(k, d, n=260):
    s0 = start_point(k, d)
    return flow(s0, np.linspace(0, k * TT, n * k))


def spiral(rmin, rmax, n=400):
    r = np.geomspace(rmin, rmax, n)
    return np.column_stack([r * np.cos(C * np.log(r)), r * np.sin(C * np.log(r))])


def rhh_F(u):
    v, n = u
    an = 0.01 * (v + 55) / (1 - np.exp(-(v + 55) / 10)); bn = 0.125 * np.exp(-(v + 65) / 80)
    am = 0.1 * (v + 40) / (1 - np.exp(-(v + 40) / 10)); bm = 4 * np.exp(-(v + 65) / 18)
    mi = am / (am + bm)
    dv = 10 - 120 * mi ** 3 * (0.8 - n) * (v - 50) - 36 * n ** 4 * (v + 77) - 0.3 * (v + 54.4)
    return np.array([dv, an * (1 - n) - bn * n])


def rhh_segments(P, T, idx):
    """Orbit segments of duration T from the isochron points P[idx] (forward integration is well conditioned)."""
    from scipy.integrate import solve_ivp
    out = []
    for i in idx:
        s = solve_ivp(lambda t, u: rhh_F(u), (0, T), P[i], method="LSODA", rtol=1e-9, atol=1e-11, max_step=0.01)
        out.append(s.y.T[::3])
    return out


class Deck11(Deck):
    chapter = "Chapter 12"
    title = "The Osinga–Moehlis Method"
    subtitle = "Isochrons as families of orbit segments (paper §3)"

    def construct(self):
        self.title_card()
        self.big_idea()
        self.poincare()
        self.step_one()
        self.grow_and_concatenate()
        self.not_backward()
        self.algorithm()
        self.twist_result()
        self.show_figure("ch11_twist", "The method on the twisted oscillator",
                         "(a) Start points of the continuation (dots) on the exact spiral (grey), both sides of $\\Gamma$.\n"
                         "(b) Orbit segments with $k = 1, 2, 3$: each ends on $L_\\eta$ next to $\\vec\\gamma_0$ after exactly $k$ periods.",
                         height=4.9)
        self.neuron()
        self.show_figure("ch11_segments", "Four orbit segments of the neuron's continuation (cf. paper Fig. 4)",
                         "Each starts on the isochron (dot) and ends within $10^{-5}$ of $\\vec\\gamma_0$ (star) after one period.",
                         height=5.2)
        if (FIGS / "ch11_validate.png").exists():
            self.show_figure("ch11_validate", "Check: continuation versus brute force",
                             "Thick grey: level curves of the forward-integrated phase map. Thin coloured: continuation. They agree.",
                             height=5.2)
        self.accuracy()
        self.recap(["The isochron is the unknown Poincaré section of the time-$T$ map",
                    "Its points are start points of orbit segments of duration $kT$ ending on $L_\\eta(\\vec\\gamma)$",
                    "Each segment is a BVP, solved over the whole segment: no sensitivity to initial conditions",
                    "Continuation traces the isochron with points spaced by arclength",
                    "When $\\lvert\\delta\\rvert = \\eta$: concatenate with $\\Gamma$, continue with $k+1$ periods",
                    "Validated: exact spirals, and the neuron's brute-force isochrons"])

    # ------------------------------------------------------------------
    def big_idea(self):
        h = heading("The whole method in one sentence")
        self.play(m.FadeIn(h))
        s = prose("An isochron is the set of start points of orbit segments\nthat take exactly $k$ periods to land on $L_\\eta$.\n"
                  "As the landing point slides along $L_\\eta$,\nthe start point slides along the isochron.",
                  font_size=32, color=YELLOW, width=12.4).move_to(pt(0, 0.3))
        self.play(m.FadeIn(s, shift=m.UP * 0.2), run_time=1.5)
        rest = prose("everything else is bookkeeping: the BVP, continuation, and what to do when $\\delta$ reaches $\\eta$",
                     font_size=24, color=MUTED, width=12.6).to_edge(m.DOWN, buff=0.5)
        self.play(m.FadeIn(rest))
        self.wait(HOLD)
        self.next_slide("idea")
        self.clear()

    def twist_axes(self, at=pt(-3.0, -0.45), size=6.2, R=2.8):
        ax = eplane((-R, R, 1), (-R, R, 1), width=size, height=size, at=at, ticks=False)
        rad = ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0]
        circ = m.Circle(radius=rad, color=INK, stroke_width=3.5).move_to(ax.c2p(0, 0))
        return ax, circ

    def poincare(self):
        h = heading("The isochron is an unknown Poincaré section")
        ax, circ = self.twist_axes()
        iso = curve(ax, spiral(0.12, 2.75), color=MAGENTA, width=3)
        e = tex(r"\Phi_T\big(I(\vec\gamma)\big) = I(\vec\gamma)", font_size=40).move_to(pt(3.8, 2.3))
        self.play(m.FadeIn(h), m.Create(ax), m.Create(circ), run_time=1)
        self.play(m.Create(iso), m.Write(e), run_time=1.5)
        txt = prose("start on the curve, flow for one period $T$:\nyou are back on the same curve", font_size=24).move_to(pt(3.8, 1.1))
        self.play(m.FadeIn(txt))
        self.next_slide()
        starts = [p for p in [spiral(0.35, 0.35, 1)[0], spiral(0.65, 0.65, 1)[0], spiral(1.4, 1.4, 1)[0], spiral(2.0, 2.0, 1)[0]]]
        tk = m.ValueTracker(0.0)
        ts = np.linspace(0, TT, 300)
        paths = [flow(p, ts) for p in starts]
        trails = m.always_redraw(lambda: m.VGroup(*[curve(ax, P[: max(2, int(tk.get_value() / TT * 299) + 1)], color=YELLOW, width=2) for P in paths]))
        dots = m.always_redraw(lambda: m.VGroup(*[dot(ax, P[int(tk.get_value() / TT * 299)], color=YELLOW, r=0.09) for P in paths]))
        self.add(trails, dots)
        self.play(tk.animate.set_value(TT), run_time=6, rate_func=m.linear)
        t2 = prose("usually the section is known and the map\nunknown; here the map is the time-$T$ map\nand the section is what we want", font_size=24, color=YELLOW).move_to(pt(3.8, -0.6))
        self.play(m.FadeIn(t2))
        t3 = prose("near $\\vec\\gamma$ we know a piece of it already:\nthe straight segment $L_\\eta(\\vec\\gamma)$", font_size=24, color=ORANGE).move_to(pt(3.8, -2.3))
        Ls = m.Line(ax.c2p(*(GAM - 0.35 * NS)), ax.c2p(*(GAM + 0.35 * NS)), color=ORANGE, stroke_width=7)
        self.play(m.FadeIn(t3), m.Create(Ls))
        self.wait(HOLD)
        self.next_slide("section")
        self.clear()

    def step_one(self):
        h = heading("Step 1: orbit segments that end on the local piece")
        self.play(m.FadeIn(h))
        e1 = tex(r"\vec u\,'(s) = T\,\vec F(\vec u(s)),\qquad 0\le s\le 1", font_size=40).move_to(pt(0, 2.3))
        l1 = prose("(3.2): an orbit segment lasting exactly one period", font_size=22, color=MUTED).next_to(e1, m.DOWN, buff=0.15)
        self.play(m.Write(e1), m.FadeIn(l1))
        self.next_slide()
        e2 = tex(r"\vec u(1) = \vec\gamma + \delta\,\vec n^s_\gamma,\qquad \lvert\delta\rvert\le\eta", font_size=40, color=ORANGE).move_to(pt(0, 1.0))
        l2 = prose("(3.3): its end point lies on $L_\\eta$", font_size=22, color=MUTED).next_to(e2, m.DOWN, buff=0.15)
        self.play(m.Write(e2), m.FadeIn(l2))
        self.next_slide()
        e3 = tex(r"\vec u(0) = \Phi_{-T}\big(\vec\gamma + \delta\,\vec n^s_\gamma\big)\ \in\ I(\vec\gamma)", font_size=40, color=MAGENTA).move_to(pt(0, -0.4))
        l3 = prose("why on the isochron: $L_\\eta$ is (almost) part of $I(\\vec\\gamma)$,\nand the flow maps the isochron into itself", font_size=24, color=MUTED).next_to(e3, m.DOWN, buff=0.2)
        self.play(m.Write(e3), m.FadeIn(l3))
        self.next_slide("start point on isochron")
        e4 = tex(r"\delta = 0:\quad \vec u(s) = \text{the cycle } \Gamma \text{ from } \vec\gamma \text{ back to } \vec\gamma", font_size=36, color=TEAL).move_to(pt(0, -2.3))
        l4 = prose("(3.4): a known solution to start the continuation from", font_size=22, color=MUTED).next_to(e4, m.DOWN, buff=0.15)
        self.play(m.Write(e4), m.FadeIn(l4))
        self.wait(HOLD)
        self.next_slide("start solution")
        self.clear()

    def grow_and_concatenate(self):
        # progress parameter: [0,1] -> k=1, delta 0..eta ; [1,2] -> k=2 ; [2,3] -> k=3 up to the window edge
        def r_of(k, d):
            p = start_point(k, d)
            return np.hypot(*p) if np.all(np.isfinite(p)) else np.inf
        lo, hi = LAM * ETA, ETA
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if r_of(3, mid) < 2.6 else (lo, mid)
        dmax = {1: ETA, 2: ETA, 3: lo}

        def d0(k):
            return 0.0 if k == 1 else LAM * ETA      # after concatenation the end sits at Phi_T(gamma + eta n)

        def kd(p):
            k = min(3, int(p) + 1)
            return k, d0(k) + (p - (k - 1)) * (dmax[k] - d0(k))

        trace_full = []
        for k in (1, 2, 3):
            for d in np.linspace(d0(k), dmax[k], 120):
                trace_full.append((k - 1 + (d - d0(k)) / (dmax[k] - d0(k)), start_point(k, d)))
        tp = np.array([a for a, _ in trace_full]); tP = np.array([b for _, b in trace_full])

        h = heading("Sliding the end point grows the isochron")
        ax, circ = self.twist_axes(at=pt(-3.2, -0.45), size=6.3)
        ghost = curve(ax, spiral(1.0, 2.75), color=MUTED, width=2)
        ghost = m.DashedVMobject(ghost, num_dashes=50)
        star = m.Star(n=5, outer_radius=0.16, color=YELLOW, fill_opacity=1).move_to(ax.c2p(*GAM))
        self.play(m.FadeIn(h), m.Create(ax), m.Create(circ), m.FadeIn(star), m.Create(ghost), run_time=1)
        # inset zoom on L_eta
        box = m.Square(2.2, color=MUTED, stroke_width=2).move_to(pt(4.9, 1.65))
        bl = prose("zoom on $L_\\eta$ near $\\vec\\gamma$", font_size=22, color=MUTED).next_to(box, m.DOWN, buff=0.12)
        ctr = box.get_center()
        dirn = np.array([NS[0], NS[1], 0.0])
        Lz = m.Line(ctr - 0.95 * dirn, ctr + 0.95 * dirn, color=ORANGE, stroke_width=6)
        gz = m.Star(n=5, outer_radius=0.12, color=YELLOW, fill_opacity=1).move_to(ctr)
        etal = tex(r"\delta = \eta", font_size=34, color=ORANGE).next_to(box, m.RIGHT, buff=0.1).align_to(box, m.UP)
        self.play(m.Create(box), m.Create(Lz), m.FadeIn(gz), m.FadeIn(bl), m.FadeIn(etal))
        pr = m.ValueTracker(0.0)

        def seg_m():
            k, d = kd(pr.get_value())
            return curve(ax, segment(k, d), color=BLUE, width=2.5)

        def start_m():
            k, d = kd(pr.get_value())
            return dot(ax, start_point(k, d), color=MAGENTA, r=0.1)

        def trace_m():
            n = max(2, int(np.searchsorted(tp, pr.get_value())) + 1)
            return curve(ax, tP[:n], color=MAGENTA, width=5)

        def end_m():
            k, d = kd(pr.get_value())
            return m.Dot(ctr + 0.95 * (d / ETA) * dirn, radius=0.08, color=BLUE)
        sg, st, tr, en = (m.always_redraw(f) for f in (seg_m, start_m, trace_m, end_m))
        kl = prose("$k$ =", font_size=30).move_to(pt(1.7, 2.3))
        kn = m.Integer(1, font_size=40, color=BLUE).next_to(kl, m.RIGHT, buff=0.2)
        kn.add_updater(lambda mob: mob.set_value(kd(pr.get_value())[0]))
        leg = m.VGroup(prose("blue: the orbit segment ($k$ periods)", font_size=22, color=BLUE),
                       prose("magenta: its start point, on the isochron", font_size=22, color=MAGENTA)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.15).move_to(pt(3.9, -0.4))
        self.add(sg, tr, st, en)
        self.play(m.FadeIn(kl), m.FadeIn(kn), m.FadeIn(leg))
        self.next_slide("k=1 start")
        self.play(pr.animate.set_value(0.999), run_time=5, rate_func=m.linear)
        msg = prose("$\\delta$ has reached $\\eta$: the end of the\ntrustworthy piece. Stretch factor $1/\\lambda \\approx 5$", font_size=22, color=YELLOW).move_to(pt(3.9, -1.9))
        self.play(m.FadeIn(msg))
        self.next_slide("delta = eta")
        self.play(m.FadeOut(msg), m.FadeOut(leg))
        # concatenation (3.5)
        cyc_copy = m.Circle(radius=ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0], color=TEAL, stroke_width=5).move_to(ax.c2p(0, 0))
        c1 = tex(r"\vec u_{\text{new}}(s) = \vec u_{\text{old}}\Big(\frac{k}{k-1}\,s\Big),\quad 0\le s\le \frac{k-1}{k}", font_size=34).move_to(pt(3.4, 0.0))
        c2 = tex(r"\vec u_{\text{new}}(s) = \vec u_{\Gamma}\big(k s-(k-1)\big),\quad \frac{k-1}{k}\le s\le 1", font_size=34, color=TEAL).next_to(c1, m.DOWN, buff=0.25)
        for c in (c1, c2):
            if c.width > 7.0:
                c.scale_to_fit_width(7.0)
            c.set_x(3.6)
        cl = prose("(3.5): old segment, then one more lap of $\\Gamma$", font_size=22, color=MUTED).next_to(c2, m.DOWN, buff=0.2)
        self.play(m.FadeOut(bl), run_time=0.3)
        self.play(m.Create(cyc_copy), m.Write(c1), m.Write(c2), m.FadeIn(cl), run_time=2)
        self.next_slide("concatenate")
        why = prose("lasts $kT$; starts at the old start point; ends at $\\vec\\gamma$,\nso $\\delta \\approx 0$ (precisely $\\lambda\\eta$). The jump of size $\\eta$\nat the join is removed by Newton at once.", font_size=22, color=YELLOW).move_to(pt(3.6, -2.6))
        self.play(m.FadeIn(why))
        self.next_slide()
        self.play(m.FadeOut(cyc_copy), m.FadeOut(c1), m.FadeOut(c2), m.FadeOut(cl), m.FadeOut(why))
        pr.set_value(1.0)
        leg2 = prose("reset $\\delta \\approx 0$, continue again:\nthe start point moves further out", font_size=24).move_to(pt(3.9, -0.6))
        self.play(m.FadeIn(leg2))
        self.play(pr.animate.set_value(1.999), run_time=4, rate_func=m.linear)
        self.next_slide("k=2")
        pr.set_value(2.0)
        self.play(pr.animate.set_value(2.999), run_time=4, rate_func=m.linear)
        kn.clear_updaters()
        fin = prose("each extra period stretches by another $1/\\lambda$:\nthe isochron leaves the window", font_size=22, color=YELLOW).move_to(pt(3.9, -2.0))
        self.play(m.FadeIn(fin))
        self.wait(HOLD)
        self.next_slide("k=3")
        self.clear()

    def not_backward(self):
        h = heading("\"But that is just backward integration!\"")
        self.play(m.FadeIn(h))
        l = m.VGroup(m.Text("mathematically: yes", font_size=30, weight=m.BOLD, color=MUTED),
                     prose("the start point is the same point\n$\\Phi_{-kT}(\\vec\\gamma + \\delta\\vec n^s)$", font_size=26)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.3).move_to(pt(-3.5, 1.3))
        r = m.VGroup(m.Text("numerically: no", font_size=30, weight=m.BOLD, color=TEAL),
                     prose("the segment is a BVP solved by\ncollocation; nothing is ever\nintegrated backwards from $\\vec u(1)$", font_size=26)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.3).move_to(pt(3.5, 1.2))
        self.play(m.FadeIn(l))
        self.next_slide()
        self.play(m.FadeIn(r))
        self.next_slide()
        q = prose("\"AUTO measures the variation between two orbit segments along the entire\norbit segment and not just at the initial value. This avoids the extreme\nsensitivity on initial conditions that is so typical for multiple-time-scale systems.\"",
                  font_size=24, color=YELLOW, width=12.6).move_to(pt(0, -1.6))
        self.play(m.FadeIn(q))
        self.wait(HOLD)
        self.next_slide("quote")
        self.clear()

    def algorithm(self):
        self.bullets("Algorithm Isochron (paper §3.1)", [
            "start from $\\Gamma$: $\\delta = 0$, $k = 1$; continue until $\\lvert\\delta\\rvert = \\eta$",
            "for $k = 2, 3, \\dots$: concatenate with $\\Gamma$, NTST $\\to k\\times$NTST,\nreset $\\delta \\approx 0$, continue until $\\lvert\\delta\\rvert = \\eta$",
            "the start points $\\vec u(0)$ of all runs, in order, form the isochron",
            "repeat for both signs of $\\delta$ (both sides of $\\Gamma$) and for each $\\vec\\gamma$",
        ], font_size=28)

    def twist_result(self):
        d = data("ch11_twist"); P1, P2 = d["P1"], d["P2"]
        h = heading("Test case with a known answer")
        ax, circ = self.twist_axes(at=pt(-3.0, -0.45), size=6.3)
        ex = curve(ax, spiral(0.12, 2.75), color=MUTED, width=6)
        ex.set_stroke(opacity=0.6)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(circ), run_time=1)
        e = tex(r"\dot r = 0.1\,r(1-r^2),\quad \dot\theta = 1 - 0.25\,r^2", font_size=34).move_to(pt(3.7, 2.4))
        e2 = tex(r"\text{exact: } \theta = \frac{b}{a}\ln r = 2.5\ln r", font_size=36, color=MUTED).move_to(pt(3.7, 1.4))
        self.play(m.Write(e), m.Write(e2), m.Create(ex), run_time=1.6)
        self.next_slide()
        k = m.ValueTracker(0)
        n1, n2 = len(P1), len(P2)
        pts = m.always_redraw(lambda: m.VGroup(
            *[dot(ax, p, color=MAGENTA, r=0.045) for p in P1[: int(k.get_value() * n1) + 1]],
            *[dot(ax, p, color=TEAL, r=0.045) for p in P2[: int(k.get_value() * n2) + 1]]))
        self.add(pts)
        self.play(k.animate.set_value(0.999), run_time=6, rate_func=m.linear)
        t = prose("dots: start points from continuation\n(magenta outside, teal inside $\\Gamma$)\nerror at most $5\\times 10^{-4}$ rad $\\sim \\eta^2$:\nset by the curvature neglected in the\nstraight piece $L_\\eta$ ($\\eta = 0.02$)", font_size=24, width=5.4).move_to(pt(3.7, -1.3))
        self.play(m.FadeIn(t))
        self.wait(HOLD)
        self.next_slide("twist result")
        self.clear()

    def neuron(self):
        P = data("ch11_rhh_inner")["P"]
        d = data("rhh_cycle"); X, T = d["X"], float(d["T"])
        idx = np.unique(np.linspace(1, len(P) - 1, 70).astype(int))
        segs = rhh_segments(P, T, idx)
        h = heading("The neuron: one period is enough (paper Fig. 4)")
        ax = eplane((-90, 60, 30), (0.3, 0.8, 0.1), width=7.6, height=5.4, at=pt(-2.4, -0.5), xlabel="V", ylabel="n")
        cyc = curve(ax, X, color=MUTED, width=2.5, step=40)
        star = m.Star(n=5, outer_radius=0.17, color=YELLOW, fill_opacity=1).move_to(ax.c2p(*X[0]))
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), m.FadeIn(star), run_time=1)
        info = prose("$\\eta = 10^{-4}$, $k = 1$\nend point pinned within $10^{-5}$\nof $\\vec\\gamma_0$ (star)", font_size=24).move_to(pt(4.5, 2.0))
        self.play(m.FadeIn(info))
        self.next_slide()
        j = m.ValueTracker(0)
        nseg = len(idx)
        sg = m.always_redraw(lambda: curve(ax, segs[min(int(j.get_value()), nseg - 1)], color=BLUE, width=2.5))
        tr = m.always_redraw(lambda: curve(ax, P[: max(2, idx[min(int(j.get_value()), nseg - 1)] + 1)], color=MAGENTA, width=5))
        sd = m.always_redraw(lambda: dot(ax, P[idx[min(int(j.get_value()), nseg - 1)]], color=MAGENTA, r=0.1))
        self.add(sg, tr, sd)
        self.bring_to_front(star)
        self.play(j.animate.set_value(nseg - 1), run_time=12, rate_func=m.linear)
        self.next_slide("boomerang")
        turn = P[np.argmin(P[:, 0])]
        ar = m.Arrow(ax.c2p(-20, 0.33), ax.c2p(*turn), buff=0.1, color=YELLOW, stroke_width=4)
        bl = prose("boomerang turn near $V \\approx -47$", font_size=22, color=YELLOW).next_to(ax.c2p(-20, 0.33), m.RIGHT, buff=0.1)
        self.play(m.GrowArrow(ar), m.FadeIn(bl))
        why = prose("contraction $\\lambda \\approx 6.6\\times 10^{-21}$:\n$\\lvert\\delta\\rvert$ stays below $10^{-5}$, yet one\nbackward period stretches that tiny\npiece across the whole picture", font_size=22, color=MUTED).move_to(pt(4.5, -0.3))
        self.play(m.FadeIn(why))
        q = prose("\"start with a local approximation in a very\nsmall neighborhood of $\\Gamma$ and still grow\nthe manifold rapidly away from $\\Gamma$\"", font_size=22, color=YELLOW, width=4.4).move_to(pt(4.6, -2.5))
        self.play(m.FadeIn(q))
        self.wait(HOLD)
        self.next_slide("one period enough")
        self.clear()

    def accuracy(self):
        h = heading("Accuracy: the straight piece misses by about η²")
        ax = eplane((0.75, 1.35, 0.1), (-0.2, 0.6, 0.1), width=4.4, height=4.4 * 0.8 / 0.6, at=pt(-3.6, -0.6), ticks=False)
        r = np.linspace(0.92, 1.22, 300)
        iso = curve(ax, np.column_stack([r * np.cos(C * np.log(r)), r * np.sin(C * np.log(r))]), color=MAGENTA, width=4)
        star = m.Star(n=5, outer_radius=0.14, color=YELLOW, fill_opacity=1).move_to(ax.c2p(*GAM))
        self.play(m.FadeIn(h), m.Create(ax), m.Create(iso), m.FadeIn(star), run_time=1)
        eta = m.ValueTracker(0.2)
        rr = np.linspace(0.92, 1.22, 4000)
        cpts = np.column_stack([rr * np.cos(C * np.log(rr)), rr * np.sin(C * np.log(rr))])

        def gap():
            e = eta.get_value(); q = GAM + e * NS
            dists = np.linalg.norm(cpts - q, axis=1)
            return q, cpts[np.argmin(dists)], dists.min()

        def draw():
            e = eta.get_value(); q, c, _ = gap()
            return m.VGroup(m.Line(ax.c2p(*(GAM - e * NS)), ax.c2p(*q), color=ORANGE, stroke_width=6),
                            m.Line(ax.c2p(*q), ax.c2p(*c), color=YELLOW, stroke_width=4),
                            m.Dot(ax.c2p(*q), radius=0.07, color=ORANGE))
        g = m.always_redraw(draw)
        self.add(g)
        lab = m.VGroup(prose("magenta: true isochron", font_size=22, color=MAGENTA),
                       prose("orange: $L_\\eta$, the tangent piece", font_size=22, color=ORANGE),
                       prose("yellow: the gap at its end", font_size=22, color=YELLOW)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.15).move_to(pt(3.6, 2.3))
        l1 = prose("$\\eta$ =", font_size=26).move_to(pt(2.4, 0.9))
        n1 = m.DecimalNumber(0.2, num_decimal_places=3, font_size=36).next_to(l1, m.RIGHT, buff=0.2)
        n1.add_updater(lambda mob: mob.set_value(eta.get_value()))
        l2 = prose("gap / $\\eta^2$ =", font_size=26).move_to(pt(2.75, 0.2))
        n2 = m.DecimalNumber(1.0, num_decimal_places=2, font_size=36, color=YELLOW).next_to(l2, m.RIGHT, buff=0.2)
        n2.add_updater(lambda mob: mob.set_value(gap()[2] / eta.get_value() ** 2))
        self.play(m.FadeIn(lab), m.FadeIn(l1), m.FadeIn(n1), m.FadeIn(l2), m.FadeIn(n2))
        self.next_slide()
        self.play(eta.animate.set_value(0.04), run_time=4)
        n1.clear_updaters(); n2.clear_updaters()
        why = prose("a tangent line misses a curve quadratically:\nhalve $\\eta$, quarter the error", font_size=24, color=YELLOW).move_to(pt(3.6, -0.8))
        self.play(m.FadeIn(why))
        self.next_slide("eta^2")
        more = prose("also: collocation error (NCOL = 4, mesh grows with $k$).\nFinal check the theory guarantees:\ncomputed isochrons never cross each other.", font_size=22, color=MUTED, width=6.0).move_to(pt(3.5, -2.4))
        self.play(m.FadeIn(more))
        self.wait(HOLD)
        self.next_slide("accuracy")
        self.clear()
