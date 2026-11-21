"""Deck for Chapter 13: The reduced Hodgkin-Huxley model (paper Section 4, first part)."""
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

GREEN_NC = GREEN
RED_NC = RED


def minf(v):
    am = 0.1 * (v + 40) / (1 - np.exp(-(v + 40) / 10)); bm = 4 * np.exp(-(v + 65) / 18)
    return am / (am + bm)


def rhh_F(u):
    v, n = u
    an = 0.01 * (v + 55) / (1 - np.exp(-(v + 55) / 10)); bn = 0.125 * np.exp(-(v + 65) / 80)
    mi = minf(v)
    dv = 10 - 120 * mi ** 3 * (0.8 - n) * (v - 50) - 36 * n ** 4 * (v + 77) - 0.3 * (v + 54.4)
    return np.array([dv, an * (1 - n) - bn * n])


def pieces(P, xr, yr):
    """Split a nan-separated polyline into pieces lying inside the window."""
    P = np.asarray(P, float)
    ok = np.all(np.isfinite(P), axis=1) & (P[:, 0] >= xr[0]) & (P[:, 0] <= xr[1]) & (P[:, 1] >= yr[0]) & (P[:, 1] <= yr[1])
    out, cur = [], []
    for p, o in zip(P, ok):
        if o:
            cur.append(p)
        elif cur:
            out.append(np.array(cur)); cur = []
    if cur:
        out.append(np.array(cur))
    return [c for c in out if len(c) > 3]


def stage(dv, v):
    if dv > 3:
        return 0      # upstroke
    if v > -10 and dv > -3:
        return 1      # peak
    if dv < -3:
        return 2      # downstroke
    return 3          # recovery


STAGE_COL = [ORANGE, YELLOW, MAGENTA, TEAL]
STAGE_NAME = ["upstroke", "peak", "downstroke", "recovery"]


class Deck12(Deck):
    chapter = "Chapter 13"
    title = "The Reduced Hodgkin–Huxley Model"
    subtitle = "From four equations to two (paper §4, first part)"

    def construct(self):
        self.title_card()
        self.hodgkin_huxley()
        self.spike_steps()
        self.m_fast()
        self.substitution()
        self.show_figure("ch12_reduction", "Checking the reduction against the full model (XPPAUT)",
                         "(a) Full (black) and reduced (dashed) spike trains. (b) $m$ stays close to $m_\\infty(V)$.\n"
                         "(c) $h + n$ stays between 0.78 and 0.85, close to 0.8.", height=4.3)
        self.model()
        self.phase_plane()
        self.spike_tour()
        self.numbers()
        self.spiral_source()
        self.bifurcation()
        self.show_figure("ch12_auto", "AUTO: equilibria and periodic orbits in the applied current",
                         "Teal: stable cycles, magenta: unstable cycles born in the subcritical Hopf point (orange).\n"
                         "(b) Zoom: rest and spiking coexist between the fold of cycles and the Hopf point.", height=4.6)
        self.why_example()
        self.recap(["Reduction: $m = m_\\infty(V)$ (fast) and $h = 0.8 - n$, leaving the plane $(V, n)$",
                    "At $I_{app} = 10$: relaxation-type cycle, $T_\\Gamma = 11.8463$, zero phase at the spike peak",
                    "$\\vec\\gamma_0 = (44.7064, 0.4597)$; one equilibrium $\\vec x^* = (-59.6044, 0.4026)$",
                    "$\\vec x^*$ is a weak spiral source on the repelling middle branch: nearly 2 turns per e-fold",
                    "Most of the period is spent on the left branch; upstroke and downstroke are fast"])

    # ------------------------------------------------------------------
    def hodgkin_huxley(self):
        h = heading("Hodgkin & Huxley (1952): the membrane as a circuit")
        self.play(m.FadeIn(h))
        e = tex(r"C\dot V = I_{\text{app}}", r"- \bar g_{\text{Na}}\,m^3h\,(V - V_{\text{Na}})", r"- \bar g_{\text{K}}\,n^4(V - V_{\text{K}})",
                r"- g_L(V - V_L)", font_size=40)
        if e.width > 13.2:
            e.scale_to_fit_width(13.2)
        e.move_to(pt(0, 1.6))
        self.play(m.Write(e), run_time=2)
        self.next_slide()
        labs = []
        for part, txt, col in [(e[1], "sodium in", ORANGE), (e[2], "potassium out", BLUE), (e[3], "leak", MUTED)]:
            b = m.Brace(part, m.DOWN, color=col)
            l = prose(txt, font_size=24, color=col).next_to(b, m.DOWN, buff=0.15)
            labs += [b, l]
            self.play(m.GrowFromCenter(b), m.FadeIn(l), run_time=0.7)
        self.next_slide("currents")
        g = tex(r"\dot x = \alpha_x(V)(1-x) - \beta_x(V)\,x,\qquad x\in\{m, h, n\}", font_size=38, color=TEAL).move_to(pt(0, -1.3))
        gl = prose("gating variables between 0 (closed) and 1 (open), rates fitted to experiments:\n$m$ sodium activation, $h$ sodium inactivation, $n$ potassium activation",
                   font_size=24, color=MUTED, width=12.6).next_to(g, m.DOWN, buff=0.3)
        self.play(m.Write(g), m.FadeIn(gl))
        self.wait(HOLD)
        self.next_slide("gating")
        self.clear()

    def spike_steps(self):
        d = data("ch12_pp"); X, t, T = d["X"], d["t"], float(d["T"])
        # two periods starting at the spike peak
        tt = np.concatenate([t, t + T])[::4]; XX = np.concatenate([X, X])[::4]
        dv = np.gradient(XX[:, 0], tt)
        st = np.array([stage(a, v) for a, v in zip(dv, XX[:, 0])])
        h = heading("A spike in four steps")
        ax = eplane((0, 24, 4), (-90, 60, 30), width=11.5, height=3.6, at=pt(0, 1.0), xlabel="t", ylabel="V")
        self.play(m.FadeIn(h), m.Create(ax), run_time=1)
        # coloured pieces by stage
        segs = m.VGroup()
        i0 = 0
        for i in range(1, len(tt) + 1):
            if i == len(tt) or st[i] != st[i0]:
                segs.add(curve(ax, np.column_stack([tt[i0:i + 1], XX[i0:i + 1, 0]]), color=STAGE_COL[st[i0]], width=4))
                i0 = i
        k = m.ValueTracker(0.0)
        dd = m.always_redraw(lambda: dot(ax, (k.get_value(), np.interp(k.get_value(), tt, XX[:, 0])), color=INK, r=0.09))
        texts = [
            "1. Upstroke: $V$ rises, $m$ opens fast,\nsodium rushes in: positive feedback",
            "2. Peak: $h$ closes and $n$ opens,\nboth slowly",
            "3. Downstroke: potassium flows out,\n$V$ falls below rest",
            "4. Recovery: $n$ closes, $h$ reopens,\nslowly; then fire again",
        ]
        boxes = m.VGroup(*[prose(tx, font_size=22, color=c) for tx, c in zip(texts, STAGE_COL)])
        boxes.arrange_in_grid(rows=2, cols=2, buff=(0.8, 0.35)).move_to(pt(0, -2.4))
        self.add(dd)
        self.play(m.Create(segs), k.animate.set_value(24), run_time=6, rate_func=m.linear)
        self.next_slide()
        for b in boxes:
            self.play(m.FadeIn(b, shift=m.UP * 0.2), run_time=0.6)
        self.wait(HOLD)
        self.next_slide("four steps")
        self.clear()

    def m_fast(self):
        h = heading("From four variables to two: m is fast")
        ax = eplane((-90, 60, 30), (0, 1, 0.25), width=6.6, height=4.6, at=pt(-3.0, -0.5), xlabel="V", ylabel="m")
        v = np.linspace(-90, 60, 300)
        mc = curve(ax, np.column_stack([v, minf(v)]), color=ORANGE, width=4)
        ml = tex(r"m_\infty(V) = \frac{\alpha_m}{\alpha_m+\beta_m}", font_size=36, color=ORANGE).move_to(pt(3.7, 2.3))
        self.play(m.FadeIn(h), m.Create(ax), m.Create(mc), m.Write(ml), run_time=1.4)
        txt = prose("time constant: a fraction of a millisecond.\nWherever $m$ starts, it snaps onto $m_\\infty(V)$\nbefore $V$, $h$, $n$ have moved", font_size=24).move_to(pt(3.7, 1.0))
        self.play(m.FadeIn(txt))
        self.next_slide()
        starts = [(-60, 0.8), (-20, 0.05), (10, 0.3), (30, 0.55)]
        ds = m.VGroup(*[dot(ax, p, color=YELLOW, r=0.09) for p in starts])
        self.play(m.FadeIn(ds))
        self.play(*[d.animate.move_to(ax.c2p(p[0], minf(p[0]))) for d, p in zip(ds, starts)], run_time=1.0, rate_func=m.rush_from)
        why = prose("like the attracting branch of a critical\nmanifold (Chapter 8): set $m = m_\\infty(V)$", font_size=24, color=YELLOW).move_to(pt(3.7, -0.6))
        self.play(m.FadeIn(why))
        self.next_slide("m fast")
        hn = tex(r"h + n \approx 0.8 \ \Rightarrow\ h = 0.8 - n", font_size=40, color=TEAL).move_to(pt(3.7, -1.9))
        hl = prose("$h$ and $n$ move oppositely at similar rates:\nin the full model $0.78 \\le h + n \\le 0.85$", font_size=22, color=MUTED).next_to(hn, m.DOWN, buff=0.2)
        self.play(m.Write(hn), m.FadeIn(hl))
        self.wait(HOLD)
        self.next_slide("h+n")
        self.clear()

    def substitution(self):
        h = heading("Substitute: four equations become two")
        self.play(m.FadeIn(h))
        e1 = tex(r"C\dot V = I_{\text{app}} - \bar g_{\text{Na}}\,", r"m^3", r"h", r"(V - V_{\text{Na}}) - \bar g_{\text{K}}\,n^4(V - V_{\text{K}}) - g_L(V - V_L)", font_size=36)
        if e1.width > 13.2:
            e1.scale_to_fit_width(13.2)
        e1.move_to(pt(0, 1.8))
        self.play(m.Write(e1), run_time=1.6)
        self.next_slide()
        r1 = m.SurroundingRectangle(e1[1], color=ORANGE, buff=0.06)
        r2 = m.SurroundingRectangle(e1[2], color=TEAL, buff=0.06)
        l1 = tex(r"m^3 \to [m_\infty(V)]^3", font_size=36, color=ORANGE).move_to(pt(-3.0, 0.5))
        l2 = tex(r"h \to 0.8 - n", font_size=36, color=TEAL).move_to(pt(3.0, 0.5))
        self.play(m.Create(r1), m.Create(r2), m.Write(l1), m.Write(l2))
        self.next_slide()
        e2 = tex(r"C\dot V = I_{\text{app}} - \bar g_{\text{Na}}\,", r"[m_\infty(V)]^3", r"(0.8 - n)", r"(V - V_{\text{Na}}) - \bar g_{\text{K}}\,n^4(V - V_{\text{K}}) - g_L(V - V_L)", font_size=36)
        e2[1].set_color(ORANGE); e2[2].set_color(TEAL)
        if e2.width > 13.2:
            e2.scale_to_fit_width(13.2)
        e2.move_to(pt(0, -0.8))
        self.play(m.Write(e2), run_time=1.6)
        e3 = tex(r"\dot n = \alpha_n(V)(1-n) - \beta_n(V)\,n", font_size=38).move_to(pt(0, -1.9))
        self.play(m.Write(e3))
        res = prose("$m$ and $h$ are gone: only $V$ and $n$ remain, a planar system we can draw",
                    font_size=26, color=YELLOW, width=12.6).to_edge(m.DOWN, buff=0.45)
        self.play(m.FadeIn(res))
        self.wait(HOLD)
        self.next_slide("reduced")
        self.clear()

    def model(self):
        h = heading("The paper's model (eq. 4.1)")
        self.play(m.FadeIn(h))
        e1 = tex(r"\dot V = \frac{1}{C}\Big(I_{\text{app}} - \bar g_{\text{Na}}[m_\infty(V)]^3(0.8 - n)(V - V_{\text{Na}}) - \bar g_{\text{K}}\,n^4(V - V_{\text{K}}) - g_L(V - V_L)\Big)", font_size=36)
        e2 = tex(r"\dot n = \alpha_n(V)(1-n) - \beta_n(V)\,n", font_size=36)
        g = m.VGroup(e1, e2).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.35)
        if g.width > 13.2:
            g.scale_to_fit_width(13.2)
        g.move_to(pt(0, 1.7))
        self.play(m.Write(e1), run_time=2); self.play(m.Write(e2))
        self.next_slide()
        r = m.VGroup(
            tex(r"\alpha_n = \frac{0.01(V+55)}{1 - e^{-(V+55)/10}},\quad \beta_n = 0.125\,e^{-(V+65)/80}", font_size=34),
            tex(r"\alpha_m = \frac{0.1(V+40)}{1 - e^{-(V+40)/10}},\quad \beta_m = 4\,e^{-(V+65)/18}", font_size=34),
        ).arrange(m.DOWN, buff=0.3).move_to(pt(0, -0.6))
        self.play(m.FadeIn(r))
        p = prose("$\\bar g_{Na} = 120$, $\\bar g_K = 36$, $g_L = 0.3$, $I_{app} = 10$, $V_{Na} = 50$, $V_K = -77$, $V_L = -54.4$, $C = 1$",
                  font_size=24, color=MUTED, width=12.8).to_edge(m.DOWN, buff=0.5)
        self.play(m.FadeIn(p))
        self.wait(HOLD)
        self.next_slide("model")
        self.clear()

    def _pp_axes(self, at=pt(-2.3, -0.5), width=7.8, height=5.6):
        return eplane((-90, 60, 30), (0.3, 0.8, 0.1), width=width, height=height, at=at, xlabel="V", ylabel="n")

    def phase_plane(self):
        d = data("ch12_pp"); X, ncV, ncN, eq = d["X"], d["ncV"], d["ncN"], d["eq"]
        h = heading("The phase plane: a cubic track and a slow variable")
        ax = self._pp_axes()
        self.play(m.FadeIn(h), m.Create(ax), run_time=1)
        win = ((-90, 60), (0.3, 0.8))
        vnc = m.VGroup(*[curve(ax, p, color=GREEN_NC, width=4) for p in pieces(ncV, *win)])
        nnc = m.VGroup(*[curve(ax, p, color=RED_NC, width=4) for p in pieces(ncN, *win)])
        l1 = prose("$V$-nullcline ($\\dot V = 0$): cubic-like", font_size=22, color=GREEN_NC).move_to(pt(4.6, 2.5))
        self.play(m.Create(vnc), m.FadeIn(l1), run_time=1.5)
        self.next_slide()
        lab = m.VGroup(
            prose("left:\nattracting", font_size=22, color=GREEN_NC).move_to(ax.c2p(-84, 0.52)),
            prose("middle:\nrepelling", font_size=22, color=GREEN_NC).move_to(ax.c2p(-34, 0.48)),
            prose("right: attracting", font_size=22, color=GREEN_NC).move_to(ax.c2p(30, 0.76)),
        )
        knees = m.VGroup(dot(ax, (-61, 0.40), color=INK, r=0.07), dot(ax, (-20, 0.71), color=INK, r=0.07))
        self.play(m.FadeIn(lab), m.FadeIn(knees))
        self.next_slide("branches")
        l2 = prose("$n$-nullcline ($\\dot n = 0$)", font_size=22, color=RED_NC).next_to(l1, m.DOWN, aligned_edge=m.LEFT, buff=0.2)
        self.play(m.Create(nnc), m.FadeIn(l2), run_time=1.2)
        e = m.Circle(radius=0.1, color=INK, stroke_width=3, fill_color=m.BLACK, fill_opacity=1).move_to(ax.c2p(*eq))
        l3 = prose("$\\vec x^*$: where they cross,\non the repelling middle branch", font_size=22).next_to(l2, m.DOWN, aligned_edge=m.LEFT, buff=0.2)
        self.play(m.FadeIn(e), m.FadeIn(l3))
        self.next_slide("equilibrium")
        cyc = curve(ax, X, color=INK, width=4, step=10)
        star = m.Star(n=5, outer_radius=0.17, color=YELLOW, fill_opacity=1).move_to(ax.c2p(*X[0]))
        l4 = prose("$\\Gamma$: the spiking cycle\nstar: $\\vec\\gamma_0$, maximal $V$ (zero phase)", font_size=22).next_to(l3, m.DOWN, aligned_edge=m.LEFT, buff=0.2)
        self.play(m.Create(cyc), m.FadeIn(star), m.FadeIn(l4), run_time=2)
        self.bring_to_front(e)
        self.wait(HOLD)
        self.next_slide("cycle")
        self.clear()

    def spike_tour(self):
        d = data("ch12_pp"); X, t, T, ncV, ncN = d["X"], d["t"], float(d["T"]), d["ncV"], d["ncN"]
        Xs, ts = X[::3], t[::3]
        dv = np.gradient(Xs[:, 0], ts)
        st = np.array([stage(a, v) for a, v in zip(dv, Xs[:, 0])])
        h = heading("One spike, at the true speed")
        ax = self._pp_axes(at=pt(-2.9, -0.5), width=7.0, height=5.4)
        win = ((-90, 60), (0.3, 0.8))
        vnc = m.VGroup(*[curve(ax, p, color=GREEN_NC, width=2.5) for p in pieces(ncV, *win)]).set_stroke(opacity=0.6)
        nnc = m.VGroup(*[curve(ax, p, color=RED_NC, width=2.5) for p in pieces(ncN, *win)]).set_stroke(opacity=0.6)
        cyc = curve(ax, X, color=MUTED, width=2.5, step=10)
        self.play(m.FadeIn(h), m.Create(ax), m.FadeIn(vnc), m.FadeIn(nnc), m.Create(cyc), run_time=1)
        ax2 = eplane((0, 12, 4), (-90, 60, 30), width=4.4, height=1.9, at=pt(4.3, 1.45), xlabel="t", ylabel="V")
        self.play(m.Create(ax2))
        tk = m.ValueTracker(0.0)

        def idx():
            return min(len(ts) - 1, int(np.searchsorted(ts, tk.get_value())))
        pd = m.always_redraw(lambda: dot(ax, Xs[idx()], color=STAGE_COL[st[idx()]], r=0.12))
        vt = m.always_redraw(lambda: curve(ax2, np.column_stack([ts[: max(2, idx() + 1)], Xs[: max(2, idx() + 1), 0]]), color=TEAL, width=3))
        names = m.VGroup(*[m.Text(nm, font_size=26, color=c) for nm, c in zip(STAGE_NAME, STAGE_COL)]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.2).move_to(pt(3.4, -1.1))

        def hl(g):
            s = st[idx()]
            for i, nm in enumerate(g):
                nm.set_opacity(1.0 if i == s else 0.25)
        names.add_updater(hl)
        fr = m.Text("fraction of T", font_size=22, color=MUTED).move_to(pt(5.3, -0.7))
        frn = m.DecimalNumber(0.0, num_decimal_places=2, font_size=34).next_to(fr, m.DOWN, buff=0.15)
        frn.add_updater(lambda mob: mob.set_value(tk.get_value() / T))
        self.add(pd, vt, names, fr, frn)
        self.next_slide()
        self.play(tk.animate.set_value(T), run_time=12, rate_func=m.linear)
        names.clear_updaters(); frn.clear_updaters()
        rec = float(np.mean(st == 3))
        msg = prose(f"recovery creeps down the left branch: about {100 * rec:.0f} per cent of $T$.\nThe jumps along the bottom (upstroke) and top are fast:\na slow–fast relaxation oscillator with tiny effective $\\varepsilon$",
                    font_size=22, color=YELLOW).move_to(pt(3.6, -2.6))
        if msg.width > 6.6:
            msg.scale_to_fit_width(6.6)
        self.play(m.FadeIn(msg))
        self.wait(HOLD)
        self.next_slide("tour")
        self.clear()

    def numbers(self):
        h = heading("Our numbers match the paper's")
        self.play(m.FadeIn(h))
        rows = [("", "this book (XPPAUT)", "paper"),
                ("period $T_\\Gamma$", "11.846275", "11.8463"),
                ("$\\vec\\gamma_0$ (max $V$)", "(44.7064, 0.459736)", "(44.7064, 0.4597)"),
                ("equilibrium $\\vec x^*$", "(-59.6044, 0.402577)", "(-59.6044, 0.4026)"),
                ("eigenvalues at $\\vec x^*$", "$0.0640 \\pm 0.6943\\,i$", "a source, complex")]
        xs = [-4.4, 0.4, 4.6]
        g = m.VGroup()
        for i, row in enumerate(rows):
            y = 1.9 - 1.0 * i
            for j, cell in enumerate(row):
                if not cell:
                    continue
                col = MUTED if i == 0 else (TEAL if j == 1 else (ORANGE if j == 2 else INK))
                c = prose(cell, font_size=26 if i else 24, color=col).move_to(pt(xs[j], y))
                g.add(c)
        line = m.Line(pt(-6.4, 1.45), pt(6.4, 1.45), color=MUTED, stroke_width=1.5)
        self.play(m.FadeIn(line))
        self.play(m.LaggedStart(*[m.FadeIn(c) for c in g], lag_ratio=0.08), run_time=2.5)
        self.wait(HOLD)
        self.next_slide("numbers")
        self.clear()

    def spiral_source(self):
        from scipy.integrate import solve_ivp
        d = data("ch12_pp"); eq, eig = d["eq"], d["eig"]
        sig, om = float(np.real(eig[0])), float(abs(np.imag(eig[0])))
        s = solve_ivp(lambda t, u: rhh_F(u), (0, 38), eq + np.array([0.02, 0.0]), method="LSODA", rtol=1e-10, atol=1e-12, max_step=0.02)
        Y = s.y.T
        dv = np.abs(Y[:, 0] - eq[0]).max() * 1.15; dn = np.abs(Y[:, 1] - eq[1]).max() * 1.15
        h = heading("The equilibrium: a weak spiral source")
        ax = m.Axes(x_range=(eq[0] - dv, eq[0] + dv, dv), y_range=(eq[1] - dn, eq[1] + dn, dn), x_length=5.8, y_length=5.4, tips=False,
                    axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": False}).move_to(pt(-3.3, -0.5))
        zl = prose("zoom around $\\vec x^*$", font_size=22, color=MUTED).next_to(ax, m.DOWN, buff=0.1)
        e = m.Circle(radius=0.08, color=INK, stroke_width=3).move_to(ax.c2p(*eq))
        self.play(m.FadeIn(h), m.Create(ax), m.FadeIn(e), m.FadeIn(zl), run_time=1)
        ev = tex(r"\sigma \pm i\omega = 0.064 \pm 0.694\,i", font_size=38).move_to(pt(3.6, 2.3))
        self.play(m.Write(ev))
        k = m.ValueTracker(1)
        tr = m.always_redraw(lambda: curve(ax, Y[: max(2, int(k.get_value()))], color=MAGENTA, width=3))
        hd = m.always_redraw(lambda: dot(ax, Y[max(1, int(k.get_value())) - 1], color=MAGENTA, r=0.08))
        self.add(tr, hd)
        self.play(k.animate.set_value(len(Y)), run_time=8, rate_func=m.linear)
        self.next_slide("spiral out")
        e1 = tex(r"\text{distance} \propto e^{\sigma t},\qquad \text{angle} = \omega t", font_size=36).move_to(pt(3.6, 1.1))
        e2 = tex(r"\text{one e-fold: } t = \frac{1}{\sigma}\ \Rightarrow\ \text{angle} = \frac{\omega}{\sigma}", font_size=36).move_to(pt(3.6, 0.0))
        e3 = tex(r"\frac{0.694}{0.064} \approx 11\ \text{rad} \approx 1.7\ \text{turns}", font_size=38, color=YELLOW).move_to(pt(3.6, -1.2))
        for x in (e1, e2, e3):
            self.play(m.Write(x))
            self.next_slide()
        why = prose("small real part, fast rotation: trajectories\nleave slowly, winding round and round.\nThis will control how tightly the isochrons\nwind into $\\vec x^*$ (Chapter 14)", font_size=22, color=MUTED).move_to(pt(3.6, -2.7))
        self.play(m.FadeIn(why))
        self.wait(HOLD)
        self.next_slide("winding")
        self.clear()

    def bifurcation(self):
        d = data("ch12_auto"); E, hb = d["eq"], float(d["hb"][0])
        h = heading("Where does spiking come from? Turn up the current")
        ax = eplane((0, 20, 5), (-70, -50, 5), width=7.4, height=5.0, at=pt(-2.6, -0.5), xlabel=r"I_{\text{app}}", ylabel="V")
        self.play(m.FadeIn(h), m.Create(ax), run_time=1)
        sel = (E[:, 0] <= 20) & (E[:, 1] >= -70) & (E[:, 1] <= -50)
        E = E[sel]
        o = np.argsort(E[:, 0]); E = E[o]
        stab = E[:, 0] < hb
        c1 = curve(ax, E[stab][:, :2], color=INK, width=4)
        c2 = m.DashedVMobject(curve(ax, E[~stab][:, :2], color=INK, width=4), num_dashes=25)
        l1 = prose("equilibrium: stable (solid)\nfor small current", font_size=22).move_to(pt(4.3, 2.2))
        self.play(m.Create(c1), m.FadeIn(l1), run_time=1.5)
        self.next_slide()
        vh = float(np.interp(hb, E[:, 0], E[:, 1]))
        hp = m.Dot(ax.c2p(hb, vh), radius=0.12, color=ORANGE)
        hl = tex(r"\text{Hopf at } I_{\text{app}} \approx 8.82", font_size=34, color=ORANGE).move_to(pt(4.3, 1.0))
        self.play(m.Create(c2), m.FadeIn(hp), m.Write(hl), run_time=1.5)
        sub = prose("subcritical: the cycle born here is unstable;\nits branch turns back at a fold near 6.3,\nwhere it meets the big stable spiking cycles", font_size=22, color=MUTED).move_to(pt(4.3, -0.3))
        self.play(m.FadeIn(sub))
        self.next_slide("hopf")
        band = m.Rectangle(width=ax.c2p(hb, 0)[0] - ax.c2p(6.3, 0)[0], height=5.0, fill_color=MAGENTA, fill_opacity=0.15, stroke_width=0)
        band.move_to(pt((ax.c2p(hb, 0)[0] + ax.c2p(6.3, 0)[0]) / 2, ax.c2p(0, -60)[1]))
        bl = prose("bistable:\nrest and spiking", font_size=22, color=MAGENTA).move_to(ax.c2p(4.0, -52))
        vl = m.DashedLine(ax.c2p(10, -70), ax.c2p(10, -50), color=YELLOW, stroke_width=3)
        yl = prose("paper: $I_{app} = 10$\nequilibrium unstable, the big cycle\nattracts the whole plane minus $\\vec x^*$", font_size=22, color=YELLOW).move_to(pt(4.3, -2.0))
        self.play(m.FadeIn(band), m.FadeIn(bl))
        self.play(m.Create(vl), m.FadeIn(yl))
        self.wait(HOLD)
        self.next_slide("bistability")
        self.clear()

    def why_example(self):
        self.bullets("Why this example?", [
            "planar: isochrons are curves we can draw",
            "slow–fast: dramatic isochrons, and standard methods struggle (Chapter 10)",
            "repelling middle branch inside the cycle: isochrons pile up there (Chapter 8)",
            "one equilibrium, a weak spiral source: every isochron winds into it (Chapter 7)",
        ], font_size=28)
