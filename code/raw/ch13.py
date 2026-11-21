"""Deck for Chapter 14: The global isochrons of a neuron (paper section 4)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import manimgx as m
from deckkit import *  # noqa: F401,F403


def level_curves(xs, ys, th, c, minlen=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = np.sin(2 * np.pi * (th - c)); co = np.cos(2 * np.pi * (th - c))
    s = np.where(co > 0, s, np.nan)
    fig, ax = plt.subplots()
    cs = ax.contour(xs, ys, s, levels=[0.0])
    segs = [p for p in cs.allsegs[0] if len(p) > minlen]
    plt.close(fig)
    return segs


XR, YR = (-90, 60, 30), (0.3, 0.8, 0.1)


class Deck13(Deck):
    chapter = "Chapter 14"
    title = "The Global Isochrons of a Neuron"
    subtitle = "Accumulation, boomerangs and self-similarity (paper §4)"

    def construct(self):
        self.title_card()
        self.hundred()
        self.four_features()
        self.band()
        self.boomerang_trace()
        self.show_figure("ch13_left", "Zoom: every isochron passes through a thin band",
                         "Left: 100 isochrons; right: 400. The band hugs the repelling middle branch of the $V$-nullcline (dashed).")
        self.show_figure("ch13_eqzoom", "Self-similar at the equilibrium",
                         "A weak spiral source ($0.064 \\pm 0.694\\,i$): each zoom shows the same pattern, rotated.")
        self.consistency()
        self.recap(["88 of 100 isochrons start on the slow left branch",
                    "Outer branches run off horizontally: far away, phase is set by $n$",
                    "Inner branches squeeze along the repelling slow manifold, make boomerang turns, end at $x^*$",
                    "Near the repelling slow manifold every phase occurs: phaseless-set-like, inside the basin",
                    "Computed isochrons never cross; two methods agree"])

    def setup_plane(self, at=pt(-1.4, -0.45), width=9.0, height=6.0):
        c = data("rhh_cycle")
        ax = plane(XR, YR, width=width, height=height, at=at, xlabel="V", ylabel="n")
        cyc = curve(ax, c["X"], color=INK, width=4, step=20)
        return ax, cyc, c

    def hundred(self):
        d = data("rhh_phase_full")
        h = heading("One hundred isochrons, equally spaced in phase")
        ax, cyc, c = self.setup_plane()
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), run_time=1.2)
        X = c["X"]
        dots = m.VGroup(*[dot(ax, X[int(j / 100 * (len(X) - 1))], color=phase_color(j / 100), r=0.045) for j in range(100)])
        self.play(m.LaggedStart(*[m.FadeIn(dd) for dd in dots], lag_ratio=0.02), run_time=2)
        note = prose("100 points of $\\Gamma$,\n$1/100$ of a period apart", font_size=24).move_to(pt(5.0, 1.8))
        self.play(m.FadeIn(note))
        self.next_slide("base points")
        isos = m.VGroup()
        for j in range(100):
            for sg in level_curves(d["xs"], d["ys"], d["th"], j / 100):
                isos.add(curve(ax, sg, color=phase_color(j / 100), width=1.4))
        self.play(m.LaggedStart(*[m.Create(i) for i in isos], lag_ratio=0.01), run_time=6)
        self.bring_to_front(cyc)
        cap = prose("compare: paper, Figure 2", font_size=22, color=MUTED).move_to(pt(5.0, -3.0))
        self.play(m.FadeIn(cap))
        self.wait(LONG)
        self.next_slide("100 isochrons")
        self.clear()

    def four_features(self):
        self.bullets("Four features, all predicted earlier", [
            "most isochrons start on the slow left branch: equal phase = equal time (Ch. 2)",
            "outer branches run off horizontally: fast $V$, slow $n$ (Ch. 8)",
            "inner branches end at the equilibrium $x^*$, the phaseless point (Ch. 7)",
            "on the way they are squeezed along the repelling slow manifold and turn sharply (Ch. 8)",
        ], font_size=27)

    def band(self):
        d = data("rhh_phase_full")
        h = heading("Why the band? Fall left, fall right, or linger")
        ax, cyc, c = self.setup_plane(at=pt(-2.2, -0.45), width=8.0, height=6.0)
        F = None
        vg = np.linspace(-90.0013, 60.0017, 500); ng = np.linspace(0.3, 0.8, 400)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import iso_lib as L
        Fn = L.rhs_np("rhh", {})
        VV, NN = np.meshgrid(vg, ng)
        fig, a2 = plt.subplots(); cs = a2.contour(vg, ng, Fn(np.array([VV, NN]))[0], levels=[0]); plt.close(fig)
        nulls = m.VGroup(*[m.DashedVMobject(curve(ax, sg, color=GREEN, width=3), num_dashes=40) for sg in cs.allsegs[0] if len(sg) > 5])
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), run_time=1)
        self.play(m.Create(nulls), run_time=1.2)
        lab = prose("$\\dot V = 0$: its middle branch is the\nrepelling slow manifold", font_size=24, color=GREEN).move_to(pt(4.4, 2.0))
        self.play(m.FadeIn(lab))
        self.next_slide("nullcline")
        # three nearby states either side of the repelling branch: integrate quickly in python (illustration)
        from scipy.integrate import solve_ivp
        starts = [(-50.0, 0.52), (-49.0, 0.52), (-49.5, 0.52)]
        cols = [TEAL, MAGENTA, YELLOW]
        paths = []
        for s0 in starts:
            sol = solve_ivp(lambda t, u: Fn(u), (0, 12), s0, method="Radau", rtol=1e-9, atol=1e-11, max_step=0.02)
            paths.append(sol.y.T)
        trk = m.VGroup(*[curve(ax, P, color=col, width=3) for P, col in zip(paths, cols)])
        sd = m.VGroup(*[dot(ax, s0, color=col, r=0.07) for s0, col in zip(starts, cols)])
        self.play(m.FadeIn(sd))
        self.play(*[m.Create(t_) for t_ in trk], run_time=5, rate_func=m.linear)
        txt = prose("three states 1 mV apart:\none falls left (slow recovery),\none falls right (spikes first),\none lingers: any delay is possible", font_size=22).move_to(pt(4.4, -1.2))
        self.play(m.FadeIn(txt))
        self.wait(LONG)
        self.next_slide("fall left/right")
        self.clear()

    def boomerang_trace(self):
        try:
            d = data("ch13_arciso")
            P = d["P1"]
        except Exception:
            P = data("ch11_rhh_inner")["P"]
        h = heading("Following one isochron by continuation")
        ax, cyc, c = self.setup_plane()
        X = c["X"]
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), run_time=1)
        star = m.Star(n=5, outer_radius=0.16, color=ORANGE, fill_opacity=1).move_to(ax.c2p(*X[0]))
        self.play(m.FadeIn(star))
        k = m.ValueTracker(2)
        iso = m.always_redraw(lambda: curve(ax, P[: int(k.get_value())], color=MAGENTA, width=4))
        head = m.always_redraw(lambda: dot(ax, P[int(k.get_value()) - 1], color=YELLOW, r=0.08))
        self.add(iso, head)
        lab = prose("$I(\\gamma_0)$ traced point by point,\nequal arclength steps", font_size=24).move_to(pt(4.9, 2.0))
        self.play(m.FadeIn(lab))
        self.next_slide()
        self.play(k.animate.set_value(len(P)), run_time=7, rate_func=m.linear)
        bm = prose("the boomerang turn", font_size=26, color=YELLOW).move_to(pt(4.9, -0.3))
        self.play(m.FadeIn(bm))
        self.wait(HOLD)
        self.next_slide("boomerang")
        self.clear()

    def consistency(self):
        d = data("rhh_phase_full")
        h = heading("The test: isochrons never cross")
        ax, cyc, c = self.setup_plane(at=pt(-2.3, -0.45), width=8.0, height=6.0)
        self.play(m.FadeIn(h), m.Create(ax), m.Create(cyc), run_time=1)
        txt = prose("leaves of a foliation are disjoint:\nany crossing would expose an error", font_size=24, width=3.9).move_to(pt(4.75, 2.0))
        self.play(m.FadeIn(txt))
        self.next_slide()
        for j in range(0, 60):
            g = m.VGroup(*[curve(ax, sg, color=phase_color(j / 60), width=1.6) for sg in level_curves(d["xs"], d["ys"], d["th"], j / 60)])
            self.play(m.Create(g), run_time=0.12)
        self.bring_to_front(cyc)
        ok = prose("60 leaves, no crossings;\ncontinuation and phase map agree", font_size=24, color=YELLOW, width=3.9).move_to(pt(4.75, -1.0))
        self.play(m.FadeIn(ok))
        self.wait(HOLD)
        self.next_slide("consistency")
        self.clear()
