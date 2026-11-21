"""Deck for Chapter 9: Four oscillators, one method (van der Pol, FitzHugh-Nagumo, Morris-Lecar, reduced HH)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import manimgx as m
from deckkit import *  # noqa: F401,F403

CASES = [
    ("vdp", "van der Pol", (-3, 3, 1), (-3.4, 3.4, 1), "x", "y"),
    ("fhn", "FitzHugh–Nagumo", (-2.5, 2.5, 1), (-0.8, 2.0, 0.5), "v", "w"),
    ("ml", "Morris–Lecar", (-0.6, 0.5, 0.2), (-0.1, 0.6, 0.1), "V", "w"),
    ("rhh", "reduced Hodgkin–Huxley", (-90, 60, 30), (0.3, 0.8, 0.1), "V", "n"),
]


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


class Deck08b(Deck):
    chapter = "Chapter 9"
    title = "Four Oscillators, One Method"
    subtitle = "Van der Pol, FitzHugh–Nagumo, Morris–Lecar and the reduced Hodgkin–Huxley neuron"

    def construct(self):
        self.title_card()
        self.models()
        self.recipe()
        self.isochron_grid()
        self.prcs()
        self.table()
        self.recap(["One recipe: cycle → Floquet multiplier → adjoint PRC → phase map → interpret",
                    "All four: fast N-shaped nullcline, slow recovery, isochrons meeting at the equilibrium",
                    "Strong time-scale separation: tiny multipliers, isochrons squeezed on the middle branch",
                    "Hopf onset → type II PRC; SNIC onset (Morris–Lecar) → type I PRC"])

    def models(self):
        h = heading("Four classic oscillators")
        rows = [
            (r"\varepsilon\dot x = y - \frac{x^3}{3} + x,\quad \dot y = -x", "van der Pol: the textbook relaxation oscillator"),
            (r"\dot v = v - \frac{v^3}{3} - w + I,\quad \dot w = \varepsilon(v + a - bw)", "FitzHugh–Nagumo: cubic voltage, linear recovery"),
            (r"\dot V = I + g_L(V_L-V) + g_K w (V_K - V) - g_{Ca} m_\infty(V)(V-1)", "Morris–Lecar: calcium and potassium currents"),
            (r"\dot V = \ldots,\quad \dot n = \alpha_n(1-n) - \beta_n n", "reduced Hodgkin–Huxley: the paper's model"),
        ]
        self.play(m.FadeIn(h))
        g = m.VGroup()
        for f, w in rows:
            e = tex(f, font_size=34)
            if e.width > 7.6:
                e.scale_to_fit_width(7.6)
            g.add(m.VGroup(prose(w, font_size=24, color=MUTED, width=5.0), e).arrange(m.RIGHT, buff=0.5))
        g.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.5).move_to(pt(0, -0.3))
        for r in g:
            self.play(m.FadeIn(r, shift=m.RIGHT * 0.2), run_time=0.7)
            self.wait(BEAT)
            self.next_slide()
        note = prose("same skeleton: a fast variable with an N-shaped nullcline, a slow recovery,\none unstable equilibrium inside a stable cycle", font_size=24, color=YELLOW, width=12.4).to_edge(m.DOWN, buff=0.3)
        self.play(m.FadeIn(note))
        self.wait(HOLD)
        self.next_slide("models")
        self.clear()

    def recipe(self):
        self.bullets("The recipe, applied to every model", [
            "1. find the cycle: XPPAUT's Periodic gives $T$ and $\\vec\\gamma_0$",
            "2. stability: $\\lambda = \\exp\\int_0^T \\nabla\\cdot\\vec F\\,dt$",
            "3. infinitesimal PRC: the periodic adjoint, $\\vec Z = \\nabla\\vartheta$ on $\\Gamma$",
            "4. phase map: integrate a grid forward, read the phase, draw level curves",
            "5. interpret: where isochrons start, crowd, and wind",
        ], font_size=28)

    def isochron_grid(self):
        h = heading("Step 4 for all four: thirty isochrons each")
        self.play(m.FadeIn(h))
        axes = []
        pos = [pt(-3.4, 1.0), pt(3.4, 1.0), pt(-3.4, -2.4), pt(3.4, -2.4)]
        for (key, title, xr, yr, xl, yl), at in zip(CASES, pos):
            d = data(f"ch08b_{key}")
            ax = plane(xr, yr, width=4.6, height=2.75, at=at, xlabel=xl, ylabel=yl, ticks=False)
            ax.x_axis.set_opacity(0); ax.y_axis.set_opacity(0)
            frame = m.Rectangle(width=4.6, height=2.75, color=MUTED, stroke_width=1.5).move_to(at)
            lab = m.Text(title, font_size=20, color=MUTED).next_to(frame, m.UP, buff=0.06)
            cyc = curve(ax, d["X"], color=INK, width=3, step=max(1, len(d["X"]) // 400))
            self.play(m.Create(frame), m.FadeIn(lab), m.Create(cyc), run_time=0.6)
            axes.append((ax, d))
        self.next_slide()
        for j in range(30):
            grp = []
            for ax, d in axes:
                for sg in level_curves(d["xs"], d["ys"], d["th"], j / 30):
                    sg = sg[(sg[:, 0] >= d["xs"][0]) & (sg[:, 0] <= d["xs"][-1])]
                    if len(sg) > 1:
                        grp.append(curve(ax, sg, color=phase_color(j / 30), width=1.3))
            self.play(*[m.Create(g) for g in grp], run_time=0.15)
        for ax, d in axes:
            self.add(dot(ax, d["eq"], color=YELLOW, r=0.06))
        msg = prose("all meet at the equilibrium; with strong time-scale separation\nthey are squeezed along the middle branch first", font_size=22, color=YELLOW).to_edge(m.DOWN, buff=0.05)
        self.wait(LONG)
        self.next_slide("isochrons")
        self.clear()

    def prcs(self):
        h = heading("Step 3: infinitesimal PRCs (fast-variable kicks)")
        self.play(m.FadeIn(h))
        cols = [TEAL, BLUE, ORANGE, MAGENTA]
        ax = plane((0, 1, 0.25), (-1, 1, 0.5), width=8.4, height=4.6, at=pt(-1.6, -0.5), xlabel="\\vartheta", ylabel="Z")
        self.play(m.Create(ax))
        zero = m.DashedLine(ax.c2p(0, 0), ax.c2p(1, 0), color=MUTED, stroke_width=2)
        self.add(zero)
        legend = m.VGroup()
        for (key, title, *_), col in zip(CASES, cols):
            d = data(f"ch08b_{key}")
            z = d["Z"][:, 0]; z = z / np.abs(z).max()
            c = curve(ax, np.column_stack([d["tz"], z]), color=col, width=3.5)
            self.play(m.Create(c), run_time=1.2)
            legend.add(prose(title, font_size=22, color=col))
        legend.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.15).move_to(pt(4.6, 1.2))
        self.play(m.FadeIn(legend))
        self.next_slide()
        t = prose("Hopf onset (van der Pol, FHN, HH): negative lobes, type II\nSNIC onset (Morris–Lecar): almost all positive, type I:\nnear the bottleneck a kick can only push forward", font_size=22, color=YELLOW, width=4.9).move_to(pt(4.5, -1.6))
        self.play(m.FadeIn(t))
        self.wait(LONG)
        self.next_slide("PRC types")
        self.clear()

    def table(self):
        h = heading("The four side by side")
        self.play(m.FadeIn(h))
        header = ["", "van der Pol", "FitzHugh–Nagumo", "Morris–Lecar", "reduced HH"]
        rows = [
            ["onset", "Hopf", "Hopf (canards)", "SNIC", "subcritical Hopf"],
            ["period T", "6.66", "39.5", "16.5", "11.8"],
            ["ln λ", "−7.1", "−38.3", "−10.0", "−46.5"],
            ["backward integration", "fine", "fails", "fine", "fails"],
            ["slow share of period", "57%", "87%", "84%", "94%"],
            ["spiral turns per e-fold", "0.28", "0.21", "0.50", "1.73"],
            ["PRC type", "II", "II", "I", "II"],
        ]
        widths = [3.3, 2.3, 2.6, 2.3, 2.6]
        y0 = 2.2
        cells = m.VGroup()
        for r, row in enumerate([header] + rows):
            x = -6.6
            for c, (txt, w) in enumerate(zip(row, widths)):
                t = m.Text(txt, font_size=20 if r else 21, color=(YELLOW if r == 0 else (MUTED if c == 0 else INK)),
                           weight=(m.BOLD if r == 0 else m.NORMAL))
                if t.width > w - 0.15:
                    t.scale_to_fit_width(w - 0.15)
                t.move_to(pt(x + w / 2, y0 - 0.62 * r))
                cells.add(t)
                x += w
        lines = m.VGroup(*[m.Line(pt(-6.6, y0 - 0.62 * r + 0.31), pt(6.5, y0 - 0.62 * r + 0.31), color=GREY_LINE, stroke_width=1)
                           for r in range(1, len(rows) + 1)])
        self.play(m.FadeIn(lines), m.FadeIn(cells[:5]))
        for r in range(len(rows)):
            self.play(m.FadeIn(cells[5 * (r + 1): 5 * (r + 2)], shift=m.RIGHT * 0.1), run_time=0.5)
            self.wait(0.8)
            self.next_slide()
        self.wait(HOLD)
        self.next_slide("table")
        self.clear()


GREY_LINE = m.GREY_D
