"""Shared helpers for the isochron chapter decks (ManimGX + Manim Slides via sciviz).

Each chapter deck is one Scene in scripts/isochron/decks/chNN.py, rendered with
    .venv/bin/python -m sciviz render isochron-chNN scripts/isochron/decks/chNN.py DeckNN
    .venv/bin/python -m sciviz deck isochron-chNN --offline
Data come from rendered-anims/isochron/data/*.npz (written by scripts/isochron/figs/*.py, computed with
XPPAUT through pyauto-xpp); book figures come from rendered-anims/isochron/book/figs/*.png.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import manimgx as m
from sciviz.manim import Slides

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "rendered-anims" / "isochron" / "data"
FIGS = ROOT / "rendered-anims" / "isochron" / "book" / "figs"

# palette (matches the book figures, brightened for a dark background)
INK = m.WHITE
MUTED = m.GREY_B
TEAL = m.ManimColor("#2ec4c9")
MAGENTA = m.ManimColor("#e0559a")
ORANGE = m.ManimColor("#f0a040")
BLUE = m.ManimColor("#5b9cf0")
GREEN = m.ManimColor("#59c36a")
RED = m.ManimColor("#ef5b5b")
YELLOW = m.YELLOW
PHASE = [m.ManimColor(c) for c in ("#00b7c7", "#4a6fd6", "#9a3fc4", "#e0339a")]

BEAT, HOLD, LONG = 1.5, 3.0, 4.5


def data(name: str) -> dict:
    return dict(np.load(DATA / f"{name}.npz", allow_pickle=True))


def phase_color(c: float) -> m.ManimColor:
    """cyan -> blue -> purple -> magenta, the book's isochron colours."""
    c = float(c) % 1.0
    x = c * (len(PHASE) - 1)
    i = min(int(x), len(PHASE) - 2)
    return m.interpolate_color(PHASE[i], PHASE[i + 1], x - i)


def tex(*parts: str, font_size: float = 40, **kw) -> m.MathTex:
    safe = [p.replace("{{", "{ {").replace("}}", "} }") for p in parts]
    return m.MathTex(*safe, font_size=font_size, **kw)


def prose(text: str, font_size: float = 28, color=INK, width: float | None = None) -> m.VGroup:
    """Left-aligned lines that may contain $maths$ (lines split at \\n)."""
    lines = [m.Tex(line, font_size=1.3 * font_size, color=color) for line in text.split("\n")]
    g = m.VGroup(*lines).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
    if width and g.width > width:
        g.scale_to_fit_width(width)
    return g


def heading(text: str, font_size: float = 38) -> m.Text:
    t = m.Text(text, font_size=font_size, weight=m.BOLD).to_edge(m.UP, buff=0.35)
    if t.width > 13.2:
        t.scale_to_fit_width(13.2)
    return t


def pt(x: float, y: float) -> np.ndarray:
    return np.array([x, y, 0.0])


def figure_card(name: str, height: float = 5.6, at=pt(0, -0.35)) -> m.Group:
    """A book figure (PNG, white background) on a rounded card."""
    path = FIGS / f"{name}.png"
    if not path.exists():
        path = FIGS / f"{name}_still.png"
    im = m.ImageMobject(str(path))
    im.scale_to_fit_height(height)
    if im.width > 13.2:
        im.scale_to_fit_width(13.2)
    im.move_to(at)
    card = m.RoundedRectangle(width=im.width + 0.25, height=im.height + 0.25, corner_radius=0.12,
                              fill_color=m.WHITE, fill_opacity=1, stroke_width=0).move_to(im)
    return m.Group(card, im)


def plane(xr, yr, width=7.0, height=5.0, at=pt(0, -0.4), xlabel="x", ylabel="y", ticks=True) -> m.Axes:
    ax = m.Axes(x_range=xr, y_range=yr, x_length=width, y_length=height, tips=False,
                axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": False, "include_numbers": False,
                             "font_size": 26})
    ax.move_to(at)
    if ticks:
        # numbered axes: hide the axes through the origin and draw a frame along the left and bottom edges
        for a in (ax.x_axis, ax.y_axis):
            a.set_opacity(0)
        x0, x1, dx = xr[:3]; y0, y1, dy = yr[:3]
        frame = m.VGroup(m.Line(ax.c2p(x0, y0), ax.c2p(x1, y0), color=MUTED, stroke_width=2),
                         m.Line(ax.c2p(x0, y0), ax.c2p(x0, y1), color=MUTED, stroke_width=2))
        def fmt(v):
            return f"{v:g}"
        for v in np.arange(x0, x1 + 1e-9, dx):
            frame.add(m.Line(ax.c2p(v, y0), ax.c2p(v, y0) + np.array([0, -0.08, 0]), color=MUTED, stroke_width=2))
            frame.add(m.MathTex(fmt(round(v, 6)), font_size=24, color=MUTED).next_to(ax.c2p(v, y0), m.DOWN, buff=0.15))
        for v in np.arange(y0, y1 + 1e-9, dy):
            frame.add(m.Line(ax.c2p(x0, v), ax.c2p(x0, v) + np.array([-0.08, 0, 0]), color=MUTED, stroke_width=2))
            frame.add(m.MathTex(fmt(round(v, 6)), font_size=24, color=MUTED).next_to(ax.c2p(x0, v), m.LEFT, buff=0.15))
        ax.add(frame)
        lx = m.MathTex(xlabel, font_size=40, color=MUTED).next_to(ax.c2p(x1, y0), m.RIGHT, buff=0.55)
        ly = m.MathTex(ylabel, font_size=40, color=MUTED).next_to(ax.c2p(x0, y1), m.UP, buff=0.2)
        ax.add(lx, ly)
        return ax
    lx = m.MathTex(xlabel, font_size=40, color=MUTED).next_to(ax.x_axis.get_end(), m.RIGHT, buff=0.2)
    ly = m.MathTex(ylabel, font_size=40, color=MUTED).next_to(ax.y_axis.get_end(), m.UP, buff=0.2)
    ax.add(lx, ly)
    return ax


def curve(ax: m.Axes, X: np.ndarray, color=INK, width: float = 3.0, step: int = 1) -> m.VMobject:
    X = np.asarray(X, float)[::step]
    X = X[np.all(np.isfinite(X), axis=1)]
    pts = [ax.c2p(x, y) for x, y in X]
    v = m.VMobject(color=color, stroke_width=width)
    v.set_points_as_corners(pts) if len(pts) > 1 else None
    return v


def dot(ax: m.Axes, p, color=INK, r: float = 0.07) -> m.Dot:
    return m.Dot(ax.c2p(*p), radius=r, color=color)


def cite(text: str) -> m.Text:
    return m.Text(text, font_size=18, color=m.GREY_C).to_corner(m.DR, buff=0.15)


class Deck(Slides, m.Scene):
    """Base: title card and helpers to step through slides."""
    chapter = "Chapter"
    title = ""
    subtitle = ""

    def title_card(self):
        k = m.Text(self.chapter.upper(), font_size=24, color=MAGENTA)
        t = m.Text(self.title, font_size=54, weight=m.BOLD)
        if t.width > 12.5:
            t.scale_to_fit_width(12.5)
        s = m.Text(self.subtitle, font_size=28, color=MUTED, slant=m.ITALIC)
        if s.width > 12.5:
            s.scale_to_fit_width(12.5)
        g = m.VGroup(k, t, s).arrange(m.DOWN, buff=0.35)
        line = m.Line(pt(-5, 0), pt(5, 0), color=TEAL, stroke_width=3).next_to(g, m.DOWN, buff=0.4)
        self.play(m.FadeIn(k, shift=m.DOWN * 0.2), m.Write(t), run_time=1.6)
        self.play(m.FadeIn(s), m.Create(line), run_time=1.0)
        self.wait(BEAT)
        self.next_slide(self.title)
        self.play(m.FadeOut(m.Group(k, t, s, line)), run_time=0.6)

    def clear(self, run_time: float = 0.6):
        if self.mobjects:
            self.play(*[m.FadeOut(mob) for mob in self.mobjects], run_time=run_time)

    def show_figure(self, name: str, title: str, caption: str = "", height: float = 5.6, notes: str = ""):
        if not (FIGS / f"{name}.png").exists() and not (FIGS / f"{name}_still.png").exists():
            print(f"[deck] figure {name} not built yet: slide skipped")
            return
        h = heading(title)
        card = figure_card(name, height=height, at=pt(0, -0.35 if not caption else -0.15))
        self.play(m.FadeIn(h), m.FadeIn(card, scale=0.97), run_time=1.0)
        if caption:
            c = prose(caption, font_size=22, color=MUTED, width=12.8).to_edge(m.DOWN, buff=0.18)
            self.play(m.FadeIn(c), run_time=0.6)
        self.wait(HOLD)
        self.next_slide(notes or title)
        self.clear()

    def bullets(self, title: str, items: list[str], font_size: float = 30, notes: str = ""):
        h = heading(title)
        self.play(m.FadeIn(h), run_time=0.6)
        rows = [prose("• " + it, font_size=font_size, width=12.4) for it in items]
        g = m.VGroup(*rows).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.4).next_to(h, m.DOWN, buff=0.6)
        g.to_edge(m.LEFT, buff=0.8)
        for r in rows:
            self.play(m.FadeIn(r, shift=m.RIGHT * 0.2), run_time=0.7)
            self.wait(BEAT)
            self.next_slide(notes)
        self.clear()

    def recap(self, items: list[str]):
        self.bullets("What to carry forward", items, font_size=28, notes="recap")
