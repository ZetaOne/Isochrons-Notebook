"""Deck for Chapter 15: What it means and what came next (paper section 5)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import manimgx as m
from deckkit import *  # noqa: F401,F403


class Deck14(Deck):
    chapter = "Chapter 15"
    title = "What It Means and What Came Next"
    subtitle = "Applications, higher dimensions, Winfree's puzzle, Koopman isochrons (paper §5)"

    def construct(self):
        self.title_card()
        self.achievement()
        self.finite_kicks()
        self.show_figure("ch14_prs", "Finite kicks need global isochrons",
                         "Phase shift after a voltage kick $\\Delta V$ at phase $\\vartheta$. Small kicks follow the iPRC; large kicks jump where they land in the band.")
        self.bullets("Two more applications", [
            "desynchronising neurons: drive them to where all phases crowd, let noise scramble them",
            "the paper: aim at the repelling slow manifold, not just the equilibrium",
            "power grids: a generator must connect on the right isochron",
        ], font_size=27)
        self.bullets("Beyond the plane", [
            "in $\\mathbb{R}^n$ an isochron is an $(n-1)$-dimensional manifold",
            "grow it from orbit segments ending on a small disc of its tangent space",
            "or compute its intersection with a plane of reachable initial conditions",
        ], font_size=27)
        self.show_figure("ch14_fhn", "Winfree's puzzle (Langfield, Krauskopf & Osinga 2014)",
                         "FitzHugh–Nagumo isochrons squeeze against the repelling middle branch: the same mechanism.", height=5.2)
        self.koopman()
        self.show_figure("ch14_fourier", "Three methods, the same isochrons",
                         "Fourier averages (Koopman) on the left, direct phase map on the right; continuation agreed in Chapter 12.")
        self.summary()

    def achievement(self):
        h = heading("The paper in four steps")
        steps = [
            ("1", "isochron = stable manifold of $\\vec\\gamma$ for the time-$T$ map"),
            ("2", "grow it from $L_\\eta$ by orbit segments ending on $L_\\eta$ (BVPs)"),
            ("3", "continuation: arclength steps; add a period when $\\lvert\\delta\\rvert = \\eta$"),
            ("4", "neuron: isochrons squeezed along the repelling slow manifold"),
        ]
        self.play(m.FadeIn(h))
        rows = m.VGroup()
        for n, t in steps:
            badge = m.VGroup(m.Circle(radius=0.28, color=TEAL, fill_opacity=0.25), m.Text(n, font_size=26)).arrange(m.ORIGIN)
            rows.add(m.VGroup(badge, prose(t, font_size=27, width=10.8)).arrange(m.RIGHT, buff=0.4))
        rows.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.5).move_to(pt(0, -0.4))
        for r in rows:
            self.play(m.FadeIn(r, shift=m.RIGHT * 0.2), run_time=0.7)
            self.wait(BEAT)
            self.next_slide()
        self.clear()

    def finite_kicks(self):
        h = heading("Small kicks: a gradient. Big kicks: the global picture")
        e1 = tex(r"\Delta\vartheta", r"=", r"\vartheta(\vec\gamma + \Delta\vec x) - \vartheta(\vec\gamma)", font_size=44).move_to(pt(0, 1.4))
        self.play(m.FadeIn(h), m.Write(e1))
        self.next_slide()
        e2 = tex(r"\approx \nabla\vartheta(\vec\gamma)\cdot\Delta\vec x", font_size=44, color=TEAL).next_to(e1, m.DOWN, buff=0.5)
        l2 = prose("only for small $\\Delta\\vec x$: needs the isochrons near $\\Gamma$ only", font_size=24, color=TEAL).next_to(e2, m.DOWN, buff=0.25)
        self.play(m.Write(e2), m.FadeIn(l2))
        self.next_slide()
        l3 = prose("large $\\Delta\\vec x$ lands far from $\\Gamma$:\nyou need the global isochrons", font_size=28, color=YELLOW).move_to(pt(0, -2.2))
        self.play(m.FadeIn(l3))
        self.wait(HOLD)
        self.next_slide("finite kicks")
        self.clear()

    def koopman(self):
        h = heading("Isochrons as a Koopman eigenfunction")
        e1 = tex(r"\vartheta\big(\Phi_t(\vec x)\big) = \vartheta(\vec x) + \frac{t}{T}", font_size=42).move_to(pt(0, 1.9))
        self.play(m.FadeIn(h), m.Write(e1))
        self.next_slide()
        e2 = tex(r"\psi(\vec x) = e^{2\pi i\,\vartheta(\vec x)}\quad\Rightarrow\quad \psi\big(\Phi_t(\vec x)\big) = e^{i\omega t}\,\psi(\vec x)", font_size=40, color=TEAL).move_to(pt(0, 0.6))
        l2 = prose("phase advances uniformly, so $\\psi$ is an eigenfunction of $f \\mapsto f\\circ\\Phi_t$", font_size=24, color=MUTED).next_to(e2, m.DOWN, buff=0.25)
        self.play(m.Write(e2), m.FadeIn(l2))
        self.next_slide()
        e3 = tex(r"f^*_\omega(\vec x) = \lim_{t\to\infty}\frac1t\int_0^t f\big(\Phi_s(\vec x)\big)\,e^{-i\omega s}\,ds \;\propto\; \psi(\vec x)", font_size=38, color=YELLOW).move_to(pt(0, -1.4))
        l3 = prose("a Fourier average along one forward trajectory (Mauroy & Mezić 2012):\nisochrons are level sets of $\\arg f^*_\\omega$, no backward integration at all", font_size=24, color=MUTED).next_to(e3, m.DOWN, buff=0.3)
        self.play(m.Write(e3), m.FadeIn(l3))
        self.wait(HOLD)
        self.next_slide("Koopman")
        self.clear()

    def summary(self):
        h = heading("The whole book in one breath")
        txt = prose(
            "A stable oscillation is a clock. Phase extends from the cycle to the basin:\n"
            "which on-cycle state does each start become? Equal-phase curves, the isochrons,\n"
            "are strong stable manifolds. They encode how kicks shift the rhythm, crowd onto\n"
            "sets without phase, and in slow–fast systems pile up along repelling slow manifolds.\n"
            "Osinga & Moehlis compute them by continuing orbit segments that end near the cycle.",
            font_size=26, width=12.6).move_to(pt(0, -0.2))
        self.play(m.FadeIn(h))
        self.play(m.Write(txt), run_time=4)
        self.wait(LONG)
        self.next_slide("summary")
        self.clear()
