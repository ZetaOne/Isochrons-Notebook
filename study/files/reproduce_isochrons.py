#!/usr/bin/env python3
"""
reproduce_isochrons.py -- global isochrons of the reduced Hodgkin-Huxley model by
continuation, as in Osinga & Moehlis, SIAM J. Appl. Dyn. Syst. 9 (2010), and as in
notebook 02_ISOCHRONS_continuation. Every numbered STEP below matches the step of the
same number in isochron_recipe.pdf.

The computation runs in XPPAUT 8.2 and its AUTO, driven from Python by pyauto-xpp.
Installation (pyauto-xpp is installed from its GitHub release):
    python3 -m venv isochron-env && source isochron-env/bin/activate
    pip install pyauto-xpp numpy matplotlib tqdm \\
        --find-links https://github.com/ZetaOne/pyauto-xpp/releases/expanded_assets/v0.3.0

Usage:
    python reproduce_isochrons.py                     # the paper: 100 isochrons, Figures 2 and 3
    python reproduce_isochrons.py --workers 8         # the same, 8 isochrons at a time
    python reproduce_isochrons.py --quick             # only the 0-phase isochron (about 5 minutes)
    python reproduce_isochrons.py --phases 0 0.3      # chosen phases
    python reproduce_isochrons.py --no-show           # save the figures without opening windows

Defaults = the paper: phases theta_i = i/100 (i = 0..99), 4 stages inside the orbit (time-4T map),
1 stage outside. Finished isochrons are saved one file per phase and skipped on a rerun, so an
interrupted run continues where it stopped.

Output (in --out, default "isochron_results"):
    orbit.dat                    the orbit Gamma from gamma0:  t  V  n
    odes/reduced_hh.ode          the model (steps 1-3: orbit, period, adjoint)
    odes/isochron_bvp_template.ode   the BVP (step 4), with gamma and n_s of the first phase
    odes/isochron_<theta>_<in|out>_k<k>.ode   the BVP of every stage: per = kT, gamma, n_s filled in
    isochron_<theta>.dat         the isochron points:  V  n  delta  stage  side   (side -1 inside, +1 outside)
    isochron_<theta>.log         what each stage did (mesh, Newton iterations, points, delta at the end)
    figure2_all_isochrons.png    the paper's Figure 2: all isochrons, coloured by phase
    figure3_zero_phase.png       the paper's Figure 3: the 0-phase isochron by arclength
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon
from pyxpp.builder import ModelBuilder
from tqdm import tqdm


# =====================================================================================
# STEP 1. The model: the reduced Hodgkin-Huxley equations (4.1) of the paper.
#   V' = I - gNa m_inf(V)^3 (0.8 - n)(V - ENa) - gK n^4 (V - EK) - gL (V - EL)
#   n' = alpha_n(V)(1 - n) - beta_n(V) n
# Written once as XPP functions and right-hand sides; used by the plain model (STEP 2)
# and by the boundary value problem (STEP 4).
# =====================================================================================
FUN = [("am(v)", "0.1*(v+40)/(1-exp(-(v+40)/10))"), ("bm(v)", "4*exp(-(v+65)/18)"),
       ("an(v)", "0.01*(v+55)/(1-exp(-(v+55)/10))"), ("bn(v)", "0.125*exp(-(v+65)/80)"),
       ("minf(v)", "am(v)/(am(v)+bm(v))")]
FV = "iapp-gna*minf(v)^3*(0.8-n)*(v-ena)-gk*n^4*(v-ek)-gl*(v-el)"
FN = "an(v)*(1-n)-bn(v)*n"
PARS = dict(iapp=10, gna=120, gk=36, gl=0.3, ena=50, ek=-77, el=-54.4)
OPTS = dict(meth="rk4", bound=1e4, jac_eps=1e-7, newt_tol=1e-10)

# Numerical constants of the paper (its Table 1), and of this implementation
ETA = 1e-4            # half-length eta of the linear piece L_eta(gamma) of the isochron
DS = 1e-5             # first continuation step, 0.1 eta
NTST_GAMMA = 100      # mesh intervals per period of the orbit; stage k uses 100 k
AUTO_NUM = dict(ncol=4, dsmin=1e-8, dsmax=0.1, epsl=1e-8, epsu=1e-8, epss=1e-6,
                itmx=8, itnw=5, nwtn=3, iad=3, normmax=1e6, npr=100000)
NMX = 5000            # at most this many continuation points per stage
SC = np.array([100.0, 1.0])   # V and n scaled to similar sizes, for distances only


def plain_model(workdir):
    """the model as an XPP .ode file, loaded into XPPAUT (one private copy per model)"""
    mb = ModelBuilder("reduced_hh")
    for sig, expr in FUN:
        mb.function(sig, expr)
    mb = (mb.ode("v", FV).ode("n", FN).parameter(**PARS).initial(v=-60, n=0.3)
          .option(total=300, dt=0.01, maxstor=400000, **OPTS))
    return mb.load(Path(workdir) / "reduced_hh.ode")


# =====================================================================================
# STEP 2. The orbit Gamma, its period T and the phase-zero point gamma0.
#   * integrate onto the orbit, then XPP's Initialconds > Periodic gives T;
#   * gamma0 = the point of Gamma where V is largest, i.e. where V' = 0 at the top of the
#     spike: found by bisection on the time, to double precision.
# =====================================================================================
class Orbit:
    def __init__(self, m):
        self.m = m
        m.run()                                    # settle onto the orbit
        m.buttons.xi_vs_t("v")
        m.buttons.initialconds.periodic()          # XPP: Initialconds > Periodic; total = T
        self.T = m.numerics["total"]
        g = self.flow(np.array([m.ics["V"], m.ics["N"]]), self.T - 0.05)
        a, b = 0.0, 0.1
        for _ in range(40):                        # bisection for V' = 0 (the maximum of V)
            c = 0.5 * (a + b)
            a, b = (c, b) if m.rhs(self.flow(g, c))["V"] > 0 else (a, c)
        self.gamma0 = self.flow(g, 0.5 * (a + b))
        m.set(v=self.gamma0[0], n=self.gamma0[1])
        self.orb = m.run(keep=False, total=self.T, dt=0.001, trans=0)
        self.poly = Polygon(np.c_[self.orb["v"], self.orb["n"]])   # to tell inside from outside

        # =================================================================================
        # STEP 3. The direction n_s of the isochron at each point of Gamma.
        #   XPP's adjoint (Numerics > Averaging > New adj) gives Z(t) along Gamma, the
        #   gradient of the asymptotic phase (the infinitesimal phase response curve).
        #   The isochron is perpendicular to it: n_s = (Z_n, -Z_V) / |Z|.
        # =================================================================================
        A = m.buttons.numerics.averaging
        A.parameters(6400)                         # the finest grid XPP offers for the adjoint
        A.new_adj()
        self.adj = m.data                          # columns t, Z_V, Z_n (t = 0 at gamma0)

    def flow(self, x, t, dt=1e-4):
        """Phi(t, x): integrate the model from x for time t (RK4, step at most dt)"""
        self.m.set(v=x[0], n=x[1])
        self.m.run(keep=False, total=t, dt=t / max(1, int(np.ceil(t / dt))), trans=0)
        s = self.m.last_state()
        return np.array([s["V"], s["N"]])

    def phase_point(self, theta):
        """gamma(theta) = Phi(theta T, gamma0) and the unit tangent n_s of its isochron,
        oriented to point out of Gamma"""
        gth = self.gamma0.copy() if theta == 0 else self.flow(self.gamma0, theta * self.T)
        zv = np.interp(theta * self.T, self.adj.t, self.adj["v"])
        zn = np.interp(theta * self.T, self.adj.t, self.adj["n"])
        ns = np.array([zn, -zv]) / np.hypot(zn, zv)
        f = self.m.rhs(gth)
        if f["V"] * ns[1] - f["N"] * ns[0] > 0:    # Gamma turns counter-clockwise: outside is to the right of F
            ns = -ns
        return gth, ns


# =====================================================================================
# STEP 4. The boundary value problem (3.2)-(3.3) as an XPP .ode file.
#   time rescaled to [0, 1]:   u' = per * F(u),   per = k T at stage k
#   boundary condition:        u(1) = gamma + d * n_s      (two "bndry" lines; v' means v(1))
#   u(0) is free; d (= delta) is the continuation parameter. gv, gn = gamma; tv, tn = n_s.
# =====================================================================================
def bvp_builder(T):
    bb = ModelBuilder("isochron_bvp")
    bb.comment("orbit segments u' = per*F(u) on [0,1] that end on L_eta(gamma): u(1) = gamma + d*n_s")
    for sig, expr in FUN:
        bb.function(sig, expr)
    return (bb.ode("v", f"per*({FV})").ode("n", f"per*({FN})")
            .parameter(**PARS).parameter(per=T, gv=0, gn=0, tv=1, tn=0, d=0)
            .boundary("v'-gv-d*tv").boundary("n'-gn-d*tn")
            .option(total=1, dt=1e-4, maxstor=400000, **OPTS))


def load_bvp(T):
    B = bvp_builder(T).load(Path(tempfile.mkdtemp()) / "isochron_bvp.ode")
    B.auto.parameters("d", "per")              # AUTO's continuation parameter is d
    return B


# =====================================================================================
# STEP 5-8. Algorithm Isochron of the paper, for one phase point gamma.
# =====================================================================================
class Isochron:
    def __init__(self, B, T, poly_xy, gth, ns, bar=None, theta=None, odedir=None):
        self.B, self.T, self.g, self.ns, self.bar = B, T, gth, ns, bar
        self.theta, self.odedir = theta, odedir     # where the .ode file of every stage is kept
        self.poly = Polygon(poly_xy)               # the orbit, to tell inside from outside
        self.uG = self.gamma_solution()
        self.log = []

    # STEP 5a. The starting solution of stage 1: Gamma itself from gamma over one period
    # (delta = 0), as a solution on [0, 1], made by integrating the BVP's equations.
    def gamma_solution(self):
        self.B.set(per=self.T, v=self.g[0], n=self.g[1], d=0)
        tr = self.B.run(total=1, dt=1e-4)
        return np.asarray(tr.t, float), np.c_[tr["v"], tr["n"]].astype(float)

    @staticmethod
    def _pts(points):
        """the isochron points of a continuation run: u(0) of each solution, and its delta"""
        return np.array([[p.u0["v"], p.u0["n"], p.par["d"]] for p in points]).reshape(-1, 3)

    # STEP 5b. Start a stage: AUTO's "Bdry Value" start from a given solution (t, U).
    # XPP's AUTO puts the starting solution on a uniform mesh, which resolves the spike
    # badly, so the first Newton steps push delta off; 6 steps with delta unbounded let
    # the mesh adapt (IAD = 3), and the continuation then restarts from the last of them.
    # If that fails, more Newton iterations (ITNW 20) and then a finer mesh are tried.
    def start(self, k, initial):
        B = self.B
        tries = [(NTST_GAMMA * k, 5), (NTST_GAMMA * k, 20), (2 * NTST_GAMMA * k, 20),
                 (4 * NTST_GAMMA * k, 20)]
        for ntst, itnw in tries:
            B.set(per=k * self.T, gv=self.g[0], gn=self.g[1], tv=self.ns[0], tn=self.ns[1], d=0)
            B.auto.reset()
            first = B.auto.run("d", start="bdry value", initial=initial, ntst=ntst, ds=DS, nmax=6,
                               parmin=-1, parmax=1, **dict(AUTO_NUM, itnw=itnw, dsmax=1e-6))
            if len(first) == 6 and all(p.type != "MX" for p in first) \
                    and abs(first[-1].par["d"]) < ETA / 10:
                return first[-1], ntst, itnw
            self.log.append(f"k={k}: start with NTST={ntst}, ITNW={itnw} failed")
        raise RuntimeError(f"k={k}: AUTO did not converge from the starting solution")

    # STEP 6 (helper). Where the segment is one period before its end:
    # delta' = (u((k-1)/k) - gamma) . n_s, from the stored solution (cubic interpolation).
    def delta_back(self, sol, k):
        t, U = sol
        tt = (k - 1) / k
        j = int(np.clip(np.searchsorted(t, tt) - 2, 0, len(t) - 4))
        ts, Us = t[j:j + 4], U[j:j + 4]
        w = np.array([np.prod([(tt - ts[m]) / (ts[l] - ts[m]) for m in range(4) if m != l])
                      for l in range(4)])
        return float((w @ Us - self.g) @ self.ns)

    # STEP 6. Choose the direction. delta changes by far less than the discretisation error
    # along most of the family (the time-T map contracts by about 1e-20), so the sign of DS
    # does not choose the direction. Probe 25 steps each way (150 if undecided) and keep:
    #   stage 1:     the way that goes inside (or outside) Gamma, as asked;
    #   stage k > 1: the way that extends the isochron, recognised by |delta'| growing
    #                (the other way goes back along the previous stage).
    def score(self, u, k, direction, probe, d0):
        if k == 1:
            last = u[-max(3, len(u) // 4):, :2]
            far = last[np.hypot(*((last - self.g) / SC).T) > 1e-2]
            if len(far) == 0:
                return 0.0
            return (self.poly.contains_points(far).mean() - 0.5) * (-direction)
        d1 = self.delta_back(probe[-1].solution, k)
        return (abs(d1) - abs(d0)) / abs(d0)

    # STEP 7. One stage: continue in delta until |delta| = eta (parameter bounds +-eta).
    # Outside Gamma |delta| may never reach eta, so the outside branch is followed until it
    # leaves the window of interest instead. The u(0) of every point is an isochron point.
    def stage(self, k, direction, initial, nprobe=25):
        B = self.B
        t0 = time.time()
        if self.odedir is not None:                # the BVP of this stage as an XPP file, kept
            side = "in" if direction < 0 else "out"
            ode = bvp_builder(k * self.T).render().replace(
                "gv=0,gn=0,tv=1,tn=0,d=0",
                f"gv={self.g[0]:.10g},gn={self.g[1]:.10g},tv={self.ns[0]:.10g},tn={self.ns[1]:.10g},d=0")
            ode = ode.replace("@ total=1", f"# stage {k} ({side}side the orbit): per = {k}*T; start AUTO with "
                              f"Run > Bdry Value, main parameter d, Ntst = {NTST_GAMMA * k}\n@ total=1", 1)
            (Path(self.odedir) / f"isochron_{self.theta:.3f}_{side}_k{k}.ode").write_text(ode)
        p0, ntst, itnw = self.start(k, initial)
        d0 = self.delta_back(p0.solution, k) if k > 1 else None
        scores = {}
        for n_try in (nprobe, 6 * nprobe):
            for sign in (1, -1):
                probe = B.auto.run(grab=p0.label, ds=sign * DS, nmax=n_try, parmin=-ETA, parmax=ETA,
                                   **AUTO_NUM)
                u = self._pts(probe)
                scores[sign] = self.score(u, k, direction, probe, d0)
                if scores[sign] > 0.01:
                    break
            else:
                sign = max(scores, key=scores.get)
                if scores[sign] <= 0:
                    continue
                probe = B.auto.run(grab=p0.label, ds=sign * DS, nmax=n_try, parmin=-ETA, parmax=ETA,
                                   **AUTO_NUM)
                u = self._pts(probe)
            break
        else:
            raise RuntimeError(f"k={k}: neither direction goes the requested way {scores}")
        parts, last = [u], probe[-1]
        chunk = NMX if direction < 0 else 100
        while abs(last.par["d"]) < ETA and last.type != "MX" and sum(map(len, parts)) < NMX:
            if direction > 0 and not (-150 < last.u0["v"] < 150 and 0.15 < last.u0["n"] < 0.95):
                break                              # the outside branch has left the window
            more = B.auto.run(grab=last.label, ds=sign * DS,
                              nmax=min(chunk, NMX - sum(map(len, parts))),
                              parmin=-ETA, parmax=ETA, **AUTO_NUM)
            if len(more) <= 1:
                break
            parts.append(self._pts(more)[1:])
            last = more[-1]
        u = np.vstack(parts)
        info = dict(k=k, side="in" if direction < 0 else "out", ntst=ntst, itnw_start=itnw,
                    points=len(u), d_end=float(last.par["d"]), seconds=round(time.time() - t0, 1))
        self.log.append(json.dumps(info))
        return u, last.solution, info

    # STEP 8. The starting solution of stage k (the paper's eq. 3.5): the last segment of
    # stage k-1 (k-1 periods) followed by one period of Gamma, rescaled to [0, 1].
    def concatenate(self, k, sol):
        (ts, Us), (tG, UG) = sol, self.uG
        tt = np.linspace(0, 1, 4000 * k + 1)
        a = tt <= (k - 1) / k
        U = np.empty((len(tt), 2))
        for j in range(2):
            U[a, j] = np.interp(tt[a] * k / (k - 1), ts, Us[:, j])
            U[~a, j] = np.interp(tt[~a] * k - (k - 1), tG, UG[:, j])
        return tt, U

    # STEP 5-8 together: inside Gamma up to the time-(kmax T) map, outside up to (kout T).
    def run(self, kmax=4, kout=1):
        out = {}
        for direction, kk in ((-1, kmax), (+1, kout)):
            init = self.uG                                         # STEP 5a: start from Gamma
            for k in range(1, kk + 1):
                u, sol, info = self.stage(k, direction, init)      # STEPS 5b-7
                out[("in" if direction < 0 else "out", k)] = (u, info)
                if self.bar is not None:                           # progress: one tick per stage
                    self.bar.set_postfix(stage=f"{info['side']} k={k}", points=info["points"],
                                         NTST=info["ntst"])
                    self.bar.update(1)
                if k < kk:
                    init = self.concatenate(k + 1, sol)            # STEP 8
        return out


# =====================================================================================
# STEP 9. All phases (in parallel if asked), saving, and the paper's Figures 2 and 3.
# =====================================================================================
def iso_file(out, theta):
    return out / f"isochron_{theta:.3f}.dat"


def save(out, theta, res, log):
    rows = [np.c_[u, np.full(len(u), k), np.full(len(u), -1 if side == "in" else 1)]
            for (side, k), (u, info) in res.items()]
    np.savetxt(iso_file(out, theta), np.vstack(rows), fmt="%.12g", header="V n delta stage side")
    (out / f"isochron_{theta:.3f}.log").write_text("\n".join(log) + "\n")


_W = {}                                        # per worker process: its own XPPAUT copy of the BVP


def _init_worker(T, poly_xy, kmax, kout, out):
    _W.update(T=T, poly=poly_xy, kmax=kmax, kout=kout, out=Path(out), B=None)


def _worker(job):
    """one phase in a worker process (one isochron, all its stages)"""
    theta, gth, ns = job
    if _W["B"] is None:
        _W["B"] = load_bvp(_W["T"])
    t0 = time.time()
    iso = Isochron(_W["B"], _W["T"], _W["poly"], gth, ns, theta=theta, odedir=_W["out"] / "odes")
    res = iso.run(kmax=_W["kmax"], kout=_W["kout"])
    save(_W["out"], theta, res, iso.log)
    return theta, round(time.time() - t0, 1), sum(len(u) for u, _ in res.values())


def nullclines(window):
    V, N = np.meshgrid(np.linspace(window[0], window[1], 700), np.linspace(window[2], window[3], 500))
    with np.errstate(all="ignore"):
        am = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10)); bm = 4 * np.exp(-(V + 65) / 18)
        an = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10)); bn = 0.125 * np.exp(-(V + 65) / 80)
    vdot = 10 - 120 * (am / (am + bm)) ** 3 * (0.8 - N) * (V - 50) - 36 * N ** 4 * (V + 77) - 0.3 * (V + 54.4)
    ndot = an * (1 - N) - bn * N
    return V, N, vdot, ndot


def figures(out, o, thetas, kmax, show):
    """the paper's Figure 2 (all isochrons) and Figure 3 (the 0-phase isochron by arclength)"""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    phase_cmap = LinearSegmentedColormap.from_list(   # the paper's colours: cyan (phase 0) to magenta
        "cyan_magenta", ["#6fd6d6", "#52b6ea", "#5b86dc", "#6f5ac8", "#9a46b6", "#b8329f"])
    win = (-90, 52, 0.3, 0.8)
    V, N, vdot, ndot = nullclines(win)

    # Figure 2: all isochrons, both branches, coloured by phase
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for th in thetas:
        if not iso_file(out, th).exists():
            continue
        d = np.loadtxt(iso_file(out, th))
        for side in (-1, 1):
            for k in range(1, kmax + 1):
                s = d[(d[:, 4] == side) & (d[:, 3] == k)]
                if len(s):
                    ax.plot(s[:, 0], s[:, 1], color=phase_cmap(th), lw=0.9)
    ax.contour(V, N, vdot, [0], colors="#41b13a", linewidths=1.2)
    ax.contour(V, N, ndot, [0], colors="#e3241b", linewidths=1.2)
    ax.plot(o.orb["v"], o.orb["n"], color="0.55", lw=3)
    ax.annotate("γ0", o.gamma0, xytext=(-14, -16), textcoords="offset points")
    sm = plt.cm.ScalarMappable(cmap=phase_cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, label="phase θ")
    ax.set(xlim=win[:2], ylim=win[2:], xlabel="V", ylabel="n",
           title=f"Figure 2: {len(thetas)} isochrons by continuation (XPPAUT 8.2 AUTO)")
    fig.savefig(out / "figure2_all_isochrons.png", dpi=150)

    # Figure 3: the 0-phase isochron inside the orbit, coloured red to blue by arclength
    if 0.0 in [round(t, 6) for t in thetas] and iso_file(out, 0.0).exists():
        d = np.loadtxt(iso_file(out, 0.0))
        P = np.vstack([o.gamma0[None]] + [d[(d[:, 4] == -1) & (d[:, 3] == k)][:, :2] for k in range(1, kmax + 1)])
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]   # arclength (V and n as plotted)
        col = plt.cm.jet_r(s / s.max())
        fig3, (a, b, c) = plt.subplots(1, 3, figsize=(16, 4.6))
        for axx in (a, c):
            axx.contour(V, N, vdot, [0], colors="#41b13a", linewidths=1)
            axx.contour(V, N, ndot, [0], colors="#e3241b", linewidths=1)
            axx.plot(o.orb["v"], o.orb["n"], color="0.55", lw=3)
            axx.scatter(P[:, 0], P[:, 1], c=col, s=1.5)
        a.set(xlim=win[:2], ylim=win[2:], xlabel="V", ylabel="n", title="(a) I(γ0) inside the orbit")
        b.scatter(s, P[:, 1], c=col, s=1.5)
        b.set(xlabel="arclength", ylabel="n", title="(b) n along I(γ0): excursions")
        c.set(xlim=(-60.5, -58.5), ylim=(0.398, 0.408), xlabel="V", ylabel="n",
              title="(c) near the equilibrium: boomerang turns")
        fig3.tight_layout()
        fig3.savefig(out / "figure3_zero_phase.png", dpi=150)
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phases", type=float, nargs="*", default=None, help="phases theta in [0, 1)")
    ap.add_argument("--nphase", type=int, default=100, help="theta_i = i/N, i = 0..N-1 (the paper: 100)")
    ap.add_argument("--quick", action="store_true", help="only the 0-phase isochron")
    ap.add_argument("--kmax", type=int, default=4, help="stages inside Gamma (the paper: 4)")
    ap.add_argument("--kout", type=int, default=1, help="stages outside Gamma (the paper: 1)")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)),
                    help="isochrons computed at the same time (one XPPAUT per process)")
    ap.add_argument("--out", default="isochron_results")
    ap.add_argument("--no-show", action="store_true", help="save the figures without opening them")
    a = ap.parse_args()
    thetas = [0.0] if a.quick else (list(a.phases) if a.phases else [i / a.nphase for i in range(a.nphase)])
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    work = tempfile.mkdtemp()
    (out / "odes").mkdir(exist_ok=True)              # every .ode file used is kept here
    o = Orbit(plain_model(out / "odes"))                          # STEPS 1-3 (odes/reduced_hh.ode)
    print(f"T = {o.T:.6f} (paper 11.8463); gamma0 = ({o.gamma0[0]:.6f}, {o.gamma0[1]:.8f}) "
          f"(paper (44.7064, 0.4597))")
    np.savetxt(out / "orbit.dat", np.c_[o.orb.t, o.orb["v"], o.orb["n"]], fmt="%.10g", header="t V n")
    points = {th: o.phase_point(th) for th in tqdm(thetas, desc="step 3: gamma(theta), n_s")}
    gth, ns = points[thetas[0]]                                    # STEP 4: the BVP file, filled in
    ode = bvp_builder(o.T).render().replace(
        "gv=0,gn=0,tv=1,tn=0,d=0", f"gv={gth[0]:.10g},gn={gth[1]:.10g},tv={ns[0]:.10g},tn={ns[1]:.10g},d=0")
    (out / "odes" / "isochron_bvp_template.ode").write_text(ode)

    todo = [th for th in thetas if not iso_file(out, th).exists()]
    print(f"{len(thetas) - len(todo)} of {len(thetas)} isochrons already in {out}/; {len(todo)} to compute "
          f"with {min(a.workers, max(1, len(todo)))} worker(s)")
    poly_xy = np.c_[o.orb["v"], o.orb["n"]]
    t0 = time.time()
    if todo and (a.workers <= 1 or len(todo) == 1):               # STEPS 5-8, one phase at a time
        B = load_bvp(o.T)
        for th in tqdm(todo, desc="isochrons", unit="isochron"):
            with tqdm(total=a.kmax + a.kout, desc=f"  θ = {th:.3f}", unit="stage", leave=False) as bar:
                iso = Isochron(B, o.T, poly_xy, *points[th], bar=bar, theta=th, odedir=out / "odes")
                res = iso.run(kmax=a.kmax, kout=a.kout)
            save(out, th, res, iso.log)
    elif todo:                                                     # STEPS 5-8, several phases at once
        ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
        with ctx.Pool(min(a.workers, len(todo)), initializer=_init_worker,
                      initargs=(o.T, poly_xy, a.kmax, a.kout, str(out))) as pool, \
                tqdm(total=len(todo), desc="isochrons", unit="isochron") as bar:
            for th, sec, npts in pool.imap_unordered(_worker, [(th, *points[th]) for th in todo]):
                bar.set_postfix(last=f"θ={th:.3f}, {npts} points, {sec} s")
                bar.update(1)
    print(f"computed in {time.time() - t0:.0f} s")
    figures(out, o, thetas, a.kmax, show=not a.no_show)
    print(f"results and figures in {out}/")


if __name__ == "__main__":
    main()
