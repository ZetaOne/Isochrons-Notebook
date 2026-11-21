#!/usr/bin/env python3
"""
One global isochron of the reduced Hodgkin-Huxley model (4.1) of Osinga & Moehlis,
"Continuation-based computation of global isochrons", SIAM J. Appl. Dyn. Syst. 9 (2010),
by their Algorithm Isochron, run in XPPAUT 8.2's own AUTO (Bdry Value starts and
continuation in delta) through pyauto-xpp. Called by isochron_continuation.sh.

    python3 isochron_continuation.py THETA [KMAX] [KOUT]

THETA  phase of the isochron (0 = the maximum of V on the orbit)
KMAX   stages inside the orbit (time-KMAX*T map), default 4
KOUT   stages outside the orbit, default 1

Writes, in the current folder:
  isochron_bvp_THETA.ode   the BVP as an XPP .ode file, with gamma(THETA) and n_s filled in
  isochron_THETA.dat       the isochron points: V n delta stage side (side -1 inside, +1 outside)
  isochron_THETA.log       what each stage did (NTST, starts, number of points, delta at the end)
  orbit.dat                the orbit Gamma from gamma0: t V n
The code is that of notebook 02_ISOCHRONS_continuation (sections 1 and 2), unchanged.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon
from pyxpp.builder import ModelBuilder

FUN = [("am(v)", "0.1*(v+40)/(1-exp(-(v+40)/10))"), ("bm(v)", "4*exp(-(v+65)/18)"),
       ("an(v)", "0.01*(v+55)/(1-exp(-(v+55)/10))"), ("bn(v)", "0.125*exp(-(v+65)/80)"),
       ("minf(v)", "am(v)/(am(v)+bm(v))")]
FV = "iapp-gna*minf(v)^3*(0.8-n)*(v-ena)-gk*n^4*(v-ek)-gl*(v-el)"
FN = "an(v)*(1-n)-bn(v)*n"
PARS = dict(iapp=10, gna=120, gk=36, gl=0.3, ena=50, ek=-77, el=-54.4)
OPTS = dict(meth="rk4", bound=1e4, jac_eps=1e-7, newt_tol=1e-10)

mb = ModelBuilder("reduced_hh")
for sig, expr in FUN:
    mb.function(sig, expr)
mb = (mb.ode("v", FV).ode("n", FN).parameter(**PARS).initial(v=-60, n=0.3)
      .option(total=300, dt=0.01, maxstor=400000, **OPTS))
m = mb.load(Path(tempfile.mkdtemp()) / "reduced_hh.ode")
m.run()
m.buttons.xi_vs_t("v")
m.buttons.initialconds.periodic()          # phase 0 = maximum of V; total = T
T = m.numerics["total"]


def flow(x, t, dt=1e-4):
    """Φ(t, x) by XPP (RK4, a step of at most dt that divides t), in double precision"""
    m.set(v=x[0], n=x[1])
    m.run(keep=False, total=t, dt=t / max(1, int(np.ceil(t / dt))), trans=0)
    s = m.last_state()
    return np.array([s["V"], s["N"]])


g = flow(np.array([m.ics["V"], m.ics["N"]]), T - 0.05)
a, b = 0.0, 0.1
for _ in range(40):                        # bisection for V' = 0 (the maximum of V)
    c = 0.5 * (a + b)
    a, b = (c, b) if m.rhs(flow(g, c))["V"] > 0 else (a, c)
gamma0 = flow(g, 0.5 * (a + b))
m.set(v=gamma0[0], n=gamma0[1])
orb = m.run(keep=False, total=T, dt=0.001, trans=0)
A = m.buttons.numerics.averaging
A.parameters(6400)
A.new_adj()
adj = m.data
eq = m.equilibrium({"v": -60, "n": 0.4})
xstar = np.array([eq.x["V"], eq.x["N"]])
POLY = Polygon(np.c_[orb["v"], orb["n"]])
print(f"T = {T:.6f}, γ0 = ({gamma0[0]:.4f}, {gamma0[1]:.4f}), x* = ({xstar[0]:.4f}, {xstar[1]:.4f})")


def phase_point(theta):
    """γ(θ) = Φ(θT, γ0) and the unit tangent n_s(θ) of its isochron (perpendicular to the
    adjoint Z(θ)), oriented to point out of Γ"""
    gth = gamma0.copy() if theta == 0 else flow(gamma0, theta * T)
    zv, zn = np.interp(theta * T, adj.t, adj["v"]), np.interp(theta * T, adj.t, adj["n"])
    ns = np.array([zn, -zv]) / np.hypot(zn, zv)
    f = m.rhs(gth)
    if f["V"] * ns[1] - f["N"] * ns[0] > 0:  # Γ turns counter-clockwise: outside is to the right of F
        ns = -ns
    return gth, ns




def bvp_builder():
    bb = ModelBuilder("isochron_bvp")
    bb.comment("orbit segments u' = per*F(u) on [0,1] that end on L_eta(gamma): u(1) = gamma + d*n_s")
    for sig, expr in FUN:
        bb.function(sig, expr)
    return (bb.ode("v", f"per*({FV})").ode("n", f"per*({FN})")
            .parameter(**PARS).parameter(per=T, gv=0, gn=0, tv=1, tn=0, d=0)
            .boundary("v'-gv-d*tv").boundary("n'-gn-d*tn")
            .option(total=1, dt=1e-4, maxstor=400000, **OPTS))



ETA, DS = 1e-4, 1e-5                       # the paper's η and DS = 0.1 η
NTST_GAMMA = 100
AUTO_NUM = dict(ncol=4, dsmin=1e-8, dsmax=0.1, epsl=1e-8, epsu=1e-8, epss=1e-6,
                itmx=8, itnw=5, nwtn=3, iad=3, normmax=1e6, npr=100000)
NMX = 5000
SC = np.array([100.0, 1.0])                # V and n scaled to similar ranges (for distances)


def load_bvp():
    B = bvp_builder().load(Path(tempfile.mkdtemp()) / "isochron_bvp.ode")
    B.auto.parameters("d", "per")
    return B


def gamma_solution(B, gth):
    """u_Γ: Γ from γ over one period, as a solution on [0, 1] (XPP integration, RK4)"""
    B.set(per=T, v=gth[0], n=gth[1], d=0)
    tr = B.run(total=1, dt=1e-4)
    return np.asarray(tr.t, float), np.c_[tr["v"], tr["n"]].astype(float)


class Isochron:
    """Algorithm Isochron of the paper for one phase point γ = γ(θ)."""

    def __init__(self, B, gth, ns, keep_V=None):
        self.B, self.g, self.ns = B, gth, ns
        self.uG = gamma_solution(B, gth)
        self.keep_V = keep_V or {}         # {k: [V values]}: store the orbit segments starting there
        self.segments = []
        self.log = []

    @staticmethod
    def _pts(d):
        return np.array([[p.u0["v"], p.u0["n"], p.par["d"]] for p in d]).reshape(-1, 3)

    def _start(self, k, initial):
        """6 steps from the starting solution, δ unbounded: AUTO corrects it and adapts the
        mesh (XPP starts on a uniform mesh).  Returns the last point (δ back near 0)."""
        B = self.B
        tries = [(NTST_GAMMA * k, 5), (NTST_GAMMA * k, 20), (2 * NTST_GAMMA * k, 20),
                 (4 * NTST_GAMMA * k, 20)]
        for ntst, itnw in tries:
            B.set(per=k * T, gv=self.g[0], gn=self.g[1], tv=self.ns[0], tn=self.ns[1], d=0)
            B.auto.reset()
            first = B.auto.run("d", start="bdry value", initial=initial, ntst=ntst, ds=DS, nmax=6,
                               parmin=-1, parmax=1, **dict(AUTO_NUM, itnw=itnw, dsmax=1e-6))
            if len(first) == 6 and all(p.type != "MX" for p in first) \
                    and abs(first[-1].par["d"]) < ETA / 10:
                return first[-1], ntst, itnw
            self.log.append(f"k={k}: start with NTST={ntst}, ITNW={itnw} failed "
                            f"({[p.type for p in first]})")
        raise RuntimeError(f"k={k}: AUTO did not converge from the starting solution")

    def _delta_back(self, sol, k):
        """δ' = (u((k-1)/k) - γ)·n_s: where the segment is one period before its end.  On the
        right way it lies on stage 1 beyond L_η (|δ'| grows past η), on the wrong way on
        L_η (|δ'| shrinks).  Cubic interpolation of the stored solution."""
        t, U = sol
        tt = (k - 1) / k
        j = int(np.clip(np.searchsorted(t, tt) - 2, 0, len(t) - 4))
        ts, Us = t[j:j + 4], U[j:j + 4]
        w = np.array([np.prod([(tt - ts[m]) / (ts[l] - ts[m]) for m in range(4) if m != l])
                      for l in range(4)])
        return float((w @ Us - self.g) @ self.ns)

    def _score(self, u, k, direction, probe, d0):
        """> 0 if the probe goes the requested way.  k = 1: inside (direction -1) or outside
        Γ, judged from where its last points are.  k > 1: the right way extends the
        isochron, the wrong way goes back along the previous stage; one period before the
        end, the right way's segments lie on stage 1 beyond L_η (|δ'| grows), the wrong
        way's on L_η (|δ'| shrinks).  Returns the relative change of |δ'|."""
        if k == 1:
            last = u[-max(3, len(u) // 4):, :2]
            far = last[np.hypot(*((last - self.g) / SC).T) > 1e-2]
            if len(far) == 0:
                return 0.0
            return (POLY.contains_points(far).mean() - 0.5) * (-direction)
        d1 = self._delta_back(probe[-1].solution, k)
        return (abs(d1) - abs(d0)) / abs(d0)

    def family(self, k, direction, initial, prev=None, nprobe=25):
        """one stage: the start points u(0) of the orbit segments of the time-kT map"""
        B = self.B
        t0 = time.time()
        p0, ntst, itnw = self._start(k, initial)
        npr = 1 if k in self.keep_V and direction < 0 else AUTO_NUM["npr"]
        num = dict(AUTO_NUM, npr=npr)
        d0 = self._delta_back(p0.solution, k) if k > 1 else None
        scores = {}
        for n_try in (nprobe, 6 * nprobe):         # a longer probe if the short one cannot tell
            for sign in (1, -1):
                probe = B.auto.run(grab=p0.label, ds=sign * DS, nmax=n_try, parmin=-ETA, parmax=ETA,
                                   **num)
                u = self._pts(probe)
                scores[sign] = self._score(u, k, direction, probe, d0)
                if scores[sign] > 0.01:
                    break
            else:
                sign = max(scores, key=scores.get)
                if scores[sign] <= 0:
                    self.log.append(f"k={k}: probes of {n_try} steps undecided {scores}")
                    continue
                probe = B.auto.run(grab=p0.label, ds=sign * DS, nmax=n_try, parmin=-ETA, parmax=ETA,
                                   **num)
                u = self._pts(probe)
            break
        else:
            raise RuntimeError(f"k={k}: neither direction goes the requested way {scores}")
        parts, points = [u], list(probe)
        last = probe[-1]
        # outside Γ, δ stays below the discretisation error all the way to infinity (the
        # time-T map contracts too strongly there): follow the branch, 100 points at a time,
        # until it is well outside the window of Figure 2
        chunk = NMX if direction < 0 else 100
        while abs(last.par["d"]) < ETA and last.type != "MX" and sum(map(len, parts)) < NMX:
            if direction > 0 and not (-150 < last.u0["v"] < 150 and 0.15 < last.u0["n"] < 0.95):
                break
            more = B.auto.run(grab=last.label, ds=sign * DS,
                              nmax=min(chunk, NMX - sum(map(len, parts))),
                              parmin=-ETA, parmax=ETA, **num)
            if len(more) <= 1:
                break
            parts.append(self._pts(more)[1:])
            points += list(more)[1:]
            last = more[-1]
        u = np.vstack(parts)
        if npr == 1:
            self._keep_segments(k, points, u)
        info = dict(k=k, direction=direction, ntst=ntst, itnw_start=itnw, points=len(u),
                    d_end=float(last.par["d"]), end_type=last.type, ds_sign=sign,
                    seconds=round(time.time() - t0, 2))
        self.log.append(json.dumps(info))
        return u, last.solution, info

    def _keep_segments(self, k, points, u):
        """orbit segments that start at V = given values: linear interpolation between the two
        stored solutions on either side of each crossing"""
        for Vt in self.keep_V[k]:
            for i in np.nonzero(np.diff(np.sign(u[:, 0] - Vt)) != 0)[0]:
                pa, pb = points[i], points[i + 1]
                f = (Vt - u[i, 0]) / (u[i + 1, 0] - u[i, 0])
                (ta, Ua), (tb, Ub) = pa.solution, pb.solution
                U = (1 - f) * Ua + f * np.column_stack([np.interp(ta, tb, Ub[:, j]) for j in range(2)])
                self.segments.append(dict(k=k, V=Vt, index=int(i), f=float(f), t=ta * k * T, U=U,
                                          d=float((1 - f) * u[i, 2] + f * u[i + 1, 2])))

    def concatenate(self, k, sol):
        """(3.5): u_last (k-1 periods) followed by u_Γ, rescaled to [0, 1]"""
        (ts, Us), (tG, UG) = sol, self.uG
        tt = np.linspace(0, 1, 4000 * k + 1)
        a = tt <= (k - 1) / k
        U = np.empty((len(tt), 2))
        for j in range(2):
            U[a, j] = np.interp(tt[a] * k / (k - 1), ts, Us[:, j])
            U[~a, j] = np.interp(tt[~a] * k - (k - 1), tG, UG[:, j])
        return tt, U

    def run(self, kmax=4, kout=1):
        """inside Γ up to the time-(kmax T) map, outside up to the time-(kout T) map"""
        out = {}
        for direction, kk in ((-1, kmax), (+1, kout)):
            init, prev = self.uG, None
            for k in range(1, kk + 1):
                u, sol, info = self.family(k, direction, init, prev)
                out[("in" if direction < 0 else "out", k)] = (u, info)
                if k < kk:
                    init, prev = self.concatenate(k + 1, sol), u
        return out




def main(argv):
    theta = float(argv[1]) if len(argv) > 1 else 0.0
    kmax = int(argv[2]) if len(argv) > 2 else 4
    kout = int(argv[3]) if len(argv) > 3 else 1
    tag = f"{theta:.3f}"
    print(f"T = {T:.8f}, gamma0 = ({gamma0[0]:.6f}, {gamma0[1]:.8f})")
    gth, ns = phase_point(theta)
    print(f"theta = {theta}: gamma = ({gth[0]:.6f}, {gth[1]:.8f}), n_s = ({ns[0]:.9f}, {ns[1]:.9f})")
    ode = bvp_builder().render()
    ode = ode.replace("gv=0,gn=0,tv=1,tn=0,d=0",
                      f"gv={gth[0]:.10g},gn={gth[1]:.10g},tv={ns[0]:.10g},tn={ns[1]:.10g},d=0")
    ode = ode.replace("@ total=1", f"init v={gth[0]:.10g},n={gth[1]:.10g}\n@ total=1")
    Path(f"isochron_bvp_{tag}.ode").write_text(ode)
    np.savetxt("orbit.dat", np.c_[orb.t, orb["v"], orb["n"]], fmt="%.10g", header="t V n")
    B = load_bvp()
    iso = Isochron(B, gth, ns)
    t0 = time.time()
    out = iso.run(kmax=kmax, kout=kout)
    rows = []
    for (side, k), (u, info) in out.items():
        s = -1 if side == "in" else 1
        rows.append(np.c_[u, np.full(len(u), k), np.full(len(u), s)])
        print(f"  {side} k={k}: {info['points']} points, NTST {info['ntst']}, "
              f"delta at the end {info['d_end']:.3e}, {info['seconds']:.0f} s")
    np.savetxt(f"isochron_{tag}.dat", np.vstack(rows), fmt="%.12g", header="V n delta stage side")
    Path(f"isochron_{tag}.log").write_text("\n".join(iso.log) + "\n")
    print(f"isochron_{tag}.dat written ({sum(len(r) for r in rows)} points, {time.time() - t0:.0f} s)")
    B.close()


if __name__ == "__main__":
    main(sys.argv)
