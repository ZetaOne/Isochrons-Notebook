"""Shared numerics for the isochron book.

Every trajectory is computed by XPPAUT through pyauto-xpp (``import pyxpp``):
time integration (CVODE, tolerances 1e-10), equilibria, nullclines, periodic
orbits (Initialconds > Periodic), adjoints (nUmerics > Averaging) and AUTO.
Only the boundary-value continuation of chapter 11 uses scipy's collocation
solver (``solve_bvp``), because that is the paper's own method and XPP's AUTO
does not expose general boundary-value continuation from Python.

Models (``MODELS``):
  sl      Stuart-Landau with twist:  r' = r(1 - r^2),  th' = w - b r^2
          (b = 0 is Winfree's "radial isochron clock")
  sl2     two cycles: r' = r(r^2 - a^2)(1 - r^2), th' = w - b r^2
          (unstable cycle r = a is the basin boundary of the stable cycle r = 1)
  vdp     van der Pol in Lienard form:  x' = (y - x^3/3 + x)/eps,  y' = -x
  fhn     FitzHugh-Nagumo:  v' = v - v^3/3 - w + I,  w' = eps (v + a - b w)
  ml      Morris-Lecar (SNIC parameter set, iapp = 0.1)
  rhh     reduced Hodgkin-Huxley, Osinga & Moehlis (2010) eq. (4.1)
"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
from pathlib import Path

import numpy as np

import pyxpp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "rendered-anims" / "isochron"
DATA = OUT / "data"
DATA.mkdir(parents=True, exist_ok=True)

NUMERICS = "@ meth=cvode,tol=1e-10,atol=1e-10,dt=0.01,maxstor=400000,bound=100000\n"

MODELS = {
    "sl": """# Stuart-Landau oscillator with twist b (b=0: radial isochron clock)
x'=x*(1-x^2-y^2)-y*(w-b*(x^2+y^2))
y'=y*(1-x^2-y^2)+x*(w-b*(x^2+y^2))
par w=1.0,b=0.0
init x=0.5,y=0
""",
    "sl2": """# two concentric cycles: unstable r=a, stable r=1
f(r)=(r*r-a*a)*(1-r*r)
x'=x*f(sqrt(x^2+y^2))-y*(w-b*(x^2+y^2))
y'=y*f(sqrt(x^2+y^2))+x*(w-b*(x^2+y^2))
par w=1.0,b=0.5,a=0.5
init x=0.8,y=0
""",
    "vdp": """# van der Pol oscillator, Lienard form
x'=(y-x^3/3+x)/eps
y'=-x
par eps=0.1
init x=2,y=0
""",
    "fhn": """# FitzHugh-Nagumo
v'=v-v^3/3-w+i
w'=eps*(v+a-b*w)
par i=0.5,a=0.7,b=0.8,eps=0.08
init v=-1,w=-0.5
""",
    "ml": """# Morris-Lecar (Rinzel & Ermentrout's dimensionless form, SNIC parameter set)
par iapp=0.1,phi=0.333,v1=-0.01,v2=0.15,v3=0.1,v4=0.145,gca=1,vk=-0.7,vl=-0.5,gk=2.0,gl=0.5
minf(v)=0.5*(1+tanh((v-v1)/v2))
ninf(v)=0.5*(1+tanh((v-v3)/v4))
lamn(v)=phi*cosh((v-v3)/(2*v4))
v'=iapp+gl*(vl-v)+gk*w*(vk-v)-gca*minf(v)*(v-1)
w'=lamn(v)*(ninf(v)-w)
init v=-0.3,w=0
""",
    "rhh": """# Reduced Hodgkin-Huxley model, Osinga & Moehlis (2010) eq. (4.1)
par iapp=10,gna=120,gk=36,gl=0.3,vna=50,vk=-77,vl=-54.4,c=1
an(v)=0.01*(v+55)/(1-exp(-(v+55)/10))
bn(v)=0.125*exp(-(v+65)/80)
am(v)=0.1*(v+40)/(1-exp(-(v+40)/10))
bm(v)=4*exp(-(v+65)/18)
minf(v)=am(v)/(am(v)+bm(v))
v'=(iapp-gna*minf(v)^3*(0.8-n)*(v-vna)-gk*n^4*(v-vk)-gl*(v-vl))/c
n'=an(v)*(1-n)-bn(v)*n
init v=-60,n=0.4
""",
}

# Python versions of the right-hand sides (for scipy collocation and Jacobians)
def rhs_np(name, pars):
    p = dict(pars)
    if name == "sl":
        w, b = p.get("w", 1.0), p.get("b", 0.0)
        def F(u):
            x, y = u[0], u[1]; r2 = x * x + y * y
            return np.array([x * (1 - r2) - y * (w - b * r2), y * (1 - r2) + x * (w - b * r2)])
        return F
    if name == "vdp":
        eps = p.get("eps", 0.1)
        return lambda u: np.array([(u[1] - u[0] ** 3 / 3 + u[0]) / eps, -u[0]])
    if name == "fhn":
        I, a, b, eps = p.get("i", 0.5), p.get("a", 0.7), p.get("b", 0.8), p.get("eps", 0.08)
        return lambda u: np.array([u[0] - u[0] ** 3 / 3 - u[1] + I, eps * (u[0] + a - b * u[1])])
    if name == "ml":
        q = dict(iapp=0.1, phi=0.333, v1=-0.01, v2=0.15, v3=0.1, v4=0.145, gca=1, vk=-0.7, vl=-0.5, gk=2.0, gl=0.5); q.update(p)
        def F(u):
            v, w = u[0], u[1]
            mi = 0.5 * (1 + np.tanh((v - q["v1"]) / q["v2"])); ni = 0.5 * (1 + np.tanh((v - q["v3"]) / q["v4"]))
            ln = q["phi"] * np.cosh((v - q["v3"]) / (2 * q["v4"]))
            return np.array([q["iapp"] + q["gl"] * (q["vl"] - v) + q["gk"] * w * (q["vk"] - v) - q["gca"] * mi * (v - 1),
                             ln * (ni - w)])
        return F
    if name == "rhh":
        q = dict(iapp=10, gna=120, gk=36, gl=0.3, vna=50, vk=-77, vl=-54.4, c=1); q.update(p)
        def F(u):
            v, n = u[0], u[1]
            an = 0.01 * (v + 55) / (1 - np.exp(-(v + 55) / 10))
            bn = 0.125 * np.exp(-(v + 65) / 80)
            am = 0.1 * (v + 40) / (1 - np.exp(-(v + 40) / 10))
            bm = 4 * np.exp(-(v + 65) / 18)
            mi = am / (am + bm)
            dv = (q["iapp"] - q["gna"] * mi ** 3 * (0.8 - n) * (v - q["vna"])
                  - q["gk"] * n ** 4 * (v - q["vk"]) - q["gl"] * (v - q["vl"])) / q["c"]
            return np.array([dv, an * (1 - n) - bn * n])
        return F
    raise KeyError(name)


def load(name, extra="", **pars):
    """Load a model into its own XPP instance; ``pars`` override parameters."""
    m = pyxpp.load(text=MODELS[name] + NUMERICS + extra + "done\n")
    if pars:
        m.set(**pars)
    return m


def xy_names(m):
    return [v.lower() for v in m.vars[:2]]


def traj(m, x0, total, dt=0.01, back=False, **pars):
    """One trajectory from x0 (sequence) for time ``total``; returns (t, X) with X shape (N, 2)."""
    a, b = xy_names(m)
    m.set(**{a: float(x0[0]), b: float(x0[1])}, total=abs(total), dt=dt, trans=0, nout=1, **pars)
    if back:
        m.buttons.initialconds.backward()
        tr = m.data
    else:
        tr = m.run()
    return np.asarray(tr.t, float), np.column_stack([tr[a], tr[b]]).astype(float)


def periodic_orbit(m, x0, var=None, settle=400.0, dt=0.002):
    """Settle onto the stable cycle from x0, then XPP's Initialconds > Periodic:
    exactly one period starting at the maximum of ``var`` (the paper's gamma_0).
    Returns (t, X, T)."""
    a, b = xy_names(m)
    var = var or a
    m.set(**{a: float(x0[0]), b: float(x0[1])}, total=settle, dt=dt, trans=0, nout=1)
    m.run()
    m.buttons.xi_vs_t(var)
    m.buttons.initialconds.periodic()
    T = float(m.numerics["total"])
    # re-integrate one full period from gamma_0 with a step that divides T
    g0 = m.data.array[0, 1:3].astype(float)
    nstep = int(np.ceil(T / dt))
    m.set(**{a: g0[0], b: g0[1]}, total=T, dt=T / nstep, trans=0, nout=1)
    tr = m.run()
    X = np.column_stack([tr[a], tr[b]]).astype(float)
    return np.asarray(tr.t, float), X, T


def floquet(m):
    """Nontrivial Floquet exponent kappa and multiplier exp(kappa T) of the orbit
    that Initialconds > Periodic left in the browser (nUmerics > Averaging)."""
    n = len(m.log)
    m.buttons.numerics.averaging.new_adj()
    adj = m.data
    txt = m.log[n:]
    mk = re.search(r"kappa\s*=\s*([-+0-9.eE]+)", txt)
    mn = re.search(r"nu = exp\(kappa T\)\s*=\s*([-+0-9.eE]+)", txt)
    kap = float(mk.group(1)) if mk else None          # None: multiplier numerically zero
    nu = float(mn.group(1)) if mn else 0.0
    return kap, nu, adj


class PhaseLookup:
    """Phase of a point on (or very near) the cycle, from one densely sampled period."""

    def __init__(self, X, scale=(1.0, 1.0)):
        self.X = np.asarray(X, float)
        self.s = np.asarray(scale, float)
        self.N = len(X) - 1                     # last point = first point (one full period)

    def __call__(self, p):
        Y = (self.X[:-1] - p) / self.s
        k = int(np.argmin((Y ** 2).sum(1)))
        best = (k, 0.0, np.inf)
        for j in (k - 1, k):                    # refine on the two neighbouring segments
            j %= self.N
            A, B = self.X[j] / self.s, self.X[j + 1] / self.s
            d = B - A
            u = np.clip(np.dot(p / self.s - A, d) / max(np.dot(d, d), 1e-300), 0, 1)
            dist = np.linalg.norm(A + u * d - p / self.s)
            if dist < best[2]:
                best = (j, u, dist)
        j, u, dist = best
        return ((j + u) / self.N) % 1.0, dist


# ---------------------------------------------------------------- phase maps
_W = {}


def _init_worker(name, pars, X, scale):
    _W["m"] = load(name, **pars)
    _W["m"].set(maxstor=5000)
    _W["look"] = PhaseLookup(X, scale)


def _phase_of(args):
    x0, T, nper, tol, maxper = args
    m, look = _W["m"], _W["look"]
    a, b = xy_names(m)
    m.set(**{a: float(x0[0]), b: float(x0[1])})
    done = 0
    while True:
        dt = T / 1000.0                          # whole periods end exactly on a step
        m.set(total=nper * T, trans=nper * T - 5 * dt, nout=1, dt=dt)
        m.run() if done == 0 else m.run_last()
        done += nper
        p = np.array(list(m.last_point().values())[:2], float)
        if not np.all(np.isfinite(p)):
            return np.nan, np.nan, done
        th, dist = look(p)
        if dist < tol or done >= maxper:
            return (th if dist < 10 * tol else np.nan), dist, done


def phase_map(name, pars, X, T, xs, ys, scale=(1, 1), nper=3, tol=1e-4, maxper=60, procs=6):
    """Asymptotic phase on the grid xs x ys: integrate whole periods with XPP
    until the point is within ``tol`` (in scaled units) of the cycle, then read
    its phase off the cycle.  Returns theta[j, i] in [0, 1) (NaN = no phase)."""
    pts = [(x, y) for y in ys for x in xs]
    jobs = [((x, y), T, nper, tol, maxper) for (x, y) in pts]
    ctx = mp.get_context("spawn")
    with ctx.Pool(procs, initializer=_init_worker, initargs=(name, pars, X, scale)) as pool:
        res = pool.map(_phase_of, jobs, chunksize=64)
    th = np.array([r[0] for r in res]).reshape(len(ys), len(xs))
    nper_used = np.array([r[2] for r in res]).reshape(len(ys), len(xs))
    return th, nper_used


def isochron_curves(xs, ys, th, levels):
    """Extract isochrons {theta = c} from a phase map as polylines, using the
    zero set of sin(2 pi (theta - c)) on the branch where cos > 0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = {}
    fig, ax = plt.subplots()
    for c in levels:
        s = np.sin(2 * np.pi * (th - c))
        co = np.cos(2 * np.pi * (th - c))
        cs = ax.contour(xs, ys, s, levels=[0.0])
        segs = []
        for path in cs.allsegs[0]:
            if len(path) < 2:
                continue
            # keep the pieces where cos > 0 (theta = c, not c + 1/2)
            ii = np.clip(np.searchsorted(xs, path[:, 0]), 1, len(xs) - 1)
            jj = np.clip(np.searchsorted(ys, path[:, 1]), 1, len(ys) - 1)
            good = co[jj, ii] > 0
            cur = []
            for p, g in zip(path, good):
                if g:
                    cur.append(p)
                elif len(cur) > 1:
                    segs.append(np.array(cur)); cur = []
                else:
                    cur = []
            if len(cur) > 1:
                segs.append(np.array(cur))
        out[c] = segs
        cs.remove()
    plt.close(fig)
    return out


# ------------------------------------------------- BVP continuation (ch. 11)
def jac_fd(F, u, h=1e-7):
    n = len(u); J = np.zeros((n, n)); f0 = F(u)
    for i in range(n):
        e = np.zeros(n); e[i] = h * max(1.0, abs(u[i]))
        J[:, i] = (F(u + e) - f0) / e[i]
    return J


def monodromy(F, gamma0, T, steps=20000):
    """D Phi_T(gamma0) by RK4 on the variational equation (used only to get the
    stable Floquet vector for the linear approximation of the isochron)."""
    def G(z):
        u, M = z[:2], z[2:].reshape(2, 2)
        return np.concatenate([F(u), (jac_fd(F, u) @ M).ravel()])
    z = np.concatenate([gamma0, np.eye(2).ravel()]); h = T / steps
    for _ in range(steps):
        k1 = G(z); k2 = G(z + h / 2 * k1); k3 = G(z + h / 2 * k2); k4 = G(z + h * k3)
        z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z[2:].reshape(2, 2)


def isochron_bvp(F, orbit_t, orbit_X, T, vs, eta=1e-3, h=None, arclen=1.0, kmax=6,
                 side=+1, scale=(1.0, 1.0), tol=1e-8, max_nodes=200000, verbose=False,
                 hmax=0.01, record=None):
    """The Osinga-Moehlis method: grow the isochron of gamma_0 = orbit_X[0] by
    continuing orbit segments u(s), 0 <= s <= 1, of  u' = k T F(u)  whose end
    point u(1) lies on the linear approximation L_eta = {gamma_0 + delta vs}.
    The start point u(0) traces the isochron.  Continuation is in arclength of
    u(0) (a pseudo-arclength condition on u(0)); when |delta| reaches eta the
    segment is extended by one more period (k -> k+1) by concatenating the last
    segment with one period of the orbit, as in section 3 of the paper.
    Returns the isochron as an array of points (scaled units are only used for
    the step length).  With record=n, also returns every n-th orbit segment as
    (k, delta, s-mesh, u) tuples: (points, segments)."""
    from scipy.integrate import solve_bvp
    g0 = orbit_X[0].astype(float)
    vs = np.real(np.asarray(vs)).astype(float); vs = vs / np.linalg.norm(vs) * side
    nrm = np.array([-vs[1], vs[0]])                  # normal to L_eta
    sc = np.asarray(scale, float)
    h = h or hmax / 10

    # Gamma as a function of s in [0, 1] (one period)
    s_orb = orbit_t / T

    def gamma_of(s):
        return np.vstack([np.interp(s, s_orb, orbit_X[:, i]) for i in range(2)])

    pts = [g0.copy()]
    segs = []
    k = 1
    # first segment: u(s) = Phi_{(s-1) k T}(gamma0 + small) ~ Gamma itself with delta = 0
    s_mesh = np.linspace(0, 1, 400)
    U = gamma_of(s_mesh)
    tangent = (vs / sc) / np.linalg.norm(vs / sc)     # unit tangent in scaled units; starts along vs
    u0_prev = g0.copy()
    total_len = 0.0
    while total_len < arclen and k <= kmax:
        def fun(s, u, k=k):
            return k * T * F(u)

        def bc(ua, ub, u0p=u0_prev.copy(), tg=tangent.copy()):
            return np.array([np.dot(nrm, ub - g0),                        # u(1) on the line L
                             np.dot(tg, (ua - u0p) / sc) - h])        # step h along the isochron
        sol = solve_bvp(fun, bc, s_mesh, U, tol=tol, max_nodes=max_nodes)
        if not sol.success:
            if verbose:
                print("bvp failed:", sol.message, "h =", h)
            h /= 2
            if h < 1e-7:
                break
            continue
        u0 = sol.sol(0.0); delta = np.dot(vs, sol.sol(1.0) - g0)
        if abs(delta) > eta:                          # past the end of L_eta: extend by one period
            k += 1
            s_new = np.concatenate([sol.x * (k - 1) / k, (k - 1) / k + s_mesh[1:] / k])
            U = np.hstack([sol.y, gamma_of(s_mesh[1:])])
            s_mesh = s_new
            if verbose:
                print(f"k -> {k}, isochron length {total_len:.3f}")
            continue
        step = np.linalg.norm((u0 - u0_prev) / sc)
        if step > 2.5 * h:                            # jumped to another part of the curve: retry smaller
            h /= 2
            if h < 1e-7:
                break
            continue
        total_len += step
        tangent = (u0 - u0_prev) / sc / max(step, 1e-300)
        pts.append(u0.copy()); u0_prev = u0
        if record and (len(pts) % record == 0):
            segs.append((k, float(delta), sol.x.copy(), sol.y.copy()))
        s_mesh, U = sol.x, sol.y
        # adapt the step: grow when the solver is happy
        h = min(h * 1.2, hmax)
        if len(s_mesh) > max_nodes * 0.8:
            break
    if record:
        return np.array(pts), segs
    return np.array(pts)


def phase_points(name, pars, X, T, pts, scale=(1, 1), nper=3, tol=1e-5, maxper=80, procs=6):
    """Asymptotic phase of an arbitrary list of points (same method as phase_map)."""
    jobs = [((float(p[0]), float(p[1])), T, nper, tol, maxper) for p in pts]
    ctx = mp.get_context("spawn")
    with ctx.Pool(procs, initializer=_init_worker, initargs=(name, pars, X, scale)) as pool:
        res = pool.map(_phase_of, jobs, chunksize=32)
    return np.array([r[0] for r in res]), np.array([r[2] for r in res])


def cycle_bundle_bvp(F, orbit_t, orbit_X, T, v0, kappa0, phase_index=0, tol=1e-4, nodes=2000):
    """Paper section 2.2 (logarithmic form): compute the cycle and its stable normal bundle together,
        u' = T F(u),   w' = T (DF(u) w - kappa w),   u(0) = u(1),  w(0) = w(1),
        f_phase(u(0)) = 0 (u(0) at the maximum of coordinate `phase_index`),  |w(0)| = 1,
    with unknown period T and Floquet exponent kappa (multiplier mu = exp(kappa T)).
    Returns (s, u, w, T, kappa)."""
    from scipy.integrate import solve_bvp
    s = np.linspace(0, 1, nodes)
    U = np.vstack([np.interp(s, orbit_t / T, orbit_X[:, i]) for i in range(2)])
    # initial guess for the bundle: transport v0 with the variational equation would be best;
    # a constant guess normalised to v0 is enough for the solver to converge here
    # initial guess for the bundle: carry v0 BACKWARDS along the (interpolated) orbit with
    # w' = DF w - kappa0 w.  Backwards in time the stable direction dominates, so this is stable.
    from scipy.integrate import solve_ivp
    def uorb(tt):
        return np.array([np.interp(tt % T, orbit_t, orbit_X[:, i]) for i in range(2)])
    def var(tt, w):
        return jac_fd(F, uorb(tt)) @ w - kappa0 * w
    w_end = np.asarray(v0, float) / np.linalg.norm(v0)
    iv = solve_ivp(var, (T, 0.0), w_end, method="Radau", rtol=1e-9, atol=1e-12, dense_output=True)
    Wg = iv.sol(s * T)
    Wg = Wg / np.linalg.norm(Wg[:, 0])
    Y0 = np.vstack([U, Wg])

    def fun(s, Y, p):
        T_, k_ = p
        U_, W_ = Y[:2], Y[2:]
        f0 = F(U_)
        DFw = np.zeros_like(W_)
        for i in range(2):                               # vectorised finite-difference Jacobian columns
            h = 1e-7 * np.maximum(1.0, np.abs(U_[i]))
            E = np.zeros_like(U_); E[i] = h
            DFw += (F(U_ + E) - f0) / h * W_[i]
        return np.vstack([T_ * f0, T_ * (DFw - k_ * W_)])

    def bc(ya, yb, p):
        return np.array([ya[0] - yb[0], ya[1] - yb[1], ya[2] - yb[2], ya[3] - yb[3],
                         F(ya[:2])[phase_index], ya[2] ** 2 + ya[3] ** 2 - 1.0])
    sol = solve_bvp(fun, bc, s, Y0, p=[T, kappa0], tol=tol, max_nodes=200000)
    return sol.x, sol.y[:2], sol.y[2:], sol.p[0], sol.p[1], sol
