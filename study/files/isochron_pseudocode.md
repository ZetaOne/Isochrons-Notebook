---
title: "Isochrons in pseudocode"
subtitle: "What an isochron is, how to get T and the phase, and the two ways to compute isochrons"
date: "18 November 2026"
---

The pseudocode works for any planar oscillator x' = F(x) with x = (x1, x2) that has one
stable limit cycle Γ. The settings used for FitzHugh–Nagumo (FHN) and the reduced
Hodgkin–Huxley model (HH) of Osinga & Moehlis are given as examples. Functions in CAPITALS
are defined in this document; `integrate(x, t)` means "solve x' = F(x) from x for time t"
with any accurate method (we use RK4 in XPPAUT, step 0.01 or smaller).

# 0. The ideas in one page

* **Phase on the cycle.** Pick one point γ0 of Γ and call its phase 0. The point reached
  from γ0 after time t has phase θ = t/T (mod 1), where T is the period. Phase is a clock.
* **Asymptotic phase of any point x.** The trajectory from x converges to Γ and ends up
  moving in step with exactly one point of Γ. That point's phase is θ(x):

  $$\lim_{t\to\infty} \|\Phi(t, x) - \Phi(t + \theta(x)\,T,\ \gamma_0)\| = 0$$

* **Isochron** of a point γ on Γ: all points x with θ(x) = θ(γ). A curve through γ, a contour
  line of the function θ(x).
* **Two ways to compute isochrons.**
    1. *Brute force:* compute θ(x) on a grid by simulation, draw contour lines. Simple, but it
       cannot resolve regions where θ changes faster than the grid (the strip inside the HH
       cycle).
    2. *Boundary value problem (Osinga & Moehlis):* compute the points of one isochron
       directly, as start points of orbit segments that land on a tiny piece of the isochron
       near γ after exactly k periods; follow them by continuation. Works where brute force
       fails.

# 1. The limit cycle, the period T and phase zero γ0

```
FUNCTION FIND_CYCLE(F, x_start, threshold):
    # 1. get onto the cycle: integrate long enough to forget the start
    x = integrate(x_start, t = 50 * (a guess of the period))

    # 2. period: time between two successive spikes (x1 crossing `threshold` upward)
    record x1(t) while integrating x for a few more periods
    s = list of up-crossing times of x1 through threshold (linear interpolation)
    T = s[-1] - s[-2]
    # (XPP: Initialconds > Periodic does this with a Newton refinement)

    # 3. phase zero: gamma0 = the point of the cycle where x1 is largest (top of the spike);
    #    there x1' = F1(x) changes sign from + to -, so bisect on time
    x = state at a time a little before the maximum of x1
    a, b = 0, small time interval that contains the maximum
    repeat 40 times:                                  # 40 halvings: double precision
        c = (a + b) / 2
        if F1(integrate(x, c)) > 0:  a = c            # still rising
        else:                        b = c            # past the top
    gamma0 = integrate(x, (a + b) / 2)

    # 4. the time from gamma0 to its next spike: the reference for brute-force phases
    t_c0 = first up-crossing time of x1 through threshold, integrating from gamma0

    RETURN T, gamma0, t_c0
```

|  | FHN | reduced HH |
|---|---|---|
| model | v' = v − v³/3 − w + I, w' = ε(v + a − bw) | the paper's eq. (4.1) |
| parameters | I = 0.5, a = 0.7, b = 0.8, ε = 0.08 | I = 10 (and the paper's conductances) |
| spike threshold | v = 0 | V = −20 mV |
| T | 39.4744 | 11.8463 |
| γ0 | (1.8521, 0.2343) | (44.7064, 0.4597) |

```
FUNCTION POINT_OF_PHASE(theta):                       # gamma(theta) on the cycle
    RETURN integrate(gamma0, theta * T)
```

# 2. The asymptotic phase of one point (brute force)

The trajectory from x and the cycle from γ0 eventually fire in step. If x fires its spike a
time Δ earlier than the cycle (modulo T), x is ahead by Δ/T of a cycle.

```
FUNCTION ASYMPTOTIC_PHASE(x, T, t_c0, threshold, n_periods = 8):
    integrate from x for n_periods * T, recording x1(t)
    s = up-crossing times of x1 through threshold
    if fewer than 2 spikes, or |s[-1] - s[-2] - T| > 1e-3 * T:
        RETURN undefined          # not settled yet: x was near the unstable equilibrium
    t_s = s[-1]                              # a spike after settling
    RETURN ((t_c0 - t_s) / T) mod 1
```

Notes:

* If many points are needed, integrate them together: XPP can run N uncoupled copies of the
  model in one run (a network `v[0..N-1]`, `w[0..N-1]`), so one run gives N phases.
* Points that never settle (the unstable equilibrium x\*) have no phase: the **phaseless set**.

# 3. Isochrons by brute force

```
FUNCTION BRUTE_FORCE_ISOCHRONS(window, nx, ny, levels = 0, 0.1, ..., 0.9):
    T, gamma0, t_c0 = FIND_CYCLE(...)
    for each grid point (i, j) of an nx-by-ny grid over window:
        theta[i, j] = ASYMPTOTIC_PHASE(grid point, T, t_c0, threshold)

    # the phase jumps from 0.99 to 0.01 somewhere: a contour across that jump is false.
    # Hide grid points next to any jump larger than 0.5, and draw levels near 0 and 1
    # from the phase shifted by half a cycle, whose own jump lies elsewhere.
    mask = grid points with a neighbour differing by more than 0.5
    S    = (theta + 0.5) mod 1
    for L in levels:
        if 0.25 <= L <= 0.75:  draw contour of theta at level L, skipping masked points
        else:                  draw contour of S at level (L + 0.5) mod 1, skipping masked points
    RETURN the contour lines   # each one is (part of) the isochron of phase L
```

Cost: a 101 × 81 grid with 8 periods each takes about 3 minutes for HH in XPPAUT.
Limitation: where θ changes by O(1) over less than one grid step (inside the HH cycle, along
the middle branch of the V-nullcline, the change happens within less than 10⁻¹⁴ mV), the
contours cannot be drawn. That is why the next method exists.

# 4. The isochron's direction at the cycle (needed by the BVP method)

Near γ the isochron is a straight line. Two equivalent ways to get its unit direction n_s:

```
FUNCTION ISOCHRON_DIRECTION_BY_ADJOINT(theta):
    # adjoint Z(t) along the cycle = gradient of the asymptotic phase (infinitesimal PRC)
    solve  Z' = -J(gamma(t))^T Z  backward in time along the cycle until Z is T-periodic,
    and normalise so that  Z(t) . F(gamma(t)) = 1  for all t
    # (XPP: Numerics > Averaging > New adj does exactly this)
    Z = Z(theta * T)
    n_s = (Z2, -Z1) / |Z|                    # perpendicular to the gradient
    if n_s points into the cycle: n_s = -n_s # choose it to point outward
    RETURN n_s

FUNCTION ISOCHRON_DIRECTION_BY_MONODROMY(theta):
    g = POINT_OF_PHASE(theta)
    M = derivative of the time-T map at g     # integrate M' = J(x(t)) M, M(0) = I, for time T
    # M has eigenvalues 1 (along the cycle) and mu (tiny: about 1e-17 for FHN, 7e-21 for HH);
    # the isochron direction is the one M squeezes: the right singular vector of the
    # smallest singular value
    RETURN that vector, normalised
```

The size of the squeeze is the Floquet multiplier, from Liouville's formula:

$$\mu = \exp\!\left(\int_0^T \nabla\cdot F(\gamma(t))\,dt\right)$$

# 5. Isochrons by a boundary value problem (Osinga & Moehlis)

**The idea.** Take an orbit segment that starts at an unknown point u(0) and ends, after
exactly k periods, on the short straight piece L_η = {γ + δ n_s : |δ| ≤ η} of the isochron.
Its start point has γ's phase, so it is a point of the isochron. Following the family of such
segments as δ changes traces the isochron. Time is rescaled to [0, 1]:

$$u'(s) = k\,T\,F(u(s)),\quad 0 \le s \le 1, \qquad u(1) = \gamma + \delta\, n_s, \qquad u(0)\ \text{free}$$

Two equations and two boundary conditions plus the free parameter δ give a one-parameter
family. As an XPP file (θ = 0 of HH; `v'` in a `bndry` line means v at s = 1):

```
v'=per*(iapp-gna*minf(v)^3*(0.8-n)*(v-ena)-gk*n^4*(v-ek)-gl*(v-el))
n'=per*(an(v)*(1-n)-bn(v)*n)
par per=11.846275,gv=44.706382,gn=0.459736,tv=0.999999991,tn=-0.000137108,d=0
bndry v'-gv-d*tv
bndry n'-gn-d*tn
```

```
FUNCTION ISOCHRON_BY_BVP(theta, K_inside = 4, K_outside = 1, eta = 1e-4):
    g   = POINT_OF_PHASE(theta)
    n_s = ISOCHRON_DIRECTION_BY_ADJOINT(theta)
    points = empty list

    for side in (inside, outside):
        u_start = CYCLE_SEGMENT(g)        # the cycle from g, one period, on [0, 1]
        for k = 1 .. (K_inside if side is inside else K_outside):
            set per = k * T, gamma = g, direction = n_s, delta = 0
            solution = START_STAGE(u_start, mesh = 100 * k)          # section 5.1
            direction = CHOOSE_DIRECTION(solution, k, side)          # section 5.2
            family = CONTINUE_IN_DELTA(solution, direction, |delta| <= eta)   # section 5.3
            append the start points u(0) of every member of family to points
            if k < K: u_start = CONCATENATE(last member of family, CYCLE_SEGMENT(g), k + 1)
    RETURN points                             # the isochron of phase theta

FUNCTION CYCLE_SEGMENT(g):
    RETURN u(s) = integrate(g, s * T), 0 <= s <= 1      # u(0) = u(1) = g, delta = 0

FUNCTION CONCATENATE(u_last, u_cycle, k):     # the paper's eq. (3.5)
    # u_last lasts k-1 periods and ends at gamma + eta n_s; append one more period of the cycle
    for s in [0, 1]:
        if s <= (k-1)/k:  u(s) = u_last( s * k / (k-1) )
        else:             u(s) = u_cycle( k*s - (k-1) )
    RETURN u   # almost a solution: a jump of size eta at s = (k-1)/k, closed by Newton
```

## 5.1 Starting a stage

```
FUNCTION START_STAGE(u_start, mesh):
    # AUTO, "Bdry Value" start: collocation with `mesh` intervals, 4 Gauss points each.
    # XPP's AUTO puts the start on a uniform mesh, which resolves the spike badly and pushes
    # delta away from 0 on the first Newton steps; so take a few unbounded steps first,
    # letting the mesh adapt, and start the real continuation from the last of them.
    for (mesh_try, newton_iterations) in [(mesh, 5), (mesh, 20), (2*mesh, 20), (4*mesh, 20)]:
        run 6 continuation steps from u_start in delta, step <= 1e-6, delta unbounded,
            mesh = mesh_try, adapt mesh every 3 steps
        if all 6 converged and |delta at the last step| < eta / 10:
            RETURN the last step
    FAIL "AUTO did not converge from the starting solution"
```

## 5.2 Choosing the direction

Along most of the family δ changes by about μ ≈ 10⁻²⁰, below the numerical noise, so the sign
of the first step does not tell which way the family goes. Try both:

```
FUNCTION CHOOSE_DIRECTION(solution, k, side):
    for sign in (+1, -1):
        probe = 25 continuation steps from solution with first step sign * 1e-5
        if k == 1:
            good = the probe's last points lie on the requested side of the cycle
        else:
            # one way extends the isochron, the other goes back over stage k-1.
            # delta' = (u((k-1)/k) - gamma) . n_s is where the segment is one period before
            # its end: it grows on the way that extends the isochron
            good = |delta'| grows along the probe
        if good: RETURN sign
    retry with 150-step probes; keep the better one
```

## 5.3 Following the family

```
FUNCTION CONTINUE_IN_DELTA(solution, sign, bound eta):
    # pseudo-arclength continuation (AUTO): each step = a predictor along the family's tangent
    # + Newton corrections of the whole collocated segment; the step is measured in the change
    # of the whole segment, not of delta, so the family moves on even when delta barely changes
    settings: first step 1e-5, smallest 1e-8, largest 0.1, tolerances 1e-8 (Newton),
              1e-6 (locating the stop), at most 5000 steps
    continue from `solution` with first step sign * 1e-5 until
        |delta| reaches eta  (inside the cycle), or
        u(0) leaves the window of interest  (outside: there |delta| may never reach eta)
    RETURN all members of the family (each with its u(0) and delta)
```

## 5.4 All phases, and checks

```
FOR theta in 0, 0.01, ..., 0.99:              # independent: run them in parallel
    isochron[theta] = ISOCHRON_BY_BVP(theta)

CHECKS:
    T, gamma0 against the paper (HH: 11.8463, (44.7064, 0.4597))
    every stage ends at |delta| = eta; each stage starts where the previous ended
    the isochron leaves gamma along n_s
    at isochron points where the phase is well conditioned (moving the point by the accuracy
        of the computed points changes ASYMPTOTIC_PHASE by less than 1e-3),
        ASYMPTOTIC_PHASE(point) = theta to within about 1e-4
    isochrons of different phases do not cross (apparent crossings only where points are
        in the sensitive set)
```

Costs in XPPAUT: one FHN isochron (3 stages inside) takes about 8 s; one HH isochron
(4 stages inside) 2–6 minutes; all 100 HH isochrons about 4.5 core-hours.

# 6. Where each piece is in the code

| pseudocode | `fhn_isochrons.py` / `hh_isochrons.py` | `reproduce_isochrons.py` |
|---|---|---|
| FIND_CYCLE | parts 1–2 | STEP 2 (class Orbit) |
| ASYMPTOTIC_PHASE | part 3 (`asymptotic_phase`) | (brute-force check in the notebooks) |
| BRUTE_FORCE_ISOCHRONS | parts 4–5 | — |
| ISOCHRON_DIRECTION_ BY_ADJOINT | — | STEP 3 (`phase_point`) |
| the BVP | — | STEP 4 (`bvp_builder`) |
| START_STAGE | — | STEP 5 (`start`) |
| CHOOSE_DIRECTION | — | STEP 6 (`score`, `delta_back`) |
| CONTINUE_IN_DELTA | — | STEP 7 (`stage`) |
| CONCATENATE | — | STEP 8 (`concatenate`) |
| all phases | — | STEP 9 (`main`, `--workers`) |
