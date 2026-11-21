# Isochrons of the reduced Hodgkin-Huxley model with XPPAUT 8.2: two shell scripts

The model is (4.1) of H. M. Osinga and J. Moehlis, *Continuation-based computation of
global isochrons*, SIAM J. Appl. Dyn. Syst. 9(4), 1201-1228 (2010), with I = 10.

XPPAUT's `-silent` mode integrates, finds equilibria and draws nullclines and direction
fields, but it cannot run AUTO or the boundary value solver. So there are two scripts:

| script | method | needs |
|---|---|---|
| `isochron_bruteforce.sh` | brute force: phase from spike times on a grid, isochrons as contours | XPPAUT 8.2 (`xppaut -silent`), awk; gnuplot for the picture |
| `isochron_continuation.sh` | the paper's Algorithm Isochron: a BVP continued in AUTO, stage by stage | XPPAUT 8.2's AUTO, run headless through pyauto-xpp (`pip install pyauto-xpp numpy matplotlib`) |

## isochron_bruteforce.sh (XPPAUT alone)

    XPP=/path/to/xppaut ./isochron_bruteforce.sh 101 81

1. One silent run settles on the orbit Γ. awk reads the period T from the spike times
   (upward crossings of V = −20), γ0 = the maximum of V (phase 0), and t_c0 = the time
   from γ0 to its next spike.
2. The grid, V from −90 to 52 and n from 0.3 to 0.8, is integrated one row at a time.
   Each row is a single silent run of an XPP network `v[0..NV-1], n[0..NV-1]` that holds
   NV copies of the model. The asymptotic phase of each point is
   θ = ((t_c0 − t_s)/T) mod 1, where t_s is a spike after 6 periods. A point counts as
   settled once its last two spikes are T apart.
3. gnuplot draws the phase map, the isochrons θ = 0, 0.1, …, 0.9 as contours, and the orbit.

On a 101 × 81 grid this takes about 3 minutes. Output:
- `gamma.txt`: T, γ0 and t_c0;
- `phase.dat`: V n θ;
- `isochrons.png`: the picture.

If `isochron_*.dat` files from the continuation script are in the same folder, they are
drawn on top (white dashes), as a check of one method against the other.

## isochron_continuation.sh (the paper's method, XPPAUT 8.2's AUTO)

    PYTHON=python3 ./isochron_continuation.sh THETA KMAX KOUT     # e.g. 0 4 1

**The BVP.** The script first writes the BVP as an ordinary XPP file,
`isochron_bvp_THETA.ode`. It contains:
- `v'=per*(...)` and `n'=per*(...)` on t ∈ [0, 1];
- `bndry v'-gv-d*tv` and `bndry n'-gn-d*tn`, which say u(1) = γ + δ n_s;
- `par per, gv, gn, tv, tn, d`, filled in with γ(θ) and n_s(θ).

The start point u(0) is free. Continuation in d = δ therefore gives a family of orbit
segments, and the start points u(0) of the segments trace the isochron.

**The stages.** For each stage k:
1. per = kT.
2. AUTO's Bdry Value start from Γ (k = 1), or from the last segment of stage k − 1
   followed by one period of Γ (the paper's eq. 3.5).
3. NTST = 100k.
4. Continuation in δ until |δ| = η = 1e-4.

**XPP's AUTO specifics.**
- The starting solution goes onto a uniform mesh. So each stage first takes 6 steps with δ
  unbounded, with IAD = 3 adapting the mesh, and then restarts from the last of them.
- The direction is chosen by probing 25 steps each way.

**Output.**
- `isochron_THETA.dat`: V n δ stage side, where side −1 is inside Γ and +1 outside.
- `isochron_THETA.log`: what each stage did.
- `orbit.dat`: the orbit Γ.
- `isochron_THETA.png`: the picture.

For θ = 0 the four stages inside Γ give 718, 808, 751 and 677 points, and take about
5 minutes in all. Stage 3 needs NTST = 1200 to start.

`example-output/` holds both scripts' results for θ = 0.
