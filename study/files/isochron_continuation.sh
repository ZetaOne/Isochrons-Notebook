#!/bin/sh
# One global isochron of the reduced Hodgkin-Huxley model by Osinga & Moehlis's
# continuation method (SIAM J. Appl. Dyn. Syst. 9, 2010), run in XPPAUT 8.2's own AUTO.
#
#   ./isochron_continuation.sh [THETA] [KMAX] [KOUT]      e.g.  ./isochron_continuation.sh 0 4 1
#
# XPPAUT's -silent mode cannot run AUTO, so AUTO is driven headless through pyauto-xpp
# (XPPAUT 8.2 built as a library):   pip install pyauto-xpp numpy matplotlib
# Set PYTHON to the interpreter that has it (default python3).
#
# Output: isochron_bvp_THETA.ode (the BVP, loadable in XPPAUT), isochron_THETA.dat
# (V n delta stage side), isochron_THETA.log, orbit.dat, and isochron_THETA.png if gnuplot is found.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}
THETA=${1:-0}; KMAX=${2:-4}; KOUT=${3:-1}
"$PYTHON" -c "import pyxpp" 2>/dev/null || {
  echo "pyauto-xpp is not importable by $PYTHON: pip install pyauto-xpp (or set PYTHON=...)"; exit 1; }
"$PYTHON" "$HERE/isochron_continuation.py" "$THETA" "$KMAX" "$KOUT"
TAG=$(printf "%.3f" "$THETA")
if command -v gnuplot >/dev/null 2>&1; then
  gnuplot <<GP
set term pngcairo size 900,700 noenhanced
set out "isochron_$TAG.png"
set xlabel "V"; set ylabel "n"; set xrange [-90:52]; set yrange [0.3:0.8]
set title "Isochron of phase $THETA: stages k = 1..$KMAX inside, 1..$KOUT outside (XPPAUT 8.2 AUTO)"
plot "orbit.dat" u 2:3 w l lw 3 lc rgb "grey" t "orbit", \
     for [k=1:$KMAX] "isochron_$TAG.dat" u (\$4==k && \$5<0 ? \$1 : 1/0):2 w l lw 1.5 t sprintf("inside, k = %d", k), \
     "isochron_$TAG.dat" u (\$5>0 ? \$1 : 1/0):2 w l lw 1.5 lc rgb "cyan" t "outside"
GP
  echo "isochron_$TAG.png written"
fi
