#!/bin/sh
# Isochrons of the reduced Hodgkin-Huxley model (Osinga & Moehlis 2010, model 4.1) by brute
# force, with XPPAUT 8.2 alone in silent mode (plus awk, and gnuplot for the picture).
#
#   XPP=/path/to/xppaut ./isochron_bruteforce.sh [NV] [NN]        e.g.  ./isochron_bruteforce.sh 101 81
#
# The asymptotic phase of a point x is read from spike times: integrate x until its trajectory
# has settled on the orbit Gamma, take the time t_s of a spike (upward crossing of V = -20) and
# compare it with the spike of Gamma after gamma0 (the maximum of V, phase 0):
#     theta(x) = ((t_c0 - t_s) / T) mod 1.
# Each XPP run integrates one row of the grid (NV copies of the model at the same n, an XPP
# network v[0..NV-1], n[0..NV-1]); points that have not settled after NPER periods are NaN.
# This is the brute-force method, not the continuation of the paper (XPPAUT's -silent mode cannot
# run AUTO); see isochron_continuation.sh for that. Contours of the phase map are the isochrons.
#
# Output: gamma.txt (T, gamma0, t_c0), phase.dat (V n theta, a grid for gnuplot), isochrons.png.
set -e
XPP=${XPP:-xppaut}
NV=${1:-101}; NN=${2:-81}                    # grid: NV values of V, NN values of n
VLO=-90; VHI=52; NLO=0.30; NHI=0.80          # the window of the paper's Figure 2
NPER=${NPER:-8}                              # periods to settle
WORK=${WORK:-bruteforce_work}; mkdir -p "$WORK"

MODEL='am(v)=0.1*(v+40)/(1-exp(-(v+40)/10))
bm(v)=4*exp(-(v+65)/18)
an(v)=0.01*(v+55)/(1-exp(-(v+55)/10))
bn(v)=0.125*exp(-(v+65)/80)
minf(v)=am(v)/(am(v)+bm(v))'
FV='10-120*minf(v)^3*(0.8-n)*(v-50)-36*n^4*(v+77)-0.3*(v+54.4)'
FN='an(v)*(1-n)-bn(v)*n'

# 1. Gamma, T, gamma0 (the maximum of V) and t_c0, from one run settled on the orbit
cat > "$WORK/hh.ode" <<ODE
$MODEL
v'=$FV
n'=$FN
init v=-60,n=0.3
@ total=300,trans=270,dt=0.001,meth=rungekutta,maxstor=100000,bound=100000
done
ODE
"$XPP" "$WORK/hh.ode" -silent -outfile "$WORK/hh.dat" > "$WORK/hh.log" 2>&1
awk 'NR>1 && pv<-20 && $2>=-20 { c[++nc] = pt + (-20-pv)/($2-pv)*($1-pt) }
     { t[NR]=$1; V[NR]=$2; N[NR]=$3; pt=$1; pv=$2; nr=NR }
     END { T = c[nc]-c[nc-1]
           # the maximum of V between the last two spikes: gamma0 (parabola through 3 samples)
           best=-1e9; for (i=2;i<nr;i++) if (t[i]>c[nc-1] && t[i]<c[nc] && V[i]>best) {best=V[i]; im=i}
           a=V[im-1]; b=V[im]; cc=V[im+1]; h=t[im]-t[im-1]; s=0.5*(a-cc)/(a-2*b+cc)
           tm=t[im]+s*h; Vm=b-0.25*(a-cc)*s; Nm=N[im]+s*0.5*(N[im+1]-N[im-1])
           printf "T %.8f\ngamma0 %.6f %.8f\ntc0 %.8f\n", T, Vm, Nm, c[nc]-tm }' "$WORK/hh.dat" > gamma.txt
T=$(awk '$1=="T"{print $2}' gamma.txt); TC0=$(awk '$1=="tc0"{print $2}' gamma.txt)
echo "T = $T, t_c0 = $TC0; gamma0 = $(awk '$1=="gamma0"{print $2, $3}' gamma.txt)"

# 2. the grid, one XPP run per row of constant n
: > phase.dat
i=0
while [ $i -lt "$NN" ]; do
  n0=$(awk -v a=$NLO -v b=$NHI -v i=$i -v N=$NN 'BEGIN{printf "%.8f", a+(b-a)*i/(N-1)}')
  dv=$(awk -v a=$VLO -v b=$VHI -v N=$NV 'BEGIN{printf "%.10f", (b-a)/(N-1)}')
  cat > "$WORK/row.ode" <<ODE
$MODEL
v[0..$((NV-1))]'=$(echo "$FV" | sed 's/(v)/(v[j])/g; s/(v+/(v[j]+/g; s/(v-/(v[j]-/g; s/\*n\^/*n[j]^/g; s/(0.8-n)/(0.8-n[j])/g')
n[0..$((NV-1))]'=an(v[j])*(1-n[j])-bn(v[j])*n[j]
v[0..$((NV-1))](0)=$VLO+[j]*$dv
n[0..$((NV-1))](0)=$n0
@ total=$(awk -v T=$T -v k=$NPER 'BEGIN{print k*T}'),trans=$(awk -v T=$T -v k=$NPER 'BEGIN{print (k-2.2)*T}')
@ dt=0.01,meth=rungekutta,maxstor=20000,bound=1000000
done
ODE
  "$XPP" "$WORK/row.ode" -silent -outfile "$WORK/row.dat" > "$WORK/row.log" 2>&1
  awk -v NV=$NV -v T=$T -v tc0=$TC0 -v n0=$n0 -v vlo=$VLO -v dv=$dv '
    { for (j=0;j<NV;j++) { v=$(2+j)
        if (NR>1 && pv[j]<-20 && v>=-20) { s1[j]=s2[j]; s2[j]=pt+(-20-pv[j])/(v-pv[j])*($1-pt) }
        pv[j]=v }
      pt=$1 }
    END { for (j=0;j<NV;j++) {
            th="NaN"
            if (s1[j]!="" && (s2[j]-s1[j]-T)^2 < (1e-3*T)^2) { x=(tc0-s2[j])/T; th=x-int(x); if (th<0) th+=1 }
            printf "%.6f %.8f %s\n", vlo+j*dv, n0, th }
          print "" }' "$WORK/row.dat" >> phase.dat
  i=$((i+1))
  printf "\r row %d of %d" $i "$NN"
done
echo; echo "phase.dat: $(grep -c NaN phase.dat) of $((NV*NN)) points did not settle within $NPER periods (near the unstable equilibrium)"

# 3. the picture: phase map, isochrons theta = 0, 0.1, ..., 0.9 as contours, and the orbit.
# The phase wraps from 1 to 0, and a contour level next to the wrap would be drawn along it; so
# theta = 0.3 ... 0.7 are contours of theta, and theta = 0.8, 0.9, 0, 0.1, 0.2 are contours of
# (theta + 0.5) mod 1, whose wrap lies elsewhere.
# If isochron_*.dat files of isochron_continuation.sh are in this folder, they are drawn on top.
if command -v gnuplot >/dev/null 2>&1; then
  awk 'NR>1' "$WORK/hh.dat" > "$WORK/orbit.dat"
  awk 'NF==3 && $3!="NaN" {s=$3+0.5; if (s>=1) s-=1; print $1, $2, s; next} NF==3 {print; next} {print}' phase.dat > "$WORK/shifted.dat"
  CONT=""
  for f in isochron_[0-9]*.dat; do [ -f "$f" ] && CONT="$CONT, \"$f\" u (\$5<0 ? \$1 : 1/0):2 w l lw 2.5 dt 2 lc rgb \"white\" t \"continuation (AUTO): $f\""; done
  gnuplot <<GP
set datafile missing "NaN"
set contour base; unset surface; set view map
set table "$WORK/cont_a.dat"
set cntrparam levels discrete 0.3,0.4,0.5,0.6,0.7
splot "phase.dat" u 1:2:3 w l
unset table
set table "$WORK/cont_b.dat"
set cntrparam levels discrete 0.3,0.4,0.5,0.6,0.7
splot "$WORK/shifted.dat" u 1:2:3 w l
unset table
reset
set term pngcairo size 1000,760 noenhanced
set out "isochrons.png"
set xrange [$VLO:$VHI]; set yrange [$NLO:$NHI]; set xlabel "V"; set ylabel "n"
set palette model HSV defined (0 0 0.55 1, 1 1 0.55 1); set cbrange [0:1]
set cblabel "asymptotic phase"
set key bottom right opaque
set title "XPPAUT 8.2 -silent, brute force ($NV x $NN points): isochrons theta = 0, 0.1, ..., 0.9"
plot "phase.dat" u 1:2:3 w image notitle, \
     "$WORK/cont_a.dat" u 1:2 w l lc rgb "black" lw 1.2 t "isochrons", \
     "$WORK/cont_b.dat" u 1:2 w l lc rgb "black" lw 1.2 notitle, \
     "$WORK/orbit.dat" u 2:3 w l lc rgb "grey30" lw 3.5 t "orbit" $CONT
GP
  echo "isochrons.png written"
fi
