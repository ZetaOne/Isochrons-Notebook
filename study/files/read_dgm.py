"""read an XPP AUTO diagram file (File -> Save diagram): one record per point"""
import sys, numpy as np
def read_dgm(path, nvar=2):
    L = open(path).read().split("\n")
    i = [k for k, l in enumerate(L) if len(l.split()) == 12][0]   # first point record
    out = []
    while i + 1 + nvar < len(L) and len(L[i].split()) == 12:
        h = [int(x) for x in L[i].split()]
        par = [float(x) for x in L[i + 1].split()]
        u0 = [float(L[i + 2 + j].split()[0]) for j in range(nvar)]
        out.append(dict(ibr=h[1], pt=h[2], itp=h[3], lab=h[4], d=par[0], u0=u0))
        i += 2 + nvar
    return out
if __name__ == "__main__":
    P = read_dgm(sys.argv[1])
    print(len(P), "points")
    for p in P[:12] + P[-3:]:
        print(p)
