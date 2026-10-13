"""Regression test for the exact capped-prior anchors (round 7).
(1) Brute-force check of gacert2.bvs.capped_anchors against enumeration of all 2^p
    models for small p with a model-size cap m < p.
(2) For every p = 200 instance of the paper, the size of the omitted binomial tails
    (the difference between the capped and the uncapped anchors used by the reported
    runs): expected max 2.35e-53 (total mass) and 7.64e-32 (C mass).
Usage: python test_capped_anchor.py"""
import itertools, json, math, os, sys, glob
import numpy as np
from scipy.stats import binom
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(V2, 'code'))
from gacert2.bvs import capped_anchors

# (1) brute force
for (p, h, m, r) in ((12, 0.2, 4, 3), (10, 0.3, 6, 2), (8, 0.5, 8, 3)):
    X = C = 0.0
    for g in itertools.product((0, 1), repeat=p):
        k = sum(g)
        if k > m:
            continue
        mass = h ** k * (1 - h) ** (p - k)
        X += mass
        if all(g[:r]):
            C += mass
    lX, lC, lD = capped_anchors(p, h, m, r)
    err = max(abs(lX - math.log(X)), abs(lC - math.log(C)), abs(lD - math.log(X - C)))
    assert err < 1e-12, (p, h, m, r, err)
    print(f'brute force p={p} h={h} m={m} r={r}: max |log error| = {err:.1e}  OK')
# (2) tail sizes for the reported p = 200 instances
mx0 = mxC = 0.0
for f in sorted(glob.glob(os.path.join(V2, 'results', 'bvs', '*200*.json'))):
    d = json.load(open(f))
    p, h = d['p'], d['h']
    r = d['candidates'][d['witness']['source']]['size']
    lX, lC, lD = capped_anchors(p, h, 60, r)
    uX, uC, uD = 0.0, r * math.log(h), math.log1p(-h ** r)
    t0, tC = binom.sf(60, p, h), binom.sf(60 - r, p - r, h)
    mx0, mxC = max(mx0, t0), max(mxC, tC)
    print(f"{d['name']}: r={r} tail(X)={t0:.2e} tail(C)={tC:.2e} |capped-uncapped| log anchors: "
          f"{abs(lX-uX):.1e} {abs(lC-uC):.1e} {abs(lD-uD):.1e}")
print(f'max tails: {mx0:.2e} (total mass), {mxC:.2e} (C mass)')
assert mx0 < 3e-53 and mxC < 8e-32
print('OK')
