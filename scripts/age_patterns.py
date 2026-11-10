"""Area-level age patterns for the 'Age and the vote' page (observed data only, no individual-level estimates).

Inputs: ECT 2026 party-list results (results.py) and DOPA registered population by age, January 2026,
aggregated to 2026 constituencies (data/age/age_by_constituency_2569.txt).
Output: out/age.js with
  ids, bands (share of eligible-age residents in 18-29, 30-44, 45-59, 60+, per mille),
  y: party-list shares among valid party votes per constituency (per mille) for 6 outcomes, plus turnout,
  cmp: for each 'group' (under 30, under 45, 60 and over) and outcome, the within-province comparison:
       average difference between the constituencies above and below their province's median group share,
       weighted by eligible voters, with a 90% interval from resampling provinces.
  quint: national quintiles of group share and the outcome average in each (not province-adjusted).
"""
import json, sys, os
ROOT = os.environ.get('TEA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'ei'))
from results import load, LABELS

ids, Y, N, _ = load()
raw = {}
for l in open(os.path.join(ROOT, 'data', 'age', 'age_by_constituency_2569.txt')):
    if l.startswith('#'): continue
    q = l.split(); raw[q[0]] = np.array(list(map(float, q[1:5])))
B = np.array([raw[k] / raw[k].sum() for k in ids])
V = Y[:, :6] / Y[:, :6].sum(1, keepdims=True)
turn = 1 - Y[:, 7] / Y.sum(1)
OUT = np.column_stack([V, turn])                       # 7 outcomes
OLAB = LABELS[:6] + ['Turnout']
GROUPS = [('u30', 'under 30', B[:, 0]), ('u45', 'under 45', B[:, 0] + B[:, 1]), ('o60', '60 and over', B[:, 3])]
prov = np.array([k.split('_')[0] for k in ids])
import re as _re
_D = json.loads(_re.search(r'const DATA=(.*?);\nconst GEO', open(os.path.join(ROOT, 'data.js')).read(), _re.S).group(1))
REGS = ['Bangkok', 'Central', 'East', 'West', 'North', 'Northeast', 'South']
regn = np.array([_D['prov'][p]['reg'] for p in prov])
assert set(regn) <= set(REGS), set(regn)
P = sorted(set(prov))
rng = np.random.default_rng(2026)


def within(x, y, w, provs):
    """Eligible-weighted mean of (outcome in the 'more' half) minus (outcome in the 'less' half), within provinces."""
    num = den = 0.0
    for p in provs:
        m = prov == p
        if m.sum() < 2: continue
        xm, ym, wm = x[m], y[m], w[m]
        med = np.median(xm); hi = xm > med; lo = xm < med
        if hi.sum() == 0 or lo.sum() == 0:
            hi = xm >= med; lo = xm < med
            if hi.sum() == 0 or lo.sum() == 0: continue
        d = np.average(ym[hi], weights=wm[hi]) - np.average(ym[lo], weights=wm[lo])
        wt = wm.sum(); num += d * wt; den += wt
    return num / den


usable = [p for p in P if (prov == p).sum() >= 2]
out = dict(ids=ids, bandLabels=['18–29', '30–44', '45–59', '60+'], bands=np.rint(B * 1000).astype(int).T.tolist(),
           outcomes=OLAB, y=np.rint(OUT * 1000).astype(int).T.tolist(), groups=[g[1] for g in GROUPS], gkeys=[g[0] for g in GROUPS],
           cmp={}, quint={}, spread={}, regions=REGS, meta=dict(
               provinces_used=len(usable), constituencies_used=int(sum((prov == p).sum() for p in usable)),
               source_age='DOPA registered Thai population by age, January 2026 (file 3_6901), aggregated to 2026 constituencies',
               source_results='Election Commission of Thailand 2026 result files, party-list ballot'))
for key, lab, x in GROUPS:
    res = []
    for j in range(OUT.shape[1]):
        d = within(x, OUT[:, j], N, usable)
        boots = []
        for _ in range(1000):
            bs = list(rng.choice(usable, len(usable), replace=True))
            # duplicate provinces count twice: emulate by summing over the resampled list
            num = den = 0.0
            for p in bs:
                m = prov == p; xm, ym, wm = x[m], OUT[m, j], N[m]; med = np.median(xm); hi = xm > med; lo = xm < med
                if hi.sum() == 0 or lo.sum() == 0: hi = xm >= med
                if hi.sum() == 0 or lo.sum() == 0: continue
                num += (np.average(ym[hi], weights=wm[hi]) - np.average(ym[lo], weights=wm[lo])) * wm.sum(); den += wm.sum()
            boots.append(num / den)
        lo_, hi_ = np.percentile(boots, [5, 95])
        res.append([round(1000 * d), round(1000 * lo_), round(1000 * hi_)])
    out['cmp'][key] = res
    # how different are the halves in age make-up, on average (so readers see how small the contrast is)
    gap = within(x, x, N, usable)
    out['spread'][key] = round(1000 * gap)
    qs = np.quantile(x, [0, .2, .4, .6, .8, 1])
    qi = np.clip(np.searchsorted(qs, x, side='right') - 1, 0, 4)
    out['quint'][key] = dict(edges=np.rint(qs * 1000).astype(int).tolist(),
                             reg=[[round(1000 * N[(qi == b) & (regn == r)].sum() / N[qi == b].sum()) for r in REGS] for b in range(5)],
                             y=[[round(1000 * np.average(OUT[qi == b, j], weights=N[qi == b])) for b in range(5)] for j in range(OUT.shape[1])])
open(os.path.join(ROOT, 'age.js'), 'w').write('const AGE=' + json.dumps(out, ensure_ascii=False, separators=(',', ':')) + ';\n')
for key in out['cmp']:
    print(key, 'age-share gap between halves %.1f pts' % (out['spread'][key] / 10))
    for j, l in enumerate(OLAB): print('   %-15s %+5.1f  (%+.1f to %+.1f)' % (l, *(np.array(out['cmp'][key][j]) / 10)))
print(out['meta'])
