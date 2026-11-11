"""How did different kinds of places vote? Observed area patterns, 2026 party-list ballot.

For each place characteristic (share of adults in an age group, population density, ...):
  quint : constituencies cut into five equal piles by the characteristic, with the vote-weighted
          outcome in each pile and the region mix of each pile (not province-adjusted)
  cmp   : within-province comparison. In each province with 2+ constituencies, the half above the
          provincial median vs the half below; difference in each outcome, averaged over provinces
          weighted by eligible voters, with a 90% interval from resampling provinces (1,000 draws)
  spread: how different the two halves are in the characteristic itself
  value : the characteristic for every constituency (for maps and cards)
Inputs: ECT 2026 results (scripts/ei/results.py, TEA_RESULTS); data/age/age_by_constituency_2569.txt;
data/area/constituency_area_km2_2569.csv; data/urban/urban_by_constituency_2569.csv; data.js (regions, eligible voters).
Output: places.js (const PLACES)
"""
import json, os, re, sys
import numpy as np

ROOT = os.environ.get('TEA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'ei'))
import csv
from results import load, LABELS

ids, Y, N, _ = load()
raw = {}
for l in open(os.path.join(ROOT, 'data', 'age', 'age_by_constituency_2569.txt')):
    if l.startswith('#'): continue
    q = l.split(); raw[q[0]] = np.array(list(map(float, q[1:7])))
R6 = np.array([raw[k] for k in ids])
adults = R6[:, :4].sum(1)
B = R6[:, :4] / adults[:, None]
area = {r['constituency_id']: float(r['area_km2']) for r in csv.DictReader(open(os.path.join(ROOT, 'data', 'area', 'constituency_area_km2_2569.csv')))}
AREA = np.array([area[k] for k in ids])
D = json.loads(re.search(r'const DATA=(.*?);\nconst GEO', open(os.path.join(ROOT, 'data.js')).read(), re.S).group(1))
V = Y[:, :6] / Y[:, :6].sum(1, keepdims=True)
turn = 1 - Y[:, 7] / Y.sum(1)
OUT = np.column_stack([V, turn])
OLAB = LABELS[:6] + ['Turnout']
prov = np.array([k.split('_')[0] for k in ids])
REGS = ['Bangkok', 'Central', 'East', 'West', 'North', 'Northeast', 'South']
regn = np.array([D['prov'][p]['reg'] for p in prov])
usable = [p for p in sorted(set(prov)) if (prov == p).sum() >= 2]
rng = np.random.default_rng(2026)

# key, label, short (plural noun for the people/places), unit, values, spread kind ('pts' or 'ratio')
CHARS = [
    ('u30', 'Share of adults under 30', 'under-30s', '%', B[:, 0] * 100, 'pts'),
    ('u45', 'Share of adults under 45', 'under-45s', '%', (B[:, 0] + B[:, 1]) * 100, 'pts'),
    ('o60', 'Share of adults aged 60+', 'over-60s', '%', B[:, 3] * 100, 'pts'),
    ('dens', 'Eligible voters per square kilometre', 'people per km²', '/km²', np.array([D['cons'][k]['el'] for k in ids]) / AREA, 'ratio'),
]
U = {r['constituency_id']: float(r['municipal_share']) for r in csv.DictReader(open(os.path.join(ROOT, 'data', 'urban', 'urban_by_constituency_2569.csv')))}
CHARS.append(('urb', 'Share of Thai residents registered in municipal areas', 'town and city residents', '%',
              np.array([100 * U[k] for k in ids]), 'pts'))


def halves(x, m):
    xm = x[m]; med = np.median(xm); hi = xm > med; lo = xm < med
    if hi.sum() == 0 or lo.sum() == 0:
        hi = xm >= med; lo = xm < med
    return hi, lo


def within(x, y, provs, ratio=False):
    num = den = 0.0
    for p in provs:
        m = prov == p
        hi, lo = halves(x, m)
        if hi.sum() == 0 or lo.sum() == 0: continue
        w = N[m]; yy = np.log(y[m]) if ratio else y[m]
        d = np.average(yy[hi], weights=w[hi]) - np.average(yy[lo], weights=w[lo])
        num += d * w.sum(); den += w.sum()
    return num / den


out = dict(ids=ids, outcomes=OLAB, regions=REGS, chars=[], meta=dict(
    provinces_used=len(usable), constituencies_used=int(sum((prov == p).sum() for p in usable)),
    sources=['ECT 2026 result files, party-list ballot', 'DOPA registered population by age, January 2026',
             'Constituency boundaries: Thai-ECT-election-map (MIT), area computed on an equal-area projection']))
for key, label, short, unit, x, kind in CHARS:
    cmp = []
    for j in range(OUT.shape[1]):
        d = within(x, OUT[:, j], usable)
        boots = [within(x, OUT[:, j], list(rng.choice(usable, len(usable), replace=True))) for _ in range(1000)]
        lo_, hi_ = np.percentile(boots, [5, 95])
        cmp.append([round(1000 * d), round(1000 * lo_), round(1000 * hi_)])
    spread = within(x, x, usable, ratio=(kind == 'ratio'))
    spread = round(float(np.exp(spread)), 2) if kind == 'ratio' else round(float(spread), 2)
    qs = np.quantile(x, [0, .2, .4, .6, .8, 1])
    qi = np.clip(np.searchsorted(qs, x, side='right') - 1, 0, 4)
    quint = dict(edges=[round(float(v), 1) for v in qs],
                 reg=[[round(1000 * N[(qi == b) & (regn == r)].sum() / N[qi == b].sum()) for r in REGS] for b in range(5)],
                 y=[[round(1000 * np.average(OUT[qi == b, j], weights=N[qi == b])) for b in range(5)] for j in range(OUT.shape[1])])
    out['chars'].append(dict(key=key, label=label, short=short, unit=unit, kind=kind, spread=spread,
                             value=[round(float(v), 1) for v in x], quint=quint, cmp=cmp))
    print(key, 'spread', spread, 'edges', quint['edges'])
    for j, l in enumerate(OLAB): print('   %-15s %+5.1f (%+.1f to %+.1f)   quint %s' % (l, *(np.array(cmp[j]) / 10), ' '.join('%.1f' % (v / 10) for v in quint['y'][j])))
with open(os.path.join(ROOT, 'places.js'), 'w') as f:
    f.write('const PLACES=' + json.dumps(out, ensure_ascii=False, separators=(',', ':')) + ';\n')
