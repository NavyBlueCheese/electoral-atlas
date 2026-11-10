"""Assign every sub-district (tambon) to 2026 constituencies by area overlap.
Sub-district boundaries: OpenGISData-Thailand (CC BY 4.0). Constituencies: Thai-ECT-election-map (MIT).
Output: tambon_code, constituency_id, weight (share of the tambon's area inside that constituency)."""
import json, csv, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get('TEA_SOURCES', os.path.join(ROOT, 'sources'))
from shapely.geometry import shape
from shapely.strtree import STRtree
from shapely.validation import make_valid

provs = {r[1]: r[0] for r in list(csv.reader(open(os.path.join(SRC, 'election69/data/2569/provinces.csv'), encoding='utf-8-sig')))[1:]}
C = json.load(open(os.path.join(SRC, 'cons2569_wgs.json')))['features']
cg, cid = [], []
for f in C:
    g = make_valid(shape(f['geometry'])).buffer(0)
    cg.append(g); cid.append('%s_%d' % (provs[f['properties']['P_name']], int(f['properties']['CONS_no'])))
tree = STRtree(cg)
T = json.load(open(os.path.join(SRC, 'subdistricts.geojson')))['features']
rows, nomatch, splits = [], [], 0
for f in T:
    p = f['properties']; g = make_valid(shape(f['geometry'])).buffer(0)
    hits = []
    for i in tree.query(g):
        a = g.intersection(cg[i]).area
        if a > 0: hits.append((cid[i], a))
    tot = sum(a for _, a in hits)
    hits = [(c, a / tot) for c, a in hits if tot and a / tot >= 0.10]
    s = sum(w for _, w in hits)
    if not hits:
        nomatch.append(p['tam_code']); continue
    # constituencies never cross provinces: keep only those in the tambon's province
    same = [(c, w) for c, w in hits if c.split('_')[0] == p['pro_code']]
    if same: hits = same; s = sum(w for _, w in hits)
    if len(hits) > 1: splits += 1
    for c, w in hits: rows.append((p['tam_code'], c, round(w / s, 4)))
with open(os.path.join(ROOT, 'data/crosswalk/tambon_constituency_2569.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['tambon_code', 'constituency_id', 'weight']); w.writerows(rows)
print('tambons', len(T), 'rows', len(rows), 'split tambons', splits, 'no match', len(nomatch), nomatch[:10], 'constituencies covered', len({r[1] for r in rows}))
