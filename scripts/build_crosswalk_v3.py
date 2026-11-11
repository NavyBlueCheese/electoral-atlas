"""Sub-district (tambon) to 2026 constituency crosswalk, version 3.

Each sub-district is assigned using the official constituency descriptions ("อำเภอ X (เฉพาะ/ยกเว้น ...)")
from the ECT result files, matched on DOPA sub-district codes and names. Sub-districts the descriptions
cannot place fall back to the area-overlap crosswalk (data/crosswalk/tambon_constituency_2569.csv).

Inputs (paths via environment variables):
  TEA_CONS       election69 data/2569/constituencies.csv (column 'areas')
  TEA_TAMBONS    sub_district.json and district.json from kongvut/thai-province-data (api/latest), DOPA codes
Output: data/crosswalk/tambon_constituency_2569_v3.csv (tambon_code, constituency_id, weight, method)
"""
import csv, collections, json, os, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONS = os.environ.get('TEA_CONS', 'sources/election69/data/2569/constituencies.csv')
TAMB = os.environ.get('TEA_TAMBONS', 'sources/thai-province-data/api/latest')
sub = json.load(open(os.path.join(TAMB, 'sub_district.json')))
dist = {x['id']: x['name']['th'] for x in json.load(open(os.path.join(TAMB, 'district.json')))}
K = {'%06d' % x['id']: (('%06d' % x['id'])[:2], dist.get(x['district_id'], ''), x['name']['th']) for x in sub}
rows = list(csv.reader(open(CONS, encoding='utf-8-sig')))[1:]
norm = lambda s: s.replace(' ', '').replace('กิ่งอำเภอ', '').strip()
clauses = collections.defaultdict(list)
for cid, prov_th, areas, *_ in rows:
    prov = cid.split('_')[0]
    for part in areas.split('|'):
        m = re.match(r'(อำเภอ|เขต)(.+?)\s*(?:\((เฉพาะ|ยกเว้น)(.*)\))?$', part.strip())
        toks = set(norm(t) for t in (m.group(4) or '').replace('แขวง', '').replace('ตำบล', '').split())
        clauses[(prov, norm(m.group(2)))].append((cid, m.group(3) or 'all', toks))
area = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(ROOT, 'data', 'crosswalk', 'tambon_constituency_2569.csv'))):
    area[r['tambon_code']].append((r['constituency_id'], float(r['weight'])))
out = []
for code, (prov, amp, tam) in sorted(K.items()):
    cl = clauses.get((prov, norm(amp)), [])
    hit = [c for c, k, s in cl if k == 'all' or (k == 'เฉพาะ' and norm(tam) in s) or (k == 'ยกเว้น' and norm(tam) not in s)]
    if len(hit) == 1:
        out.append((code, hit[0], 1.0, 'description'))
    elif code in area:
        out += [(code, c, w, 'area') for c, w in area[code]]
with open(os.path.join(ROOT, 'data', 'crosswalk', 'tambon_constituency_2569_v3.csv'), 'w') as f:
    f.write('tambon_code,constituency_id,weight,method\n')
    for row in out: f.write('%s,%s,%s,%s\n' % row)
print(collections.Counter(r[3] for r in out))
