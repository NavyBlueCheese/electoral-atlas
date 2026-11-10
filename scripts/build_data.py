"""Build data.js for Thai Electoral Atlas from the source datasets.

Usage: python3 scripts/build_data.py   (run scripts/build_geo.sh first)
Source folders come from the environment variables below; defaults assume sources/ inside the repository.
"""
import csv, json, glob, os, collections, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get('TEA_SOURCES', os.path.join(ROOT, 'sources'))
GEO_DIR = os.environ.get('TEA_GEO', os.path.join(ROOT, 'build', 'geo'))

D = os.path.join(SRC, 'election69', 'data', '2569') + '/'
OCR = os.path.join(SRC, 'election-69-OCR-result', 'data', 'matched', 'constituency') + '/'


def rd(f):
    with open(D + f, encoding='utf-8-sig') as fh:
        r = list(csv.reader(fh))
    return r[0], r[1:]


EN = {'ภูมิใจไทย': 'Bhumjaithai', 'ประชาชน': "People's Party", 'เพื่อไทย': 'Pheu Thai', 'กล้าธรรม': 'Kla Tham',
      'ประชาธิปัตย์': 'Democrat', 'ไทรวมพลัง': 'Thai Ruam Palang', 'ประชาชาติ': 'Prachachat', 'พลังประชารัฐ': 'Palang Pracharath',
      'เศรษฐกิจ': 'Economic Party', 'รวมไทยสร้างชาติ': 'United Thai Nation', 'เพื่อชาติไทย': 'Puea Chat Thai',
      'ไทยสร้างไทย': 'Thai Sang Thai', 'รวมใจไทย': 'Ruam Jai Thai', 'ใหม่': 'Mai Party', 'ไทยทรัพย์ทวี': 'Thai Sap Thawi',
      'ประชาธิปไตยใหม่': 'New Democracy', 'มิติใหม่': 'Miti Mai', 'ไทยภักดี': 'Thai Pakdee', 'รวมพลังประชาชน': 'Ruam Palang Prachachon',
      'เสรีรวมไทย': 'Thai Liberal', 'ทางเลือกใหม่': 'New Alternative', 'โอกาสใหม่': 'Okat Mai'}

# ---------- parties
_, prow = rd('parties.csv')
parties = {}
for r in prow:
    parties[r[1]] = dict(th=r[1], en=EN.get(r[1], r[1]), c=r[2] or '#9A9AA8', no=int(r[0]),
                         cs=int(r[10] or 0), ls=int(r[11] or 0), lv=int(r[12] or 0), cv=int(r[13] or 0))

# Display colours: keep each party's hue, but make the darkest ones readable on both white and the black bands.
COLFIX = {'ภูมิใจไทย': '#2E45C8', 'ไทรวมพลัง': '#B03CC4', 'พลังประชารัฐ': '#4C8CC9', 'รวมไทยสร้างชาติ': '#6B7FB0'}
for k, v in COLFIX.items():
    parties[k]['c'] = v

# ---------- provinces
_, pr = rd('provinces.csv')
REG = {'กรุงเทพมหานคร': 'Bangkok', 'กลาง': 'Central', 'เหนือ': 'North', 'ตะวันออกเฉียงเหนือ': 'Northeast',
       'ตะวันออก': 'East', 'ตะวันตก': 'West', 'ใต้': 'South'}
prov = {}
for r in pr:
    prov[r[0]] = dict(code=r[0], th=r[1], en=r[2], ab=r[4], reg=REG.get(r[5], r[5]), n=int(r[6]),
                      dens=float(r[9] or 0), pop=int(r[10] or 0))
th2code = {v['th']: k for k, v in prov.items()}

# ---------- constituencies
_, cr = rd('constituencies.csv')
cons = {}
for r in cr:
    k = r[0]
    num = lambda x: int(x) if x not in ('', None) else None
    cons[k] = dict(id=k, p=k.split('_')[0], n=int(k.split('_')[1]), area=r[2], el=num(r[3]), tu=num(r[4]),
                   va=num(r[5]), iv=num(r[6]), nv=num(r[7]), lva=num(r[9]))
_, cand = rd('candidates.csv')
by = collections.defaultdict(list)
for r in cand:
    by[r[0]].append([r[2].strip(), r[3], int(r[4]) if r[4] else 0, int(r[1])])
for k, v in by.items():
    v.sort(key=lambda x: -x[2])
    c = cons[k]
    c['cand'] = [[x[0], x[1], x[2]] for x in v[:6]]
    c['ncand'] = len(v)
    c['w'] = v[0][1]
    c['mar'] = round(100 * (v[0][2] - v[1][2]) / c['va'], 2) if c['va'] else None
    c['ws'] = round(100 * v[0][2] / c['va'], 2) if c['va'] else None

# party-list per constituency: top 5
h, pl = rd('party_list_votes.csv')
pnames = [x for x in h[1:]]
for r in pl:
    vals = sorted([(int(x or 0), pnames[i]) for i, x in enumerate(r[1:])], reverse=True)
    t = sum(v for v, _ in vals)
    cons[r[0]]['pl'] = [[n, v] for v, n in vals[:5]]
    cons[r[0]]['plt'] = t

# ---------- province-level party vote shares (all candidates and full party-list rows)
seated = [p['th'] for p in parties.values() if p['cs'] + p['ls'] > 0]
pst = {k: dict(cv=collections.Counter(), lv=collections.Counter(), va=0, lva=0, el=0, tu=0) for k in prov}
for k, v in by.items():
    ps = pst[cons[k]['p']]
    for x in v:
        if x[1] in seated:
            ps['cv'][x[1]] += x[2]
    ps['va'] += cons[k]['va']; ps['el'] += cons[k]['el']; ps['tu'] += cons[k]['tu']
for r in pl:
    ps = pst[r[0].split('_')[0]]
    for i, x in enumerate(r[1:]):
        if pnames[i] in seated:
            ps['lv'][pnames[i]] += int(x or 0)
    ps['lva'] += sum(int(x or 0) for x in r[1:])
for k, ps in pst.items():
    prov[k]['cv'] = dict(ps['cv']); prov[k]['lv'] = dict(ps['lv'])
    prov[k]['va'] = ps['va']; prov[k]['lva'] = ps['lva']; prov[k]['el'] = ps['el']; prov[k]['tu'] = ps['tu']

# ---------- cross-check with OCR of signed forms (ส.ส. 6/1)
xc = dict(ocr_files=0, agree=0, disagree=[], diffs=[], missing=[])
ocr_ids = set()
for f in glob.glob(OCR + '*.json'):
    j = json.load(open(f))
    k = os.path.basename(f)[:-5]
    ocr_ids.add(k)
    xc['ocr_files'] += 1
    b = sorted(j['results'], key=lambda x: -x['votes'])
    if b[0]['party'] == cons[k]['w']:
        xc['agree'] += 1
    else:
        xc['disagree'].append(k)
    web = {x[0].replace(' ', ''): x[2] for x in by[k]}
    for x in j['results']:
        n = x['name'].replace(' ', '')
        if n in web and web[n] != x['votes']:
            xc['diffs'].append([k, x['name'], web[n], x['votes']])
xc['missing'] = sorted(set(cons) - ocr_ids)
for k in cons:
    cons[k]['ocr'] = 0 if k in xc['missing'] else (2 if any(d[0] == k for d in xc['diffs']) else 1)

# ---------- national totals
tot = dict(el=sum(c['el'] for c in cons.values()), tu=sum(c['tu'] for c in cons.values()),
           va=sum(c['va'] for c in cons.values()), iv=sum(c['iv'] for c in cons.values()),
           nv=sum(c['nv'] for c in cons.values()))
wins = collections.Counter(c['w'] for c in cons.values())
for p in parties.values():
    assert wins.get(p['th'], 0) == p['cs'], (p['th'], wins.get(p['th']), p['cs'])

# ---------- geometry
def load(f):
    j = json.load(open(os.path.join(GEO_DIR, f)))
    if 'features' in j:
        return j['features']
    if j.get('type') == 'GeometryCollection':
        return [dict(properties={}, geometry=g) for g in j['geometries']]
    return [j]


country = load('country.json')
allx, ally = [], []
def walk(g, fn):
    polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
    for poly in polys:
        for ring in poly:
            fn(ring)
for ft in country:
    walk(ft['geometry'], lambda ring: [(allx.append(x), ally.append(y)) for x, y in ring])
minx, maxx, miny, maxy = min(allx), max(allx), min(ally), max(ally)
H = 1000.0
S = H / (maxy - miny)
W = (maxx - minx) * S
PAD = 6


def proj(x, y, s=S, ox=minx, oy=maxy, pad=PAD):
    return ((x - ox) * s + pad, (oy - y) * s + pad)


def path(g, pr=proj, dec=1):
    out = []
    polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
    m = 10 ** dec
    for poly in polys:
        for ring in poly:
            pts = [pr(x, y) for x, y in ring]
            q = [(round(x * m), round(y * m)) for x, y in pts]
            # drop consecutive duplicates
            qq = [q[0]]
            for p in q[1:]:
                if p != qq[-1]:
                    qq.append(p)
            if len(qq) < 4:
                continue
            s = 'M%s %s' % (fmt(qq[0][0], m), fmt(qq[0][1], m))
            px, py = qq[0]
            parts = []
            for x, y in qq[1:-1]:
                parts.append('%s %s' % (fmt(x - px, m), fmt(y - py, m)))
                px, py = x, y
            out.append(s + 'l' + ' '.join(parts) + 'z')
    return ''.join(out)


def fmt(v, m):
    s = ('%.1f' % (v / m)).rstrip('0').rstrip('.') if m == 10 else str(v)
    if s.startswith('0.'):
        s = s[1:]
    elif s.startswith('-0.'):
        s = '-' + s[2:]
    return s if s not in ('-0', '') else '0'


def centroid(g, pr=proj):
    best = None
    polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
    for poly in polys:
        ring = [pr(x, y) for x, y in poly[0]]
        a = cx = cy = 0
        for i in range(len(ring) - 1):
            x0, y0 = ring[i]; x1, y1 = ring[i + 1]
            f = x0 * y1 - x1 * y0
            a += f; cx += (x0 + x1) * f; cy += (y0 + y1) * f
        if a == 0:
            continue
        a /= 2
        c = (cx / (6 * a), cy / (6 * a), abs(a))
        if best is None or c[2] > best[2]:
            best = c
    return [round(best[0], 1), round(best[1], 1)]


def bbox(g, pr=proj):
    xs, ys = [], []
    walk(g, lambda ring: [(xs.append(pr(x, y)[0]), ys.append(pr(x, y)[1])) for x, y in ring])
    return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]


geo_cons = {}
for ft in load('cons.json'):
    p = ft['properties']
    k = '%s_%d' % (th2code[p['P_name']], int(p['CONS_no']))
    geo_cons[k] = dict(d=path(ft['geometry']), c=centroid(ft['geometry']))
assert set(geo_cons) == set(cons), set(cons) ^ set(geo_cons)
geo_prov = {}
for ft in load('prov.json'):
    k = th2code[ft['properties']['P_name']]
    geo_prov[k] = dict(d=path(ft['geometry']), c=centroid(ft['geometry']), b=bbox(ft['geometry']))
outline = path(country[0]['geometry'])

# Bangkok inset
bk = load('bkk.json')
bx, byy = [], []
for ft in bk:
    if ft['properties']['P_name'] == 'กรุงเทพมหานคร':
        walk(ft['geometry'], lambda ring: [(bx.append(x), byy.append(y)) for x, y in ring])
bminx, bmaxx, bminy, bmaxy = min(bx), max(bx), min(byy), max(byy)
BS = 420.0 / (bmaxy - bminy)
BW = (bmaxx - bminx) * BS
bpr = lambda x, y: proj(x, y, BS, bminx, bmaxy, 10)
geo_bkk = {}
for ft in bk:
    p = ft['properties']
    if p['P_name'] != 'กรุงเทพมหานคร':
        continue
    k = '%s_%d' % (th2code[p['P_name']], int(p['CONS_no']))
    geo_bkk[k] = dict(d=path(ft['geometry'], bpr), c=centroid(ft['geometry'], bpr))
bkk_nb = [path(ft['geometry'], bpr) for ft in load('bkkprov.json') if ft['properties']['P_name'] != 'กรุงเทพมหานคร']
bkk_box = None
# where Bangkok sits in the main map
for ft in load('prov.json'):
    if ft['properties']['P_name'] == 'กรุงเทพมหานคร':
        bkk_box = bbox(ft['geometry'])

GEO = dict(w=round(W + 2 * PAD, 1), h=round(H + 2 * PAD, 1), outline=outline, cons=geo_cons, prov=geo_prov,
           bkk=dict(w=round(BW + 20, 1), h=440, cons=geo_bkk, nb=bkk_nb, box=bkk_box))

# ---------- 2023 national seats (sourced: US-ASEAN Business Council; The Phuket Express, 25 May 2023)
Y2023 = dict(date='14 May 2023', turnout=75.71, eligible=52195920, voted=39514964,
             seats=[['Move Forward', 151, 112, 39, '#F47920'], ['Pheu Thai', 141, 112, 29, '#E30613'],
                    ['Bhumjaithai', 71, 68, 3, '#0F1E64'], ['Palang Pracharath', 40, None, None, '#1B4F9C'],
                    ['United Thai Nation', 36, None, None, '#1D2951']])

DATA = dict(parties=sorted(parties.values(), key=lambda p: -(p['cs'] + p['ls'])), prov=prov, cons=cons, tot=tot,
            xc=dict(ocr_files=xc['ocr_files'], agree=xc['agree'], disagree=xc['disagree'], missing=xc['missing'],
                    ndiff=len(xc['diffs']), ndiffcons=len(set(d[0] for d in xc['diffs'])),
                    maxdiff=max(xc['diffs'], key=lambda d: abs(d[2] - d[3])), diffs=xc['diffs']),
            y2023=Y2023)
with open(os.path.join(ROOT, 'data.js'), 'w') as f:
    f.write('const DATA=' + json.dumps(DATA, ensure_ascii=False, separators=(',', ':')) + ';\n')
    f.write('const GEO=' + json.dumps(GEO, ensure_ascii=False, separators=(',', ':')) + ';\n')
print('size', os.path.getsize(os.path.join(ROOT, 'data.js')) // 1024, 'KB', 'map', round(W), 'x', H,
      'bkk', round(BW), tot, 'turnout', round(100 * tot['tu'] / tot['el'], 2),
      'xc', xc['agree'], len(xc['diffs']), xc['missing'])
