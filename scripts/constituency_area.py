"""Area of each 2026 constituency in km², from the boundary polygons (WGS84), using the Lambert cylindrical
equal-area projection on the authalic sphere (R = 6371.0072 km).

Usage: python3 scripts/constituency_area.py BOUNDARIES.geojson
BOUNDARIES: 2026 constituencies from Thai-ECT-election-map (properties P_name, CONS_no).
Output: data/area/constituency_area_km2_2569.csv
"""
import json, os, re, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
R = 6371.0072


def ring_area(ring):
    a = np.radians(np.asarray(ring, float))
    x, y = R * a[:, 0], R * np.sin(a[:, 1])
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def poly_area(rings):
    return ring_area(rings[0]) - sum(ring_area(r) for r in rings[1:])


D = json.loads(re.search(r'const DATA=(.*?);\nconst GEO', open(os.path.join(ROOT, 'data.js')).read(), re.S).group(1))
code = {v['th']: k for k, v in D['prov'].items()}
area = {}
for f in json.load(open(sys.argv[1]))['features']:
    g = f['geometry']; polys = [g['coordinates']] if g['type'] == 'Polygon' else g['coordinates']
    k = '%s_%d' % (code[f['properties']['P_name']], f['properties']['CONS_no'])
    area[k] = area.get(k, 0) + sum(poly_area(p) for p in polys)
with open(os.path.join(ROOT, 'data', 'area', 'constituency_area_km2_2569.csv'), 'w') as fo:
    fo.write('constituency_id,area_km2\n')
    for k in sorted(area, key=lambda k: (int(k.split('_')[0]), int(k.split('_')[1]))):
        fo.write('%s,%.2f\n' % (k, area[k]))
print(len(area), 'constituencies, total %.0f km2' % sum(area.values()))
