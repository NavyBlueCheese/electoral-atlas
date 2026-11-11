"""Share of registered Thai residents living in municipal areas, per 2026 constituency.

Input: data/urban/dopa_tambon_population_6812.csv (tambon_code, thai_pop_total, thai_pop_municipal), from the
DOPA registry statistics (population and houses, year 2568 = December 2025) by registration office and
sub-district; an office whose name contains 'เทศบาล' (or ทต./ทม./ทน.) or 'เมืองพัทยา' is municipal.
Bangkok (not covered by the sub-district file) counts as entirely urban. Output: data/urban/urban_by_constituency_2569.csv
"""
import csv, collections, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
xw = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(ROOT, 'data', 'crosswalk', 'tambon_constituency_2569_v3.csv'))):
    xw[r['tambon_code']].append((r['constituency_id'], float(r['weight'])))
urb = collections.Counter(); tot = collections.Counter(); missing = []
for r in csv.DictReader(open(os.path.join(ROOT, 'data', 'urban', 'dopa_tambon_population_6812.csv'))):
    t, u = int(r['thai_pop_total']), int(r['thai_pop_municipal'])
    if r['tambon_code'] not in xw:
        missing.append(r['tambon_code']); continue
    for c, w in xw[r['tambon_code']]:
        tot[c] += t * w; urb[c] += u * w
with open(os.path.join(ROOT, 'data', 'urban', 'urban_by_constituency_2569.csv'), 'w') as f:
    f.write('constituency_id,thai_pop_municipal,thai_pop_total,municipal_share\n')
    for c in sorted(tot, key=lambda k: (int(k.split('_')[0]), int(k.split('_')[1]))):
        f.write('%s,%d,%d,%.4f\n' % (c, round(urb[c]), round(tot[c]), urb[c] / tot[c] if tot[c] else 0))
    for i in range(1, 34):   # Bangkok: the whole province is one municipal-level authority
        f.write('10_%d,,,1.0000\n' % i)
print('constituencies', len(tot) + 33, 'unmapped sub-districts (zero population):', missing)
