#!/usr/bin/env bash
# Simplify the 2026 (B.E. 2569) constituency boundaries into web-sized GeoJSON.
# Needs Node.js (uses npx mapshaper). Run from the repository root.
set -euo pipefail
SRC="${TEA_SOURCES:-sources}/Thai-ECT-election-map/ECT Constituencies/2569/ShapeFile/2569_Election_Constituencies.shp"
OUT="${TEA_GEO:-build/geo}"
mkdir -p "$OUT"
UTM="+proj=utm +zone=47 +datum=WGS84"
npx -y mapshaper -i "$SRC" encoding=utf8 -proj $UTM -clean -simplify interval=550 keep-shapes \
  -filter-islands min-area=6km2 remove-empty -o "$OUT/cons.json" format=geojson precision=10
npx -y mapshaper -i "$OUT/cons.json" -dissolve P_name -o "$OUT/prov.json" format=geojson precision=10
npx -y mapshaper -i "$OUT/cons.json" -dissolve -o "$OUT/country.json" format=geojson precision=10
# Bangkok and neighbouring provinces at higher detail, for the enlarged inset
npx -y mapshaper -i "$SRC" encoding=utf8 \
  -filter "['กรุงเทพมหานคร','นนทบุรี','ปทุมธานี','สมุทรปราการ','สมุทรสาคร','นครปฐม'].indexOf(P_name)>-1" \
  -proj $UTM -clean -simplify interval=120 keep-shapes -filter-islands min-area=0.5km2 remove-empty \
  -o "$OUT/bkk.json" format=geojson precision=5
npx -y mapshaper -i "$OUT/bkk.json" -dissolve P_name -o "$OUT/bkkprov.json" format=geojson precision=5
