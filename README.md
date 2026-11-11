# Thai Electoral Atlas

An independent, nonpartisan research platform for Thailand's elections: results, political geography and demographic voting patterns.
This repository holds the public website and the scripts that produce its data.

## Rebuild the data

1. Put the source datasets in `sources/` (ignored by git):
   - `sources/election69` from https://github.com/jjpeechaphand/election69
   - `sources/election-69-OCR-result` from https://github.com/killernay/election-69-OCR-result
   - `sources/Thai-ECT-election-map` from https://github.com/KittapatR/Thai-ECT-election-map
2. `bash scripts/build_geo.sh`
3. `python3 scripts/build_data.py`
4. `python3 scripts/build_crosswalk_v3.py`, `python3 scripts/urban_share.py` and
   `python3 scripts/constituency_area.py sources/Thai-ECT-election-map/<2026 constituencies>.geojson`
   (the crosswalk also needs `sources/thai-province-data` from https://github.com/kongvut/thai-province-data)
4b. `TEA_RESULTS=sources/election69/data/2569/ python3 scripts/place_patterns.py`
5. `TEA_RESULTS=sources/election69/data/2569/ python3 scripts/age_groups.py` (needs numpy, jax, numpyro)

## Data sources and licences

| Data | Source | Terms |
|---|---|---|
| 2026 results by constituency and party | Election Commission of Thailand result files, copied with checksums by the election69 project | Government data; reuse terms not stated |
| Signed result forms (ส.ส. 6/1), for cross-checking | election-69-OCR-result | Free public use with attribution |
| Constituency boundaries, 2023 and 2026 | Thai-ECT-election-map by Kittapat Ratanaphupha | MIT. Approximate, not a legal reference |
| Registered population by age, January 2026 | Department of Provincial Administration (DOPA) | Government open data |
| Registered population by registration office and sub-district, December 2025 | DOPA registry statistics (stat.bora.dopa.go.th) | Government open data |
| Sub-district codes and names | thai-province-data by Kongvut Sangkla | MIT |
| Sub-district boundaries | OpenGISData-Thailand | CC BY 4.0 |
| 2026 certified seat totals | Reuters via The Star, 6 March 2026 | News report |
| 2023 seat totals and turnout | US-ASEAN Business Council; The Phuket Express, 25 May 2023 | News reports |

Party colours are approximate display colours and not official branding.


## Licence

Code is released under the MIT licence (see LICENSE). Data keeps the terms of its original sources, listed above.
