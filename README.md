# Thai Electoral Atlas

An independent, nonpartisan research platform for Thailand's elections: results, political geography and demographic voting patterns, with every number traced to its source.

This repository holds the public website (a static site, no build step) and the scripts that produce its data.

## Structure

```
index.html            the site (HTML, CSS and JS in one file)
data.js               generated data: 2026 results, boundaries, cross-checks
age.js                generated data: age make-up of each constituency and area-level age patterns
img/                  logo and mascot illustrations
data/crosswalk/       sub-district to constituency crosswalk, 2026
data/age/             adults by age group in each 2026 constituency
scripts/build_geo.sh  simplifies constituency boundaries with mapshaper
scripts/build_data.py joins results, boundaries and the cross-check into data.js
scripts/age_patterns.py builds age.js
scripts/xwalk_spatial.py builds the crosswalk
scripts/ei/           experimental group-vote estimation engine (not used on the site, see its README)
vercel.json           static hosting settings
```

## Run locally

Open `index.html` through any static server, for example `python3 -m http.server`, then visit http://localhost:8000.

## Rebuild the data

1. Put the source datasets in `sources/` (ignored by git):
   - `sources/election69` from https://github.com/jjpeechaphand/election69
   - `sources/election-69-OCR-result` from https://github.com/killernay/election-69-OCR-result
   - `sources/Thai-ECT-election-map` from https://github.com/KittapatR/Thai-ECT-election-map
2. `bash scripts/build_geo.sh`
3. `python3 scripts/build_data.py`
4. `TEA_RESULTS=sources/election69/data/2569/ python3 scripts/age_patterns.py`

## Data sources and licences

| Data | Source | Terms |
|---|---|---|
| 2026 results by constituency and party | Election Commission of Thailand result files, copied with checksums by the election69 project | Government data; reuse terms not stated |
| Signed result forms (ส.ส. 6/1), for cross-checking | election-69-OCR-result | Free public use with attribution |
| Constituency boundaries, 2023 and 2026 | Thai-ECT-election-map by Kittapat Ratanaphupha | MIT. Approximate, not a legal reference |
| Registered population by age, January 2026 | Department of Provincial Administration (DOPA) | Government open data |
| Sub-district boundaries | OpenGISData-Thailand | CC BY 4.0 |
| 2026 certified seat totals | Reuters via The Star, 6 March 2026 | News report |
| 2023 seat totals and turnout | US-ASEAN Business Council; The Phuket Express, 25 May 2023 | News reports |

Party colours are approximate display colours, not official branding.

## Principles

Nonpartisan. Official results are shown as published. Estimates are labelled as modeled and carry their uncertainty. Where two official records disagree, the site shows both.

## Licence

Code is released under the MIT licence (see LICENSE). Data keeps the terms of its original sources, listed above.
