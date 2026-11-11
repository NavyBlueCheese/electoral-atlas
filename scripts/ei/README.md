# Results-only estimation engine (experimental, not used on the site)

These scripts try to estimate how each age group voted in each constituency from aggregate data only: official results plus the age make-up of each constituency (ecological inference). They are published for transparency. **The site does not show their output.**

## What was tried

- `ei.py`: Bayesian ecological inference in two stages (NumPyro): a pattern by age group with region and province effects, then constituency estimates adjusted to match each official result exactly. Version 2 adds a smooth age profile. Version 3 adds survey information on the age gradient (`poll_prior.py`).
- `validate.py`: tests the engine on simulated elections built on the real constituency structure, where the true answer is known.
- `variants.py`: re-runs the model on the real 2026 data under several reasonable settings, to check whether the answers are stable.

## Why the output is not published

In simulations the engine beat the naive baseline when age groups vote very differently. On the real 2026 data it did not give stable answers. Within a province, the share of young adults differs between constituencies by only about two percentage points. That is too little variation to separate age groups. Across ten reasonable settings, the estimated People's Party share among 18–29s ranged from about 5% to 50%. The national pattern (younger areas lean People's Party, older areas lean Bhumjaithai and Pheu Thai) is mostly regional, which is the ecological fallacy at work.

The site instead takes the age differences from published polls and fits them to each constituency's official result (`scripts/age_groups.py`), and shows the observed area patterns as a check (`scripts/place_patterns.py`).

## Town and country (municipal vs non-municipal residents)

`ei_twogroup.py` runs the same engine for a split by where people live, and `ei_twogroup_sim.py` tests it on
simulated elections built on the real constituencies and municipal shares. The municipal share differs between
neighbouring constituencies far more than age does, and on the real data the answer was stable across five
settings (People's Party about 42% among municipal residents and 26% among the rest). But in simulation:

- with no contextual effect, national estimates were accurate, yet constituency estimates were no better than
  assuming everyone in a constituency votes alike;
- with a moderate contextual effect (a place shifting the vote of everyone living there, by 0.3 on the log-odds
  scale per standard deviation of municipal share), national 90% intervals contained the truth only 13% of the
  time, even when widened.

Contextual effects cannot be ruled out from the results, so these estimates are **not published**. The site shows
the observed town/country patterns between places instead.

Input for both scripts: a JSON file `{constituency_id: [municipal population, total population]}`, which can be
built from `data/urban/urban_by_constituency_2569.csv` (Bangkok: `[1, 1]`). Example:
`TEA_RESULTS=... python3 ei_twogroup.py urban.json base` and `python3 ei_twogroup_sim.py urban.json 0.8 0.3 1`.

## Running

Set `TEA_RESULTS` to the `data/2569/` folder of https://github.com/jjpeechaphand/election69, then for example `V3=1 REAL_A=1 SMOOTH=1 python3 validate.py` or `python3 variants.py g2_tau05`. Requires numpy, jax and numpyro.
