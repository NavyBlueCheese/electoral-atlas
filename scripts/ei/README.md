# Group-vote estimation engine (experimental, not used on the site)

These scripts try to estimate how each age group voted in each constituency from aggregate data only: official results plus the age make-up of each constituency (ecological inference). They are published for transparency. **The site does not show their output.**

## What was tried

- `ei.py`: Bayesian ecological inference in two stages (NumPyro): a pattern by age group with region and province effects, then constituency estimates adjusted to match each official result exactly. Version 2 adds a smooth age profile. Version 3 adds survey information on the age gradient (`poll_prior.py`).
- `validate.py`: tests the engine on simulated elections built on the real constituency structure, where the true answer is known.
- `variants.py`: re-runs the model on the real 2026 data under several reasonable settings, to check whether the answers are stable.

## Why the output is not published

In simulations the engine beat the naive baseline when age groups vote very differently. On the real 2026 data it did not give stable answers. Within a province, the share of young adults differs between constituencies by only about two percentage points. That is too little variation to separate age groups. Across ten reasonable settings, the estimated People's Party share among 18–29s ranged from about 5% to 50%. The national pattern (younger areas lean People's Party, older areas lean Bhumjaithai and Pheu Thai) is mostly regional, which is the ecological fallacy at work.

The site therefore shows observed area patterns only (`scripts/age_patterns.py`). Group estimates will be added when survey microdata or full poll cross-tabulations are available to combine with the official results.

## Running

Set `TEA_RESULTS` to the `data/2569/` folder of https://github.com/jjpeechaphand/election69, then for example `V3=1 REAL_A=1 SMOOTH=1 python3 validate.py` or `python3 variants.py g2_tau05`. Requires numpy, jax and numpyro.
