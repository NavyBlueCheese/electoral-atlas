# Adults by age group in each 2026 constituency

`age_by_constituency_2569.txt` gives, for each of the 400 constituencies of the 2026 (B.E. 2569) election, the number of registered Thai residents aged 18–29, 30–44, 45–59 and 60+, plus men and women aged 18+.

- Source: Department of Provincial Administration (DOPA), registered population by single year of age, January 2026 (file 3_6901), by sub-district.
- Method: sub-district counts are added up to constituencies with the area-weighted crosswalk in `data/crosswalk/`. Nine sub-district codes newer than the boundary file were placed by hand using the official constituency descriptions.
- Check: the national total of adults is within 0.3% of the Election Commission's count of eligible voters. The median constituency differs by 0.4%. Eleven constituencies in split city areas differ by more than 10% (for example Nakhon Ratchasima 1 and 2, Lampang 1 and 3), because the crosswalk splits sub-districts by area, not by where people live.
- Caveat: registered residence is not always where people live. Many young people are registered in their home province but live and work elsewhere.
