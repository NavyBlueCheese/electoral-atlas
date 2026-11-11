# Sub-district to constituency crosswalk, 2026

`tambon_constituency_2569.csv` assigns each sub-district (tambon, DOPA 6-digit code) to the 2026 (B.E. 2569) constituencies it falls in.

- `weight` is the share of the sub-district's area inside that constituency. Overlaps under 10% are treated as boundary noise and dropped; the remaining weights are rescaled to sum to 1.
- 7,367 sub-districts, 400 constituencies, 126 sub-districts split between two or more constituencies.

Sources: sub-district boundaries from OpenGISData-Thailand (CC BY 4.0); constituency boundaries from Thai-ECT-election-map (MIT), which are approximate and not a legal reference.
Method: area intersection in WGS84 coordinates, built by `scripts/xwalk_spatial.py`.

## Version 3 (`tambon_constituency_2569_v3.csv`)

Used for the town/country figures. Each sub-district is placed using the official constituency descriptions
published with the ECT results (district lists with "เฉพาะ"/"ยกเว้น" sub-district clauses), matched on DOPA
codes and names (kongvut/thai-province-data, MIT). 7,423 sub-districts are placed this way; 16 that the
descriptions cannot place fall back to area weights. Built by `scripts/build_crosswalk_v3.py`.

Known limitation: the age figures (`data/age/`) were aggregated with the area crosswalk above, before version 3
existed. In split urban districts this can move a few thousand people between neighbouring constituencies.
