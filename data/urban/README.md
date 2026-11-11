# Town and country: municipal population by constituency, 2026

`dopa_tambon_population_6812.csv`: registered Thai population per sub-district (DOPA code), December 2025
(DOPA month code 6812), with the part registered at municipal offices (เทศบาลนคร/เมือง/ตำบล and เมืองพัทยา).
7,257 sub-districts outside Bangkok; totals 59,501,128 Thai residents, 17,026,698 of them municipal. Province
totals agree with the DOPA province tables.

`urban_by_constituency_2569.csv`: the same aggregated to the 2026 constituencies with
`data/crosswalk/tambon_constituency_2569_v3.csv` (`scripts/urban_share.py`). Bangkok is one municipal-level
authority, so its 33 constituencies are counted as fully urban.

Source: Department of Provincial Administration, registry statistics (stat.bora.dopa.go.th), population and
houses by registration office and sub-district.

Caveats: "municipal" is an administrative category, not a measure of how built-up a place is; some
sub-district municipalities (เทศบาลตำบล) are largely rural, and some fast-growing suburbs remain outside any
municipality. Registered residence can differ from where people actually live.
