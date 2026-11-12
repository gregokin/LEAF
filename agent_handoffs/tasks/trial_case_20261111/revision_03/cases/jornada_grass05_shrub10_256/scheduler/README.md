# Scheduler: alternating wind / rain year (DESIGN EXAMPLE)

All paths in manifests and CSV values are relative to the **case root** `cases/jornada_grass05_shrub10_256/` (e.g. `scheduler/examples/wind.json`).
In this README, shorthand such as `examples/wind.json` is only a display name local to `scheduler/`; manifest/CSV values always use the full case-root path.

## What this directory is — and is not

| File | Format status |
|---|---|
| `examples/wind.json`, `examples/rain.json`, `examples/interevent.json` | **Native** MASY event files (`masy.wind_event.v1`, `masy.rain_event.v1`, `masy.interevent.v1`), strict keys only |
| `annual_plan.csv` | **Planning / design table**. NOT a native `masy.scenario.v1` document |
| `settings.toml` | **Planning manifest**. Read by no code; no CLI exists |

There is **no** end-to-end `leaf run case.toml` driver, and the native scenario `gap_policy` accepts only
`refuse` or `idle`. **MASY idle gaps run NO evaporation, transpiration or growth.** The 72-row plan is therefore a
reviewed *design example*: a future driver has to fill every interevent period at run time from the **actual**
terminal time of the previous accepted transport block up to the next fixed start. Do not invent
`gap_policy = "interevent"`; it does not exist. This package does not claim a qualified full coupling.

## The year in one view

* 36 **pairs**, 72 transport rows, strictly alternating wind → rain (`wind_p01, rain_p01, … wind_p36, rain_p36`).
* Each rain starts 09:00 MST. Its wind event starts 09:00 on the **previous day** and forces for 43 200 s (12 h), so the
  nominal gap wind-end (21:00) → rain-start (09:00) is 43 200 s of interevent biology.
* Wind: `u* = 0.6 m/s`, directions 22.5° and 157.5° *to* (135° apart, vector mean east/90°). This is the reference experimental
  forcing, **not** an average wind climate. The compact native example uses two 21 600 s blocks; the MAPLE reference toggles with
  600 s blocks, so the example differs there but keeps the directions and the 12 h total.
* Rain counts per month `[2,2,3,3,3,3,4,4,4,3,3,2]` (= 36), dates 4/18 (2 per month), 4/14/24 (3 per month), 3/10/17/24 (4 per month).
* Rain durations: 6 h Jan/Feb/Mar/Nov/Dec; 4 h Apr/May/Oct; 3 h Jun–Sep.
* **Exact monthly totals** (NOAA 1991–2020 El Paso normals × 25.4 mm/in): 9.906, 10.160, 6.096, 4.318, 10.922, 18.542, 40.132,
  42.418, 38.608, 14.986, 10.922, 16.002 mm; annual 223.012 mm; Jul–Sep 121.158 mm (≈ 54.3 %).
* Disaggregation: each monthly total is split equally into *N* events at 0.001 mm precision. The first *N−1* events carry
  total/*N* rounded to the nearest 0.001 mm (half rounds up); the **last** event carries the remainder, so every month sums exactly.
  Intensity (mm/h) = depth / duration in hours (single constant interval). Example: July first rain 10.033 mm in 3 h = 3.344333333333333 mm/h.
* No background rain. **Only the monthly totals are observed normals**; dates, counts, durations, disaggregation and the wind
  forcing are AUTHOR synthetic choices.

## Native event examples (documented here because JSON allows no comments)

`examples/wind.json` — `masy.wind_event.v1`, id `wind_bimodal_u06_12h`: two blocks, each `shear_velocity_m_s = 0.6`,
`duration_s = 21600`, `direction_deg` 22.5 then 157.5.

`examples/rain.json` — `masy.rain_event.v1`, id `rain_jul_first_10p033mm_3h`: `water_profile_id = "trial_sandy_loam"`
(the upstream water owner must bind a qualified profile of that id; none is copied here), one interval of 10 800 s at
3.344333333333333 mm/h, `recession_guard_s = 21600`.

`examples/interevent.json` — `masy.interevent.v1`, id `interevent_1d`: `duration_s = 86400` with evaporation, transpiration and
growth enabled. A driver would use *actual* spans (initial, between-block, final), split at local midnight, the 7-day growth grid and new years.

## Rain recession guard (MAX, not duration)

`recession_guard_s = 21600` is the **maximum** recession the owner may spend, not the realized duration. The water block ends **only**
at the actual SYRUP terminal recession time. The `forcing_plus_rain_guard_end_offset_s` column = nominal forcing end + rain guard (zero guard for wind rows).
It is static design arithmetic for the no-overlap review, **not** a measured or guaranteed terminal transport time. Wind settlement the owning model counts
is actual elapsed time and may shorten or eliminate the nominal 12 h biology gap before a rain; a fixed start that would collide with the realized end of the
previous block is **refused**, never truncated or shifted.

## Interevent periods

1. **Initial**: 2027-01-01T00:00 (0 s) → `wind_p01` start (205 200 s, 2027-01-03T09:00) = 57 h.
2. **Between blocks**: actual terminal time of an accepted block → the fixed start of the next block.
3. **Final**: actual terminal time of `rain_p36` (nominal rain end 30 380 400 s; static forcing-plus-full-guard arithmetic 30 402 000 s, not a measurement) → 2028-01-01T00:00 (31 536 000 s).

Inside wind, rain and recession there is **no** evaporation, transpiration or growth and **no hidden catch-up**. No second
rainfall/infiltration authority exists: accepted storm water comes only from SYRUP, `added_water_m` is zero in interevent steps, and
nothing is debited twice. If a scheduled start would precede the realized end of the previous block the driver must **refuse**;
forcing is never silently truncated or shifted.

## `annual_plan.csv` columns

`seq` 1–72; `pair` 1–36; `month`; `kind` (`wind`|`rain`); `event_id`; `start_local` (ISO, −07:00); `start_offset_s` (seconds from
2027-01-01T00:00 MST); `duration_s` (forcing/rainfall duration); `nominal_end_local`; `nominal_end_offset_s`;
`forcing_plus_rain_guard_end_offset_s` (nominal forcing end + rain guard; for wind rows the guard is zero, so it equals the nominal end; actual settlement unmodelled; not a terminal time); `forcing_ref`
(case-root path of the native example whose schema/PATTERN the row follows: the row's own depth/duration/direction produce a fresh native event, the July example is not replayed for every month);
`rain_depth_mm`, `rain_intensity_mm_h`, `recession_guard_s` (rain rows only). **`rain_depth_mm` is the authoritative planning budget**; `rain_intensity_mm_h` is a display value rounded to 9 decimals
(depth error at most about 2e-9 mm) and a future adapter must construct intensity as depth/(duration_s/3600) instead of treating the rounded number as an exact budget;
`month_total_mm_normals` (the month's normal repeated on each rain row — do **not** sum this column).

## Not provided

Full MAPLE wind/sediment calibration, SYRUP storm/infiltration profiles and transport source pairs belong upstream and are not silently
copied. Surface-moisture feedback to wind is disabled; surface-to-soil remapping is deferred. Nothing here was executed.
