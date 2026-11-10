# Independent read-only contract review

Reviewer: Codex agent `/root/jornada_cover`. No edits, external sends or runs.

Native branch and source identities match the baseline. Current column API has
no transpiration sink; negative intake, calling T evaporation, or subtracting
native water inside LEAF would violate ownership/accounting. A pure native
withdrawal is required, limited by max(water - residual_depth,0). Antecedent
below residual is valid and must remain unchanged. Hydro advances time; the
following native T operator does not advance it again. No new species wilting
law has been supplied.

Use m of water depth in native [L,N] arrays and m3 for plant demand/uptake;
convert at the interface exactly once using scalar cell area. Test nonunit area
0.0625 m2, not just 1 m2. Match flat row-major grid, layers, namespace, device,
FP64 contiguous shape and masks. Masked root weights are not silently normalized;
inactive potential demand remains explicitly unmet. Retain individuals.

Reuse an explicit water-free Simulation inside a composed coupled driver.
Keep R, traits, roots and weekly growth; skip W, scale, removed/overflow and
synthetic initialization/forcing. Extract CPU gather; reuse GPU gather. Refuse
the synthetic-water stepping/reporting path in this mode.

For transactional daily steps, preserve prior soil, uptake and counters if
native or gather validation fails. Weekly refusal may latch the already
accepted physical day and pending uptake under the existing retry contract.

## Peak-memory decomposition

Let V be water-free LEAF arrays including the current CSR:
8LN + 188P + 8SL + 12H + 8 before mirrors. Let C be column/T parameters,
F forcing = 8LN+8N, Q request = 8LN, W0 original soil = 8LN, G plant candidates.
The native hydro kernel outputs 8(3L+6)N bytes including temporary rows;
published hydro diagnostics/state retain 8(3L+2)N. Native T outputs need 32LN
bytes (fresh state, actual, unmet, scale). Admit:

C + V + F + Q + W0 + G +
max(8(3L+6)N + hydro_scratch,
    8(3L+2)N + 32LN + T_scratch)

Include any retained previous results and host mirrors separately. Reuse hydro
state as T input rather than counting it twice. Weekly rebuild adds the new
root graph and growth/build scratch while native/current plant state remain
live; obsolete hydro/T diagnostic arrays should be released before that stage.
Available-memory readings are not reservations or whole-process bounds.
