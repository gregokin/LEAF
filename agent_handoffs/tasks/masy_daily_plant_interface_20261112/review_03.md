# Independent review of revision 3 — CORRECT (narrow final items)

Frozen review_03 and candidate_03_manifest.json. All writes/reads in scope, no tool errors, structured success and exit0 reaped.
Focused CPU: 76 passed /1 failed /4 unconfigured native skips /43 subtests. Remaining test expects resource_cursor=None after a dry day; fresh() now explicitly connects the initial resource, so it must assert the cursor stays unchanged rather than disappearing. Production behavior is correct.

Independent REAL N15 eight-day harness PASS on CPU and actual CUDA. Eight daily hydrology operations, eight accepted T debits, exactly691200 native seconds, one weekly growth, day8 T-on/G-off no credit, injected weekly geometry failure leaves parent unchanged. Pending-after-debit save/restore retries biology without water replay, closed-water-boundary pairing round-trips, closure inside native bounds. Tracemalloc peaks 1364377/1445961 bytes (includes Python, excludes startup); CUDA pool live peak158720 bytes (native+LEAF, excludes context). These tiny fixtures are interface qualification, not ecological validation or domain-scale performance.

Remaining HIGH restart issue: see review_02_rng_addendum.md / probe_revision02_rng.py. Current checkpoint stores only rng_seed_set, not the actual seed/patch. Changed RNG seed or ecological patch size restores successfully with SAME token. Bind/persist these settings and reject changed caller configuration. Preserve values, no new law.

Documentation still lacks the requested complete public signatures/shape/units/backend reference, and says CPU/CUDA memory/parity are future work. Add exact argument/return contract and point qualification to the independent task evidence. Author must not claim to have executed. Correct stale revision title and >= request-sequence sentence. No code accepted/installed yet.
