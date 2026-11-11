# Pure demand/growth revision 2 — finite bounds, units and complete admission

Resume exact session f071b8f7-f17b-4046-88f9-25ed8aba25ee, preserve saved work.
User-authorized automatic source sharing/corrections; sole writer in THIS
candidate. Codex executes/reviews/integrates/publishes. READ only this prompt
and candidate/**; Read/Edit/Write only, no execution/Bash/agents/network/settings/
installs/Git or external reads. WRITES exactly leaf/growth.py,
tests/test_growth.py, examples/phase1/species.json, docs/phase1-growth.md, report.md.
No other implementation edits, no full phase-completion claim.

Exact original read-only context files: leaf/__init__.py, leaf/traits.py,
leaf/state.py, leaf/canopy.py, leaf/stepwat.py, tests/test_phase0.py,
examples/phase0/species.json and docs/stepwat-growth.md. The candidate is selected
SOURCE context, not a complete runnable package. Codex composes it with the full
accepted package for execution. Do not guess absent modules/tests again.

## Independent checks, not your runtime claims

Codex froze revision1, found zero scope violations, seven ordinary in-tree Read
errors (preserved), and composed it with the full package for review. CPU49 tests
pass with2CUDA skips; on the selected actual GPU all49 pass, including real CUDA
compilation and parity. The requested-water plan times delivered/requested law
works on those cases. Keep tests NOT RUN in your report: you cannot execute.

## Required corrections

1. **Nonfinite ledger bounds accepted.** Reproduced on NumPy and Numba:
   mature biomass=1e308, B=[1e308], E=D=U=[0], species_index=[0], live=[True],
   Aint=Tint=[.01,.01], ordinary update. The result has population_tolerance_kg=inf
   but population_closed=True. All reported packet totals themselves are finite,
   so _unfinite(pk) does not catch later aggregate scale/tolerance overflow.
   Validate EVERY computed ledger scale/bound/residual and enforce closure before
   returning. Reject nonfinite arithmetic or compute an equivalent finite bound
   stably; never certify inf/NaN or merely return population_closed=False. Apply
   the same rule to per-plant bounds/reference/intermediate values in every
   backend. Do not hide overflow by clipping normal/extra plans; known zero or
   above-mature growth may be short-circuited before unsafe divisions/products.
   Preserve accepted above-mature/no-growth behavior for ordinary finite values,
   and never loosen tolerances. Add regression on all available backends.

2. **All allocating public helpers need admission too.** species_response and
   integral_increments allocate S arrays but have no budget argument/admission;
   temperature_factor also accepts arbitrarily large arrays and broadcasts them.
   Their public behavior contradicts the complete preflight guarantee. Make the
   public temperature_factor scalar-only if useful, with a private pre-admitted
   vector routine. Add required max_bytes/optional extra_reserve_bytes to public
   S-sized response/integral helpers and itemized estimates, including live tables,
   outputs and all simultaneous S scratch BEFORE copies/conversion. Internal
   demand/grow may use a private routine after their already-complete admission.
   No arbitrary-array conversion before preflight. Add refusal/trap tests and
   keep docs/signatures consistent. Scalar helpers need no array allocation.

3. **Expose the time basis unambiguously.** The docs already recognize that
   support/deficit are kg*year and extra-growth coefficient has units1/year, but
   the new API calls these outputs *_support_kg / maintenance_deficit_kg and the
   species key says biomass_support_kg_per_m3. Before public acceptance rename
   the NEW law's coefficient to biomass_support_kg_year_per_m3, diagnostics to
   required/requested/actual_support_kg_year and maintenance_deficit_kg_year
   (including packet sum name), and extra_growth_fraction to
   extra_growth_rate_per_year (same qualified numeric range0..1 per year).
   This is a units/name correction, no numerical change. Update every new test,
   fixture and doc. Do not alter the OLD law/module/candidate-study records or
   claim existing fits are activated. Retain dry AGB kg versus biomass-support
   kg*year distinction; k is not a production yield or a physical maintenance
   percentage. Activity/temperature integrals numerically represent elapsed
   years weighted by activity/temperature, not carbon mass.

4. **Canonical backend naming.** The new allocator/native bridge use 'cuda'.
   Use canonical 'cuda' for this new module too (an explicit documented 'cupy'
   compatibility alias is optional), to avoid a coupled run silently selecting a
   different backend label. Keep exact device guards and no implicit fallback.
   This is naming, no algorithm or trait-law change.

Preserve all existing successful requested-water/S equations, two explicit
demand laws, bounded Euler rate range and common supplied canopy geometry.
Keep GPU data residency, bounded packets and O(P+S) memory. No growth-loop Python
population iteration on production backends. Save corrections incrementally;
update report with precise changes, limitations and all tests NOT RUN. Codex
will freeze and independently rerun tests/finite/admission/memory probes.
