# Interim independent review — draft_01 (not final acceptance)

Stable source capture while R1 writer continues; manifest identifies exact bytes. Independent probe_recruitment_01.py ran through run_check, exit1 (check_failed). Ledger, ground samples, row permutation, physical patch invariance under hydraulic refinement, signed IDs, last-int64 birth, no-birth max-ID and empty-population seeds pass. Four public-input guards fail:

1. Duplicate live_ids accepted (e.g.[5,5,10]); canonical IDs must be unique.
2. max_existing_id=9 accepted with live ID10, allowing collision when the first newborn gets10. Refuse inconsistent global-ID metadata.
3. n_total_records=0 accepted with three live rows. Require total>=live and consistent empty metadata.
4. Wrong patch type{} causes secondary AttributeError rather than ReproductionError.

Also review metadata-before-allocation: evaluate calls np.ndim/np.shape on live_ids before confirming an ndarray, so a Python list can allocate conversion scratch before admission. Public scalar metadata/count products should be validated strictly and without integer overflow. This latter point is source-review concern, not yet an executed failing guard.

Primary stdout/stderr/launch/exit artifacts use draft01_reproduction prefix. No correction has been dispatched yet; R1 remains sole implementation writer. No live production changes.
