# Independent review — revision 2: CORRECT

Ordinary tool scope passes; same session reaped with structured success. CPU full suite:727 tests,674 passed,53 skips. Nonperiodic golden and adaptive canopy union/tangency integration pass. Synthetic recruitment memory check:25,079 daughters/1,000 parents; peak traced6.62MB periodic and8.01MB default, both below9.09MB conservative admitted arrays; interior birth arrays byte-identical and biomass ledger closes.

Two remaining boundary correctness issues, reproduced independently outside author tests:

1. Exact upper Domain.wrap_positions still fails5 of48 decimal-grid cases. Example Domain(3,2,.1,origin_x_m=.3,origin_y_m=-.3,periodic=(True,True)): x_max=.6000000000000001; wrap returns.30000000000000004 rather than exact lower.3. Checking only the RESULT against vmax misses input subtraction rounded ABOVE span. Canonicalize ORIGINAL input v==vmax to origin explicitly, as well as result rounding; preserve every ordinary interior value exactly. Sampling, public canopy centers and root/PET mapper upper-alias comparisons already pass these48 cases.
2. Mixed-boundary canopy PET refuses5 of32 legal upper-corner cases: e.g. nx3,ny7,dx.1,origin(.3,-.3),periodic(False,True). Its nonperiodic local x rounds slightly above W and fails local-bound validation. In the new any-periodic branch validate both ORIGINAL landscape axes against Domain bounds, then canonicalize periodic upper aliases and clip rounding on the closed nonperiodic local axes. Do not accept an originally outside position. Keep the default nonperiodic branch/golden behavior. Approved independent oracle verify_periodic_mixed_endpoints.py captures all32 cases.

Do not weaken oracles. Tests/CUDA/native not accepted yet. Source-law/memory/doc corrections from prior review are otherwise satisfactory. Historical data/locks and upstreams unchanged; Phase3 remains paused.
