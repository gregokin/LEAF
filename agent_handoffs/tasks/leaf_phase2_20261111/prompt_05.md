# Phase2 revision5 — final actual-CUDA guard / fixture corrections

Resume exact session711c5973-64c3-45f8-af8a-379ba0e357a7, normal Read/Edit/Write,
read this prompt + candidate/** only. Narrow writes: leaf/demography.py,
tests/test_demography.py, tests/test_interevent.py, report.md. No execution,
outside reads, agents/settings/Git/network/install or other production changes.

R4 reaped/frozen; scope/toolerrors0. Codex ran full456 CPU (432pass/24GPUskips),
all11 independent refusal probes and real native biology/restart: PASS. R3's
independent signed-int64 IDs including endpoints / RNG / permutation also pass.
The final full456-test run on ACTUAL selected CUDA has2errors,1skip; every other
test passes. Preserve schema2, numerical laws, ID domain and corrected budgets.

1. tests/test_demography.py line546 deliberately passes NumPy data into a CUDA
   table; expected DemographyError, gets AttributeError because NumPy2.5 ndarray
   has device='cpu', so hasattr(device) does not establish a CuPy array. In
   demography.decide CUDA input guard use robust actual namespace/type validation
   before reading device.id, then validate dtype/shape/device normally. Foreign
   objects / NumPy arrays should fail clearly before kernels, not crash a guard.
   Do not coerce or upload wrong-namespace arrays silently.
2. test_interevent.Phase2CudaParity.test_cpu_cuda_phase2_parity_and_device_residency
   creates fixture(..., mortality=False), then line1974 asserts gpu._b.stress.device.
   stress is correctly None when that law is inactive. Fix the test's expectation;
   DO NOT allocate unused stress arrays to satisfy it. Retain inactive and active
   coverage: a genuinely active fixture must have device-resident counters/annual
   accumulators and work through an actual annual decision + own-backend restart.

All author checks NOT RUN; update report briefly with exact corrections. Code/
tests incremental, uncommitted. Codex reruns final actual CPU/CUDA/independent
fixtures before accepting and publishing whole Phase2. No new ecology or limits.
