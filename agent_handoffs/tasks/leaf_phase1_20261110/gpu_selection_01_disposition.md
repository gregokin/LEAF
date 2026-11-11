# First CUDA-check launch guard correction

The last successful device inventory before the check reported physical GPU 1
(`GPU-b02858c1-29c3-9ea9-33b1-831c33f8cf75`) at 0 MiB / 0% utilization.
An immediate pre-launch inventory attempted through a sandboxed subprocess then
failed with exit 255. Codex incorrectly continued to the bounded CUDA unit test
without conditioning that launch on the fresh probe's success.

The test process has exited and was reaped (receipt:
`candidate_01_uptake_gpu.exit.json`). A subsequent direct escalated inventory
succeeded and again reported GPU 1 at 0 MiB / 0%; the other devices retained
their existing allocations. No other job was signalled or modified.

The check executed 13 allocator tests and failed one test at
`tests/test_uptake.py:420`: multiplying a randomly reduced accepted amount by
1.01 did not necessarily exceed the proposal, so the expected rejection was
invalid. Earlier CUDA compilation and numerical comparisons in that test passed;
this is not complete CUDA qualification.

Future CUDA checks use `run_gpu_check.py`, which performs inventory and checks
idle status in the same controller before launching anything. A query failure,
missing device or occupied device records the refusal and exits without a launch.
The helper keeps GPU 1 fixed, records its actual UUID, and caps the fresh process's
CuPy pool at the existing 1.5-billion-byte budget.
