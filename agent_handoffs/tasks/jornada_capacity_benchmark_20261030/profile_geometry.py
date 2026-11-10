"""Independent measurement wrapper for the bounded synthetic benchmark.

Counts live rounded CuPy pool allocations, including temporary arrays, through
the installed MemoryHook API. This excludes driver/context allocations and
allocations outside that pool. It does not implement model physics.
"""
from pathlib import Path
import gc
import hashlib
import json
import os
import sys
import time

PROCESS_START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmarks import spatial_mapping as sm


class AllocationPeak(sm.cp.cuda.MemoryHook):
    name = "bounded_geometry_allocation_peak"

    def __init__(self):
        self.active = {}
        self.live_bytes = 0
        self.peak_bytes = 0
        self.case_peak_bytes = 0
        self.unknown_frees = 0
        self.allocations = 0

    def malloc_postprocess(self, **kwargs):
        if not kwargs["mem_ptr"]:
            return
        key = (kwargs["device_id"], kwargs["pmem_id"])
        if key in self.active:
            raise RuntimeError("allocation instrumentation saw a duplicate live object")
        size = int(kwargs["mem_size"])
        self.active[key] = size
        self.live_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.live_bytes)
        self.case_peak_bytes = max(self.case_peak_bytes, self.live_bytes)
        self.allocations += 1

    def free_postprocess(self, **kwargs):
        key = (kwargs["device_id"], kwargs["pmem_id"])
        if key not in self.active:
            self.unknown_frees += 1
            return
        self.live_bytes -= self.active.pop(key)


def main():
    hook = AllocationPeak()
    original_case = sm.run_case

    def profiled_case(*args, **kwargs):
        hook.case_peak_bytes = hook.live_bytes
        baseline = hook.live_bytes
        record = original_case(*args, **kwargs)
        record["allocation_profile"] = {
            "cupy_live_rounded_peak_bytes": hook.case_peak_bytes,
            "cupy_live_rounded_at_case_start_bytes": baseline,
            "cupy_live_rounded_at_case_end_bytes": hook.live_bytes,
            "unknown_free_events_cumulative": hook.unknown_frees,
            "scope": "CuPy pool malloc/free hooks, including transient arrays; excludes CUDA context and other allocators",
        }
        return record

    sm.run_case = profiled_case
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    with hook:
        code = sm.main(sys.argv[1:])
        gc.collect()
    result = json.loads(output.read_text())
    cpu_model = next(
        (line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
         if line.startswith("model name")), "unknown"
    )
    result["independent_profile"] = {
        "whole_process_wall_seconds_including_imports": time.perf_counter() - PROCESS_START,
        "cpu_model": cpu_model,
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "gpu_uuid_selection": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cupy_live_rounded_peak_bytes": hook.peak_bytes,
        "cupy_live_rounded_at_end_bytes": hook.live_bytes,
        "unknown_free_events": hook.unknown_frees,
        "allocation_events": hook.allocations,
        "scope": "Allocation-hook peak is exact for observed rounded pool allocations; excludes CUDA context and other allocators. Stage pool reservation is recorded separately.",
        "source_sha256": hashlib.sha256((ROOT / "benchmarks/spatial_mapping.py").read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
