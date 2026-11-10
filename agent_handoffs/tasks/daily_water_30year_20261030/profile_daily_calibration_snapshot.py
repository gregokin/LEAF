"""Independent wall/memory instrumentation; does not change benchmark equations."""
from pathlib import Path
import contextlib
import gc
import hashlib
import json
import os
import sys
import time

PROCESS_START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmarks import daily_water_growth as dw
from benchmarks import spatial_mapping as sm


class AllocationPeak(sm.cp.cuda.MemoryHook if sm.cp is not None else object):
    name = "bounded_daily_water_allocation_peak"

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
            raise RuntimeError("duplicate live allocation in instrumentation")
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
    backend = sys.argv[sys.argv.index("--backend") + 1]
    gpu = backend in ("gpu", "both")
    if gpu and sm.cp is None:
        raise RuntimeError("GPU allocation profiling requires the existing CuPy environment")
    hook = AllocationPeak() if gpu else None
    original = dw.run_case

    def measured_case(*args, **kwargs):
        if hook:
            hook.case_peak_bytes = hook.live_bytes
            baseline = hook.live_bytes
        t0 = time.perf_counter()
        record = original(*args, **kwargs)
        wall = time.perf_counter() - t0
        if not isinstance(record, dict):
            raise TypeError("benchmark run_case must return its result dictionary")
        record["independent_case_wall_seconds"] = wall
        if hook:
            record["allocation_profile"] = {
                "cupy_live_rounded_peak_bytes": hook.case_peak_bytes,
                "cupy_live_rounded_at_case_start_bytes": baseline,
                "cupy_live_rounded_at_case_end_bytes": hook.live_bytes,
                "unknown_free_events_cumulative": hook.unknown_frees,
                "scope": "CuPy pool malloc/free hooks including temporaries; excludes driver/context and other allocators",
            }
        return record

    dw.run_case = measured_case
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    with hook if hook else contextlib.nullcontext():
        code = dw.main(sys.argv[1:])
        gc.collect()
    if not output.is_file():
        return code or 1
    result = json.loads(output.read_text())
    result["independent_profile"] = {
        "whole_process_wall_seconds_including_imports": time.perf_counter() - PROCESS_START,
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "gpu_uuid_selection": os.environ.get("CUDA_VISIBLE_DEVICES") if gpu else None,
        "cupy_live_rounded_peak_bytes": hook.peak_bytes if hook else None,
        "cupy_live_rounded_at_end_bytes": hook.live_bytes if hook else None,
        "unknown_free_events": hook.unknown_frees if hook else None,
        "allocation_events": hook.allocations if hook else None,
        "scope": "Actual rounded CuPy pool allocations only; reserved pool memory and CPU RSS reported separately.",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT / "benchmarks/daily_water_growth.py", ROOT / "benchmarks/spatial_mapping.py")},
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
