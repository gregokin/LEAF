"""Independent timing/allocation measurements of explicit native daily soil/plant steps."""
from pathlib import Path
import argparse
import contextlib
import gc
import hashlib
import json
import os
import resource
import time

PROCESS_START=time.perf_counter()
import numpy as np
from benchmarks import daily_water_growth as dw
from benchmarks import spatial_mapping as sm
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st
from leaf.coupling import CoupledSimulation

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

def prepare(case, backend):
    xp=sm.cp if backend=='cuda' else np
    N=case['nx']*case['ny'];L=case['layers']
    edges=[0.,.1,1.] if L==2 else [0.,.1,.3,1.]
    layers=ss.soil_layers(edges,theta_residual=.05,theta_field_capacity=.25,theta_saturation=.45,
                          diffusivity_m2_s=2e-7,theta_evaporation_stress=.2,evaporation_factor=1.2,
                          gravity_drainage=True)
    mask=xp.ones(N,dtype=bool)
    column=ss.prepare_column(layers,mask,case['dx']**2,backend=backend)
    del mask
    initial=ss.uniform_state(column,[.15]*L)
    zero=xp.zeros((L,N),dtype=np.float64)
    pulse=xp.zeros((L,N),dtype=np.float64);pulse[0]=.010
    pet=xp.full(N,3e-8,dtype=np.float64)
    # One of the two reusable intake fields is unused by each step and is an explicit caller reserve.
    sim=CoupledSimulation(case,column,initial,extra_reserve_bytes=zero.nbytes)
    return sim,zero,pulse,pet


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['numba','cuda'],required=True)
    parser.add_argument('--case',required=True)
    parser.add_argument('--days',type=int,default=365)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert args.days>0 and not args.output.exists()
    task=Path(__file__).resolve().parent
    case=next(c for c in json.loads((task/'performance_cases.json').read_text()) if c['name']==args.case)
    gpu=args.backend=='cuda'
    policy=None
    if gpu:
        pool=sm.cp.get_default_memory_pool()
        prior=pool.get_limit()
        chosen=min(prior,case['max_bytes']) if prior else case['max_bytes']
        pool.set_limit(size=chosen)
        policy={'pool_limit_bytes':chosen,'prior_limit_bytes':prior,'resource_cap_raised':False,
                'state_discarded':False,'scope':'CuPy pool only; context/other allocators excluded'}
    hook=AllocationPeak() if gpu else None
    reserved_peak=used_peak=0
    with hook if hook else contextlib.nullcontext():
        # Warm actual native hydrology/T and vegetation gather on a tiny coupled problem.
        tiny=dict(case,name='warm',nx=2,ny=2,dx=1.,grass_count=2,shrub_count=1,
                  grass_diameter=.2,shrub_diameter=.6,max_edges=10000,max_bytes=case['max_bytes'])
        t=time.perf_counter();w,z,p,e=prepare(tiny,args.backend);w.step_day(z,e)
        if gpu:sm.cp.cuda.Device().synchronize()
        del w,z,p,e;gc.collect();warm_s=time.perf_counter()-t
        t=time.perf_counter();sim,zero,pulse,pet=prepare(case,args.backend)
        if gpu:sm.cp.cuda.Device().synchronize()
        setup_s=time.perf_counter()-t
        progress=[];t=time.perf_counter()
        for day in range(args.days):
            sim.step_day(pulse if day%14==0 else zero,pet)
            if gpu:
                pool=sm.cp.get_default_memory_pool()
                reserved_peak=max(reserved_peak,int(pool.total_bytes()))
                used_peak=max(used_peak,int(pool.used_bytes()))
            if (day+1)%30==0 or day+1==args.days:
                report=sim.summary();assert report['ok'],report['checks']
                progress.append({'day':day+1,'water_residual_m3':report['water']['residual'],
                                 'water_bound_m3':report['water']['bound'],'checks':report['checks']})
        if gpu:sm.cp.cuda.Device().synchronize()
        loop_s=time.perf_counter()-t
        report=sim.summary();assert report['ok'],report['checks']
        counters=sim.vegetation.counters()
        result={'status':'ok','case':case,'backend':args.backend,'days':args.days,'summary':report,
                'counters':counters,'progress':progress,'timing':{'warm_compile_s':warm_s,'setup_s':setup_s,
                'loop_with_reports_s':loop_s,'ms_per_model_day':1000*loop_s/args.days,
                'model_years_per_wall_min':args.days/365/(loop_s/60)},
                'memory':{'peak_host_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                           'cupy_live_rounded_peak_bytes':hook.peak_bytes if hook else None,
                           'sampled_pool_reserved_peak_bytes':reserved_peak if gpu else None,
                           'sampled_pool_used_peak_bytes':used_peak if gpu else None,'allocator_policy':policy},
                'environment':dw.environment_info('gpu' if gpu else 'cpu'),
                'gpu_uuid_selection':os.environ.get('CUDA_VISIBLE_DEVICES') if gpu else None,
                'forcing_fixture':{'initial_theta':.15,'matrix_intake_depth_m':.010,'pulse_period_days':14,
                                   'potential_evaporation_m_s':3e-8,'recharge_computed_in_leaf':False,
                                   'note':'Explicit synthetic accepted-soil-intake forcing; no rainfall/infiltration/storm solver run.'}}
        del sim,zero,pulse,pet;gc.collect()
    result['memory']['cupy_live_at_end_bytes']=hook.live_bytes if hook else None
    result['memory']['unknown_free_events']=hook.unknown_frees if hook else None
    result['whole_process_timer_s']=time.perf_counter()-PROCESS_START
    result['source_sha256']={str(Path(m.__file__).resolve()):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                             for m in [dw,sm,ss,st,__import__('leaf.coupling',fromlist=[''])]}
    result['profiler_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'ok','case':args.case,'backend':args.backend,'days':args.days,
                      'loop_s':loop_s,'live_peak_bytes':result['memory']['cupy_live_rounded_peak_bytes']}),flush=True)


if __name__=='__main__':
    main()
