"""Compare default standalone mode with the preserved pre-task implementation."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
from benchmarks import daily_water_growth as current
p=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('leaf_pre_coupling_benchmark',p/'baseline_sources/benchmarks/daily_water_growth.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
case={'name':'legacy','nx':12,'ny':8,'dx':.5,'grass_count':20,'shrub_count':3,'grass_diameter':.15,'shrub_diameter':1.,'root_multiplier':3.,'pattern':'uniform','seed':19,'layers':3,'max_edges':100000,'max_bytes':100000000}
a=old.Simulation(case,'cpu');b=current.Simulation(case,'cpu')
for n in [7,7,9]:
 a.step_days(n);b.step_days(n)
 assert a.counters()==b.counters(),(a.counters(),b.counters())
 for name in old._LIVE_NAMES:np.testing.assert_array_equal(getattr(a,name),getattr(b,name),err_msg=name)
 for name in ['cells','weights','offsets','counts']:np.testing.assert_array_equal(a.graph[name],b.graph[name],err_msg=name)
 assert a.summary()==b.summary()
print(json.dumps({'status':'pass','days':a.day,'weeks':a.weeks_done,'all_live_arrays_bit_identical':True,'graphs_bit_identical':True,'counters_and_summary_identical':True}))
