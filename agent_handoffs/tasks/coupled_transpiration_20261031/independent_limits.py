"""Independent reserve/weekly admission oracles using the actual native column."""
from dataclasses import replace
from unittest import mock
import json,numpy as np
from independent_oracles import fixture,zero,host
from benchmarks import daily_water_growth as dw
from benchmarks import spatial_mapping as sm
from maple_syrup import subsurface as ss
from leaf.coupling import CoupledSimulation
available={'n':10**9}
case,col,state,inp,D=fixture('numpy',available=available)
large=replace(col,reserve_bytes=100_000_000)
small=dict(case,max_bytes=1_000_000)
with mock.patch.object(dw,'Simulation',side_effect=AssertionError('VEGETATION_ALLOCATED_BEFORE_RESERVE_ADMISSION')):
 try:CoupledSimulation(small,large,state,inputs=inp)
 except (ss.SubsurfaceMemoryError,sm.AdmissionRejected):pass
 else:raise AssertionError('Native reserve omitted from max_bytes')
sim=CoupledSimulation(case,col,state,inputs=inp)
ledger=sim.vegetation.last_ledger
available['n']=ledger['total_bytes']-1
try:sim._vegetation_admission(ledger)
except (ss.SubsurfaceMemoryError,sm.AdmissionRejected):pass
else:raise AssertionError('Full ledger peak exceeds availability')
available['n']=10**9
for i in range(6):sim.step_day(*zero(sim))
v=sim.vegetation;geometry={n:np.copy(host(getattr(v,n))) for n in ['B','canopy','root','Dp']}
with mock.patch.object(v,'_build_graph',side_effect=sm.AdmissionRejected('fixture','forced weekly latch',{})):
 try:sim.step_day(*zero(sim))
 except sm.AdmissionRejected:pass
 else:raise AssertionError('Expected weekly refusal')
assert sim.days==v.day==7 and sim.weekly_pending and v.weeks_done==0
accepted=sim.state;uptake=np.copy(host(v.uptake_cum));cum=dict(sim.cum)
available['n']=0
with mock.patch.object(v,'_build_graph',side_effect=AssertionError('NEW_GRAPH_ALLOCATED_WITH_ZERO_AVAILABILITY')):
 try:sim.retry_weekly_update()
 except (ss.SubsurfaceMemoryError,sm.AdmissionRejected):pass
 else:raise AssertionError('Expected native weekly refusal')
assert sim.state is accepted and sim.cum==cum and sim.days==v.day==7 and sim.weekly_pending
for n,a in geometry.items():np.testing.assert_array_equal(host(getattr(v,n)),a)
np.testing.assert_array_equal(host(v.uptake_cum),uptake)
available['n']=10**9;sim.retry_weekly_update()
assert not sim.weekly_pending and v.weeks_done==1 and sim.days==v.day==7
np.testing.assert_array_equal(host(v.uptake_cum),uptake)
assert sim.cum==cum and sim.state is accepted and sim.summary()['ok']
m=sim.memory_report();assert m['peak_required_bytes']>=m['vegetation_rebuild_peak_bytes']
assert m['peak_required_bytes']<=case['max_bytes']
print(json.dumps({'status':'pass','native_reserve_before_vegetation_allocation':True,'full_ledger_availability_enforced':True,'weekly_zero_availability_before_graph_allocation':True,'retry_no_duplicate_day_or_uptake':True,'overall_peak_includes_weekly_rebuild':True}))
