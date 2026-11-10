from unittest import mock
from independent_oracles import fixture,zero
from leaf.coupling import CoupledSimulation
from benchmarks import spatial_mapping as sm
from maple_syrup import subsurface as ss
available={'n':10**9}
case,col,state,inp,D=fixture('numpy',available=available)
sim=CoupledSimulation(case,col,state,inputs=inp)
for i in range(6):sim.step_day(*zero(sim))
with mock.patch.object(sim.vegetation,'_build_graph',side_effect=sm.AdmissionRejected('forced','latch fixture',{})):
 try:sim.step_day(*zero(sim))
 except sm.AdmissionRejected:pass
assert sim.days==7 and sim.weekly_pending
available['n']=0
with mock.patch.object(sim.vegetation,'_build_graph',side_effect=AssertionError('WEEKLY_ALLOCATION_REACHED_WITH_ZERO_NATIVE_AVAILABILITY')):
 try:sim.retry_weekly_update()
 except (ss.SubsurfaceMemoryError,sm.AdmissionRejected):print('PASS: native availability checked before weekly allocation')
 except AssertionError as e:print('CONFIRMED_REVISION3_DEFECT:',str(e))
 else:raise AssertionError('Zero availability admitted')
