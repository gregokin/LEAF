"""Independent reserve-bound reproductions; no production writes."""
from dataclasses import replace
from independent_oracles import fixture
from leaf.coupling import CoupledSimulation
from maple_syrup import subsurface_transpiration as st
available={'n':10**9}
case,col,state,inp,D=fixture('numpy',available=available)
large=replace(col,reserve_bytes=100_000_000)
small=dict(case,max_bytes=1_000_000)
try:sim=CoupledSimulation(small,large,state,inputs=inp)
except Exception as e:print('reserve refused:',type(e).__name__)
else:print('CONFIRMED_RESERVE_DEFECT',{'cap':small['max_bytes'],'reported_peak':sim.memory_report()['peak_required_bytes']})
sim=CoupledSimulation(case,col,state,inputs=inp)
ledger=sim.vegetation.last_ledger
part=ledger['total_bytes']-ledger['external_reserve_bytes']
comb=st.estimate_combined_memory(col,extra_reserve_bytes=part+sim.buffer_bytes+sim.extra_reserve_bytes,query_available=False)
peak=max(comb.required_bytes,ledger['total_bytes']+col.reserve_bytes)
assert peak>comb.required_bytes
available['n']=(peak+comb.required_bytes)//2
try:sim._vegetation_admission(ledger)
except Exception as e:print('gap refused:',type(e).__name__)
else:print('CONFIRMED_AVAILABILITY_GAP',{'available':available['n'],'admitted_peak':peak,'checked_smaller_bound':comb.required_bytes})
