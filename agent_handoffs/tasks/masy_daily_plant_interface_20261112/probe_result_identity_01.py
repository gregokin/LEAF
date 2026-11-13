import importlib.util, json, pathlib, sys, types
import numpy as np
TASK=pathlib.Path(__file__).resolve().parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
e=load("native_example",TASK/"public_evidence/syrup_phase15/docs/manuals/examples/native_daily_recharge/daily_recharge_example.py")
b=load("bridge_under_review",TASK/"public_evidence/early_review_01/lifecycle_bridge.py")
conf=json.loads((TASK/"verification_environment.json").read_text())
args=types.SimpleNamespace(bundle=conf["cpu_bundle"],scenario="recharge",backend="numpy",device=0)
work=TASK/"public_evidence/result_probe_01";work.mkdir(exist_ok=False)
setup,profile=e.prepare_case(args,work);lop=setup.lifecycle_operator
from maple_syrup import water_lifecycle as wl, daily_water as dw
start=dw.start_daily_lineage(lop,setup.state,expected_parent=wl.expect_parent(setup.state),run_digest="cd"*32)
p=dw.next_plan(start,2.,10.,5.);pet=np.full(lop.n_cells,1e-8,dtype=np.float64)
a=dw.daily_water_action(lop,start,expected_parent=wl.expect_parent(start),plan=p,pet_terrain_m_s=pet).state
c=dw.daily_water_action(lop,start,expected_parent=wl.expect_parent(start),plan=p,pet_terrain_m_s=pet).state
read=b.read_availability(b.bind_source(profile.syrup_package_digest),lop,a)
identity="review-wrong-instance"
q=np.zeros((lop.n_layers,lop.n_cells));q[0]=1e-6
after1,feedback1=wl.request_withdrawal(lop,a,expected_parent=wl.expect_parent(a),depth_m=q,identity=identity)
q2=q*2
after2,feedback2=wl.request_withdrawal(lop,a,expected_parent=wl.expect_parent(a),depth_m=q2,identity=identity)
try:
 result=b.accepted_withdrawal(read,identity,after1,feedback2,lop=lop);accepted=True;error=None
except b.BridgeError as exc:
 accepted=False;error=str(exc)
ledger=wl.audit_lifecycle(lop,after1)
print(json.dumps({"mixed_successor_and_candidate_accepted":accepted,"native_accepted_transpiration_m3":ledger["transpiration_m3"],"reported_plant_accepted_m3":float(result.actual_total_m3) if accepted else None,"error":error},indent=2))
assert not accepted, "feedback from a different withdrawal candidate was paired with the accepted physical state"
