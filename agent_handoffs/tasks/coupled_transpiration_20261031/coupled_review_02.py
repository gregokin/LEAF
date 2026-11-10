"""Independent ordering and gather-metadata reproductions of the complete candidate."""
from pathlib import Path
import hashlib
import json
from unittest import mock
import numpy as np
from maple_syrup import subsurface as ss
from leaf.coupling import CoupledSimulation
from benchmarks import daily_water_growth as dw

case={'name':'review','nx':1,'ny':1,'dx':1.,'grass_count':1,'shrub_count':0,
      'grass_diameter':.1,'shrub_diameter':1.,'root_multiplier':3.,'pattern':'uniform',
      'seed':1,'layers':2,'max_edges':100,'max_bytes':50_000_000}
layers=ss.soil_layers([0,.1,.3],theta_residual=.05,theta_field_capacity=.25,theta_saturation=.45,
                      diffusivity_m2_s=0,theta_evaporation_stress=.2,gravity_drainage=False)
a={'bytes':10**9}
col=ss.prepare_column(layers,np.ones(1,bool),1.,available_bytes_fn=lambda:a['bytes'])
state=ss.uniform_state(col,[.2,.2])
a['bytes']=0
reached=[]

def reached_before_native_guard(*args,**kwargs):
    reached.append(True)
    raise RuntimeError('vegetation allocation reached before native zero-memory refusal')

with mock.patch.object(dw,'Simulation',side_effect=reached_before_native_guard):
    try:CoupledSimulation(case,col,state)
    except (RuntimeError,ss.SubsurfaceMemoryError):pass
assert reached,'Constructor now guards correctly; update this historical reproduction'
v=dw.Simulation(case,vegetation_only=True)
accepted=False
try:
    v.gather_uptake_candidate(np.ones((2,1),np.float32),np.zeros(1,np.float64))
    accepted=True
except (TypeError,ValueError):pass
assert accepted,'Gather now rejects wrong dtype; update this historical reproduction'
print(json.dumps({'vegetation_allocates_before_native_zero_memory_guard':bool(reached),
                  'gather_accepts_float32_scale':accepted,
                  'source_sha256':{str(Path(x).resolve()):hashlib.sha256(Path(x).read_bytes()).hexdigest()
                     for x in [dw.__file__,__import__('leaf.coupling',fromlist=['']).__file__]}},indent=2))
