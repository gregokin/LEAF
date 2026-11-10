"""Independent small reproductions against an immutable native candidate snapshot."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
from unittest import mock
import numpy as np
from maple_syrup import subsurface as ss

T = Path(__file__).resolve().parent
SOURCE = T/'review_snapshots/revision02_native/subsurface_transpiration.py'
spec = importlib.util.spec_from_file_location('review_native_transpiration', SOURCE)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

layers = ss.soil_layers([0,.1], theta_residual=.05,theta_field_capacity=.25,theta_saturation=.45,
                        diffusivity_m2_s=0,theta_evaporation_stress=.20,evaporation_factor=1,
                        gravity_drainage=False)
available = {'n':10**9}
column = ss.prepare_column(layers,np.array([True,False]),1.,backend='numpy',available_bytes_fn=lambda:available['n'])
state = ss.column_state(column,np.array([[.02,.03]]))
op = m.prepare_transpiration(column)
q = np.array([[.005,0.]])
r = m.withdraw_transpiration(op,state,q)
mask_bug = r.budget.storage_before_m3 != ss.storage_m3(column,state)
available['n'] = 0
calls=[]
original=np.where

def tracked(*a,**kw):
    calls.append('where allocated before rejection')
    return original(*a,**kw)

with mock.patch.object(np,'where',side_effect=tracked):
    try:m.withdraw_transpiration(op,state,q)
    except ss.SubsurfaceMemoryError:pass
    else:raise AssertionError('Expected memory rejection')
negative_admitted = False
try:
    x=m.withdraw_transpiration(op,state,q,extra_reserve_bytes=-10**12)
    negative_admitted=True
except (ss.SubsurfaceError,ss.SubsurfaceMemoryError):pass
out={'snapshot_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
     'inactive_storage_budget_mismatch':mask_bug,
     'native_active_storage_m3':ss.storage_m3(column,state),
     'withdrawal_reported_storage_m3':r.budget.storage_before_m3,
     'allocation_before_memory_rejection':bool(calls),
     'negative_reserve_bypassed_zero_available':negative_admitted}
assert mask_bug and calls and negative_admitted,out
print(json.dumps(out,indent=2))
