"""Verify real installed source origins, then execute independent analytical oracles."""
from pathlib import Path
import json,sys
from benchmarks import daily_water_growth as dw
from leaf import coupling
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st
actual={m.__name__:str(Path(m.__file__).resolve()) for m in [dw,coupling,ss,st]}
expected={dw.__name__:'/home/okin/leaf/benchmarks/daily_water_growth.py',coupling.__name__:'/home/okin/leaf/leaf/coupling.py',ss.__name__:'/home/okin/SYRUP/src/maple_syrup/subsurface.py',st.__name__:'/home/okin/SYRUP/src/maple_syrup/subsurface_transpiration.py'}
assert actual==expected,actual
print(json.dumps({'installed_origins':'pass','origins':actual}),flush=True)
import independent_oracles
independent_oracles.main()
