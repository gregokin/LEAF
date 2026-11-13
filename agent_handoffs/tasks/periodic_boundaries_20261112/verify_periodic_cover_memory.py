"""Independent lifetime check of large wrapped footprints over several row tiles."""
from pathlib import Path
import json,os,tracemalloc
import numpy as np
from leaf import state,canopy
# One synthetic circle only: no model/case or plant generation, no population tensor.
d=state.Domain(256,256,.25,periodic=(True,True));x=np.array([0.]);y=x.copy();r=np.array([80.]);working=4<<20
tracemalloc.start();tracemalloc.reset_peak()
report=canopy.measure_cover(d,x,y,r,subsamples=4,max_cover_cells=2<<20,max_cover_bytes=4<<20,max_working_bytes=working)
_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
declared=report['raster_bytes']+report['working_bytes']+report['index_bytes']
print(json.dumps({'traced_peak_bytes':peak,'declared_array_peak_bytes':declared,'working_budget_bytes':working,'raster_bytes':report['raster_bytes'],'raster_shape':report['raster_shape'],'rows_per_chunk':report['rows_per_chunk'],'cover':report['total_union_fraction'],'scope':'tracemalloc includes Python metadata plus NumPy; 32KiB metadata tolerance; borrowed circle inputs excluded'},indent=2))
assert report['total_union_fraction']==1.
assert peak<=declared+(32<<10),'Wrapped row tiles retain an extra mask beyond declared array memory'
