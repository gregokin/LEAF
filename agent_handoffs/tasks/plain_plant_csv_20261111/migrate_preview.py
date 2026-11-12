"""One-time storage conversion: preserve exact plant rows and original render provenance."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import io
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
from leaf import state, traits

TASK=Path(__file__).resolve().parent
PREVIEW=ROOT/'cases/jornada_grass05_shrub10_256/preview'

def sha(data): return 'sha256:'+hashlib.sha256(data).hexdigest()

def main():
    before=json.loads((TASK/'preview_before.json').read_text())
    p=PREVIEW/'initial_plants.csv'; mp=Path(state.metadata_path(p))
    assert not mp.exists(), 'Never overwrite a previous conversion'
    raw=p.read_bytes(); head,body=raw.split(b'\n',1)
    assert head.startswith(b'# ')
    assert sha(body)=='sha256:'+before['plant_table_sha256']
    cat=traits.load_catalog(PREVIEW.parent/'species.json')
    old=state.parse_plant_csv(raw,cat)
    assert state.plant_csv_text(old,initial=True).encode()==body, 'Plant table serialization changed'
    meta=state.plant_metadata(old,body,initial=True,provenance=old.provenance,csv_file=p.name)
    meta_bytes=state.plant_metadata_text(meta).encode()
    state.publish_plant_files(p,body,meta_bytes,overwrite=True)
    assert p.read_bytes()==body
    new=state.read_plant_csv(p,cat)
    for field in state.PlantState.ARRAY_NAMES:
        np.testing.assert_array_equal(getattr(old,field),getattr(new,field),err_msg=field)
    assert (new.n,new.time_days,new.catalog_hash)==(25589,0.0,old.catalog_hash)
    assert new.provenance==old.provenance
    for key,value in before['embedded_metadata'].items():
        if key=='schema': continue
        assert meta[key]==value,key
    rows=list(csv.reader(io.StringIO(body.decode()),strict=True))
    assert rows[0]==list(state.CSV_COLUMNS)
    assert len(rows)==25590 and all(len(row)==9 for row in rows)
    preserved={}
    for rel,expected in before['preserved_files'].items():
        current=sha((ROOT/rel).read_bytes())
        assert current=='sha256:'+expected['sha256'],rel
        preserved[rel]=current
    original_provenance=PREVIEW/'preview_provenance.json'
    record={
        'schema':'leaf.plant_file_storage_migration','schema_version':1,
        'task':'plain_plant_csv_20261111','converted_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'STORAGE_ONLY_NO_MODEL_EVOLUTION',
        'statement':'Removed the legacy embedded metadata line from the existing preview CSV; saved its context in a separate companion. Exact header/plant-row bytes, all arrays, model time, parameters, terrain and image are unchanged. No initialization, growth, hydrology, transport or rendering was run.',
        'model_time_days':new.time_days,'n_plants':new.n,'plant_columns':list(state.CSV_COLUMNS),
        'original_csv_sha256':sha(raw),'plant_table_sha256_before':sha(body),
        'csv':{'file':p.name,'sha256':sha(p.read_bytes()),'n_bytes':p.stat().st_size},
        'metadata':{'file':mp.name,'sha256':sha(mp.read_bytes()),'schema':meta['schema']},
        'original_render_provenance':{'file':original_provenance.name,'sha256':sha(original_provenance.read_bytes()),'preserved_unchanged':True,'note':'Historical render record contains the pre-conversion CSV hash; this addendum records the current pair.'},
        'exact_plant_row_bytes_preserved':True,'all_plant_arrays_equal':True,
        'strict_nine_column_csv':True,'preserved_files':preserved,
        'writer_sources':{name:sha((ROOT/name).read_bytes()) for name in ['leaf/state.py']},
    }
    state.atomic_write_text(PREVIEW/'plant_files_migration.json',json.dumps(record,indent=2,sort_keys=True)+'\n')
    (TASK/'preview_migration_verification.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'n_plants':new.n,'time_days':new.time_days,'csv':record['csv'],'metadata':record['metadata'],'plant_rows_unchanged':True,'image_unchanged':True}))

if __name__=='__main__':main()
