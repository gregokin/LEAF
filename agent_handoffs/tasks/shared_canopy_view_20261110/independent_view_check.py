"""Freeze legacy outputs, then independently verify shared geometry and compatibility."""
from pathlib import Path
from unittest.mock import patch
import importlib.abc
import json
import sys
import numpy as np

TASK = Path(__file__).resolve().parent
ROOT = Path('/home/okin/leaf')


class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numba', 'cupy', 'maple_syrup', 'maple', 'benchmarks'}:
            raise AssertionError('Shared canopy imported a heavy/native module: '+fullname)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k:plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [plain(v) for v in value]
    return value


def main():
    mode = sys.argv[1]
    sys.meta_path.insert(0, BlockNative())
    from leaf import canopy, state, traits
    catalog = traits.SpeciesCatalog.from_file(ROOT/'examples/phase0/species.json')
    domain = state.Domain(4,4,1.0,origin_x_m=5.0,origin_y_m=-3.0)
    records=[]
    for pid,sp,x,y,B,status in [(11,'grass',5.5,-2.5,.2,'live'),
                                (22,'shrub',7.5,-.5,4.0,'live'),
                                (33,'grass',8.5,.5,.05,'dead_standing'),
                                (44,'shrub',6.5,-1.5,1.0,'removed')]:
        records.append(dict(plant_id=pid,species=sp,x_m=x,y_m=y,
                            establishment_ground_m=10.0,agb_dry_kg=B,
                            seasonal_agb_dry_kg=0.0,age_days=100,status=status))
    population=state.PlantState.from_records(catalog,domain,records)
    flat=state.constant_surface(domain,10.0); partial=flat.copy();partial[0,0]+=.01
    disturbed=partial.copy();disturbed[2,2]-=.5;disturbed[3,3]+=100.0
    cases=[('flat',flat,{}),('partial',partial,{}),
           ('disturbed_flag',disturbed,{'unsupported_policy':'flag'}),
           ('disturbed_exclude',disturbed,{'unsupported_policy':'exclude'}),
           ('diagnostic_flag',disturbed,{'unsupported_policy':'flag','height_mode':'intact_diagnostic'})]
    reference_path=TASK/'legacy_outputs.json'
    if mode=='freeze':
        with reference_path.open('x') as out:
            json.dump({name:plain(canopy.maple_vegetation_view(population,catalog,surface,**options))
                       for name,surface,options in cases},out,indent=2);out.write('\n')
        print(json.dumps({'pass':True,'legacy_cases_frozen':len(cases)}));return
    reference=json.loads(reference_path.read_text())
    before=[getattr(population,key).copy() for key in population.ARRAY_NAMES]
    results={}
    for name,surface,options in cases:
        old=reference[name]
        with patch.object(canopy,'canopy_view',wraps=canopy.canopy_view) as shared_call, \
             patch.object(canopy,'derive_geometry',wraps=canopy.derive_geometry) as geometry_call:
            legacy=canopy.maple_vegetation_view(population,catalog,surface,**options)
            assert shared_call.call_count==1,(name,'adapter must call shared view exactly once')
            assert geometry_call.call_count==1,(name,'geometry must be derived once')
        for section in ['fields','sidecar']:
            for key,expected in old[section].items():
                np.testing.assert_array_equal(legacy[section][key],expected,err_msg=name+'/'+section+'/'+key)
        for key in ['schema','schema_version','n_objects','fields','coordinates','domain',
                    'catalog_hash','time_days','height_mode','native_maple_valid',
                    'statuses_included','excluded_by_status','excluded_zero_diameter',
                    'unsupported_policy','n_unsupported','n_unsupported_excluded',
                    'n_unsupported_flagged_kept']:
            assert plain(legacy['meta'][key])==old['meta'][key],(name,'legacy metadata changed',key)
        generic=canopy.canopy_view(population,catalog,surface,
                                  height_mode=options.get('height_mode','exposed'))
        assert generic['meta']['schema']=='leaf.canopy_view'
        assert generic['meta']['schema_version']==1
        np.testing.assert_array_equal(generic['sidecar']['plant_id'],[11,22,33])
        lookup={int(pid):i for i,pid in enumerate(generic['sidecar']['plant_id'])}
        selected=[lookup[int(pid)] for pid in legacy['sidecar']['plant_id']]
        for key,value in legacy['fields'].items():
            np.testing.assert_array_equal(value,generic['fields'][key][selected],err_msg=name+'/'+key)
        for key,value in legacy['sidecar'].items():
            np.testing.assert_array_equal(value,generic['sidecar'][key][selected],err_msg=name+'/'+key)
        if name=='disturbed_flag':
            np.testing.assert_array_equal(generic['sidecar']['fully_buried'],[False,False,True])
            np.testing.assert_array_equal(generic['sidecar']['elevated_base'],[False,True,False])
            assert generic['fields']['height_m'][2]==0.0
            assert generic['sidecar']['exposure_depth_m'][1]==.5
            np.testing.assert_array_equal(generic['sidecar']['status'],['live','live','dead_standing'])
        results[name]=True
    for key,expected in zip(population.ARRAY_NAMES,before):
        np.testing.assert_array_equal(getattr(population,key),expected)
    assert canopy.CANOPY_FIELDS==canopy.MAPLE_FIELDS
    print(json.dumps({'pass':True,'legacy_cases':results,'shared_single_geometry_call':True,
                      'same_geometry_by_id_for_both_consumers':True,
                      'buried_and_exposed_retained_without_mortality':True,
                      'state_unmodified':True,'native_imports_blocked':True},indent=2))


if __name__=='__main__':
    main()
