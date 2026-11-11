"""Independent deterministic reproduction ledgers, identity, geometry, limits and malformed inputs."""
import json
from pathlib import Path
import numpy as np
from leaf import state,traits,recruitment as rp


def raw_catalog():
    raw=json.loads(Path('examples/phase1/species.json').read_text())
    for sid,item in raw['species'].items():
        item['reproduction']={'law':'simple_seed_clonal_v1','traits':{
            'seed':({'establishment_probability':1.,'max_recruits_per_patch':1,
                     'initial_agb_dry_kg':.02,'initial_seasonal_fraction':.5} if sid=='shrub' else None),
            'clonal':({'min_age_days':100.,'min_agb_dry_kg':.5,'probability':1.,
                       'max_daughters_per_parent':1,'initial_agb_dry_kg':.1,
                       'max_donation_fraction':.5,'dispersal_radius_m':.1,
                       'max_attempts_per_daughter':1} if sid=='grass' else None)}}
    return raw


def inputs(cat):
    return dict(live_ids=np.array([-5,0,10],np.int64),live_species=np.full(3,cat.index('grass'),np.int32),
        live_age_days=np.full(3,1000.),live_B=np.ones(3),live_E=np.full(3,.2),
        live_x=np.array([.5,1.5,2.5]),live_y=np.full(3,1.5),max_existing_id=10,n_total_records=3,
        seed=123,counter=0,max_new_plants=100,max_total_records=100,max_attempts=100,max_bytes=10_000_000)


def main():
    cat=traits.SpeciesCatalog(raw_catalog());table=rp.compile_species(cat,max_bytes=10_000_000)
    domain=state.Domain(4,4,1.);patch=rp.PatchConfig(2.);surface=np.arange(16,dtype=np.float64).reshape(4,4)
    data=inputs(cat)
    result=rp.evaluate(table,patch,domain,surface,**data);births=result['births']
    np.testing.assert_array_equal(births['ids'],np.arange(11,18,dtype=np.int64))
    np.testing.assert_allclose(result['parent_B_new'],.9,rtol=0,atol=1e-15)
    np.testing.assert_allclose(result['parent_E_new'],.18,rtol=0,atol=1e-15)
    np.testing.assert_array_equal(births['establishment_ground_m'],domain.sample(surface,births['x_m'],births['y_m']))
    np.testing.assert_array_equal(births['parent_id'][4:],[-5,0,10])
    assert result['diagnostics']['ledger']['closed']
    assert abs(result['diagnostics']['ledger']['e_transfer_residual_kg'])<1e-14
    rev={k:(v[::-1].copy() if k.startswith('live_') else v) for k,v in data.items()}
    reverse=rp.evaluate(table,patch,domain,surface,**rev)
    for name,value in births.items():np.testing.assert_array_equal(value,reverse['births'][name])
    fine=rp.evaluate(table,patch,state.Domain(8,8,.5),np.ones((8,8)),**data)
    coarse=rp.evaluate(table,patch,domain,np.ones((4,4)),**data)
    for name,value in coarse['births'].items():np.testing.assert_array_equal(value,fine['births'][name])
    reports=[{'check':'ledger/ground/signedIDs/roworder/physical-patch-resolution','pass':True}]
    failures=[]
    cases={'duplicate_IDs':{'live_ids':np.array([5,5,10],np.int64)},
           'max_existing_below_live_ID':{'max_existing_id':9},'total_records_below_live':{'n_total_records':0},
           'int64_exhausted':{'max_existing_id':2**63-1},'one_byte_cap':{'max_bytes':1},
           'wrong_ID_dtype':{'live_ids':np.array([-5,0,10],np.int32)},'boolean_limit':{'max_new_plants':True}}
    for name,changes in cases.items():
        bad=dict(data);bad.update(changes)
        try:rp.evaluate(table,patch,domain,surface,**bad)
        except rp.ReproductionError as exc:reports.append({'check':name,'refused':True,'message':str(exc)})
        except Exception as exc:failures.append({'check':name,'secondary_exception':type(exc).__name__,'message':str(exc)})
        else:failures.append({'check':name,'incorrectly_accepted':True})
    try:rp.evaluate(table,{},domain,surface,**data)
    except rp.ReproductionError:reports.append({'check':'wrong_patch_type','refused':True})
    except Exception as exc:failures.append({'check':'wrong_patch_type','secondary_exception':type(exc).__name__,'message':str(exc)})
    else:failures.append({'check':'wrong_patch_type','incorrectly_accepted':True})
    endpoint_raw=raw_catalog();endpoint_raw['species']['shrub']['reproduction']=None
    endpoint_cat=traits.SpeciesCatalog(endpoint_raw);endpoint_table=rp.compile_species(endpoint_cat,max_bytes=10_000_000)
    one=inputs(endpoint_cat)
    for name in list(one):
        if name.startswith('live_'):one[name]=one[name][:1].copy()
    one.update(max_existing_id=2**63-2,n_total_records=1)
    edge=rp.evaluate(endpoint_table,None,domain,surface,**one)
    np.testing.assert_array_equal(edge['births']['ids'],[2**63-1])
    reports.append({'check':'birth_at_last_signed_int64_ID','pass':True})
    for item in endpoint_raw['species'].values():
        if item['reproduction'] is not None:item['reproduction']['traits']['clonal']['probability']=0.
    zero_cat=traits.SpeciesCatalog(endpoint_raw);zero_table=rp.compile_species(zero_cat,max_bytes=10_000_000)
    no_birth=rp.evaluate(zero_table,None,domain,surface,**{**one,'max_existing_id':2**63-1})
    assert no_birth['births'] is None
    reports.append({'check':'max_ID_with_no_birth','pass':True})
    empty=inputs(cat)
    for name in list(empty):
        if name.startswith('live_'):empty[name]=empty[name][:0].copy()
    empty.update(max_existing_id=None,n_total_records=0)
    seeded=rp.evaluate(table,patch,domain,surface,**empty)
    np.testing.assert_array_equal(seeded['births']['ids'],np.arange(1,5,dtype=np.int64))
    reports.append({'check':'empty_population_seed_establishment','pass':True})
    print(json.dumps({'reports':reports,'failures':failures},indent=2))
    return int(bool(failures))

if __name__=='__main__':raise SystemExit(main())
