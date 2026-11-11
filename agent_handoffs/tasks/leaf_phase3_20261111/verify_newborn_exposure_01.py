"""Independent newborn background-hazard golden values for all three backends."""
import json,math,sys
from pathlib import Path
import numpy as np
from leaf import traits,demography as dm

def host(a):return a.get() if hasattr(a,'get') else np.asarray(a)

def main():
    backend=sys.argv[1] if len(sys.argv)>1 else 'numba'
    xp=np
    if backend=='cuda':import cupy as xp
    raw=json.loads(Path('examples/phase2/species.json').read_text())
    for item in raw['species'].values():
        item['mortality']={'law':'age_background_stress_v1','traits':{
            'age_hazard':None,'background_hazard_per_year':math.log(2.),'stress':None}}
    cat=traits.SpeciesCatalog(raw)
    table=dm.compile_species(cat,backend,device=0 if backend=='cuda' else None,max_bytes=10_000_000)
    ages=np.array([0.,1.,30.,365.,730.])
    p=len(ages)
    z=xp.zeros(p,dtype=np.float64)
    result=dm.decide(table,species_index=xp.full(p,cat.index('grass'),dtype=np.int32),
        plant_ids=np.array([-2**63,-1,0,1,2**63-1],dtype=np.int64),age_end_days=ages,
        stress=xp.zeros(p,dtype=np.int32),required_support=z,actual_support=z,planned_growth=z,actual_growth=z,
        period_days=365.,seed=12,counter=0,max_bytes=10_000_000)
    expected=np.array([-math.expm1(-math.log(2.)*min(float(age),365.)/365.) for age in ages])
    np.testing.assert_allclose(host(result['p_age']),expected,rtol=2e-14,atol=1e-16)
    assert host(result['cause'])[0]==dm.CAUSE_NONE,'zero-age newborn was killed by time before birth'
    print(json.dumps({'backend':backend,'ages_days':ages.tolist(),'probabilities':host(result['p_age']).tolist(),
                      'golden_probabilities':expected.tolist(),'prebirth_exposure_excluded':True},indent=2))

if __name__=='__main__':main()
