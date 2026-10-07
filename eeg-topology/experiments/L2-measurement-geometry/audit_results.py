"""Independent algebraic checks plus result/preregistration integrity checks."""
import sys
sys.dont_write_bytecode=True
import hashlib
import json
from pathlib import Path
import numpy as np
from common import ROOT, layouts, real_sh

def main():
    data={i:json.loads((ROOT/f't{i}_results.json').read_text()) for i in range(4)}
    checks={}
    for i,d in data.items():
        k='preregistration_sha256' if i==1 else 'prereg_sha256'
        checks[f'T{i}_prereg_hash_matches']=(d[k]==hashlib.sha256((ROOT/f'T{i}_PREREG.md').read_bytes()).hexdigest())
        if 'script_sha256' in d:
            name={1:'t1_quadrature.py',2:'t2_reference.py',3:'t3_information.py'}[i]
            checks[f'T{i}_script_hash_matches']=(d['script_sha256']==hashlib.sha256((ROOT/name).read_bytes()).hexdigest())
    layout=layouts()
    # SH addition theorem, independent of coefficient sampling or quadrature.
    sh_error=0.
    for item in layout.values():
        for L in (6,10):
            p=real_sh(item['unit'],L)
            sh_error=max(sh_error,float(abs(np.sum(p*p,axis=1)*4*np.pi/(L+1)**2-1).max()))
    checks['SH_addition_theorem']=sh_error<1e-12
    # Independent scalar-axis affine form of the change of minmax frame.
    drift_error=0.
    for name,result in data[0]['layouts'].items():
        x=layout[name]['xyz']; names=layout[name]['names']
        for label,case in result.items():
            cases=case['runs'] if label=='random50' else [case]
            for c in cases:
                keep=c.get('kept_indices')
                if keep is None: keep=[j for j,n in enumerate(names) if n not in c['deleted']]
                sub=x[keep]
                slopes=1/(np.ptp(sub,axis=0)+1e-8)-1/(np.ptp(x,axis=0)+1e-8)
                intercept=x.min(axis=0)/(np.ptp(x,axis=0)+1e-8)-sub.min(axis=0)/(np.ptp(sub,axis=0)+1e-8)
                delta=sub*slopes+intercept
                drift_error=max(drift_error,abs(float(abs(delta).max())-c['linf']),abs(float(abs(delta).mean())-c['mae']))
    checks['T0_independent_affine_recalculation']=drift_error<1e-12
    weights_error=0.
    for pair in data[1]['pairs'].values():
        for key in ('cap_weights','full_sphere_weights'):
            for full,sub in pair[key].values():
                for w in (full,sub):
                    weights_error=max(weights_error,abs(sum(w)-1))
                    assert min(w)>=0
    checks['T1_nonnegative_normalized_weights']=weights_error<1e-10
    checks['T1_orthonormal_cap_features']=data[1]['cap_basis_orthonormality_error']<1e-10
    checks['T1_grid_stability']=data[1]['grid_sensitivity']['relative_ratio_change']<.05
    checks['T2_graph_ranks']=all(g['rank_check_pass'] for g in data[2]['graphs'].values())
    checks['T3_invariance']=data[3]['decisions']['invertible_expression_passed']
    checks['T3_primary_exact_intersection_rank_crosschecks']=all(
        p['dimension_checks_agree'] and p['exact_common_dimension']==p['direct_H_stacked_rank_check_dimension']
        for mode in data[3]['modes'].values() for level in mode.values() for p in level['pairs'])
    def finite(v):
        if isinstance(v,dict): return all(finite(a) for a in v.values())
        if isinstance(v,list): return all(finite(a) for a in v)
        if isinstance(v,float): return bool(np.isfinite(v))
        return True
    checks['all_json_numbers_finite']=all(finite(d) for d in data.values())
    result=dict(checks=checks,passed=all(checks.values()),
                errors=dict(SH_addition_theorem=sh_error,T0_independent_affine=drift_error,weight_sum=weights_error))
    (ROOT/'audit_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']: raise SystemExit(1)

if __name__=='__main__': main()
