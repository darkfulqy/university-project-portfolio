#!/usr/bin/env python3
"""T3: signed measurement information and exact/approximate observability.
Run only after reading T3_PREREG.md. No downloads; sibling environment read only.
"""
from pathlib import Path
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('_MNE_FAKE_HOME_DIR', str(Path(__file__).resolve().parent / 'runtime-home'))
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import scipy
from scipy.linalg import helmert, solve_triangular
import common

HERE = Path(__file__).resolve().parent
SEED = 20260930
LEVELS = [1, 2, 3, 4, 6, 8, 10]
NOISES = [.01, .05, .2, .5, 1.]
RTOL = 1e-10


def np_json(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    raise TypeError(type(x).__name__)


def numerical_rank(s, tol=RTOL):
    return int(np.sum(s > tol * s[0])) if s.size and s[0] else 0


def spectral(H):
    _, s, vt = np.linalg.svd(H, full_matrices=False)
    r = numerical_rank(s)
    return dict(q=vt[:r].T, s=s[:r], all_s=s, rank=r,
                sensitivity={str(t): numerical_rank(s,t) for t in [1e-8,1e-10,1e-12]})


def subspace_pair(q1, q2):
    u, co, vt = np.linalg.svd(q1.T @ q2, full_matrices=False)
    co = np.clip(co, 0, 1)
    degrees = np.rad2deg(np.arccos(co))
    common_mask = co >= 1-1e-10
    common_dim = int(common_mask.sum())
    # The same dimension is checked independently with joined orthonormal bases.
    joined_s = np.linalg.svd(np.concatenate([q1, q2], axis=1), compute_uv=False)
    joined_rank = numerical_rank(joined_s)
    checked_dim = q1.shape[1] + q2.shape[1] - joined_rank
    if common_dim:
        # Averaging paired principal vectors makes residual error symmetric.
        candidate = q1 @ u[:,common_mask] + q2 @ vt.T[:,common_mask]
        basis, _ = np.linalg.qr(candidate, mode='reduced')
        residual = max(np.linalg.norm(basis-q@q.T@basis, ord=2) for q in [q1,q2])
    else:
        basis = np.empty((q1.shape[0],0))
        residual = 0.
    summary = dict(exact_common_dimension=common_dim,
                   orthonormal_stacked_rank_check_dimension=int(checked_dim),
                   dimension_checks_agree=bool(common_dim==checked_dim),
                   intersection_basis_max_projection_residual=float(residual),
                   principal_angles_degrees=degrees,
                   approximate_angle_counts={str(a):int(np.sum(degrees<=a)) for a in [5,15,30]})
    return summary, basis


def make_model(layout, L, mode):
    bipolar = 'A' in layout and layout['A'] is not None
    if bipolar:
        unit = np.asarray(layout.get('electrode_unit',layout.get('unit')))
        A = np.asarray(layout['A'],dtype=float)
        if len(unit) != A.shape[1]:
            raise ValueError('TCP requires electrode_unit matching signed A columns; midpoints are invalid')
    else:
        unit = np.asarray(layout['unit'])
        A = np.eye(len(unit)) if mode=='raw' else helmert(len(unit),full=False)
    Phi = np.asarray(common.real_sh(unit,L))
    # common.real_sh is scipy-orthonormal (surface area measure); convert to probability measure.
    Phi = Phi*np.sqrt(4*np.pi)
    if mode=='contrast': Phi=Phi[:,1:]
    H = A@Phi
    distances = np.linalg.norm(unit[:,None]-unit[None,:], axis=-1)
    K = .7*np.eye(len(unit)) + .3*np.exp(-distances**2/(2*.35**2))
    base_sigma = A@K@A.T + .05*np.eye(len(A))
    ch = np.linalg.cholesky(base_sigma)
    Hw = solve_triangular(ch,H,lower=True)
    sp = spectral(H)
    wsp = spectral(Hw)
    return dict(H=H,A=A,base_sigma=base_sigma,Hw=Hw,sp=sp,wsp=wsp,d=H.shape[1])


def model_summary(m):
    H=m['H']; sp=m['sp']
    return dict(observation_count=len(H), coefficient_dimension=H.shape[1],
                rank=sp['rank'], nullity=H.shape[1]-sp['rank'],
                rank_sensitivity=sp['sensitivity'], singular_values=sp['all_s'],
                condition_of_observable_part=float(sp['s'][0]/sp['s'][-1]),
                signed_measurement_operator_shape=m['A'].shape)


def stable_layout(m, sigma):
    sp=m['wsp']; eig=sp['s']**2/(sigma*sigma*m['d'])
    mask=eig>=1
    return dict(effective_dimension=int(mask.sum()), information_eigenvalues=eig,
                basis=sp['q'][:,mask])


def stable_common(m1,m2,basis,sigma):
    if basis.shape[1]==0:
        return dict(dimension=0, difference_noise_eigenvalues=[], max_variance_threshold=1.)
    # BLUE covariance factored before SVD avoids eigenvalue cancellation in a Gram matrix.
    factors=[]
    for m in [m1,m2]:
        sp=m['wsp']
        factor=(sp['q'].T@basis)/sp['s'][:,None]
        factors.append(sigma*np.sqrt(m['d'])*factor)
    s=np.linalg.svd(np.vstack(factors),compute_uv=False)
    noise_eig=s*s
    return dict(dimension=int(np.sum(noise_eig<=1)),difference_noise_eigenvalues=noise_eig,
                max_variance_threshold=1.)


def invariant_trials(models,rng):
    output=[]
    for name,m in models.items():
        H=m['H']; Sig=.2**2*m['base_sigma']; n=len(H)
        max_h=max_j=max_bad_h=max_bad_j=0.
        for _ in range(10):
            U,_=np.linalg.qr(rng.standard_normal((n,n)))
            V,_=np.linalg.qr(rng.standard_normal((n,n)))
            M=U@np.diag(np.geomspace(.5,2,n))@V.T
            a=rng.standard_normal(H.shape[1])/np.sqrt(H.shape[1])
            y=H@a+np.linalg.cholesky(Sig)@rng.standard_normal(n)
            h=H.T@np.linalg.solve(Sig,y); J=H.T@np.linalg.solve(Sig,H)
            H2=M@H; y2=M@y; Sig2=M@Sig@M.T
            h2=H2.T@np.linalg.solve(Sig2,y2); J2=H2.T@np.linalg.solve(Sig2,H2)
            badh=H2.T@np.linalg.solve(Sig,y2); badJ=H2.T@np.linalg.solve(Sig,H2)
            max_h=max(max_h,np.linalg.norm(h-h2)/np.linalg.norm(h))
            max_j=max(max_j,np.linalg.norm(J-J2)/np.linalg.norm(J))
            max_bad_h=max(max_bad_h,np.linalg.norm(h-badh)/np.linalg.norm(h))
            max_bad_j=max(max_bad_j,np.linalg.norm(J-badJ)/np.linalg.norm(J))
        output.append(dict(layout=name,trials=10,M_condition_number=4.,
                           max_relative_h_error=max_h,max_relative_J_error=max_j,
                           pass_threshold=1e-10,passed=bool(max_h<1e-10 and max_j<1e-10),
                           negative_control_fixed_covariance_max_h_error=max_bad_h,
                           negative_control_fixed_covariance_max_J_error=max_bad_j))
    return output



def position_sensitivity(selected):
    rng=np.random.default_rng(20260931)
    base=selected['eegmmidb_64']
    directions=np.array(base['unit'],copy=True)
    tangent=rng.standard_normal(directions.shape)
    tangent-=np.sum(tangent*directions,axis=1,keepdims=True)*directions
    tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
    out=dict(exploratory=True,seed=20260931,
             prereg_sha256=hashlib.sha256((HERE/'T3_POSITION_PREREG.md').read_bytes()).hexdigest(),
             tangent_directions=tangent,results=[])
    for L in [6,10]:
        for angle in [0.,.5,1.]:
            perturbed=dict(base)
            theta=np.deg2rad(angle)
            perturbed['unit']=np.cos(theta)*directions+np.sin(theta)*tangent
            m2=make_model(perturbed,L,'contrast')
            for name in ['standard_1020_19','tuh_tcp_22']:
                m1=make_model(selected[name],L,'contrast')
                overlap,basis=subspace_pair(m1['sp']['q'],m2['sp']['q'])
                r=min(m1['sp']['rank'],m2['sp']['rank'])
                c=overlap['exact_common_dimension']
                out['results'].append(dict(L=L,angle_degrees=angle,layout1=name,
                    layout2='independently_shifted_eegmmidb_64',coefficient_dimension=m1['d'],
                    rank1=m1['sp']['rank'],rank2=m2['sp']['rank'],**overlap,
                    stable_exact_common_sigma_02=stable_common(m1,m2,basis,.2),
                    exploratory_kill_triggered=bool(c<=1 or c<=.1*r)))
    return out

def main():
    layouts=common.layouts()
    required=['standard_1020_19','eegmmidb_64','tuh_tcp_22','hydrocel_129']
    # Accept aliases but store the common module's exact source keys.
    aliases={
        'standard_1020_19':['ten20_19','standard_1020_19','1020_19','10-20_19','ten_twenty_19','10-20-19','10-20_19ch'],
        'eegmmidb_64':['eegmmidb_64','EEGMMIDB_64','EEGMMIDB64','EEGMMIDB-64','eegmmidb64'],
        'tuh_tcp_22':['tcp22','tuh_tcp_22','tcp22','TCP22','TUEG_TCP22','tueg_tcp22','tueg_tcp_22'],
        'hydrocel_129':['hydrocel_129','HydroCel129','HydroCel-129','hydro129','hydrocel129']}
    selected={}
    sourcekeys={}
    for name in required:
        keys=[k for k in aliases[name] if k in layouts]
        if not keys: raise KeyError(f'Cannot find {name}; keys={list(layouts)}')
        selected[name]=layouts[keys[0]]; sourcekeys[name]=keys[0]
    pairs=[('standard_1020_19','eegmmidb_64'),('tuh_tcp_22','eegmmidb_64'),('hydrocel_129','standard_1020_19')]
    rng=np.random.default_rng(SEED)
    results=dict(test='T3',seed=SEED,created_utc=datetime.now(timezone.utc).isoformat(),
                 prereg_sha256=hashlib.sha256((HERE/'T3_PREREG.md').read_bytes()).hexdigest(),
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 numpy_version=np.__version__,scipy_version=scipy.__version__,layout_source_keys=sourcekeys,
                 common_sha256=hashlib.sha256((HERE/'common.py').read_bytes()).hexdigest(),
                 layouts_sha256=hashlib.sha256((HERE/'layouts.json').read_bytes()).hexdigest(),
                 numerical_rank_relative_tolerance=RTOL,noise_levels=NOISES,levels=LEVELS,
                 modes={})
    for mode in ['contrast','raw']:
        results['modes'][mode]={}
        for L in LEVELS:
            models={name:make_model(layout,L,mode) for name,layout in selected.items()}
            level=dict(layouts={name:model_summary(m) for name,m in models.items()},pairs=[])
            if mode=='contrast' and L==6:
                results['invertible_expression']=invariant_trials(models,rng)
            for n1,n2 in pairs:
                m1,m2=models[n1],models[n2]
                overlap,basis=subspace_pair(m1['sp']['q'],m2['sp']['q'])
                # Direct stacked-H rank is a separate diagnostic with actual operator scalings.
                stacked_rank=numerical_rank(np.linalg.svd(np.vstack([m1['H'],m2['H']]),compute_uv=False))
                overlap['direct_H_stacked_rank_check_dimension']=m1['sp']['rank']+m2['sp']['rank']-stacked_rank
                pair=dict(layout1=n1,layout2=n2,smaller_rank=min(m1['sp']['rank'],m2['sp']['rank']),**overlap)
                pair['noise']=[]
                for sigma in NOISES:
                    e1,e2=stable_layout(m1,sigma),stable_layout(m2,sigma)
                    eoverlap,_=subspace_pair(e1['basis'],e2['basis'])
                    stable=stable_common(m1,m2,basis,sigma)
                    pair['noise'].append(dict(sigma=sigma,layout1_effective_dimension=e1['effective_dimension'],
                        layout2_effective_dimension=e2['effective_dimension'],stable_exact_common=stable,
                        effective_subspace_angles=eoverlap))
                level['pairs'].append(pair)
            results['modes'][mode][str(L)]=level
    kills=[]
    for L in [6,10]:
        for pair in results['modes']['contrast'][str(L)]['pairs']:
            c=pair['exact_common_dimension']; r=pair['smaller_rank']
            kills.append(dict(L=L,pair=[pair['layout1'],pair['layout2']],dimension=c,smaller_rank=r,
                              ratio=c/r,kill_triggered=bool(c<=1 or c<=.1*r),
                              numerical_checks_agree=pair['dimension_checks_agree']))
    feasible=[]
    for pair in results['modes']['contrast']['6']['pairs']:
        noise=next(n for n in pair['noise'] if n['sigma']==.2)
        c=noise['stable_exact_common']['dimension']; r=pair['smaller_rank']
        feasible.append(dict(L=6,sigma=.2,pair=[pair['layout1'],pair['layout2']],stable_dimension=c,
                             smaller_rank=r,ratio=c/r,kill_triggered=bool(c<=1 or c<=.1*r)))
    results['supplemental_position_sensitivity']=position_sensitivity(selected)
    results['decisions']=dict(invertible_expression_passed=all(x['passed'] for x in results['invertible_expression']),
                              strict_common_space= kills,noise_feasibility=feasible)
    (HERE/'t3_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,default=np_json)+'\n')
    print(json.dumps(results['decisions'],indent=2,default=np_json))

if __name__=='__main__': main()
