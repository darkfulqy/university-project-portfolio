"""Explicit axisymmetric heat and moisture calculation on a half cylinder.
Uses uniform radial and axial grids.
The mid-plane is treated as a symmetry boundary.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
OUT=Path(__file__).resolve().parents[2]

def D(c):
    return 7e-9*np.exp(-.89/c)

def kernel(air,nr,nz,dt,axis_copy):
    dr=.02/nr;dz=.125/nz;alpha=.36/(820*2600)
    T=np.full((nr+1,nz+1),28.);C=np.full_like(T,2.55)
    ii=np.arange(1,nr)[:,None]
    for step in range(round(1800/dt)):
        t=step*dt
        ta=np.interp(t,air[:,0],air[:,1]);ca=np.interp(t,air[:,0],air[:,2])
        oldT=T.copy();oldC=C.copy()
        rt=np.empty((nr,nz));rc=np.empty_like(rt)
        rt[0]=4*(oldT[1,:nz]-oldT[0,:nz])/dr**2
        rc[0]=4*D(oldC[0,:nz])*(oldC[1,:nz]-oldC[0,:nz])/dr**2
        rt[1:]=((ii+.5)*(oldT[2:,:nz]-oldT[1:nr,:nz])-(ii-.5)*(oldT[1:nr,:nz]-oldT[:nr-1,:nz]))/(ii*dr**2)
        ff=D((oldC[1:,:nz]+oldC[:-1,:nz])/2)*np.diff(oldC[:,:nz],axis=0)
        rc[1:]=((ii+.5)*ff[1:]-(ii-.5)*ff[:-1])/(ii*dr**2)
        zm=np.empty((nr,nz));zc=np.empty_like(zm)
        zm[:,0]=2*(oldT[:nr,1]-oldT[:nr,0])/dz**2
        zm[:,1:]=(oldT[:nr,2:]-2*oldT[:nr,1:nz]+oldT[:nr,:nz-1])/dz**2
        ff=D((oldC[:nr,1:]+oldC[:nr,:-1])/2)*np.diff(oldC[:nr],axis=1)
        zc[:,0]=2*ff[:,0]/dz**2;zc[:,1:]=np.diff(ff,axis=1)/dz**2
        T[:nr,:nz]=oldT[:nr,:nz]+dt*alpha*(rt+zm)
        C[:nr,:nz]=oldC[:nr,:nz]+dt*(rc+zc)
        if axis_copy:
            T[0]=T[1];C[0]=C[1]
        T[nr]=(.36*T[nr-1]+25*dr*ta)/(.36+25*dr)
        dd=D(oldC[nr]);C[nr]=(dd*C[nr-1]+8e-7*dr*ca)/(dd+8e-7*dr)
        T[:,nz]=(.36*T[:,nz-1]+25*dz*ta)/(.36+25*dz)
        dd=D(oldC[:,nz]);C[:,nz]=(dd*C[:,nz-1]+8e-7*dz*ca)/(dd+8e-7*dz)
    return T,C

def run(nr=100,nz=32,dt=.05,axis_copy=False):
    air=np.load(OUT/'data/processed/attachment1.npy');start=time.perf_counter()
    T,C=kernel(air,nr,nz,dt,axis_copy)
    js=[0,round(.75*nz),round(.875*nz),nz]
    result={'nr':nr,'nz_half':nz,'dt':dt,'axis_copy':axis_copy,'runtime_s':time.perf_counter()-start,
            'axial_grid':'uniform on half cylinder; mirrored at mid-plane',
            'end_distance_cm':[12.5,3.125,1.5625,0.],
            'center_T_C':T[0,js].tolist(),'center_C':C[0,js].tolist(),
            'T_range':[float(T.min()),float(T.max())],'C_range':[float(C.min()),float(C.max())]}
    z=np.arange(nz+1)*.125/nz
    for tol in [.1,1.]:
        bad=np.flatnonzero(abs(T[0]-T[0,0])>tol)
        result[f'end_layer_T_gt_{tol}_cm']=float((.125-z[bad[0]])*100) if len(bad) else 0.
    bad=np.flatnonzero(abs(C[0]-C[0,0])>.01)
    result['end_layer_C_gt_0.01_cm']=float((.125-z[bad[0]])*100) if len(bad) else 0.
    tag=f'q1_2d_nr{nr}_nz{nz}_dt{dt:g}_copy{int(axis_copy)}'
    (OUT/'results/raw'/f'{tag}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    np.savez_compressed(OUT/'results/raw'/f'{tag}.npz',T_C=T,C=C,r_m=np.arange(nr+1)*.02/nr,z_m=z)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--nr',type=int,default=100);p.add_argument('--nz',type=int,default=32);p.add_argument('--dt',type=float,default=.05);p.add_argument('--axis-copy',action='store_true');a=p.parse_args();run(a.nr,a.nz,a.dt,a.axis_copy)
