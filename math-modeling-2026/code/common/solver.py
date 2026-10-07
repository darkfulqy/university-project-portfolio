"""Solve radial heat and moisture transport with finite differences and control-volume fluxes.
Tridiagonal heat systems are solved with LAPACK.
"""
from pathlib import Path
import json,time
import numpy as np
from scipy.linalg import solve_banded
OUT=Path(__file__).resolve().parents[2]

def props(c,model):
    if model==1:return np.full_like(c,820.),np.full_like(c,2600.),np.full_like(c,.36)
    if model==2:return 650+128*c,1450+2736*c/(1+c),.21+.38*c/(1+c)
    return 760+90*c,1850+2150*c/(1+c),.12+.20*c/(1+c)

def diffusivity(c,T,model):
    if model==1:return 7e-9*np.exp(-.89/c)
    if model==2:return 2.4e-3*np.exp(-.45/c-3850/T)
    return 4.2e-4*np.exp(-.30/c-3850/T)

def run(model,n=100,dt=.5,end=72*3600,theta=1.,ds=1.,hs=1.,hms=1.,tshift=0.,airmode=0,face_mode=0,boundary_mode=0,tag=None,continue_after_event=False,record_interval_s=60.,heat_face_mode=None):
    started=time.perf_counter();air=np.load(OUT/'data/processed/attachment1.npy');rad=np.load(OUT/'data/processed/attachment2.npy')
    tail=air[air[:,0]>=10800,1:].mean(axis=0);steps=round(end/dt)
    times=np.arange(steps+1)*dt
    Ta=np.interp(times,air[:,0],air[:,1])+273.15+tshift;Ca=np.interp(times,air[:,0],air[:,2])
    if airmode==0:
        Ta[times>14400]=tail[0]+273.15+tshift;Ca[times>14400]=tail[1]
    radii=np.interp(times,rad[:,0],rad[:,1])/100
    h=25*hs;hm=8e-7*hms;x=np.arange(n+1)/n;dx=1/n;idx=np.arange(1,n)
    xf=np.r_[0.,(x[1:]+x[:-1])/2,1.];vol=np.diff(xf**2)/2
    mass=np.pi*.02**2*(760+90*2.55)/3.55;dm=mass/n;mv=np.full(n+1,dm);mv[[0,-1]]*=.5
    weight=mv if model==4 else vol;water0=np.sum(weight*2.55)
    c=np.full(n+1,2.55);T=np.full(n+1,301.15);history=[];Ts=[];Cs=[];Rs=[]
    outflow=0.;balance=0.;maxbalance=0.;maxnum=0.;minC=2.55;minT=301.15;maxT=301.15;incC=0.;incT=0.;event=None;failed=None
    band=np.zeros((3,n+1));lo=np.zeros(n+1);up=np.zeros(n+1);loss=np.zeros(n+1)
    def geometry(cc,step):
        if model==4:
            cm=(cc[1:]+cc[:-1])/2
            return np.sqrt(np.r_[0.,np.cumsum(dm*(1+cm)/(760+90*cm))/np.pi])
        return x*(.02 if model<4 else radii[step])
    for step in range(steps+1):
        t=times[step];r=geometry(c,step);R=r[-1]
        stop=event is not None and not continue_after_event
        if abs(t/record_interval_s-round(t/record_interval_s))<1e-8 or (model==1 and t==100) or step==steps or stop:
            history.append([t,R,c[0],c[-1],balance]);Ts.append(T.copy()-273.15);Cs.append(c.copy());Rs.append(r.copy())
        if step==steps or stop:break
        ai=step+1 if model<4 and boundary_mode!=2 else step
        ta=Ta[ai];ca=Ca[ai];oldc=c;oldT=T;cm=(c[1:]+c[:-1])/2;tm=(T[1:]+T[:-1])/2
        waterD=ds*diffusivity(cm,tm,model)
        if model<4 or boundary_mode==1:
            spacing=R/n;axis=4*ds*diffusivity(c[0],T[0],model)/spacing**2
            ap=(idx+.5)/idx*waterD[1:]/spacing**2;am=(idx-.5)/idx*waterD[:-1]/spacing**2
            cn=c.copy();cn[0]+=dt*axis*(c[1]-c[0]);cn[1:-1]+=dt*(ap*(c[2:]-c[1:-1])-am*(c[1:-1]-c[:-2]))
            dface=ds*diffusivity(c[-1],T[-1],model);cn[-1]=(dface*cn[-2]+hm*spacing*ca)/(dface+hm*spacing)
            if boundary_mode==2:cn[0]=cn[1]
            maxnum=max(maxnum,dt*max(axis,np.max(ap+am)));outflow+=dt*hm/R*(c[-1]-ca)
        else:
            if face_mode==1:
                dd=ds*diffusivity(c,T,model);waterD=(dd[1:]+dd[:-1])/2
            if model==4:
                rf2=(r[1:]**2+r[:-1]**2)/2
                coeff=4*np.pi**2*rf2*((760+90*cm)/(1+cm))**2*waterD/dm
                boundary=2*np.pi*R*(760+90*c[-1])/(1+c[-1])*hm
            else:coeff=xf[1:-1]*waterD/(R**2*dx);boundary=hm/R
            flux=np.r_[0.,coeff*np.diff(c),boundary*(ca-c[-1])]
            cn=c+dt*np.diff(flux)/weight
            sums=np.r_[0.,coeff]+np.r_[coeff,boundary]
            maxnum=max(maxnum,dt*np.max(sums/weight));outflow-=dt*flux[-1]
        rh,cp,kk=props(cn,model);cap=rh*cp;cmh=(cn[1:]+cn[:-1])/2
        if model<4 or boundary_mode==1:
            spacing=R/n;km=props(cmh,model)[2]
            lo[:]=0;up[:]=0;loss[:]=0
            up[0]=4*kk[0]/(cap[0]*spacing**2)
            up[1:-1]=(idx+.5)/idx*km[1:]/(cap[1:-1]*spacing**2)
            lo[1:-1]=(idx-.5)/idx*km[:-1]/(cap[1:-1]*spacing**2)
        else:
            km=props(cmh,model)[2] if (face_mode if heat_face_mode is None else heat_face_mode)==0 else (kk[1:]+kk[:-1])/2
            if model==4:
                hc=4*np.pi**2*rf2*(760+90*cmh)/(1+cmh)*km/dm
                capacity=(1+cn)*cp*mv;outer=2*np.pi*R*h
            else:hc=xf[1:-1]*km/(R**2*dx);capacity=cap*vol;outer=h/R
            lo[1:]=hc/capacity[1:];up[:-1]=hc/capacity[:-1];lo[0]=0;up[-1]=0
            loss[:]=0;loss[-1]=outer/capacity[-1]
        band[:]=0;band[1]=1+theta*dt*(lo+up+loss)
        band[0,1:]=-theta*dt*up[:-1];band[2,:-1]=-theta*dt*lo[1:]
        rate=-(lo+up+loss)*T
        rate[1:]+=lo[1:]*T[:-1];rate[:-1]+=up[:-1]*T[1:]
        rhs=T+(1-theta)*dt*rate+dt*loss*(theta*ta+(1-theta)*Ta[step])
        if model<4 or boundary_mode==1:
            ks=kk[-1]/spacing;band[2,-2]=-ks;band[1,-1]=ks+h;rhs[-1]=h*ta
            if boundary_mode==2:band[1,0]=1;band[0,1]=-1;rhs[0]=0
        T=solve_banded((1,1),band,rhs,check_finite=False);c=cn
        if np.any(~np.isfinite(c)) or np.any(~np.isfinite(T)) or c.min()<=0:
            failed=step+1;break
        minC=min(minC,c.min());minT=min(minT,T.min());maxT=max(maxT,T.max());incC=max(incC,np.diff(c).max());incT=max(incT,c[0]-oldc[0])
        balance=(weight@c+outflow-water0)/water0;maxbalance=max(maxbalance,abs(balance))
        if event is None and oldc[0]>=.15 and c[0]<.15:event=t+dt*(oldc[0]-.15)/(oldc[0]-c[0])
    hist=np.array(history);Tsave=np.array(Ts);Csave=np.array(Cs);Rsave=np.array(Rs)
    tag=tag or f'm{model}_n{n}_dt{dt:g}_theta{theta:g}'
    result={'engine':'NumPy + LAPACK equivalent recurrence','model':model,'N':n,'dt_s':dt,'theta_heat':theta,'face_mode':face_mode,'boundary_mode':boundary_mode,'airmode':airmode,'scales':{'D':ds,'h':hs,'hm':hms,'T_C_shift':tshift},'event_h':event/3600 if event is not None else None,'last_time_h':hist[-1,0]/3600,'runtime_s':time.perf_counter()-started,'final_radius_cm':Rsave[-1,-1]*100,'final_center_C':Csave[-1,0],'final_surface_C':Csave[-1,-1],'final_center_T_C':Tsave[-1,0],
            'diagnostics':{'event_interpolated_s':event,'failed_at_step':failed,'max_explicit_coefficient_sum':float(maxnum),'min_C':float(minC),'min_T_C':float(minT-273.15),'max_T_C':float(maxT-273.15),'max_radial_C_increase':float(incC),'max_center_C_time_increase':float(incT),'final_relative_water_balance':float(balance),'max_relative_water_balance':float(maxbalance)}}
    result['heat_face_mode']=face_mode if heat_face_mode is None else heat_face_mode
    result['record_interval_s']=record_interval_s
    np.savez_compressed(OUT/'results/raw'/f'{tag}.npz',history=hist,T_C=Tsave,C=Csave,r_m=Rsave)
    (OUT/'results/raw'/f'{tag}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'tag':tag,'event_h':result['event_h'],'C0':result['final_center_C'],'Cs':result['final_surface_C'],'balance':balance,'CFL':float(maxnum),'failed':failed,'runtime_s':result['runtime_s']}),flush=True)
    return result
