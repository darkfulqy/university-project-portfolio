"""Sealed passive fly cable with explicit continuous conductance replay.

The CNS activity is not voltage. A separately specified conductance per contact
converts continuous presynaptic state into current-generating membrane channels.
No electrode injection and no feedback into the CNS are used.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy import sparse
from neuron import h
from lfpykit import CellGeometry, RecExtElectrode


@dataclass(frozen=True)
class CableParameters:
    cm_uF_cm2: float = 1.0
    ra_ohm_cm: float = 100.07105
    leak_S_cm2: float = 1e-5
    leak_reversal_mV: float = -65.0
    exc_reversal_mV: float = 0.0
    inh_reversal_mV: float = -70.0
    conductance_uS_per_contact_per_au: float = 1e-5
    radius_scale: float = 1.0
    max_segment_um: float = 2.0
    sigma_S_m: float = .3


def live_section_names():
    return [sec.name() for sec in h.allsec()]


class PassiveFlyCable:
    def __init__(self, swc_path: Path, p=CableParameters()):
        # NEURON initialization and integration are global. Refuse to share the
        # simulator with sections this object does not own; never delete them.
        existing = live_section_names()
        if existing:
            raise RuntimeError(f'{len(existing)} pre-existing NEURON sections; '
                               f'construct, run and close cells serially: {existing[:5]}')
        self.p = p
        self.sections = {}; self.segments = []
        try:
            self._build(swc_path, p)
        except BaseException:
            self.close()
            raise

    def _build(self, swc_path, p):
        a = np.loadtxt(swc_path)
        ids = a[:, 0].astype(int); parents = a[:, 6].astype(int)
        if len(np.unique(ids)) != len(ids) or (parents == -1).sum() != 1:
            raise ValueError('A single rooted SWC tree with unique IDs is required')
        self.xyz_um = a[:, 2:5] * .008
        radius = a[:, 5] * .008 * p.radius_scale
        if np.any(radius <= 0) or not np.isfinite(a).all():
            raise ValueError('Invalid SWC geometry')
        lookup = {int(n): i for i,n in enumerate(ids)}
        root = int(ids[parents == -1][0])
        edges = {}
        for j,parent in enumerate(parents):
            if parent == -1:
                continue
            k = lookup[parent]
            start, end = self.xyz_um[k], self.xyz_um[j]
            length = np.linalg.norm(end-start)
            if length <= 0:
                raise ValueError('Zero length SWC edge')
            sec = h.Section(name=f'fly_edge_{ids[j]}')
            h.pt3dclear(sec=sec)
            h.pt3dadd(*start, 2*radius[k], sec=sec)
            h.pt3dadd(*end, 2*radius[j], sec=sec)
            sec.nseg = max(1, int(np.ceil(length / p.max_segment_um)))
            sec.Ra = p.ra_ohm_cm; sec.cm = p.cm_uF_cm2
            sec.insert('pas')
            self.sections[int(ids[j])] = sec
            edges[int(ids[j])] = (start,end,radius[k],radius[j])
        roots = [int(n) for n,parent in zip(ids,parents) if parent == root]
        for n,parent in zip(ids,parents):
            if parent == -1:
                continue
            if parent == root:
                if n != roots[0]:
                    self.sections[int(n)].connect(self.sections[roots[0]](0), 0)
            else:
                self.sections[int(n)].connect(self.sections[int(parent)](1), 0)
        parent_of = {int(n): int(parent) for n,parent in zip(ids,parents)}
        starts=[]; ends=[]; diameters=[]; node=[]; position=[]
        for n,sec in self.sections.items():
            start,end,r0,r1=edges[n]
            for k,seg in enumerate(sec):
                low=seg.x-.5/sec.nseg; high=seg.x+.5/sec.nseg
                starts.append(start+low*(end-start)); ends.append(start+high*(end-start))
                diameters.append(2*(r0+seg.x*(r1-r0)))
                node.append(n); position.append(k)
                self.segments.append(seg)
        self.starts=np.array(starts); self.ends=np.array(ends)
        # Audit topology: SWC child node of each edge, its SWC parent, and the
        # segment position along that edge (0 = parent end). Root-adjacent
        # edges share the SWC root point.
        self.diameters_um=np.array(diameters)
        self.segment_swc_node_id=np.array(node,dtype=np.int64)
        self.segment_swc_parent_node_id=np.array([parent_of[n] for n in node],dtype=np.int64)
        self.segment_position_in_edge=np.array(position,dtype=np.int64)
        self.swc_root_node_id=root
        self.area_um2=np.array([seg.area() for seg in self.segments])
        self.geometry=CellGeometry(x=np.c_[self.starts[:,0], self.ends[:,0]],
            y=np.c_[self.starts[:,1],self.ends[:,1]], z=np.c_[self.starts[:,2],self.ends[:,2]],
            d=np.array(diameters))
        h.load_file('stdrun.hoc')
        h.CVode().active(0); h.CVode().use_fast_imem(1)

    def map_contacts(self, xyz_um, presynaptic_columns, signs, n_sources):
        """Nearest centerline segment projection; retain and report distances."""
        xyz_um=np.asarray(xyz_um)
        direction=self.ends-self.starts
        fraction=np.clip(np.einsum('csi,si->cs', xyz_um[:,None,:]-self.starts,
                                  direction)/(direction*direction).sum(axis=1),0,1)
        projections=self.starts+fraction[:,:,None]*direction
        distance=np.linalg.norm(xyz_um[:,None,:]-projections,axis=2)
        closest=distance.argmin(axis=1)
        self.contact_segment_indices=closest
        self.contact_projection_distance_um=distance[np.arange(len(closest)),closest]
        signs=np.asarray(signs); columns=np.asarray(presynaptic_columns)
        self.exc_map=sparse.csr_matrix((np.ones(np.sum(signs>0)),
            (closest[signs>0], columns[signs>0])),shape=(len(self.segments),n_sources))
        self.inh_map=sparse.csr_matrix((np.ones(np.sum(signs<0)),
            (closest[signs<0], columns[signs<0])),shape=(len(self.segments),n_sources))

    def conductance(self, activity):
        if np.any(activity < 0) or not np.isfinite(activity).all():
            raise ValueError('Continuous conductance drive must be nonnegative and finite')
        return (self.p.conductance_uS_per_contact_per_au*(self.exc_map@activity),
                self.p.conductance_uS_per_contact_per_au*(self.inh_map@activity))

    def set_conductance(self, exc_uS, inh_uS):
        # One pas channel with conductance-weighted reversal is algebraically
        # identical to parallel ohmic leak/excitatory/inhibitory channels.
        # S/cm² * area[µm²] * .01 = µS; no electrode current is introduced.
        exc=exc_uS/(self.area_um2*.01); inh=inh_uS/(self.area_um2*.01)
        total=self.p.leak_S_cm2+exc+inh
        reversal=(self.p.leak_S_cm2*self.p.leak_reversal_mV+
                  exc*self.p.exc_reversal_mV+inh*self.p.inh_reversal_mV)/total
        for seg,g,e in zip(self.segments,total,reversal):
            seg.pas.g=float(g); seg.pas.e=float(e)

    def run(self, activity, source_dt_ms=1.0, dt_ms=.025, warmup_ms=2000.):
        if not np.isclose(source_dt_ms/dt_ms,round(source_dt_ms/dt_ms)):
            raise ValueError('Cable dt must divide source sample interval')
        nsteps=int(round(source_dt_ms/dt_ms)); activity=np.asarray(activity)
        ntime=len(activity); nseg=len(self.segments)
        voltage=np.empty((ntime,nseg)); imem=np.empty_like(voltage)
        gexc,ginh=self.conductance(activity.T)
        self.set_conductance(gexc[:,0],ginh[:,0])
        h.dt=dt_ms; h.steps_per_ms=1/dt_ms; h.secondorder=0
        h.finitialize(self.p.leak_reversal_mV)
        for _ in range(int(round(warmup_ms/dt_ms))):
            h.fadvance()
        voltage[0]=[s.v for s in self.segments]
        imem[0]=[s.i_membrane_ for s in self.segments]
        for t in range(ntime-1):
            self.set_conductance(gexc[:,t],ginh[:,t])
            for _ in range(nsteps):
                h.fadvance()
            voltage[t+1]=[s.v for s in self.segments]
            imem[t+1]=[s.i_membrane_ for s in self.segments]
        if not np.isfinite(voltage).all() or not np.isfinite(imem).all():
            raise FloatingPointError('Nonfinite cable result')
        return {'membrane_voltage_mV':voltage, 'transmembrane_current_nA':imem,
                'interval_exc_conductance_uS':gexc.T[:-1],
                'interval_inh_conductance_uS':ginh.T[:-1],
                'applied_exc_conductance_at_current_sample_uS':np.vstack([gexc[:,0],gexc.T[:-1]]),
                'applied_inh_conductance_at_current_sample_uS':np.vstack([ginh[:,0],ginh.T[:-1]])}

    def lead_field(self, electrodes_um, sigma=None, method='linesource'):
        pos=np.asarray(electrodes_um)
        electrode=RecExtElectrode(self.geometry,sigma=self.p.sigma_S_m if sigma is None else sigma,
                                 x=pos[:,0],y=pos[:,1],z=pos[:,2],method=method)
        return electrode.get_transformation_matrix()  # mV / nA

    def close(self):
        # Delete only sections created by this object; idempotent.
        self.segments.clear()
        for sec in self.sections.values():
            h.delete_section(sec=sec)
        self.sections.clear()


def synthetic_probe(xyz_um, offset_um=50.0):
    """16 point contacts at 50 µm pitch; separately check cable clearance."""
    center=np.mean(xyz_um,axis=0)
    _,_,axes=np.linalg.svd(xyz_um-center,full_matrices=False)
    return center+offset_um*axes[1]+50*np.arange(-7.5,8)[:,None]*axes[0][None,:]
