"""Shared, offline template geometry. No writes to the borrowed environment."""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('_MNE_FAKE_HOME_DIR', str(ROOT / 'runtime-home'))
os.environ.setdefault('MNE_DONTWRITE_HOME', 'true')
import numpy as np
import mne
from scipy.special import sph_harm_y

SEED = 20260930
TCP22 = ('FP1-F7 F7-T3 T3-T5 T5-O1 FP2-F8 F8-T4 T4-T6 T6-O2 '
         'T3-C3 C3-CZ CZ-C4 C4-T4 FP1-F3 F3-C3 C3-P3 P3-O1 '
         'FP2-F4 F4-C4 C4-P4 P4-O2 A1-T3 T4-A2').split()
TEN20 = 'Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4 P8 O1 O2'.split()
EEG64 = ('FC5 FC3 FC1 FCz FC2 FC4 FC6 C5 C3 C1 Cz C2 C4 C6 '
         'CP5 CP3 CP1 CPz CP2 CP4 CP6 Fp1 Fpz Fp2 AF7 AF3 AFz AF4 AF8 '
         'F7 F5 F3 F1 Fz F2 F4 F6 F8 FT7 FT8 T7 T8 T9 T10 TP7 TP8 '
         'P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2 Iz').split()

def unit(xyz):
    xyz = np.asarray(xyz)
    return xyz / np.linalg.norm(xyz, axis=1, keepdims=True)

def template_positions(template):
    montage = mne.channels.make_standard_montage(template)
    # BioFoundation calls info.set_montage, which transforms native coordinates
    # to the fiducial head frame. Reproduce that rather than using MRI xyz.
    info = mne.create_info(montage.ch_names, 256., 'eeg')
    info.set_montage(montage, on_missing='raise')
    return info.get_montage().get_positions()['ch_pos']

def layouts():
    p = template_positions('standard_1005')
    lookup = {k.upper(): v for k, v in p.items()}
    out = {}
    def add(key, names, positions=lookup):
        xyz = np.array([positions[n.upper()] for n in names])
        out[key] = dict(names=names, xyz=xyz, unit=unit(xyz))
    add('ten20_19', TEN20)
    add('eegmmidb64', EEG64)
    # Explicit extended standard base grid: all labels before intermediate h
    # labels in MNE's 1005 list. Name list is saved; not a claim of one canonical
    # universal 10-10 cap. Includes inferior I1/Iz/I2 boundary positions.
    base = list(p)[:list(p).index('AFp9h')]
    add('ten_ten', base)
    for n in (19, 20, 22):
        edges = TCP22[:n]
        electrodes = list(dict.fromkeys(e for edge in edges for e in edge.split('-')))
        xyz = np.array([lookup[e] for e in electrodes])
        A = np.zeros((n, len(electrodes)))
        for i, edge in enumerate(edges):
            a, b = edge.split('-')
            A[i, electrodes.index(a)] = 1
            A[i, electrodes.index(b)] = -1
        mids = np.array([(lookup[a] + lookup[b])/2 for a,b in (e.split('-') for e in edges)])
        out[f'tcp{n}'] = dict(names=edges, xyz=mids, unit=unit(mids), A=A,
                             electrode_names=electrodes, electrode_xyz=xyz,
                             electrode_unit=unit(xyz))
    hydro = template_positions('GSN-HydroCel-129')
    add('hydrocel129', list(hydro), {k.upper():v for k,v in hydro.items()})
    return out

def real_sh(points, L):
    """Real orthonormal SH on dOmega: l ascending, m=-l,...,l; l0 first."""
    points = unit(points)
    theta = np.arccos(np.clip(points[:,2], -1, 1))
    phi = np.arctan2(points[:,1], points[:,0])
    cols = []
    for ell in range(L+1):
        for m in range(-ell, ell+1):
            y = sph_harm_y(ell, abs(m), theta, phi)
            cols.append(y.real if m == 0 else
                        np.sqrt(2)*(-1)**m*(y.imag if m < 0 else y.real))
    return np.stack(cols, axis=1)

def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: jsonable(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [jsonable(v) for v in value]
    return value

if __name__ == '__main__':
    import json
    import scipy
    data = dict(seed=SEED, versions=dict(numpy=np.__version__, scipy=scipy.__version__, mne=mne.__version__),
                frame='MNE fiducial head frame, xyz in metres; unit=xyz/||xyz||, no per-layout fit',
                tcp19_definition='BioFoundation CHN_ORDER[:19], not asserted a universal TCP19',
                layouts=layouts())
    (ROOT/'layouts.json').write_text(json.dumps(jsonable(data), indent=2)+'\n')
    print({k:len(v['names']) for k,v in data['layouts'].items()})
