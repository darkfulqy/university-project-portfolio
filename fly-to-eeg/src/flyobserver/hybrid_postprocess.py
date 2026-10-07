"""Sum per-cell extracellular potential contributions with upstream hybridLFPy.

This module executes the unchanged ``hybridLFPy/postproc.py`` from INM-6/hybridLFPy
commit 479c2c4825abbcf047fbdb707c9234b32dd19c75 (GPL-3.0, local copy verified by
SHA-256 before loading). Only ``PostProcess.__init__`` and
``PostProcess.calc_measure`` are invoked. The stock ``Population``/``CachedNetwork``
classes are not imported, no hybridLFPy network or spike replay is run, and NEURON
is not imported.

Input contract (``aggregate``)
------------------------------
contributions_uV : mapping label -> array (n_time, n_channel)
    Signed, finite, model-derived extracellular potential of one cell or cell type
    at every electrode, all computed on the SAME time grid, electrode geometry and
    (unreferenced) potential convention. Labels match ``[A-Za-z0-9][A-Za-z0-9_.-]*``,
    at most 128 characters, and are unique ignoring case. Insertion order is kept.
time_s : (n_time,), n_time >= 2, strictly increasing and uniformly sampled.
electrodes_um : (n_channel, 3) finite electrode coordinates in micrometres.
reference_index : int in [0, n_channel); defines the common referenced view.
units : must be exactly ``'uV'``.
output_dir : must not exist; its parent must exist.
sample_rate_hz : optional; when given it must match ``1 / dt``.
provenance : optional JSON-serialisable dict stored verbatim in the record.

No normalisation, rescaling or re-referencing of individual contributions occurs.
All checks run before any file is created.

Output contract
---------------
``output_dir/populations/<label>_contribution_extracellular_potential_uV.h5``
    upstream input files: ``data`` (n_channel, n_time) float64, ``srate`` (Hz),
    ``time_s``, ``electrodes_um``, ``reference_index``; attrs ``units``,
    ``data_axes``, ``source_sha256``.
``output_dir/cells/``, ``output_dir/figures/``
    empty folders created by upstream ``PostProcess._set_up_savefolder``.
``output_dir/summed_potential_uV.h5``
    ``unreferenced_uV`` and ``referenced_uV`` (n_time, n_channel), ``time_s``,
    ``electrodes_um``, ``reference_index``, ``labels``.
``output_dir/postprocess_record.json``
    upstream module path/commit/hashes/license, methods invoked, MPI and package
    versions, grid/geometry, input array and HDF5 hashes before/after
    ``calc_measure``, output hashes, independent-sum check and provenance.

``referenced_uV = unreferenced_uV - unreferenced_uV[:, [reference_index]]``.
"""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = ROOT / 'references/source-cache/population_field/INM-6__hybridLFPy'
UPSTREAM_POSTPROC = UPSTREAM_DIR / 'hybridLFPy/postproc.py'
UPSTREAM_POSTPROC_SHA256 = 'eaf5d32a693d3da0ef6f683f7aabe6f0ee88d45e41bdcb572d64ace96cd5f54e'
UPSTREAM_LICENSE = UPSTREAM_DIR / 'LICENSE'
UPSTREAM_LICENSE_SHA256 = 'fe3eea6c599e23a00c08c5f5cb2320c30adc8f8687db5fcec9b79a662c53ff6b'
UPSTREAM_REPOSITORY = 'INM-6/hybridLFPy'
UPSTREAM_COMMIT = '479c2c4825abbcf047fbdb707c9234b32dd19c75'

MEASURE = 'extracellular_potential_uV'
OUTPUT_FILE = '{}_contribution_{}'
UNITS = 'uV'
SIGNAL_KIND = ('model-derived extracellular potential, signed sum of per-cell '
               'contributions; not a validated LFP or scalp EEG')
LABEL_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}')
TIME_RTOL = 1e-6

MPI_ENV_KEYS = ('FI_PROVIDER', 'FI_TCP_IFACE')
INTERFACE_PATTERN = re.compile(r'[A-Za-z0-9]{1,15}')
MPI_INTERFACE_NOTE = (
    'Host-specific: on the 2026-09-15 macOS pilot host the MPICH 5.0.1 wheel '
    '(ch4:ofi) chose default nic utun7 and aborted in MPI_Finalize; tcp/lo0 and '
    'sockets/lo0 failed initialization; FI_PROVIDER=tcp FI_TCP_IFACE=en0 exited cleanly.')

_loaded = {}
_mpi_interface_request = None


def configure_mpi_tcp_interface(interface):
    """Select the libfabric tcp provider on one interface for this process only.

    Must run before mpi4py.MPI is imported; if MPI is already initialised the
    environment must already match, otherwise the request cannot take effect.
    """
    global _mpi_interface_request
    if not isinstance(interface, str) or not INTERFACE_PATTERN.fullmatch(interface):
        raise ValueError(f'Unsafe network interface name: {interface!r}')
    wanted = {'FI_PROVIDER': 'tcp', 'FI_TCP_IFACE': interface}
    if 'mpi4py.MPI' in sys.modules:
        current = {k: os.environ.get(k) for k in MPI_ENV_KEYS}
        if current != wanted:
            raise RuntimeError(f'MPI already initialised with {current}; cannot apply {wanted}')
    else:
        os.environ.update(wanted)
    _mpi_interface_request = interface


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sha256_array(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype='<f8').tobytes()).hexdigest()


def load_upstream_postprocess(path=UPSTREAM_POSTPROC, expected_sha256=UPSTREAM_POSTPROC_SHA256):
    """Load the reviewed upstream module from its file path after hash verification."""
    path = Path(path).resolve()
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f'Upstream postproc.py hash mismatch: {digest} != {expected_sha256}')
    if digest in _loaded:
        return _loaded[digest]
    spec = importlib.util.spec_from_file_location(f'hybridlfpy_postproc_{UPSTREAM_COMMIT[:7]}', path)
    module = importlib.util.module_from_spec(spec)
    # Keep the preserved upstream snapshot free of __pycache__.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    if sha256_file(path) != digest:
        raise RuntimeError('Upstream postproc.py changed while loading')
    if module.SIZE != 1 or module.RANK != 0:
        raise RuntimeError(f'Serial MPI required; got rank {module.RANK} of size {module.SIZE}')
    _loaded[digest] = module
    return module


def validate_label(label):
    if not isinstance(label, str) or not LABEL_PATTERN.fullmatch(label):
        raise ValueError(f'Unsafe or empty label: {label!r}')
    return label


def validate_inputs(contributions_uV, time_s, electrodes_um, reference_index, units,
                    output_dir, sample_rate_hz=None):
    """Return validated float64 copies; raise ValueError without touching the filesystem."""
    if units != UNITS:
        raise ValueError(f'units must be {UNITS!r}, got {units!r}')
    if not hasattr(contributions_uV, 'items') or len(contributions_uV) == 0:
        raise ValueError('contributions_uV must be a non-empty mapping')
    labels = [validate_label(k) for k in contributions_uV]
    if len({k.lower() for k in labels}) != len(labels):
        raise ValueError('Labels must be unique ignoring case')

    t = np.asarray(time_s)
    if t.ndim != 1 or t.shape[0] < 2 or not np.issubdtype(t.dtype, np.number) \
            or np.iscomplexobj(t):
        raise ValueError('time_s must be a real 1-D array with at least two samples')
    t = t.astype(np.float64)
    if not np.isfinite(t).all():
        raise ValueError('time_s must be finite')
    steps = np.diff(t)
    if not (steps > 0).all():
        raise ValueError('time_s must be strictly increasing')
    dt_s = (t[-1] - t[0]) / (t.shape[0] - 1)
    if not np.allclose(steps, dt_s, rtol=TIME_RTOL, atol=0):
        raise ValueError(f'time_s is not uniformly sampled (max step deviation '
                         f'{np.abs(steps - dt_s).max():.3e} s)')
    rate = 1.0 / dt_s
    if sample_rate_hz is not None:
        if isinstance(sample_rate_hz, bool) or not np.isfinite(sample_rate_hz) \
                or sample_rate_hz <= 0 or not math.isclose(sample_rate_hz, rate, rel_tol=TIME_RTOL):
            raise ValueError(f'sample_rate_hz {sample_rate_hz!r} does not match time_s ({rate!r} Hz)')

    xyz = np.asarray(electrodes_um)
    if xyz.ndim != 2 or xyz.shape[0] < 1 or xyz.shape[1] != 3 \
            or not np.issubdtype(xyz.dtype, np.number) or np.iscomplexobj(xyz):
        raise ValueError(f'electrodes_um must be real (n_channel, 3), got shape {xyz.shape}')
    xyz = xyz.astype(np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError('electrodes_um must be finite')
    n_channel = xyz.shape[0]

    if isinstance(reference_index, (bool, np.bool_)) or not isinstance(reference_index, (int, np.integer)):
        raise ValueError(f'reference_index must be an integer, got {reference_index!r}')
    reference_index = int(reference_index)
    if not 0 <= reference_index < n_channel:
        raise ValueError(f'reference_index {reference_index} outside [0, {n_channel})')

    arrays = {}
    for label in labels:
        a = np.asarray(contributions_uV[label])
        if not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a) or a.dtype == np.bool_:
            raise ValueError(f'{label}: contribution must be real numeric, got {a.dtype}')
        if a.shape != (t.shape[0], n_channel):
            raise ValueError(f'{label}: shape {a.shape} != (n_time, n_channel) '
                             f'{(t.shape[0], n_channel)}')
        a = a.astype(np.float64)
        if not np.isfinite(a).all():
            raise ValueError(f'{label}: contribution contains non-finite values')
        arrays[label] = a

    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f'Refusing to reuse existing output: {output_dir}')
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f'Output parent does not exist: {output_dir.parent}')
    return arrays, t, float(dt_s), float(rate), xyz, reference_index, output_dir


def runtime_versions(module):
    MPI = module.MPI
    import h5py
    import mpi4py

    def dist(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    return {
        'python': sys.version,
        'platform': platform.platform(),
        'numpy': np.__version__,
        'h5py': h5py.version.version,
        'hdf5': h5py.version.hdf5_version,
        'mpi4py': mpi4py.__version__,
        'mpich_wheel': dist('mpich'),
        'mpi_library_version': MPI.Get_library_version().strip(),
        'mpi_standard_version': list(MPI.Get_version()),
        'mpi_vendor': [MPI.get_vendor()[0], list(MPI.get_vendor()[1])],
        'mpi_comm_world_size': MPI.COMM_WORLD.Get_size(),
        'mpi_comm_world_rank': MPI.COMM_WORLD.Get_rank(),
        'neuron_imported': 'neuron' in sys.modules,
        'mpi_runtime_env': {k: os.environ.get(k) for k in MPI_ENV_KEYS},
        'mpi_interface_selection': (
            'automatic (library default)' if _mpi_interface_request is None else
            f'process-local FI_PROVIDER=tcp FI_TCP_IFACE={_mpi_interface_request} '
            'via --mpi-tcp-interface'),
        'mpi_interface_note': MPI_INTERFACE_NOTE,
    }


@dataclass(frozen=True)
class HybridPostprocessResult:
    labels: tuple
    time_s: np.ndarray
    electrodes_um: np.ndarray
    reference_index: int
    unreferenced_uV: np.ndarray
    referenced_uV: np.ndarray
    contributions_uV: dict
    record: dict
    output_dir: Path


def aggregate(contributions_uV, time_s, electrodes_um, reference_index, output_dir,
              units=UNITS, sample_rate_hz=None, provenance=None,
              upstream_path=UPSTREAM_POSTPROC, upstream_sha256=UPSTREAM_POSTPROC_SHA256):
    """Validate, write upstream input files, run PostProcess.calc_measure, verify, record."""
    arrays, t, dt_s, rate, xyz, ref, output_dir = validate_inputs(
        contributions_uV, time_s, electrodes_um, reference_index, units, output_dir,
        sample_rate_hz)
    provenance = {} if provenance is None else provenance
    json.dumps(provenance, allow_nan=False)
    if sha256_file(UPSTREAM_LICENSE) != UPSTREAM_LICENSE_SHA256:
        raise ValueError('Upstream LICENSE hash mismatch')
    module = load_upstream_postprocess(upstream_path, upstream_sha256)
    import h5py

    labels = list(arrays)
    output_dir.mkdir(parents=False, exist_ok=False)
    dt_output_ms = dt_s * 1e3
    kwargs = dict(y=list(labels), dt_output=dt_output_ms,
                  mapping_Yy=[(k, k) for k in labels], savelist=[], probes=[],
                  savefolder=str(output_dir), cells_subfolder='cells',
                  populations_subfolder='populations', figures_subfolder='figures',
                  output_file=OUTPUT_FILE, compound_file='{}_sum.h5')
    post = module.PostProcess(**kwargs)

    source_hashes, input_files = {}, {}
    for k in labels:
        path = Path(post.populations_path) / OUTPUT_FILE.format(k, f'{MEASURE}.h5')
        if path.exists():
            raise FileExistsError(path)
        source_hashes[k] = sha256_array(arrays[k])
        with h5py.File(path, 'w-') as f:
            f.create_dataset('data', data=arrays[k].T)
            f['srate'] = 1e3 / dt_output_ms
            f['time_s'] = t
            f['electrodes_um'] = xyz
            f['reference_index'] = ref
            f.attrs['units'] = UNITS
            f.attrs['data_axes'] = 'channel,time'
            f.attrs['source_sha256'] = source_hashes[k]
            f.attrs['label'] = k
            f.attrs['signal_kind'] = SIGNAL_KIND
        input_files[k] = path
    before = {k: sha256_file(p) for k, p in input_files.items()}

    upstream_dict, upstream_sum = post.calc_measure(MEASURE)

    after = {k: sha256_file(p) for k, p in input_files.items()}
    if before != after:
        raise RuntimeError('Input HDF5 files changed during calc_measure')
    if list(upstream_dict) != labels:
        raise RuntimeError('calc_measure returned unexpected labels')
    for k in labels:
        if not np.array_equal(upstream_dict[k], arrays[k].T):
            raise RuntimeError(f'{k}: calc_measure contribution does not round-trip')
    unreferenced = np.ascontiguousarray(np.asarray(upstream_sum, dtype=np.float64).T)
    if unreferenced.shape != (t.shape[0], xyz.shape[0]):
        raise RuntimeError(f'Unexpected summed shape {unreferenced.shape}')

    # Independent sum in reverse order. Each n-term float64 sum, in any order, is
    # within about (n-1)*eps*S of the exact sum, S = sum_i |a_i| per element, so
    # two orders differ by at most 2(n-1)*eps*S. The referenced view adds one
    # subtraction per side, each rounding by at most eps*(S_ch + S_ref). Both
    # bounds use the conservative factor (2n+4)*eps with no absolute constant.
    independent = np.zeros_like(unreferenced)
    magnitude = np.zeros_like(unreferenced)
    for k in reversed(labels):
        independent += arrays[k]
        magnitude += np.abs(arrays[k])
    factor = (2 * len(labels) + 4) * np.finfo(np.float64).eps
    tolerance = factor * magnitude
    error = np.abs(unreferenced - independent)
    if not (error <= tolerance).all():
        raise RuntimeError(f'PostProcess sum differs from independent sum (max {error.max():.3e})')
    referenced = unreferenced - unreferenced[:, [ref]]
    referenced_independent = independent - independent[:, [ref]]
    ref_error = np.abs(referenced - referenced_independent)
    ref_tolerance = factor * (magnitude + magnitude[:, [ref]])
    if not (ref_error <= ref_tolerance).all():
        raise RuntimeError('Referenced view differs from independent referenced sum')

    summed_path = output_dir / 'summed_potential_uV.h5'
    with h5py.File(summed_path, 'w-') as f:
        f['unreferenced_uV'] = unreferenced
        f['referenced_uV'] = referenced
        f['time_s'] = t
        f['electrodes_um'] = xyz
        f['reference_index'] = ref
        f['labels'] = np.array(labels, dtype=h5py.string_dtype())
        f.attrs['units'] = UNITS
        f.attrs['data_axes'] = 'time,channel'
        f.attrs['signal_kind'] = SIGNAL_KIND

    def rel(p):
        p = Path(p).resolve()
        return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)

    record = {
        'schema': 'flyobserver.hybrid_postprocess/v1',
        'signal_kind': SIGNAL_KIND,
        'claims': {'full_hybridLFPy_network_run': False, 'spike_replay_run': False,
                   'Population_or_CachedNetwork_imported': False,
                   'validated_lfp_or_eeg': False},
        'upstream': {
            'repository': UPSTREAM_REPOSITORY, 'commit': UPSTREAM_COMMIT,
            'module_path': rel(module.__file__), 'module_name': module.__name__,
            'module_sha256': sha256_file(module.__file__),
            'license_path': rel(UPSTREAM_LICENSE), 'license': 'GPL-3.0',
            'license_sha256': sha256_file(UPSTREAM_LICENSE), 'modified': False,
            'class': 'PostProcess', 'constructor_kwargs': kwargs,
            'methods_invoked': ['PostProcess.__init__', 'PostProcess.calc_measure'],
            'calc_measure_argument': MEASURE,
            'methods_not_invoked': ['run', 'calc_measure_layer', 'create_tar_archive'],
        },
        'runtime': runtime_versions(module),
        'grid': {'n_time': int(t.shape[0]), 'n_channel': int(xyz.shape[0]),
                 't0_s': float(t[0]), 't_end_s': float(t[-1]), 'dt_s': dt_s,
                 'dt_output_ms': dt_output_ms, 'sample_rate_hz': rate,
                 'time_s_sha256': sha256_array(t)},
        'electrodes_um': xyz.tolist(),
        'electrodes_um_sha256': sha256_array(xyz),
        'reference_index': ref,
        'units': UNITS,
        'axes': {'contribution_input': 'time,channel', 'upstream_hdf5_data': 'channel,time',
                 'summed_output': 'time,channel'},
        'labels': labels,
        'inputs': {k: {'array_sha256': source_hashes[k], 'hdf5_path': rel(input_files[k]),
                       'hdf5_sha256_before_calc_measure': before[k],
                       'hdf5_sha256_after_calc_measure': after[k]} for k in labels},
        'outputs': {'summed_hdf5_path': rel(summed_path),
                    'summed_hdf5_sha256': sha256_file(summed_path),
                    'unreferenced_uV_sha256': sha256_array(unreferenced),
                    'referenced_uV_sha256': sha256_array(referenced)},
        'checks': {'input_hdf5_unchanged': True, 'contributions_round_trip_exact': True,
                   'independent_sum_max_abs_error_uV': float(error.max()),
                   'independent_sum_bitwise_equal': bool(np.array_equal(unreferenced, independent)),
                   'independent_sum_tolerance_rule': '(2*n_labels+4)*eps*sum|contribution|',
                   'referenced_tolerance_rule':
                       '(2*n_labels+4)*eps*(sum|contribution|[ch] + sum|contribution|[ref])',
                   'independent_sum_max_error_over_tolerance':
                       float(np.max(np.divide(error, tolerance, out=np.zeros_like(error),
                                              where=tolerance > 0))),
                   'referenced_max_abs_error_uV': float(ref_error.max()),
                   'per_cell_normalization': False},
        'provenance': provenance,
    }
    if record['runtime']['neuron_imported']:
        raise RuntimeError('NEURON was imported in the postprocess process')
    with open(output_dir / 'postprocess_record.json', 'x') as f:
        json.dump(record, f, indent=2, allow_nan=False)
        f.write('\n')
    return HybridPostprocessResult(
        labels=tuple(labels), time_s=t, electrodes_um=xyz, reference_index=ref,
        unreferenced_uV=unreferenced, referenced_uV=referenced,
        contributions_uV={k: arrays[k] for k in labels}, record=record, output_dir=output_dir)
