#!/usr/bin/env python3
"""Reproduce CPU geometry checks using the current interpreter; no installs."""
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
         _MNE_FAKE_HOME_DIR=str(root/'runtime-home'), MNE_DONTWRITE_HOME='true',
         OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
for name in ('common.py','t0_minmax.py','t1_quadrature.py','t2_reference.py','t3_information.py'):
    print(f'Running {name}',flush=True)
    subprocess.run([sys.executable,str(root/name)],env=env,check=True)
supplement=root/'t3_position_sensitivity.py'
if supplement.exists():
    subprocess.run([sys.executable,str(supplement)],env=env,check=True)
