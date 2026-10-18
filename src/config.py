import os
import re
import sys
import random
import platform
import warnings
from pathlib import Path

import numpy as np
import matplotlib

# Non-interactive backend makes every figure, including SHAP summary plots,
# save reliably when run from PyCharm, terminal, or headless environments.
matplotlib.use(os.environ.get('MPLBACKEND', 'Agg'))
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.environ.get(
    'R2_DATA_PATH',
    PROJECT_ROOT / 'data' / 'Admisiion Main Dataset 2025.csv'
))
OUTDIR = Path(os.environ.get('R2_OUTDIR', PROJECT_ROOT / 'outputs'))
OUTDIR.mkdir(parents=True, exist_ok=True)

# Publication-quality raster export settings
FIG_DPI = 800
CONFUSION_CMAP = 'Blues'
plt.rcParams['savefig.dpi'] = FIG_DPI


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {'0', 'false', 'no', 'off'}


# Default to deterministic multi-core CPU execution (fast for N=1,139 tabular data
# and avoids GPU/CPU floating-point divergence or DMatrix device-mismatch warnings).
# Set environment variable R2_USE_CUDA=1 to explicitly enable GPU mode.
CUDA_REQUESTED = _env_bool('R2_USE_CUDA', False)
CUDA_DEVICE = os.environ.get('R2_CUDA_DEVICE', '0')


def detect_cuda():
    if not CUDA_REQUESTED:
        return False
    try:
        import subprocess
        completed = subprocess.run(
            ['nvidia-smi', '-L'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
        return completed.returncode == 0 and 'GPU' in completed.stdout
    except Exception:
        return False


USE_CUDA = detect_cuda()
SEARCH_N_JOBS = 1 if USE_CUDA else -1
SHOW_PLOTS = _env_bool('R2_SHOW_PLOTS', False)


def safe_outcome_filename(label: str) -> str:
    """Make a readable, cross-platform-safe filename component."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(label)).strip('_')


def display(obj):
    """PyCharm-friendly replacement for notebook display()."""
    try:
        from IPython.display import display as ipy_display
        ipy_display(obj)
    except Exception:
        print(obj)


def maybe_show():
    if SHOW_PLOTS:
        plt.show()


def print_runtime_banner():
    print('=' * 80)
    print('R2_ML Analysis Framework — Reproducible Pipeline')
    print('=' * 80)
    print('Python:', sys.version.split()[0])
    print('Platform:', platform.platform())
    print('Dataset:', DATA_PATH)
    print('Output directory:', OUTDIR)
    print('Execution mode:', f'CUDA (device {CUDA_DEVICE})' if USE_CUDA else 'Deterministic Multi-Core CPU (n_jobs=-1)')
    print('=' * 80)
