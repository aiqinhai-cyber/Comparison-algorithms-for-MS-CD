"""Central configuration for the DAG-NoCurl discrete-BN benchmark."""

from __future__ import annotations

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# CPU process/thread settings. Set thread variables before importing NumPy.
# -----------------------------------------------------------------------------
N_JOBS = 10
THREADS_PER_WORKER = 1

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = str(THREADS_PER_WORKER)


# -----------------------------------------------------------------------------
# Paths and benchmark protocol.
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
BIF_DIR = PROJECT_DIR / "data" / "bif"
OUTPUT_DIR = PROJECT_DIR / "results_nocurl_threshold_sweep"

NETWORKS = [
    "alarm",
    "child3",
    "child5",
    "alarm3",
    "child10",
    "andes",
    "alarm10",
    "pigs",
]
SAMPLE_SIZES = [200, 300, 500, 1000, 2000, 5000]
SEEDS = [42, 43, 44, 45, 46]


# -----------------------------------------------------------------------------
# Original NoCurl-2 parameters from the authors' repository.
# -----------------------------------------------------------------------------
LAMBDA1 = 10.0
LAMBDA2 = 1000.0
H_TOL = 1e-8
INTERNAL_THRESHOLD = 0.3


# -----------------------------------------------------------------------------
# Final graph threshold analysis.
# WARNING: oracle selection uses ground-truth F1 and is unsuitable as an
# undisclosed primary benchmark. Use FIXED_THRESHOLD_MODE=True for a fair main
# comparison and report the oracle sweep only as sensitivity/upper-bound study.
# -----------------------------------------------------------------------------
GRAPH_THRESHOLDS = [index / 10.0 for index in range(11)]
FIXED_THRESHOLD_MODE = False
FIXED_GRAPH_THRESHOLD = 0.3


# -----------------------------------------------------------------------------
# Discrete-BN adaptation.
# State labels are mapped using the BIF-declared order, divided by r_i - 1, and
# centered. This is an engineering adaptation, not the original data protocol.
# -----------------------------------------------------------------------------
SCALE_BY_DECLARED_CARDINALITY = True
CENTER_DATA = True


# -----------------------------------------------------------------------------
# Result statistics, persistence, and logging.
# -----------------------------------------------------------------------------
STD_DDOF = 1
SAVE_DATA = True
SAVE_WEIGHTS = True
RESUME_COMPLETED = True
VERBOSE = True
CAPTURE_SAMPLING_WARNINGS = True

EVALUABLE_STATUSES = {"ok", "completed_with_optimizer_warning"}
METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "shd",
    "learned_edges",
    "fit_seconds",
]


# -----------------------------------------------------------------------------
# Attribution and versioning.
# -----------------------------------------------------------------------------
IMPLEMENTATION_VERSION = "nocurl2-discrete-bn-modular-v1"
PAPER_TITLE = "DAGs with No Curl: An Efficient DAG Structure Learning Approach"
PAPER_URL = "https://proceedings.mlr.press/v139/yu21a.html"
UPSTREAM_REPOSITORY = "https://github.com/fishmoon1234/DAG-NoCurl"
UPSTREAM_CORE_FILE = (
    "https://github.com/fishmoon1234/DAG-NoCurl/blob/master/BPR.py"
)
