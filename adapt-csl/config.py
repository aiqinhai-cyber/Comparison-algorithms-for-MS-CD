"""All user-editable settings for the Adapt-CSL benchmark."""

from __future__ import annotations

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
BIF_DIR = PROJECT_DIR / "data" / "bif"
OUTPUT_ROOT = PROJECT_DIR / "adapt_csl_results"

BIF_FILES = [
    "alarm.bif",
    "alarm3.bif",
    "alarm10.bif",
    "child3.bif",
    "child5.bif",
    "child10.bif",
    "andes.bif",
    "pigs.bif",
]
SAMPLE_SIZES = [200, 300, 500, 1000, 2000, 5000]
SEEDS = [42, 43, 44, 45, 46]
NUM_RUNS = len(SEEDS)
STD_DDOF = 1

# Preserve the uploaded script's default. Increase this value only after
# checking available CPU cores and memory.
NUM_WORKERS = 1

# Paper setting: discrete G2 CI test, significance level 0.01.
ALPHA = 0.01

# The paper does not publish the complete learnPC subroutine. We use a
# target-wise, order-independent PC-simple search. None searches every feasible
# level; 3 is a practical default for the large networks in this benchmark.
MAX_CONDITIONING_SET = 3

# The article specifies BDeu but does not report its equivalent sample size.
BDEU_EQUIVALENT_SAMPLE_SIZE = 20.0
SCORE_TOLERANCE = 1e-10

# Section 5.1.3 randomly directs any edge still undirected after Meek rules.
RANDOMLY_COMPLETE_UNDIRECTED = True

OVERWRITE_COMPLETED = False
SHOW_PROGRESS = True
CAPTURE_WARNINGS = True

# P/R/F1 formatted means are integer percentages; SHD mean is an integer.
# All formatted standard deviations retain two decimal places. Raw CSV files
# always retain the full floating-point values.
PERCENT_METRICS_IN_FORMATTED_SUMMARY = True
STD_DECIMALS = 2

IMPLEMENTATION_NAME = "Independent Python reproduction of Adapt-CSL"
IMPLEMENTATION_VERSION = "modular-paper-algorithm1-v1"
PAPER_TITLE = (
    "Enhancing causal structure learning with adaptive local-to-global "
    "skeleton construction"
)
PAPER_DOI = "10.1016/j.asoc.2026.115654"
