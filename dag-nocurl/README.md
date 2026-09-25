# DAG-NoCurl Benchmark Implementation for Discrete Bayesian Networks

This repository runs **DAG-NoCurl / NoCurl-2** on discrete Bayesian network data as a baseline for the method proposed in our study. It reorganizes the original single-file experimental script into modules that support code review, resumable execution, and reproducibility, covering BIF sampling, discrete data encoding, NoCurl-2 optimization, graph metrics, five-run experiments, threshold analysis, and result storage.

> **Important note:** This is not the official repository released by the DAG-NoCurl authors, nor is it a byte-for-byte reproduction of the official results. The core optimization procedure is based on the original paper, the authors' public `BPR.py`, and the existing experimental script. Passing sequentially encoded categorical states to a linear continuous optimizer is an engineering adaptation introduced in this study to support a common discrete BN experimental protocol. This adaptation should be explicitly disclosed in the paper and result tables.

## 1. Acknowledgment of the Original Paper and Authors

DAG-NoCurl was proposed by Yue Yu, Tian Gao, Naiyu Yin, and Qiang Ji:

- Paper: Yue Yu, Tian Gao, Naiyu Yin, Qiang Ji. **DAGs with No Curl: An Efficient DAG Structure Learning Approach**. ICML 2021, PMLR 139:12156–12166.
- Paper page: https://proceedings.mlr.press/v139/yu21a.html
- Official code: https://github.com/fishmoon1234/DAG-NoCurl
- Primary reference file for this implementation: https://github.com/fishmoon1234/DAG-NoCurl/blob/master/BPR.py

We thank the original authors for making their paper and code publicly available. This repository claims no original contribution to the DAG-NoCurl algorithm itself. The core implementation file, [`nocurl.py`](nocurl.py), retains source and author attribution; third-party attribution is provided in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). If you use this project, please cite the original paper first.

```bibtex
@InProceedings{pmlr-v139-yu21a,
  title     = {DAGs with No Curl: An Efficient DAG Structure Learning Approach},
  author    = {Yu, Yue and Gao, Tian and Yin, Naiyu and Ji, Qiang},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  pages     = {12156--12166},
  year      = {2021},
  volume    = {139},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v139/yu21a.html}
}
```

## 2. Purpose and Scope of Reproduction

This project uses the following experimental protocol:

- Networks: `alarm`, `child3`, `child5`, `alarm3`, `child10`, `andes`, `alarm10`, and `pigs`;
- Sample sizes: 200, 300, 500, 1000, 2000, and 5000;
- Random seeds: 42, 43, 44, 45, and 46;
- Five runs for each network–sample-size combination;
- Reporting of means and sample standard deviations for directed-edge Precision, Recall, F1, and SHD;
- Ten processes, with each process restricted to one BLAS thread.

The original DAG-NoCurl primarily targets linear or generalized structural equation models. Here, the inputs come from discrete BIF models: states are encoded as `0, 1, ..., r_i-1` in the order declared in the BIF file, divided by `r_i-1` by default, and then centered column-wise before being passed to NoCurl-2. This numerical ordering does not necessarily have a meaningful metric interpretation. The results therefore represent “DAG-NoCurl under this project's discrete encoding protocol” and must not be presented as official results from the original paper on discrete BNs.

## 3. Repository Structure

```text
dag-nocurl-benchmark/
├── config.py                 # All experimental parameters (edit this first)
├── data_utils.py             # BIF loading, ancestral sampling, state encoding, and preprocessing
├── nocurl.py                 # Core NoCurl-2 optimization
├── metrics.py                # Precision, Recall, F1, and SHD
├── experiment.py             # Experiment for one network/sample size/seed
├── result_utils.py           # Five-run aggregation and threshold selection
├── storage.py                # Atomic saving, hashes, and environment information
├── run_experiments.py        # Main entry point
├── data/bif/README.md        # Instructions for placing BIF files
├── tests/                    # Unit tests
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Development and testing dependencies
├── pyproject.toml            # Code quality tool configuration
├── CITATION.cff              # Citation information
├── THIRD_PARTY_NOTICES.md    # Upstream attribution and modification notices
└── LICENSE                   # Apache License 2.0
```

## 4. Environment Setup

Recommended environment:

- Linux (Ubuntu 20.04, 22.04, or 24.04);
- Python 3.10 or 3.11;
- A 64-bit CPU;
- At least 16 GB of RAM; 32 GB or more is recommended for concurrent runs on large networks;
- No GPU required.

Create a dedicated environment:

```bash
conda create -n dag-nocurl python=3.10 -y
conda activate dag-nocurl
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies are specified using compatible version ranges rather than being presented as an official fixed environment. Each run records the Python version, operating system, and installed package versions in `config.json` in the results directory for traceability. For experiments reported in a paper, also run `python -m pip freeze > environment-lock.txt` and archive the file with the experiments.

## 5. Data Preparation

Place the eight BIF files in `data/bif/`:

```text
data/bif/alarm.bif
data/bif/alarm3.bif
data/bif/alarm10.bif
data/bif/child3.bif
data/bif/child5.bif
data/bif/child10.bif
data/bif/andes.bif
data/bif/pigs.bif
```

The program performs ancestral sampling directly from the CPDs in each BIF file. It does not use the older `BayesianModelSampling` implementation in pgmpy, avoiding errors caused by the removal of `np.mat` in NumPy 2.0. The custom sampler still generates data according to the conditional probability distributions in the BIF file, but the sample sequence is not guaranteed to match that of the older pgmpy sampler row by row for the same random seed.

If `alarm3`, `alarm10`, `child3`, `child5`, and `child10` are extended networks produced by replicating, joining, or renaming base networks, disclose the construction rules or generation scripts in both the paper and the data documentation to ensure reproducibility. Do not release experimental results without explaining the provenance of these derived BIF files.

## 6. Parameter Settings

All commonly used parameters are located at the top of [`config.py`](config.py).

| Parameter | Default | Description |
|---|---:|---|
| `N_JOBS` | `10` | Number of concurrent independent experiment processes; the actual number does not exceed the CPU core count or task count. |
| `THREADS_PER_WORKER` | `1` | Number of BLAS/OpenMP threads per process, preventing each of the ten processes from spawning additional threads. |
| `NETWORKS` | 8 networks | Names of the BIF networks to evaluate. |
| `SAMPLE_SIZES` | 6 sample sizes | Number of observations sampled for each network. |
| `SEEDS` | 42–46 | Random seeds for five independent repetitions. |
| `LAMBDA1` | `10.0` | Acyclicity penalty coefficient for the first NoCurl-2 stage. |
| `LAMBDA2` | `1000.0` | Stronger acyclicity penalty coefficient for the second stage. The default matches the NoCurl-2 linear experimental setting in the authors' repository. |
| `H_TOL` | `1e-8` | `ftol` for the three L-BFGS-B optimizations. This is an optimization stopping tolerance, not a graph threshold. |
| `INTERNAL_THRESHOLD` | `0.3` | Threshold applied to the initial weights before estimating reachability and node potentials. |
| `GRAPH_THRESHOLDS` | 0.0–1.0 | List of thresholds examined when constructing graphs from the final continuous weights. |
| `FIXED_THRESHOLD_MODE` | `False` | If `True`, uses only the prespecified fixed threshold; if `False`, performs oracle threshold analysis. |
| `FIXED_GRAPH_THRESHOLD` | `0.3` | Threshold used in fixed-threshold mode. |
| `SCALE_BY_DECLARED_CARDINALITY` | `True` | Divides the encoding of variable i by `r_i-1` to reduce scale differences caused by different cardinalities. |
| `CENTER_DATA` | `True` | Subtracts the sample mean from each column to match a linear loss without an explicit intercept. |
| `STD_DDOF` | `1` | Uses sample standard deviations to summarize the five runs. |
| `SAVE_DATA` | `True` | Saves the encoded data and ground-truth adjacency matrix for each experiment. |
| `SAVE_WEIGHTS` | `True` | Saves the initial weights, final continuous weights, and potential function. |
| `RESUME_COMPLETED` | `True` | Reuses completed tasks when the configuration and BIF hashes match. |

### Threshold Selection Must Be Disclosed Accurately

With the default setting `FIXED_THRESHOLD_MODE=False`, the ground-truth graph is used to select the threshold with the highest mean F1 across five runs for each network–sample-size combination. Ties are resolved first by lower mean SHD and then by a smaller threshold. This is **oracle/ground-truth-based threshold selection** and uses test ground truth; it should therefore not be presented as a fair main result without disclosure.

Recommended practice:

1. For the main comparison table, set `FIXED_THRESHOLD_MODE=True` and fix `FIXED_GRAPH_THRESHOLD=0.3` before running the experiments, or select the threshold using independent validation data only.
2. Report the default threshold sweep separately as a sensitivity analysis or an oracle upper bound.
3. Do not select thresholds using the same test ground truth and then describe the results as fully independent test results.

## 7. Execution Procedure

Before running the experiments, perform the checks below:

```bash
python -m pytest
python run_experiments.py
```

Simply launch or execute **`run_experiments.py`**. The script proceeds as follows:

1. Checks the parameters and all BIF files;
2. Records SHA-256 hashes of the source code and BIF files, the software environment, and the complete configuration;
3. Creates 8 × 6 × 5 = 240 tasks;
4. Independently samples, encodes, scales, and centers data from the BIF model for each task;
5. Runs NoCurl-2 to obtain an initial solution that is not constrained to be acyclic, estimates node potentials from reachability, and then optimizes edge weights along the directions determined by those potentials;
6. Converts continuous weights into DAGs using the fixed threshold or threshold list;
7. Computes directed-edge Precision, Recall, F1, and SHD;
8. Saves each completed experiment atomically and regenerates the current summary whenever a task result is received, preserving completed results if execution is interrupted;
9. Writes the mean and standard deviation once all five seeds for a network–sample-size combination have completed.

If any of the three NoCurl-2 L-BFGS-B stages returns an unsuccessful status but its outputs remain finite, the program marks the result as `completed_with_optimizer_warning` and retains the diagnostics rather than silently treating it as fully converged. Check the warning counts and `optimizer_diagnostics` before reporting results.

## 8. Metric Definitions

Let E be the ground-truth directed-edge set and Ê the predicted directed-edge set:

- Precision = `|E ∩ Ê| / |Ê|`;
- Recall = `|E ∩ Ê| / |E|`;
- F1 is the harmonic mean of Precision and Recall;
- SHD = extra skeleton edges + missing skeleton edges + reversed edges.

A reversed edge contributes one reversal to SHD. Precision, Recall, and F1 require an exact directional match for an edge to count as a true positive. When no edges are predicted, Precision and F1 are defined as 0.

## 9. Output Files

Results are written to:

```text
results_nocurl_threshold_sweep/<configuration_id>/
```

Here, `<configuration_id>` is generated from the configuration settings that affect the results, the environment, and the BIF hashes. The main files are:

| File | Contents |
|---|---|
| `config.json` | Complete configuration, environment, source code hashes, and BIF hashes. |
| `all_runs.csv` | Run status and elapsed time for each network, sample size, and seed. |
| `all_threshold_runs.csv` | Raw metrics for every run at each threshold. |
| `threshold_summary.csv` | Five-run means and standard deviations for each threshold. |
| `final_best_results.csv/json` | Summary for the fixed or oracle-selected threshold. |
| `<network>/summary_<network>_all_sizes.csv` | Separate summary of all six sample sizes for one network. |
| `<network>/n_<N>/seed_<S>/result.json` | Complete status and metrics for one experiment. |
| `<network>/n_<N>/seed_<S>/data.npz` | Encoded data, ground-truth adjacency matrix, and node names. |
| `<network>/n_<N>/seed_<S>/weights.npz` | Continuous weights, initial weights, and potential function. |

Summary CSV files contain both numeric columns and `mean ± std` strings suitable for reading in a paper. If any task fails, the setting is marked `complete=False`; results from fewer than five runs are not presented as complete five-run statistics.

## 10. Engineering Changes Relative to the Original Single-File Script

- Fixed the syntax error caused by a duplicate `method` keyword in the first-stage `scipy.optimize.minimize` call;
- Removed the extra `]` following the missing-file list in the main function;
- Replaced the older pgmpy sampling path that triggers the `np.mat` error with an ancestral sampler compatible with NumPy 2.x;
- Separated training from threshold evaluation: continuous weights are fitted only once for a given dataset;
- Split configuration, data handling, the algorithm, metrics, tasks, summaries, and storage into separate modules;
- Added atomic writes, resumable execution, input/source code hashes, optimizer diagnostics, and summaries updated after each task;
- Added a fixed-threshold mode and explicit warnings for oracle mode.

These changes do not alter NoCurl-2's core objective function, analytical gradients, matrix-exponential reachability computation, potential-function projection, or three-stage L-BFGS-B procedure. They do, however, change the sampling random-number stream and experimental management relative to the older script.

## 11. Fair Comparison and Limits on Explaining Low Performance

If DAG-NoCurl's Arc-F1 on these discrete networks is substantially lower than in the original paper's continuous simulation results, first examine:

- A mismatch between the algorithm's assumptions and discrete categorical data;
- Artificial linear distances introduced by arbitrary sequential encoding of categorical states;
- Sensitivity of continuous weights and final thresholds under small sample sizes;
- Differences in the size, density, or replicated structure of the extended BIF networks;
- Whether all methods use the same data, ground-truth edge directions, and SHD definition;
- Whether oracle-threshold results and fixed-threshold results are mixed in the same table.

These possible explanations require verification through logs, threshold curves, and additional experiments. The discrepancy should not be attributed directly to implementation errors or to the algorithm itself being ineffective. Report fixed-threshold results, threshold sensitivity, optimizer warning counts, and runtime together where possible.

## 12. Checks Before Uploading to GitHub

```bash
python -m pytest
python -m ruff check .
git init
git add .
git status
git commit -m "Add modular DAG-NoCurl discrete-BN benchmark"
```

Do not commit large experimental output directories, caches, or local environments; `.gitignore` already excludes these files. Whether BIF data can be released depends on its source and license. If redistribution is not permitted, retain only `data/bif/README.md` and provide lawful download or generation instructions.

## 13. License

The upstream DAG-NoCurl repository uses the Apache License 2.0. This project incorporates a core procedure based on its public implementation and retains source attribution and third-party notices. See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details. Citation requirements are separate from the software license; the original paper should still be cited when using this project.
