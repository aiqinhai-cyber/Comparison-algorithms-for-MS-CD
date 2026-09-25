# GOLEM Discrete-BN Benchmark

Comparative experiments on discrete Bayesian network structure learning using GOLEM (an unofficial PyTorch reimplementation and experimental adaptation).

## Important Notice

This project conducts reproduction experiments based on the GOLEM method proposed by Ignavier Ng, AmirEmad Ghassami, and Kun Zhang. The algorithmic ideas, theoretical analysis, and original implementation of GOLEM belong to the original authors and their respective institutions.

- Original paper: **On the Role of Sparsity and DAG Constraints for Learning Linear DAGs**
- Authors: **Ignavier Ng, AmirEmad Ghassami, Kun Zhang**
- Venue: **Advances in Neural Information Processing Systems 33 (NeurIPS 2020)**
- Paper page: [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/hash/d04d42cdf14579cd294e5079e0745411-Abstract.html)
- Paper PDF: [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2020/file/d04d42cdf14579cd294e5079e0745411-Paper.pdf)
- arXiv: [arXiv:2006.10201](https://arxiv.org/abs/2006.10201)
- Official code: [ignavierng/golem](https://github.com/ignavierng/golem)

This repository is not an official discrete-data version released by the GOLEM authors, and it does not claim their endorsement, maintenance, or technical support. For a strict reproduction of the original paper, please prioritize the paper, its supplementary material, and the authors' official code.

## Acknowledgments and Attribution

We thank the GOLEM authors for making their paper and source code publicly available, enabling subsequent research to understand, examine, and compare the method. The GOLEM objective, the GOLEM-EV/GOLEM-NV distinction, the soft sparsity and DAG penalties, the strategy of initializing NV with EV, and the post-processing approach of thresholding and progressively removing weak edges are derived from the original paper and official code.

The authors' official repository uses the [Apache License 2.0](https://github.com/ignavierng/golem/blob/main/LICENSE). When publishing or redistributing any code directly derived from that repository, comply with its license and retain the original copyright, license, and required attribution notices. Do not remove or obscure information identifying the original authors.

The main additions in this project are engineering adaptations for a specific experimental protocol, including:

- Implementing the GOLEM optimization objective in modern PyTorch;
- Reading BIF Bayesian networks and their ground-truth DAGs;
- Encoding discrete variables according to the state order declared in the BIF files;
- Running batches of experiments across multiple networks, sample sizes, and random seeds;
- Computing directed precision, recall, F1, and SHD;
- Saving individual results, five-run results, and means ± standard deviations;
- Saving weighted adjacency matrices, binary DAGs, optimization histories, and execution-environment information.

## Overview of the Original GOLEM Method

GOLEM stands for **Gradient-based Optimization of dag-penalized Likelihood for learning linEar dag Models**. The original method learns a weighted adjacency matrix for a linear structural equation model through continuous optimization.

GOLEM uses a likelihood-based objective with two types of soft penalties:

$$
S(B;X) = L(B;X) + \lambda_1 \|B\|_1 + \lambda_2 h(B),
$$

where:

- \(B\) is the weighted adjacency matrix to be learned;
- \(L(B;X)\) is the negative log-likelihood of the linear model;
- \(\lVert B\rVert_1\) is a sparsity penalty intended to reduce unnecessary edges;
- \(h(B)=\operatorname{tr}(\exp(B\circ B))-d\) is the soft acyclicity penalty;
- \(\lambda_1\) and \(\lambda_2\) control sparsity and acyclicity, respectively.

The original paper discusses two main settings:

- **GOLEM-EV** assumes equal noise variances across variables.
- **GOLEM-NV** allows unequal noise variances. The official implementation recommends initializing GOLEM-NV with the GOLEM-EV solution to reduce the risk of converging to undesirable local solutions.

Because the optimized matrix may contain small nonzero coefficients or cycles with finite samples, this project follows the post-processing approach described in the original paper: first remove edges whose absolute weights do not exceed the threshold; if cycles remain, remove edges in ascending order of absolute weight until a DAG is obtained.

## Research Purpose

This project evaluates GOLEM as a causal structure learning baseline under a unified experimental protocol on the following discrete Bayesian network datasets:

- `alarm.bif`
- `alarm3.bif`
- `alarm10.bif`
- `child3.bif`
- `child5.bif`
- `child10.bif`
- `andes.bif`
- `pigs.bif`

The following sample sizes are used for each network:

```text
200, 300, 500, 1000, 2000, 5000
```

Each network–sample-size setting is run independently with the following five random seeds:

```text
42, 43, 44, 45, 46
```

The default configuration therefore includes:

\[
8\ \text{networks}\times 6\ \text{sample sizes}\times 5\ \text{random seeds}=240\ \text{experiments}.
\]

## Key Limitation Regarding Discrete Data

**GOLEM was originally developed for continuous linear structural equation models, particularly linear Gaussian models. This project uses discrete categorical data, which do not satisfy GOLEM's original modeling assumptions.**

Each discrete variable is encoded according to the state order declared in its BIF file:

```text
state_0 -> 0
state_1 -> 1
...
state_r -> r
```

The encoded data are centered, or optionally standardized, before being supplied to GOLEM. This introduces a numerical ordering and distances between states; for nominal variables, such an ordering generally has no natural statistical meaning. Consequently:

- These experiments compare baselines under model misspecification;
- The results do not imply that GOLEM natively supports discrete variables;
- The results do not replace evaluation of structure learning methods specifically designed for discrete data;
- This limitation should be stated explicitly when comparing GOLEM with discrete-data methods such as PC, BOSS, and MMHC;
- Poor performance should not be attributed solely to GOLEM itself, as it may also reflect a mismatch between the data type and modeling assumptions.

A statement such as the following may be used in the experimental settings of a paper:

> GOLEM was originally developed for continuous linear structural equation models. To include it as a continuous-optimization baseline on the categorical Bayesian-network benchmarks, we encoded each state according to its order declared in the corresponding BIF file and centered the resulting numerical variables. This constitutes a model-misspecified evaluation; therefore, the results should not be interpreted as evidence that GOLEM natively supports discrete data.

## Project Structure

```text
golem-discrete-benchmark/
├── README.md
├── config.py
├── run_experiments.py
├── golem.py
├── data_utils.py
├── metrics.py
├── result_utils.py
│
├── data/
│   ├── bif/
│   │   ├── alarm.bif
│   │   ├── alarm3.bif
│   │   ├── alarm10.bif
│   │   ├── child3.bif
│   │   ├── child5.bif
│   │   ├── child10.bif
│   │   ├── andes.bif
│   │   └── pigs.bif
│   └── samples/
│
└── results/
```

The responsibilities of each file are as follows:

| File | Purpose |
| --- | --- |
| `config.py` | Centralizes networks, sample sizes, random seeds, GOLEM parameters, paths, and output formatting |
| `run_experiments.py` | Runs experiment batches and coordinates data, models, evaluation, and result saving |
| `golem.py` | Implements the GOLEM objective, PyTorch training, thresholding, and DAG post-processing |
| `data_utils.py` | Handles BIF loading, sample generation, CSV lookup, state encoding, and preprocessing |
| `metrics.py` | Computes directed precision, recall, F1, SHD, and skeleton metrics |
| `result_utils.py` | Saves adjacency matrices, flattens results, aggregates five-run experiments, and records environment information |

## Environment Setup

A dedicated Python environment is recommended. Python 3.10 or later is suitable.

### Conda

```bash
conda create -n golem-benchmark python=3.10 -y
conda activate golem-benchmark
```

Install PyTorch using the appropriate command from the [official PyTorch installation page](https://pytorch.org/get-started/locally/) for your machine's CUDA version, then install the remaining dependencies:

```bash
pip install numpy pandas scipy networkx pgmpy tqdm
```

For CPU-only execution, you can also install the packages directly:

```bash
pip install torch numpy pandas scipy networkx pgmpy tqdm
```

## Data Placement

The recommended location for BIF files is:

```text
data/bif/
```

For compatibility with the earlier single-file experiment layout, the code also searches the project root for BIF files. Both of the following locations are therefore supported:

```text
data/bif/alarm.bif
```

or:

```text
golem-discrete-benchmark/alarm.bif
```

If a public repository includes third-party BIF files or pregenerated data, verify their sources, licenses, and redistribution conditions separately, and provide the corresponding references. The license for the GOLEM paper and code does not automatically cover these data files.

## Data Modes

Set `DATA_MODE` in `config.py`.

### 1. `resample`: Generate New Data for Each Seed

```python
DATA_MODE = "resample"
SAVE_RESAMPLED_DATA = True
```

The program independently generates data from the BIF probability distribution for each random seed. This is the default setting and is more appropriate for five repeated experiments.

Generated data are saved to:

```text
results/generated_samples/<network>/<network>_<size>_seed<seed>.csv
```

### 2. `seeded_files`: Load Pregenerated Seed-Specific Data

```python
DATA_MODE = "seeded_files"
```

For example:

```text
data/samples/alarm/alarm_200_seed42.csv
data/samples/alarm/alarm_200_seed43.csv
data/samples/alarm/alarm_200_seed44.csv
data/samples/alarm/alarm_200_seed45.csv
data/samples/alarm/alarm_200_seed46.csv
```

### 3. `shared_file`: Use the Same Dataset for All Five Seeds

```python
DATA_MODE = "shared_file"
```

For example:

```text
data/samples/alarm/alarm_200.csv
```

This mode is not recommended for the main five-run experiments. The current GOLEM implementation uses full-batch optimization and zero-matrix initialization; when the same data are reused, changing only the optimizer's random seed generally does not create genuinely independent data replications.

## Parameter Settings

All adjustable parameters are defined in `config.py`.

The default GOLEM-EV parameters match the example settings provided in the authors' official repository:

```python
EV_LAMBDA_1 = 0.02
EV_LAMBDA_2 = 5.0
EV_NUM_ITER = 100_000
EV_LEARNING_RATE = 1e-3
```

The default GOLEM-NV parameters are:

```python
NV_LAMBDA_1 = 0.002
NV_LAMBDA_2 = 5.0
NV_NUM_ITER = 100_000
NV_LEARNING_RATE = 1e-3
```

Other important parameters are:

```python
VARIANT = "ev"              # ev, nv, or both
NV_INITIALIZATION = "ev"   # ev or zero
GRAPH_THRESHOLD = 0.3
PREPROCESSING = "center"   # center or zscore
DEVICE = "auto"            # auto, cpu, or cuda
DTYPE = "float32"          # float32 or float64
CHECKPOINT_INTERVAL = 5_000
```

Although these parameters follow the official settings, discrete-state encoding changes the data scale and the interpretation of the model. Report these parameters in formal experimental comparisons. Sensitivity analyses of `GRAPH_THRESHOLD`, `LAMBDA_1`, and `LAMBDA_2` are recommended; parameters used in continuous-data experiments should not be assumed optimal for discrete data simply because they come from the official implementation.

## Running the Experiments

Run the following command from the project root:

```bash
python run_experiments.py
```

At startup, the program checks all requested BIF files to avoid discovering missing files only after lengthy training.

The default full benchmark consists of 240 runs, each with 100,000 optimization steps, and may take substantial time. For formal baseline comparisons, reducing the iteration count is not recommended as a substitute for the complete configuration. If a change is necessary, report it accurately in both the paper and configuration file.

## Output Files

For GOLEM-EV on the `alarm` network with 200 observations, the output layout is:

```text
results/ev/alarm/n200/
├── seed_42/
├── seed_43/
├── seed_44/
├── seed_45/
├── seed_46/
├── five_runs_results.csv
└── mean_std_metrics.csv
```

Each seed directory contains:

```text
result.json
true_adjacency.csv
ev/weighted_adjacency.npy
ev/processed_weighted_adjacency.npy
ev/estimated_dag.csv
ev/optimization_history.csv
ev/checkpoints/
```

Overall summary files include:

```text
results/ev/all_results.json
results/ev/all_runs.csv
results/ev/mean_std_results.csv
results/ev/experiment_config.json
results/ev/environment.json
```

Specifically:

- `all_runs.csv` contains the complete results of each independent run;
- `mean_std_results.csv` contains the mean and standard deviation across five runs for each network and sample size;
- Each `n<size>` directory also stores its corresponding five-run results and summary;
- Precision, recall, and F1 are converted to percentages by default;
- Means are saved as integers by default;
- Standard deviations are saved to two decimal places, for example, `46 ± 1.14`;
- SHD means are saved as integers and standard deviations to two decimal places, for example, `50 ± 1.58`.

## Evaluation Metrics

All adjacency matrices use the convention:

```text
adjacency[parent, child] = 1
```

Directed-edge precision, recall, and F1 are defined as:

\[
\mathrm{Precision}=\frac{TP}{TP+FP},
\qquad
\mathrm{Recall}=\frac{TP}{TP+FN},
\]

\[
\mathrm{F1}=\frac{2\cdot \mathrm{Precision}\cdot \mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}.
\]

In this project, SHD counts the minimum number of structural edits required to delete, add, or reverse edges, with an edge reversal counted as one edit. State this convention explicitly in the paper, as different codebases may count a reversal as either one or two operations.

## Reproducibility

The program sets random seeds for Python, NumPy, PyTorch CPU, and CUDA, and enables deterministic computation where available. Each experiment also records:

- Network name and sample size;
- Random seed;
- Data file path;
- GOLEM variant and hyperparameters;
- Preprocessing method;
- PyTorch, NumPy, pandas, and NetworkX versions;
- CUDA availability, CUDA version, and GPU name;
- Training time and final objective value.

Even with deterministic options enabled, small numerical differences may occur across operating systems, GPUs, CUDA versions, PyTorch versions, or underlying linear algebra libraries. When publishing experimental results, it is recommended to release both `experiment_config.json` and `environment.json`.

## Differences from the Authors' Official Implementation

| Aspect | Authors' official implementation | This project |
| --- | --- | --- |
| Framework | TensorFlow 1.x | PyTorch |
| Main data assumptions | Continuous linear models | Model-misspecified evaluation using ordinally encoded discrete states |
| Data source | Linear SEM experimental settings in the original paper | BIF Bayesian networks and their discrete samples |
| GOLEM objective | Authors' implementation | Modern PyTorch reimplementation based on the paper's objective |
| Repeated experiments | Official experimental protocol | Batch execution with five fixed seeds |
| Output | Outputs defined by the official repository | Additional per-run CSV files, means ± standard deviations, and environment records |

Accordingly, this project is best described as:

> An unofficial PyTorch reimplementation of GOLEM and an adaptation for discrete BIF experiments used in baseline comparisons. It is neither a file-by-file port of the authors' official code nor a newly derived GOLEM algorithm for discrete variables.

## Citation

If this project is used in a paper, report, or public experiment, please cite at least the original GOLEM paper:

```bibtex
@inproceedings{Ng2020role,
  author    = {Ng, Ignavier and Ghassami, AmirEmad and Zhang, Kun},
  title     = {On the Role of Sparsity and DAG Constraints for Learning Linear DAGs},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  year      = {2020}
}
```

If you use or refer to the authors' official implementation, also link to it in your paper or project documentation:

```text
https://github.com/ignavierng/golem
```

When using BIF networks or other third-party data in this repository, provide the appropriate dataset references according to their actual sources. Citing the GOLEM paper does not replace dataset citations.

## Research and Usage Statements

- This project is intended for research reproduction and baseline comparisons; it does not guarantee the validity of inferred causal relationships.
- Directed graphs learned from observational data should not be interpreted directly as true causal mechanisms without additional assumptions and domain validation.
- Report discrete encoding, preprocessing, parameters, failed experiments, and mismatches with modeling assumptions accurately.
- Do not describe this project as an author-released “discrete GOLEM” or “official PyTorch GOLEM.”
- If you modify this project, distinguish the original GOLEM contributions, this project's adaptations, and your own additions in the documentation.

## Licensing Recommendations

The original GOLEM repository uses the Apache License 2.0. Before publicly releasing this repository, add a clear `LICENSE` file to the repository root and determine the appropriate license based on the actual provenance of the code:

1. If the repository includes or adapts substantial code from the authors' repository, comply with Apache-2.0 and retain the relevant copyright and license notices.
2. If the code is independently implemented solely from the published paper, still provide clear citations to the original paper and official code.
3. Third-party BIF files, data, and dependencies have their own licenses, which must be checked separately.
4. This README is not legal advice. If redistribution rights are uncertain, consult your institution or a qualified professional.

---

We again thank Ignavier Ng, AmirEmad Ghassami, and Kun Zhang for their contributions to the GOLEM method and its open-source implementation.
