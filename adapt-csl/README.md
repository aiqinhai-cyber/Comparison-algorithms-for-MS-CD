# Adapt-CSL: Experimental Reproduction on Discrete Bayesian Networks

This repository provides an independent Python reproduction of Adapt-CSL for the comparative experiments in our paper. The implementation follows Algorithm 1, Theorems 3 and 4, Equations (7)–(17), and the experimental settings in the original paper. It is not official code released by the original authors.

## Original Paper and Authors

Please cite the original paper first:

- Tianci Li, Debo Cheng, Zhangling Duan, Zhaolong Ling.
- *Enhancing causal structure learning with adaptive local-to-global skeleton construction*.
- Applied Soft Computing, 201, 115654, 2026.
- DOI: [10.1016/j.asoc.2026.115654](https://doi.org/10.1016/j.asoc.2026.115654)

~~~bibtex
@article{li2026adaptcsl,
  title   = {Enhancing causal structure learning with adaptive local-to-global skeleton construction},
  author  = {Li, Tianci and Cheng, Debo and Duan, Zhangling and Ling, Zhaolong},
  journal = {Applied Soft Computing},
  volume  = {201},
  pages   = {115654},
  year    = {2026},
  doi     = {10.1016/j.asoc.2026.115654}
}
~~~

The algorithm, equations, experimental findings, and name Adapt-CSL are attributable to the original authors. This document does not reproduce figures, tables, or substantial passages from the original paper. This project provides an independent engineering implementation based on the paper's description and uses it as a baseline in the MS-CD experiments.

## Implementation Scope

The code implements the three stages of Algorithm 1 in the paper:

1. Independently learn the local PC set for each target variable and record direction-specific separating sets;
2. For asymmetric PC relationships, compare the following local BDeu scores according to Theorem 3 to adaptively select the AND or OR rule:

   $$
   S_B(\operatorname{Sep}(X,Y)\rightarrow X,D)
   -S_B(\operatorname{Sep}(X,Y)\rightarrow X\leftarrow Y,D);
   $$

3. For triples with nonadjacent endpoints, compare
   $Z_1\rightarrow Z\leftarrow Z_2$ and $Z_1\rightarrow Z\rightarrow Z_2$
   according to Theorem 4, then apply Meek rules R1–R4. Remaining unoriented edges are randomly oriented according to the paper's experimental protocol while ensuring that the output is a DAG.

Compared with the earlier attached code, this version:

- Removes the restricted hill-climbing stage that is not present in the original paper;
- Removes mandatory binarization;
- Supports multistate discrete variables in both G² testing and BDeu scoring;
- Corrects the direction of the separating set used in Theorem 3;
- Uses the complete total scores of the two structures in Theorem 4;
- Implements all four Meek rules, R1–R4;
- Saves each experiment immediately and saves the mean ± standard deviation as soon as five runs are complete;
- Uses one CPU worker process by default, consistent with the code uploaded for this revision; this can be changed in `config.py`.

## Reproduction Limitations That Must Be Disclosed

The original paper does not provide complete pseudocode for the `learnPC` subroutine or report the equivalent sample size used for BDeu. Consequently, third-party code based solely on the paper cannot be guaranteed to match the authors' MATLAB program line by line.

This project makes the following explicit, auditable choices:

- `learnPC`: target-wise, level-wise, order-independent PC-simple search;
- Conditional independence test: discrete G² with `alpha=0.01`, consistent with Section 5.1.4 of the paper;
- Maximum conditioning-set size: 3 by default, an engineering limit for large networks rather than a parameter reported in the paper;
- BDeu ESS: 20.0 by default, consistent with the code uploaded for this revision; this is a disclosed reproduction parameter, not a value reported in the paper.

The recommended description is therefore:

> paper-aligned independent Python reproduction of Adapt-CSL

Do not describe this implementation as “the authors' official code” or “an exact line-by-line reproduction.” If the authors' MATLAB source code becomes available, recheck `learnPC` and the BDeu ESS against it.

## Repository Structure

~~~text
adapt-csl-discrete-benchmark/
├── config.py              # All configurable parameters
├── data_utils.py          # BIF loading, multistate BN sampling, and encoding
├── scores.py              # Multistate G², BDeu, and caching
├── graph_utils.py         # PDAGs, Meek R1–R4, and random DAG completion
├── adapt_csl.py           # Main implementation of Algorithm 1
├── metrics.py             # Precision, Recall, F1, and SHD
├── result_utils.py        # Atomic saving, configuration hashes, and five-run summaries
├── run_experiments.py     # Main entry point with configurable multiprocessing
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── CITATION.cff
├── tests/
│   └── test_smoke.py
├── THIRD_PARTY_NOTICES.md
└── data/bif/              # Recommended location for BIF files
~~~

## Environment

### Environment Reported in the Original Paper

Section 5.1.4 of the original paper reports an experimental platform with an Intel Core i7-10700 2.90 GHz CPU, an NVIDIA GeForce RTX 3060 GPU, and 64 GB of RAM. Adapt-CSL was implemented in MATLAB for the original experiments. This information acknowledges and accurately describes the original authors' experimental conditions; it does not mean that this Python reproduction requires the same hardware.

### Environment for This Project

- Python 3.10 or 3.11;
- Windows, Linux, or macOS;
- A CPU is sufficient; the core Adapt-CSL computations do not use a GPU;
- Single-process execution by default, with at least 8 GB of RAM recommended; large networks and larger conditioning sets require more memory and time;
- Compatible version ranges for NumPy, Pandas, SciPy, NetworkX, and pgmpy are specified in `requirements.txt`;
- Each run records the Python version, operating system, logical CPU count, dependency versions, parameters, source code hashes, and BIF hashes in `config.json`.

Create the environment:

~~~bash
conda create -n adapt-csl python=3.10 -y
conda activate adapt-csl
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

The project uses its own ancestral sampler rather than the older pgmpy sampling path that depended on `np.mat`, avoiding the NumPy 2.0 `np.mat was removed` error. pgmpy is used only to parse BIF files.

## Data Preparation

Place the following files in the project root or, preferably, in `data/bif/`:

~~~text
alarm.bif
alarm3.bif
alarm10.bif
child3.bif
child5.bif
child10.bif
andes.bif
pigs.bif
~~~

The code generates multistate discrete data directly from the ground-truth CPDs in the BIF files, without median-based binarization. A new dataset is generated for each network–sample-size–random-seed combination.

## Parameters

All parameters are located at the top of `config.py`:

| Parameter | Default | Purpose |
|---|---:|---|
| `NUM_WORKERS` | 1 | Number of parallel CPU processes; consistent with the uploaded code. |
| `SAMPLE_SIZES` | 200…5000 | Sample sizes for each network. |
| `SEEDS` | 42…46 | Five independent repetitions. |
| `ALPHA` | 0.01 | G² significance level; p ≥ alpha is interpreted as conditional independence. |
| `MAX_CONDITIONING_SET` | 3 | Maximum conditioning-set size in local PC search; set to `None` to disable this explicit limit. |
| `BDEU_EQUIVALENT_SAMPLE_SIZE` | 20.0 | BDeu equivalent sample size; this value is not reported in the paper. |
| `SCORE_TOLERANCE` | 1e-10 | Numerical tolerance for score comparisons. |
| `OVERWRITE_COMPLETED` | False | Whether to overwrite previously successful individual runs under the same configuration. |

The number of conditioning-set combinations grows rapidly with `MAX_CONDITIONING_SET`. Setting it to `None` may be very time-consuming on large networks such as Andes and Pigs.

## Complete Execution Procedure

For each network–sample-size–random-seed combination, the program performs the following steps:

1. Reads the ground-truth DAG, discrete states, and conditional probability tables from the BIF file;
2. Performs ancestral sampling in the topological order of the ground-truth DAG to generate multistate discrete observations;
3. Learns a local PC set for each target variable using G² tests and records separating sets;
4. Applies the local BDeu comparison in Theorem 3 to asymmetric PC relationships to select the AND or OR rule;
5. Identifies V-structures using Theorem 4, applies Meek rules R1–R4, and orients the remaining undirected edges without introducing cycles;
6. Compares the learned DAG with the ground-truth BIF DAG and computes Precision, Recall, F1, SHD, Missing, Extra, and Reversed;
7. Immediately saves the result and saves the mean ± sample standard deviation once all five seeds for the same network and sample size are complete.

For a fair comparison across algorithms, the preferred approach is to save a common set of sampled datasets in advance and have all algorithms read exactly the same data. Consistent with the uploaded code, this project currently regenerates data deterministically within each task from the BIF file, sample size, and seed. The data can be reproduced by other algorithms using the same sampler and seeds.

## Running the Experiments

From the project directory, run:

~~~bash
python run_experiments.py
~~~

The default configuration contains

$$8\times6\times5=240$$

tasks. One task runs at a time by default. To enable parallel execution, change `NUM_WORKERS` in `config.py` and report this setting in the paper and experimental records.

## Saving and Resuming

Immediately after each experiment, the worker process writes:

~~~text
adapt_csl_results/<configuration_id>/<network>/size_<n>/seed_<seed>/
├── result.json
├── predicted_edges.txt
└── true_edges.txt
~~~

Whenever the main process receives a result, it updates:

- `all_runs.csv`: all individual run results;
- `global_summary_raw.csv`: means and standard deviations stored in separate columns;
- `formatted_mean_std_summary.csv`: complete five-run results formatted for paper tables;
- `<network>/summary_<network>_all_sizes.csv`: summary of all sample sizes for each network;
- `<network>/size_<n>/runs_<network>_<n>.csv`: details of the five runs for the corresponding sample size.

When rerunning the same configuration, successful `result.json` files are loaded and the corresponding computations are skipped. Changes to the configuration, source code, or BIF contents produce a new configuration hash, preventing results from different experiments from being mixed.

## Metrics and Formatting

A true positive requires an exact directed-edge match:

$$P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN},$$

$$F1=\frac{2PR}{P+R}.$$

SHD is defined as `Missing + Extra + Reversed`, with a reversed edge counting as 1, consistent with the paper's directed structural error convention.

Raw CSV files retain full numerical values. In the formatted summaries:

- Precision, Recall, and F1 are multiplied by 100; means are reported as integers and standard deviations to two decimal places;
- SHD means are reported as integers and standard deviations to two decimal places;
- Results are displayed as `mean ± standard deviation`.

## Suggested Description for the Paper

> We used an independent Python reproduction of Adapt-CSL based on Algorithm 1 and Theorems 3-4 of Li et al. Discrete G2 tests with alpha=0.01 were used for local PC learning. As the paper does not publish the complete learnPC routine or the BDeu equivalent sample size, we used an order-independent target-wise PC-simple search with a maximum conditioning-set size of 3 and BDeu ESS=20.0. These implementation choices and all random seeds are disclosed for reproducibility.

## Development Checks

After installing the development dependencies, run:

~~~bash
pip install -r requirements-dev.txt
ruff check .
ruff format --check .
pytest -q
~~~

GitHub Actions runs the same static checks and tests on Python 3.10 and 3.11. The tests do not launch the full set of 240 experiments.

## Licensing and Data

Before publishing this repository on GitHub, choose an open-source license consistent with your institution's requirements. BIF datasets may have separate sources and licenses; do not redistribute them without checking the applicable licenses. The paper states that its benchmark data come from the [bnlearn Bayesian network repository](https://www.bnlearn.com/bnrepository/).
