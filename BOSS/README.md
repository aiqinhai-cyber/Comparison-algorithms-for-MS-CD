# Discrete BOSS+BDeu experiments

This project runs a Python BOSS implementation for the eight supplied discrete
Bayesian networks and sample sizes 200, 300, 500, 1000, 2000, and 5000. It uses
causal-learn's Grow-Shrink Tree search and a rewritten vectorized BDeu score.

## 1. Install on Windows

Open Anaconda Prompt:

```powershell
conda create -n boss-python python=3.10 -y
conda activate boss-python
cd C:\path\to\boss_discrete_experiments
pip install -r requirements.txt
```

No GPU is required.

## 2. Put BIF files here

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

## 3. Put existing sample files here

The simplest layout is:

```text
data/samples/alarm_200.csv
data/samples/alarm_300.csv
...
data/samples/pigs_5000.csv
```

CSV headers must equal the BIF variable names. The loader also recognizes
`data/samples/alarm/200.csv`, `alarm/n200.csv`, and equivalent `.txt` files.
It stops when multiple possible files exist rather than silently picking one.

If samples need to be regenerated, run:

```powershell
python generate_all_samples.py
```

Do not regenerate samples if the other algorithms were evaluated on existing
sample files. Every baseline must receive exactly the same samples.

## 4. Smoke test one small job

```powershell
python run_boss_batch.py --network alarm --sample-size 200
```

Outputs appear under `outputs/boss_bdeu/alarm/n200/`:

- `estimated_dag.csv`: representative DAG from the learned order;
- `estimated_cpdag.txt`: conventional BOSS CPDAG in readable form;
- `estimated_cpdag_endpoints.csv`: causal-learn endpoint matrix;
- `true_adjacency.csv`: true BIF DAG in the same node order;
- `result.json`: parameters, Arc_F1, SHD, score, order, and runtime.

## 5. Run all 48 settings

```powershell
python run_boss_batch.py
```

The combined table is `outputs/boss_bdeu/all_results.csv`.

Continue after a stopped run by executing the same command. Completed result
files are skipped. Use `--overwrite` only when parameters or code changed.

## 6. Tests

```powershell
pip install pytest
pytest -q
```

## 7. Parameters and reporting

The main configuration uses ESS=10, structure prior=1, maximum parents=5, and
one start. `max_parents` and `max_parent_configurations` are explicit
computational guards and must be reported in the paper. They should not be
selected separately for each network using the true-DAG Arc_F1.

For sensitivity analysis, copy the output directory and run ESS values 1, 5,
and 10 separately. Do not overwrite the main results.

The returned order defines a representative DAG. causal-learn's conventional
BOSS output is a CPDAG. Directed Arc_F1 against one true DAG and CPDAG accuracy
answer different questions. If the other baselines output CPDAGs, convert and
evaluate all methods under the same protocol.

In `estimated_cpdag_endpoints.csv`, an edge `i -> j` is represented by
`M[j, i] = 1` and `M[i, j] = -1`; an undirected edge has `-1` in both cells.

## 8. Scalability warning

Andes, Pigs, Alarm10, and Child10 can be substantially more expensive than the
small networks. Test Alarm n=200 first, then one large network with n=200.
Runtime depends mainly on variable count, state cardinalities, maximum parent
count, and parent configurations. This Python implementation should not be
presented as reproducing the Java Tetrad runtime.
