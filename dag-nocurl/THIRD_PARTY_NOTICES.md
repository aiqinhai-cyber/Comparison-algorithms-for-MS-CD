# Third-Party Notices

## DAG-NoCurl

This project includes a reorganized implementation of the NoCurl-2 computational flow based on the following upstream work:

- Yue Yu, Tian Gao, Naiyu Yin, and Qiang Ji, “DAGs with No Curl: An Efficient DAG Structure Learning Approach,” ICML 2021.
- Paper: https://proceedings.mlr.press/v139/yu21a.html
- Official repository: https://github.com/fishmoon1234/DAG-NoCurl
- Principal upstream reference file: `BPR.py`

The upstream repository is distributed under the Apache License, Version 2.0. Copyright remains with the original copyright holders.

This repository is an independent research adaptation for benchmarking ordinally encoded discrete Bayesian-network samples. It is not an official release by the DAG-NoCurl authors, and the original authors have not endorsed its discrete-data protocol or experimental conclusions.

Material engineering changes include modularization, a NumPy-2-compatible ancestral sampler, process-level parallelism, atomic persistence, five-seed aggregation, graph-threshold analysis, and evaluation metrics. The core attribution header is also retained in `nocurl.py`.
