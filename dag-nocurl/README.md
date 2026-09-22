# DAG-NoCurl 离散贝叶斯网络基准实现

本仓库用于在离散贝叶斯网络数据上运行 **DAG-NoCurl / NoCurl-2**，并将其作为本文方法的对比算法。项目把原先的单文件实验脚本拆分为可审查、可恢复、可复现的模块，覆盖 BIF 采样、离散数据编码、NoCurl-2 优化、图指标计算、五次重复实验、阈值分析与结果保存。

> **重要说明**：这不是 DAG-NoCurl 作者发布的官方仓库，也不是对官方结果的逐字节复刻。本项目的核心优化流程依据原论文、作者公开的 `BPR.py` 和现有实验脚本整理；把类别状态按顺序编码后交给线性连续优化器，属于本研究为统一离散 BN 实验协议所做的工程适配。论文和结果表中应明确披露这一点。

## 1. 对原论文和作者的致谢

DAG-NoCurl 由 Yue Yu、Tian Gao、Naiyu Yin 和 Qiang Ji 提出：

- 论文：Yue Yu, Tian Gao, Naiyu Yin, Qiang Ji. **DAGs with No Curl: An Efficient DAG Structure Learning Approach**. ICML 2021, PMLR 139:12156–12166.
- 论文主页：https://proceedings.mlr.press/v139/yu21a.html
- 官方代码：https://github.com/fishmoon1234/DAG-NoCurl
- 本实现主要参考的官方文件：https://github.com/fishmoon1234/DAG-NoCurl/blob/master/BPR.py

感谢原作者公开论文与代码。本仓库不声称拥有 DAG-NoCurl 算法本身的原创贡献。核心实现文件 [`nocurl.py`](nocurl.py) 保留了来源和作者说明；第三方归属见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。若使用本项目，请首先引用原论文。

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

## 2. 本项目的用途与复现边界

本项目面向以下实验协议：

- 网络：`alarm`、`child3`、`child5`、`alarm3`、`child10`、`andes`、`alarm10`、`pigs`；
- 样本量：200、300、500、1000、2000、5000；
- 随机种子：42、43、44、45、46；
- 每个“网络 × 样本量”运行 5 次；
- 报告有向边 Precision、Recall、F1、SHD 的均值和样本标准差；
- 使用 10 个进程，每个进程限制为 1 个 BLAS 线程。

原始 DAG-NoCurl 主要针对线性或广义结构方程模型。这里的输入来自离散 BIF：程序按照 BIF 中声明的状态顺序编码为 `0, 1, ..., r_i-1`，默认除以 `r_i-1` 后逐列中心化，再交给 NoCurl-2。该数值顺序未必具有真实的度量含义，因此结果代表“DAG-NoCurl 在本项目离散编码协议下的表现”，不能被表述为原论文在离散 BN 上的官方结果。

## 3. 目录结构

```text
dag-nocurl-benchmark/
├── config.py                 # 全部实验参数（优先修改这里）
├── data_utils.py             # BIF 读取、祖先采样、状态编码和预处理
├── nocurl.py                 # NoCurl-2 核心优化
├── metrics.py                # Precision、Recall、F1、SHD
├── experiment.py             # 单个网络/样本量/种子的实验
├── result_utils.py           # 五次实验聚合和阈值选择
├── storage.py                # 原子化保存、哈希和环境信息
├── run_experiments.py        # 唯一主入口
├── data/bif/README.md        # BIF 文件放置说明
├── tests/                    # 单元测试
├── requirements.txt          # 运行依赖
├── requirements-dev.txt      # 开发和测试依赖
├── pyproject.toml            # 代码规范工具配置
├── CITATION.cff              # 引用信息
├── THIRD_PARTY_NOTICES.md    # 上游归属与改动声明
└── LICENSE                   # Apache License 2.0
```

## 4. 运行环境

推荐环境：

- Linux（Ubuntu 20.04/22.04/24.04 均可）；
- Python 3.10 或 3.11；
- 64 位 CPU；
- 至少 16 GB 内存，大网络并发运行时建议 32 GB 或更多；
- 不要求 GPU。

创建独立环境：

```bash
conda create -n dag-nocurl python=3.10 -y
conda activate dag-nocurl
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

依赖采用兼容版本范围而非假称官方固定环境。每次运行会把 Python、操作系统和已安装包版本写入结果目录的 `config.json`，便于追溯。正式论文实验建议另外执行 `python -m pip freeze > environment-lock.txt` 并随实验归档保存。

## 5. 数据准备

将 8 个 BIF 文件放入 `data/bif/`：

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

程序直接从每个 BIF 的 CPD 做祖先采样。这里没有调用 pgmpy 旧版的 `BayesianModelSampling`，从而避开 NumPy 2.0 删除 `np.mat` 所引发的错误；自定义采样仍按 BIF 的条件概率分布生成数据，但给定同一随机种子时，不保证样本序列与 pgmpy 旧采样器逐行相同。

如果 `alarm3`、`alarm10`、`child3`、`child5`、`child10` 是由基础网络复制、拼接或重命名得到的扩展网络，应在论文和数据说明中同时公开生成规则或生成脚本，以保证数据集可复现。不要只公开实验结果而不解释这些派生 BIF 的来源。

## 6. 参数说明

所有常用参数都位于 [`config.py`](config.py) 顶部。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `N_JOBS` | `10` | 同时运行的独立实验进程数；实际值不超过 CPU 核数和任务数。 |
| `THREADS_PER_WORKER` | `1` | 每个进程内部 BLAS/OpenMP 线程数，防止 10 个进程各自再开多线程。 |
| `NETWORKS` | 8 个网络 | 要运行的 BIF 网络名称。 |
| `SAMPLE_SIZES` | 6 个样本量 | 每个网络的采样规模。 |
| `SEEDS` | 42–46 | 五次独立重复实验的随机种子。 |
| `LAMBDA1` | `10.0` | NoCurl-2 第一阶段无环惩罚的系数。 |
| `LAMBDA2` | `1000.0` | 第二阶段更强的无环惩罚系数。默认值与作者仓库 NoCurl-2 线性实验设置一致。 |
| `H_TOL` | `1e-8` | 三次 L-BFGS-B 优化的 `ftol`。它是优化停止容差，不是图阈值。 |
| `INTERNAL_THRESHOLD` | `0.3` | 在利用初始解估计可达关系和节点势函数之前，对初始权重做截断。 |
| `GRAPH_THRESHOLDS` | 0.0–1.0 | 对最终连续权重构图时分析的阈值列表。 |
| `FIXED_THRESHOLD_MODE` | `False` | `True` 时只使用预先声明的固定阈值；`False` 时执行 oracle 阈值分析。 |
| `FIXED_GRAPH_THRESHOLD` | `0.3` | 固定阈值模式采用的阈值。 |
| `SCALE_BY_DECLARED_CARDINALITY` | `True` | 将第 i 个变量的编码除以 `r_i-1`，减少基数不同造成的尺度差异。 |
| `CENTER_DATA` | `True` | 逐列减去样本均值，以匹配无显式截距的线性损失。 |
| `STD_DDOF` | `1` | 用样本标准差聚合五次结果。 |
| `SAVE_DATA` | `True` | 保存每次实验的编码数据和真实邻接矩阵。 |
| `SAVE_WEIGHTS` | `True` | 保存初始权重、最终连续权重及势函数。 |
| `RESUME_COMPLETED` | `True` | 配置和 BIF 哈希一致时复用已完成任务。 |

### 阈值选择必须如实披露

默认 `FIXED_THRESHOLD_MODE=False` 会在每个“网络 × 样本量”上，使用真实图选择五次平均 F1 最高的阈值；并列时依次选择平均 SHD 更低、阈值更小者。这是 **oracle/ground-truth-based threshold selection**，会利用测试真值，因而不应作为未说明的公平主结果。

推荐做法：

1. 主对比表设置 `FIXED_THRESHOLD_MODE=True`，并在实验前固定 `FIXED_GRAPH_THRESHOLD=0.3`；或只在独立验证数据上选阈值。
2. 默认阈值扫描作为敏感性分析或 oracle 上界单独报告。
3. 不要用同一测试真值选阈值后再称其为完全独立的测试结果。

## 7. 执行过程

运行前建议先检查：

```bash
python -m pytest
python run_experiments.py
```

只需点击或执行 **`run_experiments.py`**。脚本按以下流程工作：

1. 检查参数及全部 BIF 文件；
2. 记录源码和 BIF 的 SHA-256、软件环境及完整配置；
3. 建立 8 × 6 × 5 = 240 个任务；
4. 每个任务从 BIF 独立采样、编码、缩放并中心化；
5. NoCurl-2 先求初始非无环解，再根据可达关系求节点势函数，最后在势函数确定的方向上优化边权；
6. 将连续权重按固定阈值或阈值列表转为 DAG；
7. 计算有向边 Precision、Recall、F1 和 SHD；
8. 每完成一次实验立即原子化保存；每收到一个任务结果便重新生成当前汇总，所以中断后仍保留已完成结果；
9. 同一网络和样本量的五个种子全部完成后，写出均值与标准差。

NoCurl-2 的三个 L-BFGS-B 阶段若返回非成功状态，但输出仍为有限值，程序将结果标为 `completed_with_optimizer_warning` 并保留诊断，而不是静默当作完全收敛。正式报告前应检查警告数量及 `optimizer_diagnostics`。

## 8. 指标定义

令真实有向边集为 E，预测有向边集为 Ê：

- Precision = `|E ∩ Ê| / |Ê|`；
- Recall = `|E ∩ Ê| / |E|`；
- F1 为 Precision 与 Recall 的调和平均；
- SHD = 多余骨架边 + 缺失骨架边 + 方向反转边。

一条反向边按 1 次方向反转计入 SHD。Precision、Recall 与 F1 要求方向完全一致才算真正例。无预测边时 Precision 和 F1 定义为 0。

## 9. 输出文件

结果写入：

```text
results_nocurl_threshold_sweep/<configuration_id>/
```

其中 `<configuration_id>` 由会影响结果的配置、环境和 BIF 哈希生成。主要文件如下：

| 文件 | 内容 |
|---|---|
| `config.json` | 完整配置、环境、源码哈希和 BIF 哈希。 |
| `all_runs.csv` | 每个网络、样本量、种子的运行状态与耗时。 |
| `all_threshold_runs.csv` | 每次运行在每个阈值下的原始指标。 |
| `threshold_summary.csv` | 每个阈值的五次均值和标准差。 |
| `final_best_results.csv/json` | 固定或 oracle 选定阈值后的汇总。 |
| `<network>/summary_<network>_all_sizes.csv` | 单个网络六种样本量的独立汇总。 |
| `<network>/n_<N>/seed_<S>/result.json` | 一次实验的完整状态和指标。 |
| `<network>/n_<N>/seed_<S>/data.npz` | 编码数据、真实邻接矩阵和节点名。 |
| `<network>/n_<N>/seed_<S>/weights.npz` | 连续权重、初始权重和势函数。 |

汇总 CSV 同时保存数值列和便于论文阅读的 `mean ± std` 字符串。若有任务失败，该设置的 `complete=False`，程序不会用不足五次的结果伪装成完整五次统计。

## 10. 与原单文件脚本相比的工程改动

- 修复第一阶段 `scipy.optimize.minimize` 中重复 `method` 关键字的语法错误；
- 修复主函数缺失文件列表后的多余 `]`；
- 用兼容 NumPy 2.x 的祖先采样器替代会触发 `np.mat` 错误的旧 pgmpy 采样路径；
- 将训练与阈值评估分离：同一数据只拟合一次连续权重；
- 将配置、数据、算法、指标、任务、汇总和存储拆为独立模块；
- 增加原子写入、断点续跑、输入/源码哈希、优化器诊断和逐任务汇总；
- 增加固定阈值模式，并在 oracle 模式下显示明确警告。

这些改动不改变 NoCurl-2 的核心目标函数、解析梯度、矩阵指数可达关系、势函数投影及三阶段 L-BFGS-B 流程，但会使采样随机数流和外围实验管理不同于旧脚本。

## 11. 公平比较与低性能的解释边界

若 DAG-NoCurl 在这些离散网络上的 Arc-F1 明显低于其原论文中的连续模拟结果，首先应检查：

- 算法假设与离散类别数据不匹配；
- 类别状态的任意顺序编码引入了不存在的线性距离；
- 小样本下连续权重和最终阈值高度敏感；
- 扩展 BIF 网络的规模、密度或复制结构不同；
- 是否所有方法使用同一数据、同一真实边方向和同一 SHD 定义；
- 是否把 oracle 阈值结果与固定阈值结果混在同一表中。

这些是需要通过日志、阈值曲线和额外实验验证的解释，不应直接归因于算法实现错误或算法本身无效。建议同时报告固定阈值结果、阈值敏感性、优化器警告数和运行时间。

## 12. 上传 GitHub 前检查

```bash
python -m pytest
python -m ruff check .
git init
git add .
git status
git commit -m "Add modular DAG-NoCurl discrete-BN benchmark"
```

不要提交大型实验目录、缓存或本地环境；`.gitignore` 已排除这些文件。BIF 数据能否公开取决于其来源与许可。若不能再分发，请只保留 `data/bif/README.md` 并提供合法下载/生成说明。

## 13. 许可

上游 DAG-NoCurl 仓库采用 Apache License 2.0。本项目包含依据其公开实现整理的核心流程，并保留来源说明和第三方声明。详见 [`LICENSE`](LICENSE) 与 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。论文引用义务独立于软件许可，使用时仍应引用原论文。
