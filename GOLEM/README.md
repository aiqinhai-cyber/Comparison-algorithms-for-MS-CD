# GOLEM Discrete-BN Benchmark

基于 GOLEM 的离散贝叶斯网络结构学习对比实验（非官方 PyTorch 复现与实验适配）。

## 重要说明

本项目基于 Ignavier Ng、AmirEmad Ghassami 和 Kun Zhang 提出的 GOLEM 方法开展复现实验。GOLEM 的算法思想、理论分析和原始实现均归原作者及其所属机构所有。

- 原论文：**On the Role of Sparsity and DAG Constraints for Learning Linear DAGs**
- 原作者：**Ignavier Ng, AmirEmad Ghassami, Kun Zhang**
- 发表会议：**Advances in Neural Information Processing Systems 33 (NeurIPS 2020)**
- 论文主页：[NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/hash/d04d42cdf14579cd294e5079e0745411-Abstract.html)
- 论文 PDF：[NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2020/file/d04d42cdf14579cd294e5079e0745411-Paper.pdf)
- arXiv：[arXiv:2006.10201](https://arxiv.org/abs/2006.10201)
- 原作者官方代码：[ignavierng/golem](https://github.com/ignavierng/golem)

本仓库不是 GOLEM 原作者发布的官方离散版本，也未声明得到原作者的认可、维护或技术支持。若要严格复现原论文，请优先使用原论文、补充材料和原作者官方代码。

## 致谢与归属

我们感谢 GOLEM 原作者公开论文与源代码，使后续研究能够理解、检验和比较该方法。本项目中的 GOLEM 目标函数、GOLEM-EV/GOLEM-NV 划分、软稀疏约束、软 DAG 约束、EV 初始化 NV 的策略，以及阈值化和逐步删除弱边的后处理思想，均来源于原论文及官方代码。

原作者官方仓库采用 [Apache License 2.0](https://github.com/ignavierng/golem/blob/main/LICENSE)。公开或再分发任何直接来源于官方仓库的代码时，应遵守该许可证，保留原始版权、许可证和必要的归属声明。请勿删除或模糊原作者信息。

本项目新增的主要内容是面向特定实验协议的工程适配，包括：

- 使用现代 PyTorch 实现 GOLEM 优化目标；
- 读取 BIF 贝叶斯网络及其真实 DAG；
- 按 BIF 中声明的状态顺序编码离散变量；
- 在多个网络、样本量和随机种子上批量运行；
- 计算有向 Precision、Recall、F1 和 SHD；
- 保存单次结果、五次实验结果及平均值 ± 标准差；
- 保存加权邻接矩阵、二值 DAG、优化历史和运行环境信息。

## 原始 GOLEM 方法简介

GOLEM 的全称为 **Gradient-based Optimization of dag-penalized Likelihood for learning linEar dag Models**。原方法针对线性结构方程模型，通过连续优化学习加权邻接矩阵。

GOLEM 使用基于似然的目标函数，并加入两类软惩罚：

$$
$$
S(B;X) = L(B;X) + \lambda_1 \|B\|_1 + \lambda_2 h(B),
$$
$$

其中：

- \(B\) 是待学习的加权邻接矩阵；
- \(L(B;X)\) 是线性模型的负对数似然；
- \(\lVert B\rVert_1\) 是稀疏惩罚，用于减少不必要的边；
- \(h(B)=\operatorname{tr}(\exp(B\circ B))-d\) 是软无环惩罚；
- \(\lambda_1\) 和 \(\lambda_2\) 分别控制稀疏性与无环性。

原论文讨论了两种主要设置：

- **GOLEM-EV**：假设各变量的噪声方差相等；
- **GOLEM-NV**：允许噪声方差不同。官方实现建议使用 GOLEM-EV 的结果初始化 GOLEM-NV，以降低陷入不理想局部解的风险。

由于优化得到的矩阵在有限样本下可能包含较小的非零系数或环，本项目按照原论文描述的思路进行后处理：先移除绝对值不超过阈值的边；如果仍然存在环，则按照绝对权重从小到大删除边，直至得到 DAG。

## 本项目的研究目的

本项目用于在统一实验协议下，将 GOLEM 作为因果结构学习对比方法运行于以下离散贝叶斯网络数据：

- `alarm.bif`
- `alarm3.bif`
- `alarm10.bif`
- `child3.bif`
- `child5.bif`
- `child10.bif`
- `andes.bif`
- `pigs.bif`

每个网络使用以下样本量：

```text
200, 300, 500, 1000, 2000, 5000
```

每种“网络 × 样本量”设置在以下五个随机种子下独立运行：

```text
42, 43, 44, 45, 46
```

默认共有：

\[
8\ \text{个网络}\times 6\ \text{种样本量}\times 5\ \text{个随机种子}=240\ \text{次实验}。
\]

## 关于离散数据的关键限制

**GOLEM 原本面向连续线性结构方程模型，特别是线性高斯模型。本项目的数据是离散类别数据，因此不满足 GOLEM 的原始建模假设。**

本项目按照 BIF 文件中声明的状态顺序，将每个离散变量编码为：

```text
state_0 -> 0
state_1 -> 1
...
state_r -> r
```

编码后的数据经过中心化或可选的标准化，再输入 GOLEM。该处理会人为引入状态之间的数值顺序和距离；对于名义型变量，这种顺序通常没有自然统计含义。因此：

- 本实验是模型失配条件下的基线比较；
- 结果不能说明 GOLEM 原生支持离散变量；
- 结果不能替代专门面向离散数据的结构学习方法；
- 与 PC、BOSS、MMHC 等离散方法比较时，应在论文中明确这一限制；
- 若 GOLEM 表现较差，不能简单归因于算法本身，也可能来自数据类型与模型假设不一致。

在论文实验设置中使用类似表述：

> GOLEM was originally developed for continuous linear structural equation models. To include it as a continuous-optimization baseline on the categorical Bayesian-network benchmarks, we encoded each state according to its order declared in the corresponding BIF file and centered the resulting numerical variables. This constitutes a model-misspecified evaluation; therefore, the results should not be interpreted as evidence that GOLEM natively supports discrete data.

## 项目结构

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

各文件职责如下：

| 文件                   | 作用                              |
| -------------------- | ------------------------------- |
| `config.py`          | 集中管理网络、样本量、随机种子、GOLEM参数、路径和输出格式 |
| `run_experiments.py` | 批量运行实验并协调数据、模型、评价与结果保存          |
| `golem.py`           | GOLEM目标函数、PyTorch训练、阈值化及DAG后处理  |
| `data_utils.py`      | BIF读取、样本生成、CSV查找、状态编码和数据预处理     |
| `metrics.py`         | 有向Precision、Recall、F1、SHD及骨架指标  |
| `result_utils.py`    | 邻接矩阵保存、结果展开、五次实验聚合与环境记录         |

## 环境安装

建议使用独立的 Python 环境。Python 3.10 或更高版本较为合适。

### Conda

```bash
conda create -n golem-benchmark python=3.10 -y
conda activate golem-benchmark
```

根据计算机的 CUDA 版本，从 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择相应命令安装 PyTorch，然后安装其余依赖：

```bash
pip install numpy pandas scipy networkx pgmpy tqdm
```

仅使用 CPU 时，也可以直接安装：

```bash
pip install torch numpy pandas scipy networkx pgmpy tqdm
```

## 数据放置

推荐将 BIF 文件放入：

```text
data/bif/
```

为了兼容早期的单文件实验布局，代码也会在项目根目录查找 BIF 文件。因此，下面两种放置方式均可：

```text
data/bif/alarm.bif
```

或：

```text
golem-discrete-benchmark/alarm.bif
```

如果公开仓库中包含第三方 BIF 文件或预生成数据，请分别核实其来源、许可证和再分发条件，并在仓库中补充对应引用。GOLEM 原论文和代码的许可证不自动覆盖这些数据文件。

## 数据运行模式

在 `config.py` 中修改 `DATA_MODE`。

### 1. `resample`：每个种子重新采样

```python
DATA_MODE = "resample"
SAVE_RESAMPLED_DATA = True
```

程序使用 BIF 的概率分布，在每个随机种子下独立生成数据。这是默认设置，也是五次重复实验更合理的方式。

生成的数据保存为：

```text
results/generated_samples/<network>/<network>_<size>_seed<seed>.csv
```

### 2. `seeded_files`：读取预生成的种子数据

```python
DATA_MODE = "seeded_files"
```

例如：

```text
data/samples/alarm/alarm_200_seed42.csv
data/samples/alarm/alarm_200_seed43.csv
data/samples/alarm/alarm_200_seed44.csv
data/samples/alarm/alarm_200_seed45.csv
data/samples/alarm/alarm_200_seed46.csv
```

### 3. `shared_file`：五个种子共用同一数据

```python
DATA_MODE = "shared_file"
```

例如：

```text
data/samples/alarm/alarm_200.csv
```

不推荐将其用于主要五次重复实验。当前 GOLEM 使用全批量优化和零矩阵初始化；当相同数据被重复使用时，仅改变优化器随机种子通常不会形成真正独立的数据重复实验。

## 参数设置

所有可调参数均位于 `config.py`。

默认 GOLEM-EV 参数与原作者官方仓库给出的示例设置一致：

```python
EV_LAMBDA_1 = 0.02
EV_LAMBDA_2 = 5.0
EV_NUM_ITER = 100_000
EV_LEARNING_RATE = 1e-3
```

默认 GOLEM-NV 参数为：

```python
NV_LAMBDA_1 = 0.002
NV_LAMBDA_2 = 5.0
NV_NUM_ITER = 100_000
NV_LEARNING_RATE = 1e-3
```

其他重要参数：

```python
VARIANT = "ev"              # ev、nv 或 both
NV_INITIALIZATION = "ev"   # ev 或 zero
GRAPH_THRESHOLD = 0.3
PREPROCESSING = "center"   # center 或 zscore
DEVICE = "auto"            # auto、cpu 或 cuda
DTYPE = "float32"          # float32 或 float64
CHECKPOINT_INTERVAL = 5_000
```

说明：参数虽然参考官方设置，但离散数据编码改变了数据尺度和模型含义。正式论文实验中应报告这些参数，并建议对 `GRAPH_THRESHOLD`、`LAMBDA_1` 和 `LAMBDA_2` 进行敏感性分析，不能仅因为参数来自连续数据实验就假设它们对离散数据仍然最优。

## 运行实验

在项目根目录执行：

```bash
python run_experiments.py
```

程序启动时会先检查所有请求的 BIF 文件，避免在长时间训练后才发现文件缺失。

默认完整实验需要运行 240 次，每次迭代 100,000 步，计算时间可能很长。若用于正式论文对比，不建议通过减少迭代次数替代完整设置；如确需改变，应在论文和配置文件中如实报告。

## 输出结果

以 GOLEM-EV、`alarm` 网络、样本量 200 为例：

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

每个随机种子目录中包含：

```text
result.json
true_adjacency.csv
ev/weighted_adjacency.npy
ev/processed_weighted_adjacency.npy
ev/estimated_dag.csv
ev/optimization_history.csv
ev/checkpoints/
```

总体汇总文件包括：

```text
results/ev/all_results.json
results/ev/all_runs.csv
results/ev/mean_std_results.csv
results/ev/experiment_config.json
results/ev/environment.json
```

其中：

- `all_runs.csv` 保存每次独立实验的完整结果；
- `mean_std_results.csv` 保存每个网络和样本量下五次结果的平均值与标准差；
- 每个 `n<size>` 目录还会单独保存对应的五次结果和汇总结果；
- Precision、Recall 和 F1 默认转换为百分数；
- 平均值默认保存为整数；
- 标准差默认保留两位小数，例如 `46 ± 1.14`；
- SHD 平均值保存为整数，标准差保留两位小数，例如 `50 ± 1.58`。

## 评价指标

邻接矩阵统一采用：

```text
adjacency[parent, child] = 1
```

有向边精确率、召回率和 F1 定义为：

\[
\mathrm{Precision}=\frac{TP}{TP+FP},
\qquad
\mathrm{Recall}=\frac{TP}{TP+FN},
\]

\[
\mathrm{F1}=\frac{2\cdot \mathrm{Precision}\cdot \mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}.
\]

本项目的 SHD 计算删除、增加或反向一条边所需的最少结构修改次数，其中反向边计为一次修改。论文中应明确说明这一约定，因为不同代码库可能将反向边计为一次或两次操作。

## 可复现性说明

程序设置 Python、NumPy、PyTorch CPU 和 CUDA 随机种子，并在可用时启用确定性计算。每次实验还会保存：

- 网络名称和样本量；
- 随机种子；
- 数据文件路径；
- GOLEM变体和超参数；
- 预处理方式；
- PyTorch、NumPy、Pandas 和 NetworkX 版本；
- CUDA可用状态、CUDA版本和GPU名称；
- 训练时间与最终目标函数值。

即使启用了确定性选项，不同操作系统、GPU、CUDA、PyTorch 或底层线性代数库之间仍可能产生轻微数值差异。发表实验结果时，建议同时公开 `experiment_config.json` 和 `environment.json`。

## 与原作者官方实现的区别

| 项目      | 原作者官方实现        | 本项目                   |
| ------- | -------------- | --------------------- |
| 实现框架    | TensorFlow 1.x | PyTorch               |
| 主要数据假设  | 连续线性模型         | 将离散状态序数编码后进行失配评估      |
| 数据来源    | 原论文的线性SEM实验设置  | BIF贝叶斯网络及其离散样本        |
| GOLEM目标 | 原作者实现          | 根据论文目标函数进行现代PyTorch复现 |
| 重复实验    | 官方实验协议         | 固定5个种子批量运行            |
| 结果输出    | 官方仓库定义的输出      | 增加逐次CSV、均值±标准差和环境记录   |

因此，本项目更准确的定位是：

> 一个用于论文基线比较的、非官方的 GOLEM PyTorch 复现与离散 BIF 实验适配，而不是原作者官方代码的逐文件移植，也不是面向离散变量重新推导的 GOLEM 新算法。

## 引用

如果本项目用于论文、报告或公开实验，请至少引用 GOLEM 原论文：

```bibtex
@inproceedings{Ng2020role,
  author    = {Ng, Ignavier and Ghassami, AmirEmad and Zhang, Kun},
  title     = {On the Role of Sparsity and DAG Constraints for Learning Linear DAGs},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  year      = {2020}
}
```

如果使用或参考了原作者官方实现，也请在论文或项目文档中链接：

```text
https://github.com/ignavierng/golem
```

使用本仓库中的 BIF 网络或其他第三方数据时，还应根据其实际来源补充相应的数据集引用；引用 GOLEM 论文不能替代数据集引用。

## 学术与使用声明

- 本项目用于研究复现和基线比较，不提供因果关系真实性保证；
- 从观察数据学习到的有向图不应在缺少额外假设和领域验证时直接解释为真实因果机制；
- 请如实报告离散编码、预处理、参数、失败实验和模型假设不匹配问题；
- 请勿将本项目描述为原作者发布的“离散 GOLEM”或“官方 PyTorch GOLEM”；
- 如果修改本项目，请在文档中区分原始 GOLEM 贡献、本项目适配和自己的新增修改。

## 许可证建议

原作者 GOLEM 仓库使用 Apache License 2.0。本仓库正式公开前，应在仓库根目录加入清晰的 `LICENSE` 文件，并根据代码的实际来源确认许可证选择：

1. 如果包含或改写了原作者仓库的实质性代码，应遵守 Apache-2.0，保留相关版权和许可证声明；
2. 如果代码仅依据公开论文独立实现，也仍应明确引用原论文和官方代码；
3. 第三方 BIF 文件、数据和依赖项各自具有独立许可证，必须分别核实；
4. 本 README 不是法律意见，如对再分发权利存在疑问，应咨询所在机构或专业人员。

---

再次感谢 Ignavier Ng、AmirEmad Ghassami 和 Kun Zhang 对 GOLEM 方法及开源实现所作的贡献。
