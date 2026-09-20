# Adapt-CSL：离散贝叶斯网络实验复现

本仓库是 Adapt-CSL 的独立 Python 复现，用于论文中的对比实验。实现依据原论文的 Algorithm 1、Theorem 3、Theorem 4、公式 (7)-(17) 和实验设置编写，不是原作者发布的官方代码。

## 原论文与作者

请优先引用原论文：

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

论文算法、公式、实验结论和名称 Adapt-CSL 均归原作者。本文档不转载原论文图表或大段正文。本项目仅根据论文描述完成独立工程实现，并将其作为 MS-CD 实验中的对比算法。

## 实现范围

代码对应论文 Algorithm 1 的三个阶段：

1. 对每个目标变量独立学习局部 PC 集，并记录方向相关的分离集；
2. 对非对称 PC 关系，按 Theorem 3 比较以下局部 BDeu 分数，自适应选择 AND 或 OR：

   $$
   S_B(\operatorname{Sep}(X,Y)\rightarrow X,D)
   -S_B(\operatorname{Sep}(X,Y)\rightarrow X\leftarrow Y,D);
   $$

3. 对非相邻端点组成的三元组，按 Theorem 4 比较
   $Z_1\rightarrow Z\leftarrow Z_2$ 与 $Z_1\rightarrow Z\rightarrow Z_2$，
   之后执行 Meek R1-R4；仍未定向的边按论文实验协议随机定向，同时保证输出为 DAG。

与旧版附件代码相比，本版已经：

- 删除原论文不存在的受限爬山阶段；
- 删除强制二值化；
- G² 和 BDeu 均支持多状态离散变量；
- 修正 Theorem 3 所使用分离集的方向；
- 使用 Theorem 4 的完整两结构总分；
- 补全 Meek R1-R4；
- 每次实验立即保存，五次完成后立即保存均值 ± 标准差；
- 默认使用 1 个 CPU 工作进程，与本次上传代码保持一致；可在 `config.py` 中调整。

## 必须披露的复现边界

原论文没有给出 `learnPC` 子程序的完整伪代码，也没有报告 BDeu 的 equivalent sample size。因此，任何仅依据论文正文编写的第三方代码都无法保证与作者 MATLAB 程序逐行相同。

本项目采用以下可审计选择：

- `learnPC`：目标变量级、逐层且顺序无关的 PC-simple 搜索；
- 条件独立检验：离散 G²，`alpha=0.01`，与论文 Section 5.1.4 一致；
- 最大条件集：默认 3，是大网络上的工程限制，并非论文公布参数；
- BDeu ESS：默认 20.0，与本次上传代码保持一致；它是明确披露的复现参数，并非论文公布参数。

因此推荐称为：

> paper-aligned independent Python reproduction of Adapt-CSL

不要称为“作者官方代码”或“逐行完全复现”。如果获得作者的 MATLAB 源码，应再次核对 `learnPC` 与 BDeu ESS。

## 目录结构

~~~text
adapt-csl-discrete-benchmark/
├── config.py              # 所有可调参数
├── data_utils.py          # BIF 读取、多状态 BN 采样与编码
├── scores.py              # 多状态 G²、BDeu 与缓存
├── graph_utils.py         # PDAG、Meek R1-R4、随机 DAG 完成
├── adapt_csl.py           # Algorithm 1 主体
├── metrics.py             # Precision、Recall、F1、SHD
├── result_utils.py        # 原子保存、配置哈希与五次汇总
├── run_experiments.py     # 唯一运行入口，支持可配置多进程
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── CITATION.cff
├── tests/
│   └── test_smoke.py
├── THIRD_PARTY_NOTICES.md
└── data/bif/              # 推荐放置 BIF 文件
~~~

## 环境

### 原论文报告的环境

原论文 Section 5.1.4 报告的实验平台为 Intel Core i7-10700 2.90 GHz CPU、NVIDIA GeForce RTX 3060 GPU 和 64 GB 内存；其中 Adapt-CSL 原实验使用 MATLAB 实现。该信息用于尊重并准确说明原作者的实验条件，不代表本 Python 复现必须使用相同硬件。

### 本项目环境

- Python 3.10 或 3.11；
- Windows、Linux 或 macOS；
- CPU 即可，Adapt-CSL 核心计算不使用 GPU；
- 默认单进程，建议至少 8 GB 内存；大型网络和更大的条件集需要更多内存与时间；
- NumPy、Pandas、SciPy、NetworkX 和 pgmpy 的兼容范围见 `requirements.txt`；
- 每次运行会在 `config.json` 中记录 Python、操作系统、逻辑 CPU 数量、依赖版本、参数、源代码哈希和 BIF 哈希。

创建环境：

~~~bash
conda create -n adapt-csl python=3.10 -y
conda activate adapt-csl
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

项目使用自己的祖先采样器，不调用 pgmpy 中曾依赖 `np.mat` 的旧采样路径，因此不会触发 NumPy 2.0 的 `np.mat was removed` 错误。pgmpy 仅用于解析 BIF。

## 数据准备

将以下文件放在项目根目录，或者推荐放入 `data/bif/`：

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

代码直接从 BIF 的真实 CPD 生成多状态离散数据，不进行中位数二值化。每个“网络 × 样本量 × 随机种子”都会重新生成一份数据。

## 参数

所有参数都在 `config.py` 顶部：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `NUM_WORKERS` | 1 | 并行 CPU 进程数；与上传代码一致 |
| `SAMPLE_SIZES` | 200…5000 | 每个网络的样本量 |
| `SEEDS` | 42…46 | 五次独立重复实验 |
| `ALPHA` | 0.01 | G² 显著性水平；p ≥ alpha 判为条件独立 |
| `MAX_CONDITIONING_SET` | 3 | 局部 PC 搜索的最大条件集大小；设为 `None` 表示不人为截断 |
| `BDEU_EQUIVALENT_SAMPLE_SIZE` | 20.0 | BDeu 等效样本量；论文未报告该值 |
| `SCORE_TOLERANCE` | 1e-10 | 分数比较数值容差 |
| `OVERWRITE_COMPLETED` | False | 是否覆盖相同配置下已成功的单次实验 |

条件组合数会随 `MAX_CONDITIONING_SET` 快速增长。在 Andes、Pigs 等大网络上，将其设为 `None` 可能非常耗时。

## 完整运行过程

对每个“网络 × 样本量 × 随机种子”，程序执行以下过程：

1. 从 BIF 读取真实 DAG、离散状态和条件概率表；
2. 按真实 DAG 的拓扑顺序执行祖先采样，生成多状态离散观测数据；
3. 使用 G² 检验分别学习每个目标变量的局部 PC 集并记录分离集；
4. 对非对称 PC 关系执行 Theorem 3 的局部 BDeu 比较，选择 AND 或 OR 规则；
5. 使用 Theorem 4 识别 V 结构，执行 Meek R1-R4，并将残余无向边无环定向；
6. 将学习 DAG 与 BIF 真实 DAG 比较，计算 Precision、Recall、F1、SHD、Missing、Extra 和 Reversed；
7. 立即保存该次结果；当同一网络和样本量的五个种子齐全时，立即保存均值 ± 样本标准差。

不同算法进行公平对比，最理想的做法是预先保存同一批采样数据，并让所有算法读取完全相同的数据。本项目当前与上传代码一致，会在每个任务中按 BIF、样本量和种子确定性地重新生成数据；只要其他算法采用相同采样器和种子，数据即可复现。

## 直接运行

进入项目目录后只需运行：

~~~bash
python run_experiments.py
~~~

默认共有：

$$8\times6\times5=240$$

个任务。默认一次运行 1 个任务；如需并行，可在 `config.py` 中修改 `NUM_WORKERS`，但必须在论文和实验记录中报告该设置。

## 保存与恢复

每次实验结束，工作进程立即写入：

~~~text
adapt_csl_results/<configuration_id>/<network>/size_<n>/seed_<seed>/
├── result.json
├── predicted_edges.txt
└── true_edges.txt
~~~

主进程每收到一个结果都会更新：

- `all_runs.csv`：所有单次结果；
- `global_summary_raw.csv`：均值与标准差分列保存；
- `formatted_mean_std_summary.csv`：五次完整结果的论文表格形式；
- `<network>/summary_<network>_all_sizes.csv`：每个网络单独的全部样本量汇总；
- `<network>/size_<n>/runs_<network>_<n>.csv`：对应样本量的五次明细。

相同配置重新运行时，成功的 `result.json` 会被读取并跳过计算。配置、源代码或 BIF 内容改变都会产生新的配置哈希，避免不同实验误混。

## 指标及格式

有向边完全一致才算 TP：

$$P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN},$$

$$F1=\frac{2PR}{P+R}.$$

SHD 定义为 `Missing + Extra + Reversed`，反向边计 1，与论文的有向结构误差口径一致。

原始 CSV 保存完整数值。格式化汇总中：

- Precision、Recall、F1 乘 100 后，均值保存为整数，标准差保留两位小数；
- SHD 均值保存为整数，标准差保留两位小数；
- 显示形式为 `均值 ± 标准差`。

## 论文中的建议描述

> We used an independent Python reproduction of Adapt-CSL based on Algorithm 1 and Theorems 3-4 of Li et al. Discrete G2 tests with alpha=0.01 were used for local PC learning. As the paper does not publish the complete learnPC routine or the BDeu equivalent sample size, we used an order-independent target-wise PC-simple search with a maximum conditioning-set size of 3 and BDeu ESS=20.0. These implementation choices and all random seeds are disclosed for reproducibility.

## 开发检查

安装开发依赖后可执行：

~~~bash
pip install -r requirements-dev.txt
ruff check .
ruff format --check .
pytest -q
~~~

GitHub Actions 会在 Python 3.10 和 3.11 上执行同样的静态检查和测试。测试不会启动 240 组正式实验。

## 许可与数据

请在发布到 GitHub 前为本仓库选择与你所在单位要求一致的开源许可证。BIF 数据集可能有各自的来源和许可；不要在未核对许可时直接重新分发。论文中说明其基准数据来自 [bnlearn Bayesian network repository](https://www.bnlearn.com/bnrepository/)。
