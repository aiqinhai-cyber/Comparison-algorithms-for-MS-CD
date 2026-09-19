from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# 数据路径
BIF_DIR = PROJECT_DIR / "data" / "bif"
SAMPLE_DIR = PROJECT_DIR / "data" / "samples"
RESULT_DIR = PROJECT_DIR / "results"

# 实验网络
NETWORKS = [
    "alarm",
    "alarm3",
    "alarm10",
    "child3",
    "child5",
    "child10",
    "andes",
    "pigs",
]

# 样本量
SAMPLE_SIZES = [200, 300, 500, 1000, 2000, 5000]

# 每组实验运行5次
RANDOM_SEEDS = [42, 43, 44, 45, 46]

# GOLEM参数
GOLEM_VARIANT = "EV"
LEARNING_RATE = 1e-3
NUM_ITERATIONS = 100000
LAMBDA_1 = 0.02
LAMBDA_2 = 5.0
EQUAL_VARIANCES = True

# 图结构后处理
EDGE_THRESHOLD = 0.3
ENFORCE_DAG = True

# 数据标准化
STANDARDIZE_DATA = True

# 输出设置
PRECISION_RECALL_F1_AS_PERCENT = True
SUMMARY_MEAN_DECIMALS = 0
SUMMARY_STD_DECIMALS = 2

# 运行设备
DEVICE = "cpu"