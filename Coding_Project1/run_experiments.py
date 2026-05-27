"""超参数控制变量实验脚本。依次运行不同 LR、Dropout、BatchSize 并记录结果。"""
import subprocess
import sys
import os

# Baseline config content (full file)
BASELINE_CONFIG = '''"""
医学图像二分类 — 超参数配置文件
===============================
所有可调参数集中管理，训练/评估/数据集模块均从此处导入。
"""

import os

# ==================== 路径配置 ====================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure")

# ==================== 数据集划分 ====================
TRAIN_RATIO = 0.6
VAL_RATIO   = 0.2
TEST_RATIO  = 0.2
SPLIT_SEED = 42

# ==================== 图像参数 ====================
IMAGE_SIZE = 64
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD  = (0.229, 0.224, 0.225)

# ==================== 训练超参数 ====================
BATCH_SIZE    = {batch_size}
LR            = {lr}
EPOCHS        = 60
WEIGHT_DECAY  = 1e-4
DROPOUT       = {dropout}

# 是否使用 WeightedRandomSampler（False = 仅用 pos_weight 处理不平衡）
USE_WEIGHTED_SAMPLER = False
# pos_weight 缩放因子（1.0 = 使用完整类别反比权重）
POS_WEIGHT_SCALE = 0.5

# ==================== 学习率调度 ====================
WARMUP_RATIO = 0.1
COSINE_END_FACTOR = 0.01

# ==================== 早停策略 ====================
EARLY_STOP_PATIENCE = 15

# ==================== 设备 ====================
DEVICE = "mps"
'''

experiments = [
    # (display_name, param_changes_dict)
    ("lr1e-4",  {"lr": "1e-4",  "batch_size": "64", "dropout": "0.3"}),
    ("lr1e-3",  {"lr": "1e-3",  "batch_size": "64", "dropout": "0.3"}),
    ("drop0.2", {"lr": "5e-4",  "batch_size": "64", "dropout": "0.2"}),
    ("drop0.5", {"lr": "5e-4",  "batch_size": "64", "dropout": "0.5"}),
    ("bs32",    {"lr": "5e-4",  "batch_size": "32", "dropout": "0.3"}),
    ("bs128",   {"lr": "5e-4",  "batch_size": "128", "dropout": "0.3"}),
]

results = {}

for display, params in experiments:
    print(f"\n{'#'*60}")
    print(f"# Experiment: {display}")
    print(f"# Params: {params}")
    print(f"{'#'*60}")

    # Write baseline config with overridden parameters
    config_content = BASELINE_CONFIG.format(**params)
    with open("config.py", "w") as f:
        f.write(config_content)

    # Run training with MPS fallback and generous timeout
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # CPU fallback if MPS fails
    try:
        result = subprocess.run(
            [sys.executable, "train.py"],
            capture_output=True, text=True, timeout=14400,  # 4h per experiment
            env=env,
        )
        stdout = result.stdout
        stderr = result.stderr
        timed_out = False
    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = "TIMEOUT after 4 hours — MPS may have hung"
        timed_out = True

    # Extract test accuracy and AUC
    acc = None
    auc = None
    for line in stdout.split("\n"):
        if "准确率 (Accuracy):" in line:
            acc = float(line.split(":")[-1].strip())
        if "AUC: " in line or "AUC:" in line:
            try:
                auc = float(line.split(":")[-1].strip())
            except ValueError:
                pass

    if timed_out:
        results[display] = {"accuracy": "TIMEOUT", "auc": "TIMEOUT"}
        print(f"  => TIMEOUT (4h) — MPS may have hung, skipping")
    elif acc:
        results[display] = {"accuracy": acc, "auc": auc}
        print(f"  => Test Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    else:
        results[display] = {"accuracy": None, "auc": None}
        print(f"  => FAILED to extract results")
        print("\n".join(stdout.split("\n")[-20:]))

    # Save full output
    os.makedirs("output", exist_ok=True)
    with open(f"output/experiment_{display}.txt", "w") as f:
        f.write(stdout)
        if stderr:
            f.write("\n\nSTDERR:\n")
            f.write(stderr)

# Restore baseline config
config_content = BASELINE_CONFIG.format(lr="5e-4", batch_size="64", dropout="0.3")
with open("config.py", "w") as f:
    f.write(config_content)

print(f"\n{'='*60}")
print("Summary of Hyperparameter Experiments:")
print(f"{'='*60}")
print(f"Baseline (lr=5e-4, bs=64, drop=0.3): Test Acc = 0.8939, AUC = 0.9560")
print("-" * 60)
for display, res in results.items():
    acc_str = f"{res['accuracy']:.4f}" if res['accuracy'] else "FAILED"
    auc_str = f"{res['auc']:.4f}" if res['auc'] else "N/A"
    print(f"  {display:12s}: Test Acc = {acc_str}, AUC = {auc_str}")
print(f"{'='*60}")
