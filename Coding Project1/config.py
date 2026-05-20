"""
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
BATCH_SIZE    = 16
LR            = 5e-4
EPOCHS        = 60
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.3

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
