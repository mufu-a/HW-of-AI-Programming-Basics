"""
医学图像二分类 — 完整训练流程
=============================
- 按患者分组划分数据集
- 加权采样 + pos_weight 处理类别不平衡
- AdamW + Warmup + 余弦退火 + 早停
- 验证集搜索最优阈值，测试集输出分类报告
- 保存模型、测试结果 txt、损失/准确率曲线图
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score, classification_report
)
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，服务器兼容
import matplotlib.pyplot as plt

from config import (
    OUTPUT_DIR, FIGURE_DIR, BATCH_SIZE, LR, EPOCHS, WEIGHT_DECAY,
    DROPOUT, WARMUP_RATIO, COSINE_END_FACTOR, EARLY_STOP_PATIENCE,
    DEVICE, POS_WEIGHT_SCALE, USE_WEIGHTED_SAMPLER,
)
from dataset import collect_samples, split_data, MedicalImageDataset, get_transforms
from model import MedicalCNN


# ==================== 工具函数 ====================

def compute_pos_weight(train_samples: list[dict]) -> torch.Tensor:
    """
    根据训练集中类别样本数计算 BCEWithLogitsLoss 的 pos_weight。
    pos_weight = num_neg / num_pos（正类为患病 class=1）。
    """
    num_pos = sum(1 for s in train_samples if s["label"] == 1)
    num_neg = sum(1 for s in train_samples if s["label"] == 0)
    raw = num_neg / num_pos
    pos_weight = torch.tensor([raw * POS_WEIGHT_SCALE], dtype=torch.float32)
    print(f"pos_weight (neg/pos={num_neg}/{num_pos}={raw:.2f}, scale={POS_WEIGHT_SCALE}): {pos_weight.item():.4f}")
    return pos_weight


def create_weighted_sampler(train_samples: list[dict]) -> WeightedRandomSampler:
    """
    为训练集创建 WeightedRandomSampler，使每个 batch 中类别分布更均衡。
    权重 = 总样本数 / (类别数 × 该类样本数)
    """
    labels = np.array([s["label"] for s in train_samples])
    class_counts = np.bincount(labels)
    class_weights = len(labels) / (len(class_counts) * class_counts)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_samples),
        replacement=True,
    )
    print(f"类别权重: {class_weights}")
    return sampler


def get_lr_scheduler(optimizer: optim.Optimizer,
                     warmup_epochs: int, total_epochs: int) -> optim.lr_scheduler.LambdaLR:
    """
    构建 Warmup + 余弦退火的学习率调度器。
    - warmup 阶段：学习率从 0 线性增长到 LR
    - 余弦退火阶段：从 LR 余弦衰减到 LR * COSINE_END_FACTOR
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # 线性 warmup
            return (epoch + 1) / warmup_epochs
        else:
            # 余弦退火
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return COSINE_END_FACTOR + 0.5 * (1 - COSINE_END_FACTOR) * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def evaluate(model: nn.Module, dataloader: DataLoader,
             criterion: nn.Module) -> tuple[float, np.ndarray, np.ndarray]:
    """
    在给定 dataloader 上评估模型，返回 (loss, all_labels, all_probs)。
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE, dtype=torch.float32)
            labels = labels.to(DEVICE, dtype=torch.float32).unsqueeze(1)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    all_labels = np.concatenate(all_labels).ravel()
    all_logits = np.concatenate(all_logits).ravel()
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid
    return avg_loss, all_labels, all_probs


def find_best_threshold(labels: np.ndarray, probs: np.ndarray,
                        num_thresholds: int = 100) -> float:
    """
    在验证集上搜索使 F1 分数最大的分类阈值。
    """
    best_thresh = 0.5
    best_f1 = 0.0
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh


def _hparam_tag() -> str:
    """生成当前超参数标签用于文件命名。"""
    return f"lr{LR:.0e}_bs{BATCH_SIZE}_drop{DROPOUT}_wd{WEIGHT_DECAY:.0e}"


def plot_curves(train_losses: list[float], val_losses: list[float],
                train_accs: list[float], val_accs: list[float]):
    """绘制损失与准确率曲线并保存到 figure/ 目录。"""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    tag = _hparam_tag()
    epochs = range(1, len(train_losses) + 1)

    # 损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f"loss_curve_{tag}.png"), dpi=150)
    plt.close()

    # 准确率曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, label="Train Accuracy", marker="o")
    plt.plot(epochs, val_accs, label="Val Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve ({tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f"accuracy_curve_{tag}.png"), dpi=150)
    plt.close()

    print(f"曲线图已保存至: {FIGURE_DIR}/")


# ==================== 训练主函数 ====================

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # ---- 1. 收集样本 & 按患者划分 ----
    print("=" * 60)
    print("收集样本并划分数据集...")
    all_samples = collect_samples()
    train_samples, val_samples, test_samples = split_data(all_samples)

    # ---- 2. 构建 Dataset & DataLoader ----
    train_dataset = MedicalImageDataset(train_samples, transform=get_transforms(train=True))
    val_dataset   = MedicalImageDataset(val_samples,   transform=get_transforms(train=False))
    test_dataset  = MedicalImageDataset(test_samples,  transform=get_transforms(train=False))

    use_pin_memory = (DEVICE == "cuda")
    num_workers = 0 if DEVICE == "mps" else 2  # MPS 不支持多进程 DataLoader

    if USE_WEIGHTED_SAMPLER:
        train_sampler = create_weighted_sampler(train_samples)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                                  num_workers=num_workers, pin_memory=use_pin_memory)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=use_pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=use_pin_memory)

    # ---- 3. 模型、损失函数、优化器 ----
    model = MedicalCNN(dropout=DROPOUT).to(DEVICE)
    pos_weight = compute_pos_weight(train_samples).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warmup_epochs = int(EPOCHS * WARMUP_RATIO)
    scheduler = get_lr_scheduler(optimizer, warmup_epochs, EPOCHS)

    # ---- 4. 训练循环 ----
    print("=" * 60)
    print(f"开始训练: epochs={EPOCHS}, lr={LR}, batch_size={BATCH_SIZE}, "
          f"warmup={warmup_epochs}, sampler={USE_WEIGHTED_SAMPLER}, device={DEVICE}")
    print("=" * 60)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, EPOCHS + 1):
        # ---- 训练阶段 ----
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images = images.to(DEVICE, dtype=torch.float32)
            labels = labels.to(DEVICE, dtype=torch.float32).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_train += (preds == labels).sum().item()
            total_train += images.size(0)

        train_loss_epoch = epoch_train_loss / total_train
        train_acc_epoch = correct_train / total_train
        train_losses.append(train_loss_epoch)
        train_accs.append(train_acc_epoch)

        # ---- 验证阶段 ----
        val_loss_epoch, val_labels, val_probs = evaluate(model, val_loader, criterion)
        val_acc_epoch = ((val_probs >= 0.5).astype(int) == val_labels).mean()
        val_losses.append(val_loss_epoch)
        val_accs.append(val_acc_epoch)

        # 验证集指标
        val_auc = roc_auc_score(val_labels, val_probs)
        val_f1 = f1_score(val_labels, (val_probs >= 0.5).astype(int), zero_division=0)
        val_recall = recall_score(val_labels, (val_probs >= 0.5).astype(int), zero_division=0)
        # 搜索最优阈值下的验证准确率
        best_t = find_best_threshold(val_labels, val_probs)
        val_acc_best = ((val_probs >= best_t).astype(int) == val_labels).mean()

        # 学习率调度
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"LR: {current_lr:.2e} | "
              f"Train Loss: {train_loss_epoch:.4f} Acc: {train_acc_epoch:.4f} | "
              f"Val Loss: {val_loss_epoch:.4f} Acc@.5: {val_acc_epoch:.4f} Acc*: {val_acc_best:.4f} | "
              f"AUC: {val_auc:.4f} F1: {val_f1:.4f} Recall: {val_recall:.4f}")

        # ---- 早停 & 保存最佳模型 ----
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n早停触发: 连续 {EARLY_STOP_PATIENCE} 轮验证 loss 未下降")
                break

    print(f"\n最佳验证 loss: {best_val_loss:.4f} (Epoch {best_epoch})")

    # 恢复最佳模型
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, os.path.join(OUTPUT_DIR, "best_model.pth"))
    print(f"最佳模型已保存至: {OUTPUT_DIR}/best_model.pth")

    # ---- 5. 验证集搜索最优阈值 ----
    _, val_labels_all, val_probs_all = evaluate(model, val_loader, criterion)
    best_threshold = find_best_threshold(val_labels_all, val_probs_all)
    print(f"验证集最优分类阈值: {best_threshold:.4f}")

    # ---- 6. 测试集评估 ----
    test_result = test_evaluate(model, test_loader, criterion, best_threshold)
    tag = _hparam_tag()
    result_path = os.path.join(OUTPUT_DIR, f"test_results_{tag}.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(test_result)
    print(test_result)
    print(f"测试结果已保存至: {result_path}")

    # ---- 7. 绘制曲线 ----
    plot_curves(train_losses, val_losses, train_accs, val_accs)

    return model


def test_evaluate(model: nn.Module, test_loader: DataLoader,
                  criterion: nn.Module, threshold: float) -> str:
    """
    在测试集上做最终评估，返回格式化的分类报告字符串。
    此函数同时被 train.py 和 evaluate.py 调用。
    """
    test_loss, test_labels, test_probs = evaluate(model, test_loader, criterion)
    test_preds = (test_probs >= threshold).astype(int)

    test_auc = roc_auc_score(test_labels, test_probs)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    test_acc = (test_preds == test_labels).mean()

    # sklearn classification_report 中 label 0 = 不患病(正常), label 1 = 患病
    report = classification_report(
        test_labels, test_preds,
        target_names=["不患病(Normal)", "患病(Diseased)"],
        digits=4,
    )

    lines = [
        "=" * 55,
        "          测试集评估结果 (Test Set Evaluation)",
        "=" * 55,
        f"分类阈值 (Threshold): {threshold:.4f}",
        f"测试 Loss: {test_loss:.4f}",
        f"准确率 (Accuracy): {test_acc:.4f}",
        f"AUC:  {test_auc:.4f}",
        f"召回率 (Recall): {test_recall:.4f}",
        f"F1 分数:  {test_f1:.4f}",
        "-" * 55,
        "分类报告 (Classification Report):",
        report,
        "=" * 55,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # 如果直接运行 train.py，执行完整训练 + 测试流程
    # 如果通过 -evaluate 参数调用，则只执行测试（由 evaluate.py 触发）
    if len(sys.argv) > 1 and sys.argv[1] == "--evaluate-only":
        print("进入仅评估模式...")
        # 此模式由 evaluate.py 处理，此处不执行训练
        pass
    else:
        train()
