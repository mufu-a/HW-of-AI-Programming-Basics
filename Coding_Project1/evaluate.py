"""
医学图像二分类 — 独立测试评估脚本
================================
加载训练好的最佳模型，在测试集上输出分类报告。
用法: python evaluate.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import OUTPUT_DIR, BATCH_SIZE, DEVICE
from dataset import collect_samples, split_data, MedicalImageDataset, get_transforms
from model import MedicalCNN
from train import evaluate, find_best_threshold, test_evaluate


def evaluate_only():
    # ---- 1. 加载数据 ----
    print("正在收集样本...")
    all_samples = collect_samples()
    train_samples, val_samples, test_samples = split_data(all_samples)

    # 测试集仅归一化，不做增强
    val_dataset  = MedicalImageDataset(val_samples,  transform=get_transforms(train=False))
    test_dataset = MedicalImageDataset(test_samples, transform=get_transforms(train=False))

    use_pin_memory = (DEVICE == "cuda")
    num_workers = 0 if DEVICE == "mps" else 2

    val_loader  = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=num_workers, pin_memory=use_pin_memory)

    # ---- 2. 加载模型 ----
    model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}\n请先运行 train.py 训练模型。")

    model = MedicalCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"已加载模型: {model_path}")

    # ---- 3. 在验证集上搜索最优阈值（与训练时一致） ----
    criterion = nn.BCEWithLogitsLoss()
    _, val_labels, val_probs = evaluate(model, val_loader, criterion)
    best_threshold = find_best_threshold(val_labels, val_probs)
    print(f"验证集最优分类阈值: {best_threshold:.4f}")

    # ---- 4. 测试集评估 ----
    test_result = test_evaluate(model, test_loader, criterion, best_threshold)
    print(test_result)

    # 保存结果
    out_path = os.path.join(OUTPUT_DIR, "test_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(test_result)
    print(f"测试结果已保存至: {out_path}")


if __name__ == "__main__":
    evaluate_only()
