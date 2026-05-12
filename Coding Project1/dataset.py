"""
医学图像数据集加载模块
=====================
- 扫描 archive/ 目录，构建 (路径, 标签, 患者ID) 列表
- 按患者 ID 使用 GroupShuffleSplit 划分训练/验证/测试集
- 训练集应用 albumentations 增强，验证/测试集仅归一化
"""

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupShuffleSplit

from config import (
    DATA_DIR, IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD,
    VAL_RATIO, TEST_RATIO, SPLIT_SEED,
)


def collect_samples(data_dir: str = DATA_DIR) -> list[dict]:
    """
    遍历 archive 目录，收集所有样本的元信息。

    目录结构: archive/<patient_id>/<label>/*.png
    label 0 = 不患病（正常），label 1 = 患病

    Returns:
        samples: [{"path": str, "label": int, "patient_id": str}, ...]
    """
    samples = []
    for patient_id in sorted(os.listdir(data_dir)):
        patient_dir = os.path.join(data_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue
        for label_str in ("0", "1"):
            label_dir = os.path.join(patient_dir, label_str)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(".png"):
                    samples.append({
                        "path": os.path.join(label_dir, fname),
                        "label": int(label_str),
                        "patient_id": patient_id,
                    })
    return samples


class MedicalImageDataset(Dataset):
    """医学组织切片图像 Dataset，支持 albumentations 变换。"""

    def __init__(self, samples: list[dict], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        # cv2 读取为 BGR，转为 RGB 以匹配 albumentations / ImageNet 归一化
        img = cv2.imread(sample["path"])
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {sample['path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = sample["label"]
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]
        return img, label

    def get_patient_ids(self) -> list[str]:
        """返回每个样本对应的患者 ID，供 GroupShuffleSplit 使用。"""
        return [s["patient_id"] for s in self.samples]


def get_transforms(train: bool = True) -> A.Compose:
    """
    获取数据变换。

    Args:
        train: True 返回训练用增强（翻转、旋转、颜色抖动 + 归一化），
               False 返回仅归一化。
    """
    if train:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, p=0.5),
            A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ToTensorV2(),
        ])


def split_data(samples: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """
    按患者 ID 使用 GroupShuffleSplit 划分数据集（60/20/20）。

    步骤：
    1. 先分出 40% 作为临时集（val + test），60% 为训练集
    2. 再将临时集对半分出验证集与测试集
    禁止同一患者的切片出现在不同集合中。

    Returns:
        (train_samples, val_samples, test_samples)
    """
    n = len(samples)
    indices = np.arange(n)
    groups = np.array([s["patient_id"] for s in samples])

    # Step 1: train (60%) vs temp (40%)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=VAL_RATIO + TEST_RATIO,
                             random_state=SPLIT_SEED)
    train_idx, temp_idx = next(gss1.split(indices, groups=groups))

    # Step 2: val (20%) vs test (20%) from temp
    temp_groups = groups[temp_idx]
    temp_indices = indices[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
                             random_state=SPLIT_SEED)
    val_idx_in_temp, test_idx_in_temp = next(gss2.split(temp_indices, groups=temp_groups))

    val_idx = temp_indices[val_idx_in_temp]
    test_idx = temp_indices[test_idx_in_temp]

    train_samples = [samples[i] for i in train_idx]
    val_samples   = [samples[i] for i in val_idx]
    test_samples  = [samples[i] for i in test_idx]

    # 输出划分统计
    for name, subset in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        patients = set(s["patient_id"] for s in subset)
        c0 = sum(1 for s in subset if s["label"] == 0)
        c1 = sum(1 for s in subset if s["label"] == 1)
        print(f"[{name}] patients={len(patients)}, samples={len(subset)}, "
              f"class0(normal)={c0}, class1(diseased)={c1}")

    return train_samples, val_samples, test_samples


if __name__ == "__main__":
    print("正在收集样本...")
    all_samples = collect_samples()
    print(f"总样本数: {len(all_samples)}")
    c0 = sum(1 for s in all_samples if s["label"] == 0)
    c1 = sum(1 for s in all_samples if s["label"] == 1)
    print(f"  class 0 (不患病): {c0}")
    print(f"  class 1 (患病):   {c1}")
    print(f"  不均衡比 (0/1):   {c0 / c1:.2f}")

    print("\n划分数据集...")
    train_s, val_s, test_s = split_data(all_samples)

    print("\n测试 Dataset 加载...")
    train_ds = MedicalImageDataset(train_s, transform=get_transforms(train=True))
    img, lbl = train_ds[0]
    print(f"  图像 shape: {img.shape}, 标签: {lbl}")
