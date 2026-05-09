import os
import random
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import DATA_DIR, IMG_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED


def _collect_samples():
    samples = defaultdict(list)
    for pid in os.listdir(DATA_DIR):
        pid_path = os.path.join(DATA_DIR, pid)
        if not os.path.isdir(pid_path):
            continue
        for cls in ("0", "1"):
            cls_path = os.path.join(pid_path, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    samples[pid].append((os.path.join(cls_path, fname), int(cls)))
    return samples


def build_splits():
    samples = _collect_samples()
    pids = list(samples.keys())
    random.seed(SEED)
    random.shuffle(pids)
    n = len(pids)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    train_pids = set(pids[:train_end])
    val_pids = set(pids[train_end:val_end])
    test_pids = set(pids[val_end:])

    def gather(pid_set):
        data = []
        for pid in pid_set:
            data.extend(samples[pid])
        return data

    return gather(train_pids), gather(val_pids), gather(test_pids)


class OCTDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
