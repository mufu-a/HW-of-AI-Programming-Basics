import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score

from torch.utils.data import DataLoader

from config import BATCH_SIZE, NUM_WORKERS, DEVICE, CHECKPOINT_PATH
from dataset import build_splits, OCTDataset
from model import CNN


@torch.no_grad()
def test():
    print(f"Using device: {DEVICE}")
    _, _, test_samples = build_splits()
    test_loader = DataLoader(OCTDataset(test_samples), BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model = CNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average=None)
    precision = precision_score(all_labels, all_preds, average=None)

    print(f"Test Accuracy: {acc:.4f}")
    for i, name in enumerate(["Class 0 (Healthy)", "Class 1 (Diseased)"]):
        print(f"  {name} — Recall: {recall[i]:.4f}  Precision: {precision[i]:.4f}")


if __name__ == "__main__":
    test()
