import math
import os

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, precision_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE, EPOCHS, LR, DROPOUT, WEIGHT_DECAY, NUM_WORKERS,
    WARMUP_RATIO, DEVICE, CHECKPOINT_PATH, SEED,
)
from dataset import build_splits, OCTDataset
from model import CNN
from visualize import plot_curves


def set_seed():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)


def train_epoch(model, loader, criterion, optimizer, device, pbar):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        pbar.update(1)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, pbar):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        loss = criterion(model(imgs), labels)
        total_loss += loss.item() * imgs.size(0)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        pbar.update(1)
    return total_loss / total, correct / total


@torch.no_grad()
def run_test(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average=None)
    precision = precision_score(all_labels, all_preds, average=None)
    return acc, recall, precision


def train():
    print(f"Using device: {DEVICE}")
    set_seed()
    train_samples, val_samples, test_samples = build_splits()
    train_loader = DataLoader(OCTDataset(train_samples), BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    val_loader = DataLoader(OCTDataset(val_samples), BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    test_loader = DataLoader(OCTDataset(test_samples), BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

    model = CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warmup_epochs = int(EPOCHS * WARMUP_RATIO)
    total_epochs = EPOCHS - warmup_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - 1, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(EPOCHS):
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{EPOCHS} Train", leave=False, unit="batch")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, train_pbar)
        train_pbar.close()

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:3d}/{EPOCHS} Val  ", leave=False, unit="batch")
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, DEVICE, val_pbar)
        val_pbar.close()

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        tqdm.write(f"Epoch {epoch+1:3d}/{EPOCHS}")
        tqdm.write(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
                   f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  "
                   f"LR: {scheduler.get_last_lr()[0]:.2e}")
        tqdm.write("")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "history": history,
            }, CHECKPOINT_PATH)

    os.makedirs("figure", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    file_tag = f"{BATCH_SIZE}_{LR}_{DROPOUT}"
    figure_path = f"figure/{file_tag}.png"
    output_path = f"output/{file_tag}.txt"

    plot_curves(history, save_path=figure_path)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_acc, test_recall, test_precision = run_test(model, test_loader, DEVICE)

    class_names = ["Class 0 (Healthy)", "Class 1 (Diseased)"]
    with open(output_path, "w") as f:
        f.write(f"Hyperparameters: batch_size={BATCH_SIZE}, lr={LR}, dropout={DROPOUT}\n")
        f.write(f"Best val accuracy: {best_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        for i, name in enumerate(class_names):
            f.write(f"  {name} — Recall: {test_recall[i]:.4f}  Precision: {test_precision[i]:.4f}\n")

    print(f"Results saved to {output_path}")
    print(f"Figures saved to {figure_path}")
    print(f"Training done. Best val acc: {best_acc:.4f}  Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    train()
