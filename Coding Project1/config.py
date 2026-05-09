import torch

DATA_DIR = "archive"
IMG_SIZE = 50
BATCH_SIZE = 128
EPOCHS = 100
LR = 5e-4
DROPOUT = 0.4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
WARMUP_RATIO = 0.05
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
SEED = 42

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
CHECKPOINT_PATH = "best_model.pth"
