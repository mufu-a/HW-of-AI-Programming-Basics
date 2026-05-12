"""
医学图像二分类 CNN 模型
======================
带残差连接的高效卷积网络 + GlobalAveragePooling，二分类输出 1 个 logit。
"""

import torch
import torch.nn as nn
from config import DROPOUT


class ResConvBlock(nn.Module):
    """带残差连接的双卷积块: Conv→BN→ReLU → Conv→BN → +skip → ReLU → (optional MaxPool)."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Sequential()
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = torch.relu(out + identity)
        out = self.pool(out)
        return out


class MedicalCNN(nn.Module):
    """用于组织切片二分类的宽残差 CNN 模型。"""

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()

        # 4 个残差卷积块: 32→64→128→256
        self.block1 = ResConvBlock(3, 32, pool=True)     # 32×32×32
        self.block2 = ResConvBlock(32, 64, pool=True)    # 64×16×16
        self.block3 = ResConvBlock(64, 128, pool=True)   # 128×8×8
        self.block4 = ResConvBlock(128, 256, pool=True)  # 256×4×4

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = MedicalCNN()
    dummy = torch.randn(4, 3, 50, 50)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
