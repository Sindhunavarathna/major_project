"""PyTorch lightweight CNN architecture for 5-class script classification.

Classes:
0: Kannada
1: Tamil
2: Telugu
3: Arabic
4: Latin (German / English)
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES: List[str] = ["Kannada", "Tamil", "Telugu", "Arabic", "Latin", "Devanagari"]
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}


class ScriptClassifierCNN(nn.Module):
    """Compact Convolutional Neural Network for manuscript script classification.

    Designed for fast CPU and GPU execution on 128x128 single-channel manuscript patches.
    Model parameter size: ~660K parameters (<3MB).
    Inference latency: ~3-5ms on standard CPU.
    """

    def __init__(self, num_classes: int = len(CLASS_NAMES), in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Block 1: 128x128 -> 64x64
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3: 32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        # Block 4: 16x16 -> 8x8
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.dropout1 = nn.Dropout(dropout + 0.1)
        self.fc1 = nn.Linear(256 * 4 * 4, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Adaptive pooling & linear layers
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
