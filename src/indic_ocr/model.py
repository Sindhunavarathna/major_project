"""Convolutional Recurrent Neural Network (CRNN) for unified Indic handwriting recognition.

Architecture:
1. Deep CNN Backbone: 7 Convolutional layers with BatchNorm, ReLU, and MaxPooling.
   Downsamples (B, 1, 32, W) -> (B, 512, 1, W/4).
2. Sequence Reshaping: Transposes spatial feature map to time sequence (W/4, B, 512).
3. Recurrent Layers: 2-layer Bidirectional LSTM capturing character context bidirectionally.
4. Linear Transcription: Projects 512-dim LSTM features to character vocabulary classes.
5. CTC Optimization: Computes Connectionist Temporal Classification loss.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.indic_ocr.charset import UnifiedIndicCharset


class IndicCRNN(nn.Module):
    """Deep CRNN with Bidirectional LSTM for Kannada, Tamil, and Telugu handwriting recognition."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.num_classes = num_classes

        # CNN Feature Extractor
        # Downsamples Height: 32 -> 16 -> 8 -> 4 -> 2 -> 1
        # Downsamples Width: W -> W/2 -> W/4 -> W/4 -> W/4 -> W/4
        self.cnn = nn.Sequential(
            # Block 1: 32xW -> 16x(W/2)
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 16x(W/2) -> 8x(W/4)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 8x(W/4) -> 8x(W/4)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4: 8x(W/4) -> 4x(W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 5: 4x(W/4) -> 4x(W/4)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Block 6: 4x(W/4) -> 2x(W/4)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 7: 2x(W/4) -> 1x(W/4)
            nn.Conv2d(512, 512, kernel_size=(2, 1), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Bidirectional LSTM Sequence Modeler
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=False,  # Input/Output is (T, B, H)
        )

        # Final Transcription Linear Layer
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor of shape (Batch, 1, Height=32, Width).

        Returns:
            Log-probabilities of shape (T, Batch, num_classes) suitable for nn.CTCLoss.
        """
        # 1. CNN Feature extraction: (B, 1, 32, W) -> (B, 512, 1, W/4)
        features = self.cnn(x)

        # 2. Squeeze height: (B, 512, 1, W/4) -> (B, 512, W/4)
        features = features.squeeze(2)

        # 3. Permute to time-first sequence for LSTM: (W/4, B, 512)
        sequence = features.permute(2, 0, 1)

        # 4. Bidirectional LSTM: (T, B, 512) -> (T, B, 2 * lstm_hidden)
        recurrent, _ = self.lstm(sequence)

        # 5. Projection to classes: (T, B, num_classes)
        logits = self.fc(recurrent)

        # 6. Log-softmax over vocabulary classes
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs

    def predict_greedy(
        self,
        x: torch.Tensor,
        charset: UnifiedIndicCharset,
    ) -> Tuple[str, float]:
        """Inference helper performing greedy CTC decoding and returning (text, confidence)."""
        self.eval()
        with torch.no_grad():
            log_probs = self.forward(x)  # (T, 1, num_classes)
            # Exponentiate to get probabilities
            probs = torch.exp(log_probs).squeeze(1)  # (T, num_classes)
            max_probs, max_indices = torch.max(probs, dim=1)

            text = charset.decode(max_indices.cpu().tolist(), collapse_repeats=True, remove_blanks=True)

            # Average confidence over non-blank predicted timesteps
            non_blank_mask = max_indices != charset.BLANK_INDEX
            if non_blank_mask.sum() > 0:
                avg_conf = float(max_probs[non_blank_mask].mean().item())
            else:
                avg_conf = float(max_probs.mean().item())

            return text, round(avg_conf, 4)
