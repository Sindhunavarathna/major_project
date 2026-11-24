"""Unified Character Set and CTC Encoding/Decoding for Kannada, Tamil, and Telugu.

Covers:
- CTC Blank Token (<blank> at index 0)
- Kannada Unicode Block (U+0C80 - U+0CFF)
- Tamil Unicode Block (U+0B80 - U+0BFF)
- Telugu Unicode Block (U+0C00 - U+0C7F)
- Common Indic and ASCII digits (0-9, native numerals)
- Common punctuation and whitespace
"""

import json
import os
from typing import Dict, List, Optional, Sequence, Union
import torch


class UnifiedIndicCharset:
    """Unified Character Vocabulary spanning Kannada, Tamil, and Telugu scripts for CTC OCR."""

    BLANK_TOKEN = "<blank>"
    BLANK_INDEX = 0

    def __init__(self, custom_chars: Optional[Sequence[str]] = None):
        if custom_chars is not None:
            self._chars = list(custom_chars)
            if self.BLANK_TOKEN not in self._chars:
                self._chars.insert(0, self.BLANK_TOKEN)
        else:
            self._chars = self._build_default_charset()

        self.char_to_idx: Dict[str, int] = {char: idx for idx, char in enumerate(self._chars)}
        self.idx_to_char: Dict[int, str] = {idx: char for idx, char in enumerate(self._chars)}

    def _build_default_charset(self) -> List[str]:
        chars = [self.BLANK_TOKEN]

        # 1. Standard Whitespace & Punctuation
        punct = [" ", ".", ",", "!", "?", "-", ":", ";", "'", '"', "(", ")", "/", "\\", "|"]
        chars.extend(punct)

        # 2. ASCII digits
        chars.extend([str(i) for i in range(10)])

        # 3. Kannada Unicode Block (U+0C80 - U+0CFF)
        for code in range(0x0C80, 0x0D00):
            ch = chr(code)
            if ch.isprintable() and ch not in chars:
                chars.append(ch)

        # 4. Tamil Unicode Block (U+0B80 - U+0BFF)
        for code in range(0x0B80, 0x0C00):
            ch = chr(code)
            if ch.isprintable() and ch not in chars:
                chars.append(ch)

        # 5. Telugu Unicode Block (U+0C00 - U+0C7F)
        for code in range(0x0C00, 0x0C80):
            ch = chr(code)
            if ch.isprintable() and ch not in chars:
                chars.append(ch)

        return chars

    def __len__(self) -> int:
        return len(self._chars)

    @property
    def num_classes(self) -> int:
        return len(self._chars)

    def encode(self, text: str) -> List[int]:
        """Encode text string into list of vocabulary integer indices, skipping unknown chars."""
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    def decode(
        self,
        indices: Sequence[int],
        collapse_repeats: bool = True,
        remove_blanks: bool = True,
    ) -> str:
        """Decode integer indices to text string with CTC collapse and blank filtering."""
        decoded_chars: List[str] = []
        prev_idx: Optional[int] = None

        for idx in indices:
            idx_int = int(idx)
            if idx_int < 0 or idx_int >= len(self._chars):
                continue

            if collapse_repeats and idx_int == prev_idx:
                continue

            prev_idx = idx_int

            if remove_blanks and idx_int == self.BLANK_INDEX:
                continue

            decoded_chars.append(self.idx_to_char[idx_int])

        return "".join(decoded_chars)

    def decode_greedy(
        self,
        log_probs: Union[torch.Tensor, Sequence[Sequence[float]]],
    ) -> str:
        """Greedy CTC best-path decoding from log-probability output tensor of shape (T, C)."""
        if isinstance(log_probs, torch.Tensor):
            if log_probs.ndim == 3:
                # Shape: (T, 1, C) or (1, T, C)
                if log_probs.shape[1] == 1:
                    log_probs = log_probs.squeeze(1)
                else:
                    log_probs = log_probs.squeeze(0)
            best_indices = torch.argmax(log_probs, dim=-1).cpu().tolist()
        else:
            best_indices = [int(max(range(len(step)), key=lambda c: step[c])) for step in log_probs]

        return self.decode(best_indices, collapse_repeats=True, remove_blanks=True)

    def save(self, filepath: str) -> None:
        """Save vocabulary list to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"chars": self._chars}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "UnifiedIndicCharset":
        """Load vocabulary list from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(custom_chars=data["chars"])
