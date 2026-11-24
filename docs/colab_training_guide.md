# Google Colab GPU Training Guide: Shared Indic CRNN (Kannada, Tamil, Telugu)

This guide provides instructions to train the Shared Indic CRNN handwriting recognition model on a **Free Google Colab GPU (T4 / V100)**.

---

## 1. Quick Setup in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Go to **Runtime $\to$ Change runtime type $\to$ Hardware accelerator $\to$ T4 GPU** (Free).
3. In a code cell, clone your project repository or upload your project zip:
   ```bash
   !git clone https://github.com/<your-username>/multilingual-manuscript-system.git
   %cd multilingual-manuscript-system
   ```

4. Install requirements:
   ```bash
   !pip install -r requirements.txt
   ```

---

## 2. Prepare Data

### Option A: Using IIIT-INDIC-HW-WORDS (Recommended for large-scale training)
Download the official dataset from CVIT IIIT Hyderabad:
```bash
!mkdir -p data/iiit_indic_hw_words
# Download Kannada, Tamil, Telugu word subsets
# Extract into data/iiit_indic_hw_words/
```

### Option B: Using Team-Collected & Seed Samples
If running directly on team-collected data:
```bash
!python scripts/collect_indic_data.py --summary
```

---

## 3. Launch GPU Training

Run the training script with GPU acceleration:
```bash
!python scripts/train_indic_recognizer.py \
    --epochs 25 \
    --batch-size 32 \
    --lr 0.0005 \
    --device cuda \
    --save-dir models/indic_crnn
```

### Key Training Features
* **Zero Leakage:** Samples are split strictly by `writer_id`. A writer in the test set never appears in the training or validation sets.
* **Per-Language Logging:** Logs Training Loss, Validation Loss, and per-language CER and WER (**Kannada**, **Tamil**, and **Telugu**) at every epoch.
* **Checkpointing:** Automatically saves the best model weights to `models/indic_crnn/best_indic_crnn.pth` and vocabulary to `models/indic_crnn/vocab.json`.
* **Resuming:** If your Colab session disconnects, you can resume training with:
  ```bash
  !python scripts/train_indic_recognizer.py \
      --epochs 30 \
      --device cuda \
      --resume models/indic_crnn/best_indic_crnn.pth
  ```

---

## 4. Download Checkpoint to Local Repo

Once training is complete in Colab, download `best_indic_crnn.pth` and `vocab.json` back to your local project:
```python
from google.colab import files
files.download("models/indic_crnn/best_indic_crnn.pth")
files.download("models/indic_crnn/vocab.json")
```
Place them in `models/indic_crnn/` on your local machine. The local pipeline will automatically detect them for inference.
