# Fraud Detection System — Deep Neural Network

---

## Project Structure

```
.
├── data/               ← Place creditcard.csv here (optional)
├── demo/
│   └── demo.py         ← Live inference demo
├── docs/               ← Project documentation (reports and presentations)
├── models/             ← Saved model weights
├── results/            ← Plots & metrics
├── train.py            ← Main DNN training script
├── train_gpu.py        ← Training script optimized for GPU
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset from Kaggle
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#    Place creditcard.csv inside the data/ folder

# 3. Train the model
python train.py

# 4. Run the live demo
python demo/demo.py
```

> **No dataset?** `train.py` and `demo.py` both fall back to synthetic data automatically — you can test the full pipeline without downloading anything.

---

## Model Architecture

```
Input (30 features)
    ↓
Linear(512) → BatchNorm → GELU → Dropout(0.3)
    ↓
Linear(256) → BatchNorm → GELU → Dropout(0.3)
    ↓
Linear(128) → BatchNorm → GELU → Dropout(0.3)
    ↓
Linear(64)  → BatchNorm → GELU → Dropout(0.15)
    ↓
Linear(32)  → BatchNorm → GELU
    ↓
Linear(1)   → BCEWithLogitsLoss → Sigmoid
```

---

## Handling Class Imbalance

Two strategies are combined:

| Strategy | Description |
|---|---|
| **SMOTE** | Synthetic oversampling of the minority (fraud) class |
| **Class Weights** | `pos_weight` in BCEWithLogitsLoss penalises fraud misses more |
| **Weighted Sampler** | DataLoader samples mini-batches with fraud-boosted probability |

---

## Outputs

After running `train.py`, the `results/` folder contains:

| File | Description |
|---|---|
| `training_curves.png` | Loss, AUC, and F1 over epochs |
| `confusion_matrix.png` | TP / TN / FP / FN breakdown |
| `roc_curve.png` | ROC curve with AUC score |
| `pr_curve.png` | Precision-Recall curve |
| `metrics.json` | Final test metrics (AUC, F1, AP) |

---

## Reproducibility

All random seeds are fixed to `42`. Results are deterministic on the same hardware.

```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
```
