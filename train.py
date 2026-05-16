"""
Fraud Detection - DNN Training Script
DNN architecture, training, metrics, optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os, json, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED        = 42
BATCH_SIZE  = 512
EPOCHS      = 50
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results"
MODEL_PATH  = "models/dnn_fraud.pt"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(path="data/creditcard.csv"):
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Dataset loaded: {df.shape}")
    else:
        print(f"⚠  {path} not found — generating synthetic demo data.")
        df = generate_synthetic_data()

    # ---> FEATURE ENGINEERING <---
    # 1. Convert seconds into Hour of Day (0-23)
    df['hour'] = (df['Time'] // 3600) % 24
    # 2. Drop the original 'Time'
    df = df.drop(['Time'], axis=1)

    return df

def generate_synthetic_data(n=10000, fraud_ratio=0.02):
    """Synthetic stand-in with the same schema as the Kaggle dataset."""
    np.random.seed(SEED)
    n_fraud = int(n * fraud_ratio)
    n_legit = n - n_fraud

    legit  = np.random.randn(n_legit, 28)
    fraud  = np.random.randn(n_fraud, 28) + 2.5   # shifted distribution

    X = np.vstack([legit, fraud])
    y = np.array([0]*n_legit + [1]*n_fraud)
    t = np.random.uniform(0, 172800, n)
    amt = np.abs(np.random.randn(n)) * 100

    cols = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "Time", t)
    df["Amount"] = amt
    df["Class"]  = y
    return df
# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    scaler = StandardScaler()
    df = df.copy()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    
    # -> THE FIX: Scale 'hour' instead of 'Time' <-
    df["hour"]   = scaler.fit_transform(df[["hour"]])

    X = df.drop("Class", axis=1).values.astype(np.float32)
    y = df["Class"].values.astype(np.float32)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    print(f"Train: {X_tr.shape}  Val: {X_val.shape}  Test: {X_te.shape}")
    print(f"Fraud in train: {y_tr.sum():.0f} / {len(y_tr)}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te
# ─────────────────────────────────────────────
# 3. CLASS IMBALANCE: SMOTE + CLASS WEIGHTS
# ─────────────────────────────────────────────
def handle_imbalance(X_tr, y_tr, strategy="smote"):
    """
    strategy: 'smote'   — oversample minority with SMOTE
              'weights' — use class weights only (no resampling)
              'both'    — SMOTE then class weights
    """
    if strategy in ("smote", "both"):
        sm = SMOTE(sampling_strategy=0.35, random_state=SEED)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        print(f"After SMOTE → {int(y_tr.sum())} fraud / {int((y_tr==0).sum())} legit")

    classes = np.unique(y_tr)
    cw = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    print(f"Class weights: {dict(zip(classes.astype(int), cw.round(3)))}")
    return X_tr, y_tr, class_weights

# ─────────────────────────────────────────────
# 4. DNN ARCHITECTURE
# ─────────────────────────────────────────────
class FraudDNN(nn.Module):
    """
    Deep Neural Network for binary fraud classification.
    Architecture: Input → [512→256→128→64→32] → Output
    Each hidden block: Linear → BatchNorm → GELU → Dropout
    """
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            # Block 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            # Block 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            # Block 4
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            # Block 5
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            # Output
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(1)

# ─────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────
def make_loaders(X_tr, y_tr, X_val, y_val):
    tr_ds  = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    # Weighted sampler for extra minority-class emphasis
    sample_weights = np.where(y_tr == 1, (y_tr == 0).sum() / (y_tr == 1).sum(), 1.0)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    tr_loader  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return tr_loader, val_loader

def train(model, tr_loader, val_loader, class_weights, epochs=EPOCHS):
    pos_weight = class_weights[1] / class_weights[0]
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}
    best_auc, patience, wait = 0.0, 7, 0

    for epoch in range(1, epochs + 1):
        # — Train —
        model.train()
        tr_loss = 0.0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(tr_loader.dataset)

        # — Validate —
        model.eval()
        val_loss, preds, labels = 0.0, [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits = model(Xb)
                val_loss += criterion(logits, yb).item() * len(yb)
                preds.extend(torch.sigmoid(logits).cpu().numpy())
                labels.extend(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)

        preds_arr = np.array(preds)
        labels_arr = np.array(labels)
        auc = roc_auc_score(labels_arr, preds_arr)
        f1  = f1_score(labels_arr, (preds_arr > 0.5).astype(int), zero_division=0)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(f1)
        history["val_auc"].append(auc)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Loss {tr_loss:.4f}/{val_loss:.4f} | "
                  f"AUC {auc:.4f} | F1 {f1:.4f}")

        # Early stopping
        if auc > best_auc:
            best_auc = auc
            wait = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return history

# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, X_te, y_te, threshold=0.95):
    model.eval()
    te_ds = TensorDataset(torch.tensor(X_te), torch.tensor(y_te))
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE)

    probs, labels = [], []
    with torch.no_grad():
        for Xb, yb in te_loader:
            logits = model(Xb.to(DEVICE))
            probs.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(yb.numpy())

    probs  = np.array(probs)
    labels = np.array(labels)
    preds  = (probs > threshold).astype(int)

    print("\n" + "="*55)
    print("TEST SET RESULTS")
    print("="*55)
    print(classification_report(labels, preds, target_names=["Legit","Fraud"]))
    print(f"ROC-AUC : {roc_auc_score(labels, probs):.4f}")
    print(f"Avg Prec: {average_precision_score(labels, probs):.4f}")

    metrics = {
        "roc_auc"   : float(roc_auc_score(labels, probs)),
        "avg_prec"  : float(average_precision_score(labels, probs)),
        "f1_fraud"  : float(f1_score(labels, preds, pos_label=1)),
        "threshold" : threshold,
    }
    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return probs, labels, preds, metrics

# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────
def save_plots(history, probs, labels, preds):
    plt.style.use("seaborn-v0_8-darkgrid")

    # — Loss curves —
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss Curve"); axes[0].legend()

    axes[1].plot(history["val_auc"], label="AUC", color="green")
    axes[1].plot(history["val_f1"],  label="F1",  color="orange")
    axes[1].set_title("Val AUC & F1"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_curves.png", dpi=150)
    plt.close()

    # — Confusion matrix —
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit","Fraud"],
                yticklabels=["Legit","Fraud"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # — ROC curve —
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"DNN (AUC={auc:.4f})", color="steelblue")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/roc_curve.png", dpi=150)
    plt.close()

    # — Precision-Recall curve —
    prec, rec, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"DNN (AP={ap:.4f})", color="darkorange")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/pr_curve.png", dpi=150)
    plt.close()

    print(f"\nPlots saved to {RESULTS_DIR}/")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    X_tr, X_val, X_te, y_tr, y_val, y_te = preprocess(df)
    X_tr, y_tr, class_weights = handle_imbalance(X_tr, y_tr, strategy="smote")

    model = FraudDNN(input_dim=X_tr.shape[1]).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")
    print(model)

    tr_loader, val_loader = make_loaders(X_tr, y_tr, X_val, y_val)
    history = train(model, tr_loader, val_loader, class_weights)
    probs, labels, preds, metrics = evaluate(model, X_te, y_te)
    save_plots(history, probs, labels, preds)

    print("\n✅ Training complete. Metrics:", metrics)
