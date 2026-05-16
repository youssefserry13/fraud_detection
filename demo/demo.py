import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os, sys

# This ensures the script can find the models folder correctly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "dnn_fraud.pt")
CSV_PATH = os.path.join(BASE_DIR, "creditcard.csv") # Updated path for your data

class FraudDNN(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(dropout*0.5),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.GELU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 30 
THRESHOLD = 0.9996  # <--- YOUR BEST THRESHOLD FROM TRAINING

def load_model(path=MODEL_PATH):
    model = FraudDNN(input_dim=INPUT_DIM).to(DEVICE)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"✅ SUCCESS: Loaded trained weights from {path}")
    else:
        print(f"❌ ERROR: Model not found at {path}")
        print("   Make sure you ran train_gpu.py first!")
    model.eval()
    return model

def predict(model, X, threshold=THRESHOLD):
    t = torch.tensor(X.astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        logits = model(t)
        probs  = torch.sigmoid(logits).cpu().numpy()
    labels = (probs > threshold).astype(int)
    return probs, labels

def generate_synthetic_data(n=1000, fraud_ratio=0.02):
    """Synthetic stand-in with the same schema as the Kaggle dataset."""
    SEED = 42
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

def demo_from_csv(model, csv_path="data/creditcard.csv", n=20):
    from sklearn.preprocessing import StandardScaler
    if not os.path.exists(csv_path):
        print(f"⚠ CSV not found at {csv_path}. Generating synthetic demo data for live demo.")
        df = generate_synthetic_data(n=1000)
    else:
        df = pd.read_csv(csv_path)
    
    # -> APPLY HOUR TRICK TO MATCH TRAINING  <-
    df['hour'] = (df['Time'] // 3600) % 24
    df = df.drop(['Time'], axis=1)
    
    fraud_df = df[df["Class"] == 1].sample(min(n//2, len(df[df["Class"]==1])), random_state=42)
    legit_df = df[df["Class"] == 0].sample(n - len(fraud_df), random_state=42)
    sample   = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42)

    scaler = StandardScaler()
    sample["Amount"] = scaler.fit_transform(sample[["Amount"]])
    sample["hour"]   = scaler.fit_transform(sample[["hour"]])

    X = sample.drop("Class", axis=1).values.astype(np.float32)
    y = sample["Class"].values

    probs, labels = predict(model, X)
    correct = (labels == y).sum()

    print("\n" + "="*60)
    print(f"   LIVE DEMO: REAL KAGGLE DATA (Threshold: {THRESHOLD})")
    print("="*60)
    for i in range(len(y)):
        verdict = "🚨 FRAUD" if labels[i] == 1 else "✅ LEGIT"
        bar = "█" * int(probs[i] * 20) + "░" * (20 - int(probs[i] * 20))
        gt = "FRAUD" if y[i] == 1 else "LEGIT"
        print(f" Sample {i+1:2d} | {verdict} | Prob: {probs[i]:.4f} [{bar}] (True: {gt})")
    print("-" * 60)
    print(f" Accuracy: {correct}/{len(y)} ({100*correct/len(y):.1f}%)")

if __name__ == "__main__":
    model = load_model()
    demo_from_csv(model)