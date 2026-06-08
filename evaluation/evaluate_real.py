import sys
sys.path.append(".")

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from torch.utils.data import DataLoader, TensorDataset
from models.transformer_model import PMUTransformer
import os

os.makedirs("results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load data ──────────────────────────────────────────────────────────
X_test  = np.load("data/processed_real/X_test.npy")
y_test  = np.load("data/processed_real/y_test.npy")

with open("data/processed_real/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

class_names = list(label_encoder.classes_)

# ── Load model and predict ─────────────────────────────────────────────
model = PMUTransformer(n_features=7, n_classes=6).to(device)
model.load_state_dict(
    torch.load("models/saved_real/best_model.pt", weights_only=True)
)
model.eval()

loader = DataLoader(
    TensorDataset(
        torch.FloatTensor(X_test).to(device),
        torch.LongTensor(y_test).to(device)
    ),
    batch_size=128, shuffle=False
)

all_preds, all_labels = [], []

with torch.no_grad():
    for X_batch, y_batch in loader:
        preds = model(X_batch).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
test_acc   = (all_preds == all_labels).mean()

# ── Print report ───────────────────────────────────────────────────────
print("=" * 60)
print("FINAL EVALUATION — REAL PMU DATA")
print("=" * 60)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ── Load training history ──────────────────────────────────────────────
train_losses = np.load("results/real_train_losses.npy")
val_losses   = np.load("results/real_val_losses.npy")
train_accs   = np.load("results/real_train_accs.npy")
val_accs     = np.load("results/real_val_accs.npy")

epochs = range(1, len(train_losses) + 1)

# ── Figure: 3 panels ───────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig)

colors = {
    "train": "#1565C0",
    "val":   "#E53935",
}

# Panel 1 — Loss curves
ax1 = fig.add_subplot(gs[0])
ax1.plot(epochs, train_losses, color=colors["train"], linewidth=2, label="Train Loss")
ax1.plot(epochs, val_losses,   color=colors["val"],   linewidth=2, label="Val Loss",   linestyle="--")
ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Loss",  fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2 — Accuracy curves
ax2 = fig.add_subplot(gs[1])
ax2.plot(epochs, [a * 100 for a in train_accs], color=colors["train"], linewidth=2, label="Train Acc")
ax2.plot(epochs, [a * 100 for a in val_accs],   color=colors["val"],   linewidth=2, label="Val Acc",   linestyle="--")
ax2.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
ax2.set_xlabel("Epoch",    fontsize=11)
ax2.set_ylabel("Accuracy (%)", fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 105])

# Panel 3 — Confusion matrix
ax3 = fig.add_subplot(gs[2])
cm   = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax3, colorbar=False, cmap="Blues")
ax3.set_title(
    f"Confusion Matrix — Real PMU Data\nAccuracy: {test_acc * 100:.2f}%",
    fontsize=13, fontweight="bold"
)

plt.suptitle(
    "PMU Fault Detection & Classification — Transformer Model on Real Data",
    fontsize=14, fontweight="bold", y=1.02
)

plt.tight_layout()
plt.savefig("results/final_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: results/final_evaluation.png")