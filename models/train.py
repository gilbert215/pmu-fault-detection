import sys
sys.path.append(".")

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.transformer_model import PMUTransformer, count_parameters
import pickle

os.makedirs("models/saved", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── Device ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load preprocessed data ─────────────────────────────────────────────
print("Loading preprocessed data...")
X_train = np.load("data/processed/X_train.npy")
X_val   = np.load("data/processed/X_val.npy")
X_test  = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")
y_val   = np.load("data/processed/y_val.npy")
y_test  = np.load("data/processed/y_test.npy")

with open("data/processed/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print(f"Train : {X_train.shape}")
print(f"Val   : {X_val.shape}")
print(f"Test  : {X_test.shape}")

# ── Convert to PyTorch tensors ─────────────────────────────────────────
X_train_t = torch.FloatTensor(X_train).to(device)
X_val_t   = torch.FloatTensor(X_val).to(device)
X_test_t  = torch.FloatTensor(X_test).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
y_val_t   = torch.LongTensor(y_val).to(device)
y_test_t  = torch.LongTensor(y_test).to(device)

# ── DataLoaders ────────────────────────────────────────────────────────
BATCH_SIZE = 64

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t,   y_val_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── Model ──────────────────────────────────────────────────────────────
model = PMUTransformer(
    n_features=7,
    n_classes=6,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=128,
    dropout=0.1
).to(device)

print(f"\nModel parameters: {count_parameters(model):,}")

# ── Training setup ─────────────────────────────────────────────────────
EPOCHS        = 50
LEARNING_RATE = 1e-3
PATIENCE      = 10    # early stopping patience

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5, verbose=True
)

# ── Training loop ──────────────────────────────────────────────────────
train_losses, val_losses     = [], []
train_accs,   val_accs       = [], []
best_val_loss                = float("inf")
patience_counter             = 0

print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):

    # ── Train ──────────────────────────────────────────────────────────
    model.train()
    epoch_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item() * X_batch.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += y_batch.size(0)

    train_loss = epoch_loss / total
    train_acc  = correct / total

    # ── Validate ───────────────────────────────────────────────────────
    model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs      = model(X_batch)
            loss         = criterion(outputs, y_batch)
            val_loss_sum += loss.item() * X_batch.size(0)
            preds         = outputs.argmax(dim=1)
            val_correct  += (preds == y_batch).sum().item()
            val_total    += y_batch.size(0)

    val_loss = val_loss_sum / val_total
    val_acc  = val_correct  / val_total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    scheduler.step(val_loss)

    # ── Print progress ─────────────────────────────────────────────────
    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    # ── Save best model ────────────────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "models/saved/best_model.pt")
        print(f"  --> Best model saved (val_loss={val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# ── Save training history ──────────────────────────────────────────────
np.save("results/train_losses.npy", np.array(train_losses))
np.save("results/val_losses.npy",   np.array(val_losses))
np.save("results/train_accs.npy",   np.array(train_accs))
np.sav