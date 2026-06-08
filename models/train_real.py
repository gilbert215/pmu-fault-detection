import sys
sys.path.append(".")

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.transformer_model import PMUTransformer, count_parameters
import pickle

os.makedirs("models/saved_real", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading real preprocessed data...")
X_train = np.load("data/processed_real/X_train.npy")
X_val   = np.load("data/processed_real/X_val.npy")
X_test  = np.load("data/processed_real/X_test.npy")
y_train = np.load("data/processed_real/y_train.npy")
y_val   = np.load("data/processed_real/y_val.npy")
y_test  = np.load("data/processed_real/y_test.npy")

print(f"Train : {X_train.shape}")
print(f"Val   : {X_val.shape}")
print(f"Test  : {X_test.shape}")

X_train_t = torch.FloatTensor(X_train).to(device)
X_val_t   = torch.FloatTensor(X_val).to(device)
X_test_t  = torch.FloatTensor(X_test).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
y_val_t   = torch.LongTensor(y_val).to(device)
y_test_t  = torch.LongTensor(y_test).to(device)

BATCH_SIZE = 128

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=BATCH_SIZE, shuffle=False)

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

EPOCHS        = 50
LEARNING_RATE = 1e-3
PATIENCE      = 10

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)

train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_val_loss    = float("inf")
patience_counter = 0

print("\n" + "=" * 60)
print("Training on REAL PMU data...")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):

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
        correct    += (outputs.argmax(dim=1) == y_batch).sum().item()
        total      += y_batch.size(0)

    train_loss = epoch_loss / total
    train_acc  = correct / total

    model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs      = model(X_batch)
            loss         = criterion(outputs, y_batch)
            val_loss_sum += loss.item() * X_batch.size(0)
            val_correct  += (outputs.argmax(dim=1) == y_batch).sum().item()
            val_total    += y_batch.size(0)

    val_loss = val_loss_sum / val_total
    val_acc  = val_correct  / val_total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "models/saved_real/best_model.pt")
        print(f"  --> Best model saved (val_loss={val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

np.save("results/real_train_losses.npy", np.array(train_losses))
np.save("results/real_val_losses.npy",   np.array(val_losses))
np.save("results/real_train_accs.npy",   np.array(train_accs))
np.save("results/real_val_accs.npy",     np.array(val_accs))

print("\n" + "=" * 60)
print("Evaluating on test set...")
print("=" * 60)

model.load_state_dict(torch.load("models/saved_real/best_model.pt", weights_only=True))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
test_acc   = (all_preds == all_labels).mean()

np.save("results/real_test_preds.npy",  all_preds)
np.save("results/real_test_labels.npy", all_labels)

print(f"\nReal Data Test Accuracy: {test_acc * 100:.2f}%")
print("Training complete!")