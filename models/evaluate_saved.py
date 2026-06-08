import sys
sys.path.append(".")

import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
from models.transformer_model import PMUTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_test  = np.load("data/processed/X_test.npy")
y_test  = np.load("data/processed/y_test.npy")

X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.LongTensor(y_test).to(device)

test_loader = DataLoader(
    TensorDataset(X_test_t, y_test_t),
    batch_size=64, shuffle=False
)

model = PMUTransformer(n_features=7, n_classes=6).to(device)
model.load_state_dict(torch.load("models/saved/best_model.pt"))
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

np.save("results/test_preds.npy",  all_preds)
np.save("results/test_labels.npy", all_labels)
np.save("results/train_losses.npy", np.array([]))
np.save("results/val_losses.npy",   np.array([]))
np.save("results/train_accs.npy",   np.array([]))
np.save("results/val_accs.npy",     np.array([]))

print(f"Test Accuracy: {test_acc * 100:.2f}%")
print("Predictions saved to results/")