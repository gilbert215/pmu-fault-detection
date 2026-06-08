import sys
sys.path.append(".")

import numpy as np
import torch
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from models.transformer_model import PMUTransformer

os.makedirs("results", exist_ok=True)

device = torch.device("cpu")  # SHAP works best on CPU

# ── Load data and model ────────────────────────────────────────────────
X_test  = np.load("data/processed_real/X_test.npy")
y_test  = np.load("data/processed_real/y_test.npy")

with open("data/processed_real/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

class_names  = list(label_encoder.classes_)
feature_names = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "Fi"]

model = PMUTransformer(n_features=7, n_classes=6).to(device)
model.load_state_dict(
    torch.load("models/saved_real/best_model.pt", weights_only=True)
)
model.eval()

# ── Wrapper: flatten window → model → probabilities ───────────────────
# SHAP needs a function that takes a 2D numpy array (samples, features)
# We reshape each sample back to (1, window_size, n_features)
WINDOW_SIZE = 10
N_FEATURES  = 7

def model_predict(x_flat):
    """
    x_flat: numpy array of shape (n_samples, window_size * n_features)
    returns: numpy array of shape (n_samples, n_classes) — probabilities
    """
    x_tensor = torch.FloatTensor(x_flat).reshape(-1, WINDOW_SIZE, N_FEATURES)
    with torch.no_grad():
        logits = model(x_tensor)
        probs  = torch.softmax(logits, dim=1)
    return probs.numpy()

# ── Sample data for SHAP ───────────────────────────────────────────────
# Flatten windows: (samples, 10, 7) → (samples, 70)
X_flat = X_test.reshape(len(X_test), -1)

# Use 100 background samples and 200 explanation samples
# (SHAP is slow on large datasets — these numbers keep it fast)
np.random.seed(42)
bg_idx   = np.random.choice(len(X_flat), 100, replace=False)
exp_idx  = np.random.choice(len(X_flat), 200, replace=False)

X_background = X_flat[bg_idx]
X_explain    = X_flat[exp_idx]
y_explain    = y_test[exp_idx]

print("Running SHAP KernelExplainer...")
print("(This takes 2-4 minutes — normal)")

explainer   = shap.KernelExplainer(model_predict, X_background)
shap_values = explainer.shap_values(X_explain, nsamples=100)

# shap_values is a list of 6 arrays, one per class
# Each array shape: (200, 70) — 70 = 10 timesteps × 7 features

print("SHAP computation complete.")

# ── Aggregate SHAP across timesteps ───────────────────────────────────
# Average absolute SHAP across the 10 timesteps for each feature
# Result shape per class: (200, 7)
def aggregate_shap_by_feature(shap_class, window_size=10, n_features=7):
    """
    shap_class: (n_samples, window_size * n_features)
    returns:    (n_samples, n_features) — mean absolute SHAP per feature
    """
    reshaped = shap_class.reshape(-1, window_size, n_features)
    return np.abs(reshaped).mean(axis=1)

shap_per_feature = [
    aggregate_shap_by_feature(shap_values[i])
    for i in range(6)
]

# Mean importance per feature per class
mean_importance = np.array([s.mean(axis=0) for s in shap_per_feature])
# shape: (6 classes, 7 features)

# ── Print top features per class ──────────────────────────────────────
print("\n" + "=" * 60)
print("SHAP FEATURE IMPORTANCE PER FAULT TYPE")
print("=" * 60)
for i, cls in enumerate(class_names):
    ranked = np.argsort(mean_importance[i])[::-1]
    print(f"\n{cls}:")
    for rank, feat_idx in enumerate(ranked):
        print(f"  {rank+1}. {feature_names[feat_idx]:<4}  SHAP={mean_importance[i][feat_idx]:.4f}")

# ── Plot 1: Heatmap — feature importance per class ────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

im = ax.imshow(mean_importance, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(N_FEATURES))
ax.set_xticklabels(feature_names, fontsize=11)
ax.set_yticks(range(6))
ax.set_yticklabels(class_names, fontsize=11)
ax.set_title(
    "SHAP Feature Importance per Fault Type\n(mean absolute SHAP value)",
    fontsize=13, fontweight="bold"
)
plt.colorbar(im, ax=ax, label="Mean |SHAP|")

# Annotate cells
for i in range(6):
    for j in range(N_FEATURES):
        ax.text(j, i, f"{mean_importance[i, j]:.3f}",
                ha="center", va="center", fontsize=8,
                color="black" if mean_importance[i, j] < mean_importance.max() * 0.6 else "white")

plt.tight_layout()
plt.savefig("results/shap_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: results/shap_heatmap.png")

# ── Plot 2: Bar chart — overall feature importance (all classes) ───────
overall_importance = mean_importance.mean(axis=0)
ranked_idx         = np.argsort(overall_importance)[::-1]

colors_bar = ["#E53935", "#1565C0", "#2E7D32", "#F57F17", "#6A1B9A", "#00838F", "#4E342E"]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(
    range(N_FEATURES),
    overall_importance[ranked_idx],
    color=[colors_bar[i] for i in ranked_idx],
    edgecolor="black", linewidth=0.5
)
ax.set_xticks(range(N_FEATURES))
ax.set_xticklabels([feature_names[i] for i in ranked_idx], fontsize=12)
ax.set_ylabel("Mean |SHAP value|", fontsize=11)
ax.set_title(
    "Overall PMU Channel Importance for Fault Classification\n(averaged across all fault types)",
    fontsize=13, fontweight="bold"
)
ax.grid(True, axis="y", alpha=0.3)

for bar, val in zip(bars, overall_importance[ranked_idx]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("results/shap_bar.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/shap_bar.png")

# ── Plot 3: Per-class bar charts (2x3 grid) ────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(
    "SHAP Feature Importance by Fault Type",
    fontsize=14, fontweight="bold"
)

for idx, (cls, ax) in enumerate(zip(class_names, axes.flat)):
    imp     = mean_importance[idx]
    r_idx   = np.argsort(imp)[::-1]
    ax.bar(
        range(N_FEATURES),
        imp[r_idx],
        color=[colors_bar[i] for i in r_idx],
        edgecolor="black", linewidth=0.4
    )
    ax.set_xticks(range(N_FEATURES))
    ax.set_xticklabels([feature_names[i] for i in r_idx], fontsize=10)
    ax.set_title(cls, fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean |SHAP|", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/shap_per_class.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/shap_per_class.png")

print("\n" + "=" * 60)
print("SHAP analysis complete.")
print("=" * 60)
print("\nKey question answered:")
print("Which PMU channels matter most for fault detection?")
print(f"Top overall feature: {feature_names[np.argmax(overall_importance)]}")