import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/pmu_fault_data.csv")

fault_types = ["Normal", "LG", "LL", "LLG", "LLL", "LLLG"]
colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#F44336"]

# ── Plot 1: Voltage distribution per fault type ────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(
    "Voltage (Va) Distribution by Fault Type",
    fontsize=14, fontweight="bold"
)

for idx, (fault, color) in enumerate(zip(fault_types, colors)):
    ax = axes[idx // 3][idx % 3]
    subset = df[df["fault_type"] == fault]["Va"]
    ax.hist(subset, bins=40, color=color, alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.set_title(f"{fault}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Va (kV)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.axvline(subset.mean(), color="black", linestyle="--",
               linewidth=1.5, label=f"Mean: {subset.mean():.1f}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/voltage_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/voltage_distribution.png")

# ── Plot 2: Current distribution per fault type ────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(
    "Current (Ia) Distribution by Fault Type",
    fontsize=14, fontweight="bold"
)

for idx, (fault, color) in enumerate(zip(fault_types, colors)):
    ax = axes[idx // 3][idx % 3]
    subset = df[df["fault_type"] == fault]["Ia"]
    ax.hist(subset, bins=40, color=color, alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.set_title(f"{fault}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ia (A)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.axvline(subset.mean(), color="black", linestyle="--",
               linewidth=1.5, label=f"Mean: {subset.mean():.1f}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/current_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/current_distribution.png")

# ── Plot 3: Box plot — all features by fault type ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(
    "Feature Comparison Across Fault Types",
    fontsize=14, fontweight="bold"
)

features = ["Va", "Ia", "Freq"]
titles   = ["Voltage Phase A (Va)", "Current Phase A (Ia)", "Frequency (Hz)"]

for ax, feat, title in zip(axes, features, titles):
    data_by_fault = [df[df["fault_type"] == f][feat].values for f in fault_types]
    bp = ax.boxplot(data_by_fault, patch_artist=True, labels=fault_types)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Fault Type", fontsize=10)
    ax.set_ylabel(feat, fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/feature_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/feature_boxplots.png")

# ── Plot 4: Scatter — Voltage vs Current colored by fault type ─────────
fig, ax = plt.subplots(figsize=(10, 7))
for fault, color in zip(fault_types, colors):
    subset = df[df["fault_type"] == fault]
    ax.scatter(subset["Va"], subset["Ia"], c=color, label=fault,
               alpha=0.4, s=10, edgecolors="none")

ax.set_xlabel("Voltage Phase A — Va (kV)", fontsize=12)
ax.set_ylabel("Current Phase A — Ia (A)", fontsize=12)
ax.set_title(
    "PMU Measurements: Voltage vs Current by Fault Type",
    fontsize=13, fontweight="bold"
)
ax.legend(fontsize=10, markerscale=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/voltage_vs_current.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/voltage_vs_current.png")

# ── Print summary ──────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("KEY OBSERVATIONS")
print("=" * 50)
for fault in fault_types:
    subset = df[df["fault_type"] == fault]
    print(f"\n{fault}:")
    print(f"  Mean Va : {subset['Va'].mean():.1f} kV")
    print(f"  Mean Ia : {subset['Ia'].mean():.1f} A")
    print(f"  Mean Freq: {subset['Freq'].mean():.3f} Hz")