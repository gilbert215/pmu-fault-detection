import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

os.makedirs("data/processed", exist_ok=True)

def preprocess(csv_path="data/pmu_fault_data.csv"):

    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # ── Features and labels ────────────────────────────────────────────
    feature_cols = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "Freq"]
    X = df[feature_cols].values
    y = df["fault_type"].values

    # ── Encode labels to integers ──────────────────────────────────────
    # Normal=0, LG=1, LL=2, LLG=3, LLL=4, LLLG=5
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"\nLabel mapping:")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  {cls} → {i}")

    # ── Scale features ─────────────────────────────────────────────────
    # StandardScaler: zero mean, unit variance
    # Fit ONLY on training data to avoid data leakage
    scaler = StandardScaler()

    # ── Train / validation / test split ───────────────────────────────
    # 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.30,
        random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50,
        random_state=42, stratify=y_temp
    )

    # Fit scaler on train only, transform all splits
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"\nSplit sizes:")
    print(f"  Train      : {X_train.shape[0]} samples")
    print(f"  Validation : {X_val.shape[0]} samples")
    print(f"  Test       : {X_test.shape[0]} samples")
    print(f"  Features   : {X_train.shape[1]}")

    # ── Create sliding windows ─────────────────────────────────────────
    # PMU data is time-series — group into windows of 10 timesteps
    # This gives the model temporal context for each prediction
    def create_windows(X, y, window_size=10):
        Xw, yw = [], []
        for i in range(len(X) - window_size + 1):
            Xw.append(X[i:i + window_size])
            yw.append(y[i + window_size - 1])  # label = last step in window
        return np.array(Xw), np.array(yw)

    window_size = 10
    X_train_w, y_train_w = create_windows(X_train, y_train, window_size)
    X_val_w,   y_val_w   = create_windows(X_val,   y_val,   window_size)
    X_test_w,  y_test_w  = create_windows(X_test,  y_test,  window_size)

    print(f"\nAfter windowing (window_size={window_size}):")
    print(f"  Train shape : {X_train_w.shape}")
    print(f"  Val shape   : {X_val_w.shape}")
    print(f"  Test shape  : {X_test_w.shape}")
    print(f"  (samples, timesteps, features)")

    # ── Save everything ────────────────────────────────────────────────
    np.save("data/processed/X_train.npy", X_train_w)
    np.save("data/processed/X_val.npy",   X_val_w)
    np.save("data/processed/X_test.npy",  X_test_w)
    np.save("data/processed/y_train.npy", y_train_w)
    np.save("data/processed/y_val.npy",   y_val_w)
    np.save("data/processed/y_test.npy",  y_test_w)

    # Save scaler and label encoder for later use
    with open("data/processed/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("data/processed/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print("\nAll files saved to data/processed/")
    print("\nPreprocessing complete!")

    return X_train_w, X_val_w, X_test_w, y_train_w, y_val_w, y_test_w, label_encoder


if __name__ == "__main__":
    preprocess()