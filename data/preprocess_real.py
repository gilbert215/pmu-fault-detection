import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

os.makedirs("data/processed_real", exist_ok=True)

print("Loading real PMU dataset...")
df = pd.read_csv("data/pmu_real_data.csv")
df["fault_types"] = df["fault_types"].replace("NF", "Normal")

feature_cols = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic", "Fi"]
X = df[feature_cols].values
y = df["fault_types"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Label mapping:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"  {cls} → {i}")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

def create_windows(X, y, window_size=10):
    Xw, yw = [], []
    for i in range(len(X) - window_size + 1):
        Xw.append(X[i:i + window_size])
        yw.append(y[i + window_size - 1])
    return np.array(Xw), np.array(yw)

window_size = 10
X_train_w, y_train_w = create_windows(X_train, y_train, window_size)
X_val_w,   y_val_w   = create_windows(X_val,   y_val,   window_size)
X_test_w,  y_test_w  = create_windows(X_test,  y_test,  window_size)

print(f"\nSplit sizes after windowing:")
print(f"  Train : {X_train_w.shape}")
print(f"  Val   : {X_val_w.shape}")
print(f"  Test  : {X_test_w.shape}")

np.save("data/processed_real/X_train.npy", X_train_w)
np.save("data/processed_real/X_val.npy",   X_val_w)
np.save("data/processed_real/X_test.npy",  X_test_w)
np.save("data/processed_real/y_train.npy", y_train_w)
np.save("data/processed_real/y_val.npy",   y_val_w)
np.save("data/processed_real/y_test.npy",  y_test_w)

with open("data/processed_real/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("data/processed_real/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nSaved to data/processed_real/")
print("Preprocessing complete!")