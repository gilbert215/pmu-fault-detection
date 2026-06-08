import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)

def download_ieee_fault_dataset():
    """
    Downloads a PMU fault detection dataset.
    This dataset contains simulated PMU measurements for
    normal operation and 5 fault types on a power system.
    """

    # Direct download URL (hosted on GitHub — free, no account needed)
    url = (
        "https://raw.githubusercontent.com/arnabde05/fault-detection"
        "/main/detect_dataset.csv"
    )

    save_path = "data/pmu_fault_data.csv"

    if os.path.exists(save_path):
        print("Dataset already exists. Skipping download.")
        df = pd.read_csv(save_path)
        print(f"Loaded {len(df)} rows from {save_path}")
        return df

    print("Downloading PMU fault dataset...")

    try:
        urllib.request.urlretrieve(url, save_path)
        df = pd.read_csv(save_path)
        print(f"Downloaded {len(df)} rows successfully.")
        return df

    except Exception as e:
        print(f"Download failed: {e}")
        print("Generating synthetic PMU dataset instead...")
        df = generate_synthetic_pmu_data()
        df.to_csv(save_path, index=False)
        print(f"Saved {len(df)} synthetic rows to {save_path}")
        return df


def generate_synthetic_pmu_data(n_samples=10000):
    """
    Generates realistic synthetic PMU data if download fails.
    Each fault type has distinct voltage/current signatures.
    """
    np.random.seed(42)
    records = []

    fault_types = {
        "Normal": (1.0,  1.0,  0.05),   # (voltage_pu, current_pu, noise)
        "LG":     (0.7,  2.5,  0.08),   # single line to ground
        "LL":     (0.75, 3.0,  0.08),   # line to line
        "LLG":    (0.65, 3.5,  0.10),   # double line to ground
        "LLL":    (0.5,  4.0,  0.10),   # three phase
        "LLLG":   (0.45, 4.5,  0.12),   # three phase to ground
    }

    samples_per_class = n_samples // len(fault_types)

    for fault, (v_pu, i_pu, noise) in fault_types.items():
        for _ in range(samples_per_class):
            # Voltage measurements (3 phases)
            Va = v_pu * 230 + np.random.normal(0, noise * 230)
            Vb = v_pu * 230 + np.random.normal(0, noise * 230)
            Vc = v_pu * 230 + np.random.normal(0, noise * 230)

            # Current measurements (3 phases)
            Ia = i_pu * 100 + np.random.normal(0, noise * 100)
            Ib = i_pu * 100 + np.random.normal(0, noise * 100)
            Ic = i_pu * 100 + np.random.normal(0, noise * 100)

            # Frequency deviation
            freq = 50.0 + np.random.normal(0, 0.05 if fault == "Normal" else 0.3)

            records.append({
                "Va": round(Va, 4),
                "Vb": round(Vb, 4),
                "Vc": round(Vc, 4),
                "Ia": round(Ia, 4),
                "Ib": round(Ib, 4),
                "Ic": round(Ic, 4),
                "Freq": round(freq, 4),
                "fault_type": fault
            })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def explore_dataset(df):
    """Print basic statistics about the dataset."""
    print("\n" + "=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Total samples : {len(df)}")
    print(f"Features      : {list(df.columns)}")
    print(f"\nClass distribution:")
    print(df["fault_type"].value_counts())
    print(f"\nBasic statistics:")
    print(df.describe().round(2))


if __name__ == "__main__":
    df = download_ieee_fault_dataset()
    explore_dataset(df)
    print("\nDataset ready at: data/pmu_fault_data.csv")