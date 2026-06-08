import urllib.request
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

url = "https://zenodo.org/records/8214226/files/PMU_Fault_Data.csv"
save_path = "data/pmu_real_data.csv"

if os.path.exists(save_path):
    print("File already exists. Skipping download.")
else:
    print("Downloading real PMU dataset from Zenodo...")
    urllib.request.urlretrieve(url, save_path)
    print("Download complete.")

df = pd.read_csv(save_path)

print("\n" + "=" * 50)
print("REAL DATASET OVERVIEW")
print("=" * 50)
print(f"Shape        : {df.shape}")
print(f"Columns      : {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nUnique values in each column:")
for col in df.columns:
    unique = df[col].nunique()
    print(f"  {col}: {unique} unique values")