"""
Data Loading and Inspection Module for Hotel Booking Cancellation Analysis.
Course: Exploratory Data Analysis and Visualization (U21ADP05)
Institution: KPR Institute of Engineering and Technology
"""

import os
import urllib.request
import pandas as pd
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_PATH = os.path.join(DATA_DIR, "hotels.csv")


def ensure_dataset_exists(csv_path: str = DATA_PATH, url: str = DATA_URL) -> str:
    """
    Downloads dataset automatically if not present locally.
    Provides clear manual instructions if download fails.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        print(f"[*] Dataset not found at '{csv_path}'. Downloading from {url}...")
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, csv_path)
            print(f"[+] Download complete! Saved to '{csv_path}'.")
        except Exception as e:
            print(f"[-] Automated download failed: {e}")
            print(
                "\n================ MANUAL DOWNLOAD INSTRUCTIONS ================\n"
                "1. Open your browser and navigate to:\n"
                f"   {url}\n"
                "2. Save the file as 'hotels.csv' inside the 'data/' folder:\n"
                f"   Path: {os.path.abspath(csv_path)}\n"
                "3. Re-run this application once the file is in place.\n"
                "===============================================================\n"
            )
            raise FileNotFoundError(
                f"Dataset missing at '{csv_path}' and automatic download failed."
            )
    else:
        print(f"[+] Dataset already exists at '{csv_path}'.")
    return csv_path


def load_and_inspect_data(csv_path: str = DATA_PATH) -> tuple[pd.DataFrame, dict]:
    """
    Loads dataset, verifies schema, checks dimensions, target column,
    and returns DataFrame along with statistical summary dictionary.
    """
    ensure_dataset_exists(csv_path)
    df = pd.read_csv(csv_path)

    # Dataset verification
    n_rows, n_cols = df.shape
    target_col = "is_canceled"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset!")
    if n_cols < 15:
        raise ValueError(f"Dataset has only {n_cols} columns, expected at least 15 features.")

    # Missing values
    missing_series = df.isnull().sum()
    missing_dict = missing_series[missing_series > 0].to_dict()

    # Duplicates
    duplicate_count = int(df.duplicated().sum())

    # Target distribution
    target_dist = df[target_col].value_counts().to_dict()
    target_pct = (df[target_col].value_counts(normalize=True) * 100).round(2).to_dict()

    summary = {
        "num_rows": n_rows,
        "num_cols": n_cols,
        "target_column": target_col,
        "duplicate_count": duplicate_count,
        "missing_values": missing_dict,
        "target_distribution": target_dist,
        "target_percentage": target_pct,
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "numerical_columns": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(df.select_dtypes(include=["object"]).columns),
    }

    print("\n" + "=" * 60)
    print("        HOTEL BOOKING DATASET INSPECTION SUMMARY")
    print("=" * 60)
    print(f"Total Observations (Rows): {n_rows:,}")
    print(f"Total Variables (Columns):  {n_cols}")
    print(f"Target Variable:           '{target_col}'")
    print(f"Target Class Counts:       {target_dist}")
    print(f"Target Class Proportions:  {target_pct}%")
    print(f"Duplicate Rows Detected:   {duplicate_count:,} ({duplicate_count/n_rows*100:.2f}%)")
    print(f"Columns with Missing Data: {list(missing_dict.keys())}")
    for col, cnt in missing_dict.items():
        print(f"  - {col}: {cnt:,} missing ({cnt/n_rows*100:.2f}%)")
    print("=" * 60 + "\n")

    return df, summary


if __name__ == "__main__":
    df, summary = load_and_inspect_data()
    print("First 5 rows preview:")
    print(df.head())
