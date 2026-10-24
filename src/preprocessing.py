"""
Data Preprocessing and Pipeline Module for Hotel Booking Cancellation Analysis.
Course: Exploratory Data Analysis and Visualization (U21ADP05)
Institution: KPR Institute of Engineering and Technology
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

TOP_COUNTRIES = ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "BRA", "NLD", "USA", "CHE", "CN", "AUT", "SWE"]

NUMERIC_FEATURES = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "total_stay",
    "adults",
    "children",
    "babies",
    "total_guests",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "room_type_changed",
    "has_agent",
    "has_company",
]

CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country_grouped",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
]


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> dict:
    """
    Computes IQR boundaries and detects outlier counts for numerical columns.
    Academic justification: Boxplot IQR rule (1.5 * IQR) identifies statistical extremes
    without assuming normality.
    """
    outlier_report = {}
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            clean_series = df[col].dropna()
            q1 = float(clean_series.quantile(0.25))
            q3 = float(clean_series.quantile(0.75))
            iqr = q3 - q1
            lower_bound = float(q1 - 1.5 * iqr)
            upper_bound = float(q3 + 1.5 * iqr)
            outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
            outlier_report[col] = {
                "q1": round(q1, 2),
                "q3": round(q3, 2),
                "iqr": round(iqr, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "outlier_count": int(len(outliers)),
                "outlier_pct": round(float(len(outliers) / len(clean_series) * 100), 2),
                "min": round(float(clean_series.min()), 2),
                "max": round(float(clean_series.max()), 2),
            }
    return outlier_report


def clean_raw_data(df: pd.DataFrame, drop_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Cleans raw dataset, filters invalid observations, and engineers domain features.
    Guarantees no data leakage by not incorporating target variables in transformations.
    """
    initial_shape = df.shape
    cleaning_log = {"initial_rows": initial_shape[0], "initial_cols": initial_shape[1]}

    df_clean = df.copy()

    # Filter erroneous/impossible records
    # 1. Bookings with 0 guests (adults + children + babies == 0)
    zero_guests_mask = (df_clean["adults"] == 0) & (df_clean["children"].fillna(0) == 0) & (df_clean["babies"] == 0)
    cleaning_log["zero_guests_removed"] = int(zero_guests_mask.sum())
    df_clean = df_clean[~zero_guests_mask]

    # 2. Filter invalid negative ADR or extreme single erroneous outlier (e.g. ADR = 5400)
    invalid_adr_mask = (df_clean["adr"] < 0) | (df_clean["adr"] > 5000)
    cleaning_log["invalid_adr_removed"] = int(invalid_adr_mask.sum())
    df_clean = df_clean[~invalid_adr_mask]

    # 3. Duplicate handling
    dup_count = int(df_clean.duplicated().sum())
    cleaning_log["duplicate_rows_detected"] = dup_count
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
        cleaning_log["duplicates_action"] = "Dropped exact identical rows to prevent train-test data leakage."
    else:
        cleaning_log["duplicates_action"] = "Retained duplicate rows as potential independent bookings."

    cleaning_log["cleaned_rows"] = int(df_clean.shape[0])

    # Feature Engineering (domain-driven, zero target leakage)
    df_clean["children"] = df_clean["children"].fillna(0)
    df_clean["country"] = df_clean["country"].fillna("Unknown")
    df_clean["agent"] = df_clean["agent"].fillna(0)
    df_clean["company"] = df_clean["company"].fillna(0)

    df_clean["total_stay"] = df_clean["stays_in_weekend_nights"] + df_clean["stays_in_week_nights"]
    df_clean["total_guests"] = df_clean["adults"] + df_clean["children"] + df_clean["babies"]
    df_clean["room_type_changed"] = (df_clean["reserved_room_type"] != df_clean["assigned_room_type"]).astype(int)
    df_clean["has_agent"] = (df_clean["agent"] > 0).astype(int)
    df_clean["has_company"] = (df_clean["company"] > 0).astype(int)

    # Group country to top 15 and 'Other'
    df_clean["country_grouped"] = df_clean["country"].apply(
        lambda c: c if c in TOP_COUNTRIES else "Other"
    )

    return df_clean, cleaning_log


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs scikit-learn ColumnTransformer:
    - Numeric: Median Imputer + StandardScaler
    - Categorical: Mode Imputer + OneHotEncoder
    """
    numeric_pipe = SimpleImputer(strategy="median")
    categorical_pipe = SimpleImputer(strategy="most_frequent")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def split_and_preprocess(
    df: pd.DataFrame,
    target_col: str = "is_canceled",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    drop_duplicates: bool = True,
) -> dict:
    """
    Complete preprocessing workflow:
    1. Clean raw data and engineer features.
    2. Exclude reservation_status and reservation_status_date (outcome leakage).
    3. Stratified split into 70% Train, 15% Validation, 15% Test.
    4. Fit preprocessing ONLY on training data.
    5. Transform validation and test data.
    6. Save preprocessor and preprocessing summary to outputs/.
    """
    df_cleaned, cleaning_log = clean_raw_data(df, drop_duplicates=drop_duplicates)

    # Target variable and feature matrix
    y = df_cleaned[target_col].values

    # CRITICAL: Drop outcome-revealing leakage variables and raw identifier columns
    leakage_cols = ["reservation_status", "reservation_status_date", target_col, "country", "agent", "company"]
    feature_cols = [c for c in df_cleaned.columns if c not in leakage_cols]

    X_df = df_cleaned[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    # Outlier report on cleaned dataset
    outlier_report = detect_outliers_iqr(df_cleaned, NUMERIC_FEATURES)

    # Stratified Split: First split off test set (15%), then validation set (15% of total = ~17.65% of 85%)
    # 70% Train, 15% Val, 15% Test
    val_ratio_of_rem = val_size / (1.0 - test_size)  # 0.15 / 0.85 = 0.17647

    X_train_df, X_temp_df, y_train, y_temp = train_test_split(
        X_df, y, test_size=(test_size + val_size), stratify=y, random_state=random_state
    )

    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_temp_df, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    print(f"[*] Split sizes: Train={len(X_train_df):,} ({len(X_train_df)/len(X_df)*100:.1f}%), "
          f"Val={len(X_val_df):,} ({len(X_val_df)/len(X_df)*100:.1f}%), "
          f"Test={len(X_test_df):,} ({len(X_test_df)/len(X_df)*100:.1f}%)")

    # Fit preprocessor strictly on X_train_df to prevent data leakage
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train_df)

    X_train = preprocessor.transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    # Retrieve one-hot feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_feature_names

    print(f"[+] Total processed features: {len(all_feature_names)}")

    # Save artifacts
    preprocessor_path = os.path.join(OUTPUTS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)

    feature_names_path = os.path.join(OUTPUTS_DIR, "feature_names.json")
    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(all_feature_names, f, indent=4)

    summary = {
        "cleaning_log": cleaning_log,
        "outlier_report": outlier_report,
        "split_summary": {
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
            "train_cancellation_rate": round(float(y_train.mean() * 100), 2),
            "val_cancellation_rate": round(float(y_val.mean() * 100), 2),
            "test_cancellation_rate": round(float(y_test.mean() * 100), 2),
        },
        "num_features_engineered": len(all_feature_names),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "total_transformed_features": len(all_feature_names),
        "preprocessor_saved_at": preprocessor_path,
        "feature_names_saved_at": feature_names_path,
    }

    summary_json_path = os.path.join(OUTPUTS_DIR, "preprocessing_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    summary_txt_path = os.path.join(OUTPUTS_DIR, "preprocessing_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("HOTEL BOOKING PREPROCESSING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Initial Rows: {cleaning_log['initial_rows']:,}\n")
        f.write(f"Zero-guest Rows Removed: {cleaning_log['zero_guests_removed']:,}\n")
        f.write(f"Invalid ADR Records Removed: {cleaning_log['invalid_adr_removed']:,}\n")
        f.write(f"Duplicate Rows Handled: {cleaning_log['duplicate_rows_detected']:,}\n")
        f.write(f"Action: {cleaning_log['duplicates_action']}\n")
        f.write(f"Cleaned Rows: {cleaning_log['cleaned_rows']:,}\n\n")
        f.write("Train / Val / Test Split:\n")
        f.write(f"  Train: {len(X_train):,} ({summary['split_summary']['train_cancellation_rate']}% canceled)\n")
        f.write(f"  Val:   {len(X_val):,} ({summary['split_summary']['val_cancellation_rate']}% canceled)\n")
        f.write(f"  Test:  {len(X_test):,} ({summary['split_summary']['test_cancellation_rate']}% canceled)\n\n")
        f.write(f"Total Processed Features: {len(all_feature_names)}\n")

    print(f"[+] Preprocessing artifacts saved to '{OUTPUTS_DIR}'.")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_df": X_train_df,
        "X_val_df": X_val_df,
        "X_test_df": X_test_df,
        "df_cleaned": df_cleaned,
        "feature_names": all_feature_names,
        "preprocessor": preprocessor,
        "summary": summary,
    }


if __name__ == "__main__":
    try:
        from src.data_loading import load_and_inspect_data
    except ModuleNotFoundError:
        from data_loading import load_and_inspect_data
    df, _ = load_and_inspect_data()
    data_dict = split_and_preprocess(df)
    print("Preprocessing completed successfully!")
