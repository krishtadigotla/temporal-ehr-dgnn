import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

RAW_PATH = "data/full_cohort_data.csv"
PROCESSED_DIR = "data/processed"

def preprocess():
    # Load raw dataset
    df = pd.read_csv(RAW_PATH)
    df.columns = [c.lower() for c in df.columns]

    # --- Select target variable ---
    target = "hosp_exp_flg"   # hospital mortality (binary, present for all samples)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    # Drop columns that are >50% missing
    missing_ratio = df.isna().mean()
    cols_to_keep = missing_ratio[missing_ratio < 0.50].index
    df = df[cols_to_keep]

    # Split features and label
    y = df[target].astype(int)
    X = df.drop(columns=[target])

    # Identify numeric & categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Impute missing values
    X[numeric_cols] = SimpleImputer(strategy="median").fit_transform(X[numeric_cols])
    if len(categorical_cols) > 0:
        X[categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[categorical_cols])

    # One-hot encode categorical features
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Scale numeric features
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Train / validation / test split (70 / 15 / 15)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    # Create processed directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save processed splits
    X_train.assign(target=y_train).to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    X_val.assign(target=y_val).to_csv(f"{PROCESSED_DIR}/val.csv", index=False)
    X_test.assign(target=y_test).to_csv(f"{PROCESSED_DIR}/test.csv", index=False)

    print("\nPreprocessing complete!")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

if __name__ == "__main__":
    preprocess()
