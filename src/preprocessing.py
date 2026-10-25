"""Preprocessing pipeline for AutoValue AI.

Provides data preparation functions:
- Missing-value handling: Training-set median for numeric, mode for categorical.
- Categorical encoding: pandas pd.get_dummies().
- Feature scaling: scikit-learn StandardScaler (fitted on train split only).
- Train/Test Split: 80:20 ratio with fixed random seed (random_state=42).
- Zero data leakage: Imputers, dummy schemas, and scalers learn statistics strictly
  from X_train and transform X_test and live inference inputs consistently.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_INFO_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COL,
)

def prepare_train_test_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """Separates target and features, and splits into 80% train and 20% test sets.
    
    Returns:
        (X_train, X_test, y_train, y_test, active_num_cols, active_cat_cols)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    active_num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    active_cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    feature_cols = active_num_cols + active_cat_cols
    if not feature_cols:
        raise ValueError("No matching numerical or categorical features found in dataset.")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, active_num_cols, active_cat_cols

def compute_training_imputation_values(
    X_train: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Computes median for numerical columns and mode for categorical columns
    using ONLY the training partition to prevent data leakage.
    """
    num_medians = {}
    for col in num_cols:
        s = X_train[col].dropna()
        num_medians[col] = float(s.median()) if not s.empty else 0.0

    cat_modes = {}
    for col in cat_cols:
        s = X_train[col].dropna()
        cat_modes[col] = str(s.mode()[0]) if not s.empty else "Unknown"

    return {
        "num_medians": num_medians,
        "cat_modes": cat_modes,
    }

def apply_imputation(
    df_features: pd.DataFrame,
    imputation_values: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Fills missing values using the precomputed training statistics."""
    df_out = df_features.copy()

    # Fill numerical missing values with training median
    for col, med_val in imputation_values.get("num_medians", {}).items():
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(med_val)

    # Fill categorical missing values with training mode
    for col, mode_val in imputation_values.get("cat_modes", {}).items():
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(mode_val)

    return df_out

def encode_categorical_features(
    df_features: pd.DataFrame,
    cat_cols: List[str],
    expected_dummy_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Encodes categorical columns using pandas pd.get_dummies().
    
    If expected_dummy_columns is None (during training):
        Encodes categories and returns the encoded DataFrame and the list of generated columns.
    If expected_dummy_columns is provided (during test evaluation or live prediction):
        Encodes categories and reindexes to exactly match the training column schema,
        filling any missing category columns with 0.0 (prevents shape mismatch or unseen category crashes).
    """
    # Ensure categorical columns exist and are string type
    cols_to_encode = [c for c in cat_cols if c in df_features.columns]
    df_copy = df_features.copy()
    for col in cols_to_encode:
        df_copy[col] = df_copy[col].astype(str)

    # pandas one-hot encoding
    df_encoded = pd.get_dummies(df_copy, columns=cols_to_encode, drop_first=False, dtype=float)

    if expected_dummy_columns is None:
        # Training split: save column list
        dummy_columns = list(df_encoded.columns)
        return df_encoded, dummy_columns
    else:
        # Test split or inference: align columns exactly
        aligned_encoded = df_encoded.reindex(columns=expected_dummy_columns, fill_value=0.0)
        return aligned_encoded, expected_dummy_columns

def scale_features(
    X_encoded_train: pd.DataFrame,
    X_encoded_test_or_val: Optional[pd.DataFrame] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """Applies scikit-learn StandardScaler to encoded features.
    
    If scaler is None:
        Fits scaler on X_encoded_train and transforms it.
    If scaler is provided:
        Transforms the given DataFrame using the pre-fitted scaler.
    """
    if scaler is None:
        new_scaler = StandardScaler()
        X_train_scaled = new_scaler.fit_transform(X_encoded_train)
        X_other_scaled = None
        if X_encoded_test_or_val is not None:
            X_other_scaled = new_scaler.transform(X_encoded_test_or_val)
        return X_train_scaled, X_other_scaled, new_scaler
    else:
        transformed = scaler.transform(X_encoded_train)
        return transformed, None, scaler

def extract_and_save_feature_metadata(
    df: pd.DataFrame,
    active_num_cols: List[str],
    active_cat_cols: List[str],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Extracts available categories, ranges, and medians for UI validation and dynamic rendering."""
    path = output_path or FEATURE_INFO_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    category_options = {}
    for col in active_cat_cols:
        unique_vals = sorted([str(x) for x in df[col].dropna().unique().tolist() if str(x).strip() != ""])
        category_options[col] = unique_vals

    num_stats = {}
    for col in active_num_cols:
        s = df[col].dropna()
        num_stats[col] = {
            "min": float(s.min()) if not s.empty else 0.0,
            "max": float(s.max()) if not s.empty else 100.0,
            "median": float(s.median()) if not s.empty else 0.0,
            "mean": float(s.mean()) if not s.empty else 0.0,
        }

    metadata = {
        "numerical_features": active_num_cols,
        "categorical_features": active_cat_cols,
        "category_options": category_options,
        "numerical_stats": num_stats,
        "total_samples": len(df),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata

def load_feature_metadata(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Loads stored feature metadata if available."""
    filepath = path or FEATURE_INFO_PATH
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
