"""Data loader and inspection module for AutoValue AI.

Provides data engineering functions for:
- Dataset inspection: head(), tail(), shape, describe(), info(), dtypes
- Missing-value detection
- Duplicate detection and removal with justification
- Numerical outlier detection using the IQR method
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from config import (
    DATA_FILE,
    FALLBACK_DATA_FILE,
    TARGET_COL,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)

def extract_numeric_from_string(value: Any) -> Optional[float]:
    """Extracts the first floating-point number from a string containing units.
    
    Examples:
        '23.4 kmpl' -> 23.4
        '1248 CC' -> 1248.0
        '74 bhp' -> 74.0
        '17.3 km/kg' -> 17.3
        ' bhp' -> np.nan (handles missing digits safely)
    """
    if pd.isna(value) or value is None:
        return np.nan
    val_str = str(value).strip().lower()
    if val_str in ("", "null", "nan", "none", "bhp"):
        return np.nan
    
    # Use regular expression to find the first decimal or integer number
    match = re.search(r"(\d+(?:\.\d+)?)", val_str)
    if match:
        try:
            num = float(match.group(1))
            return num if num > 0 else np.nan
        except (ValueError, IndexError):
            return np.nan
    return np.nan

def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Loads raw dataset from CSV with rigorous path checking.
    
    Checks primary DATA_FILE ('Car details v3.csv'), then FALLBACK_DATA_FILE ('car_data.csv').
    Raises FileNotFoundError with clear instructions if neither is found.
    """
    if filepath is not None:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at specified path: {path}\n"
                f"Please ensure 'Car details v3.csv' is placed inside the 'data/' folder."
            )
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"The dataset at {path} is empty.")
        return df

    # Primary check
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        if not df.empty:
            return df

    # Fallback check
    if FALLBACK_DATA_FILE.exists():
        df = pd.read_csv(FALLBACK_DATA_FILE)
        if not df.empty:
            return df

    raise FileNotFoundError(
        "Used car dataset not found!\n"
        "Expected file location: data/Car details v3.csv\n\n"
        "Instructions to add the dataset:\n"
        "1. Download 'Car details v3.csv' from Kaggle (Vehicle Dataset from CarDekho):\n"
        "   https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho\n"
        "2. Save the CSV file as 'data/Car details v3.csv' in your project root directory.\n"
        "3. Refresh or run the training command: python src/train_models.py"
    )

def clean_car_dataset(
    df_raw: pd.DataFrame, drop_duplicates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cleans and standardizes the car dataset following standard ML steps.
    
    Steps:
    1. Duplicate detection: Counts and optionally removes duplicates.
       (Justification: Duplicate vehicle listings often result from repeat scrapes
       or re-listings of identical used car adverts).
    2. Brand extraction: Derives car brand from 'name' (e.g., 'Maruti Swift' -> 'Maruti').
    3. Unit parsing: Strips string suffixes from 'mileage', 'engine', and 'max_power'.
    4. Numeric conversion: Ensures year, km_driven, seats, and selling_price are numeric.
    5. Target validation: Filters rows with missing or non-positive selling_price.
    
    Returns:
        (cleaned_df, cleaning_metadata)
    """
    df = df_raw.copy()
    initial_shape = df.shape
    duplicate_count = int(df.duplicated().sum())

    # 1. Handle duplicate records
    if drop_duplicates and duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # 2. Extract brand from car name
    if "brand" not in df.columns:
        if "name" in df.columns:
            df["brand"] = df["name"].astype(str).apply(
                lambda x: x.strip().split()[0].title() if str(x).strip() else "Unknown"
            )
        else:
            df["brand"] = "Unknown"

    # 3. Extract numerical values from unit strings
    for col in ["mileage", "engine", "max_power"]:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric_from_string)

    # 4. Ensure numeric dtypes
    for col in ["year", "km_driven", "seats", TARGET_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Remove unusable target values or physically impossible rows
    if TARGET_COL in df.columns:
        valid_mask = (
            df[TARGET_COL].notna() &
            (df[TARGET_COL] > 0) &
            (df["year"].notna() if "year" in df.columns else True) &
            ((df["year"] >= 1990) if "year" in df.columns else True) &
            (df["km_driven"].notna() if "km_driven" in df.columns else True) &
            ((df["km_driven"] >= 0) if "km_driven" in df.columns else True)
        )
        df_clean = df[valid_mask].reset_index(drop=True)
    else:
        df_clean = df.reset_index(drop=True)

    retained_shape = df_clean.shape

    metadata = {
        "initial_rows": initial_shape[0],
        "initial_columns": initial_shape[1],
        "duplicates_detected": duplicate_count,
        "duplicates_removed": drop_duplicates,
        "rows_after_cleaning": retained_shape[0],
        "columns_after_cleaning": retained_shape[1],
        "target_column": TARGET_COL,
    }

    return df_clean, metadata

def detect_outliers_iqr(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """Calculates outlier bounds using the Interquartile Range (IQR) method:
    
    Formula:
        IQR = Q3 - Q1
        Lower Bound = Q1 - 1.5 * IQR
        Upper Bound = Q3 + 1.5 * IQR
        
    Note on Outlier Removal Policy:
    We detect and report outliers, but do NOT blindly remove them. In used car markets,
    high prices or engine capacities frequently reflect authentic luxury models
    (such as BMW, Mercedes-Benz, Audi) rather than measurement or clerical errors.
    Purging them would bias the model toward budget hatchbacks.
    """
    target_cols = columns or [c for c in NUMERICAL_FEATURES + [TARGET_COL] if c in df.columns]
    outlier_summary = {}

    for col in target_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = float(q3 - q1)
        lower_bound = float(q1 - 1.5 * iqr)
        upper_bound = float(q3 + 1.5 * iqr)

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = float((outlier_count / len(series)) * 100) if len(series) > 0 else 0.0

        outlier_summary[col] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_count,
            "outlier_pct": round(outlier_pct, 2),
            "min_value": float(series.min()),
            "max_value": float(series.max()),
        }

    return outlier_summary

def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates a comprehensive diagnostic report of the dataset."""
    missing_counts = df.isnull().sum().to_dict()
    missing_pcts = ((df.isnull().sum() / len(df)) * 100).round(2).to_dict() if len(df) > 0 else {}

    summary = {
        "shape": df.shape,
        "total_records": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": missing_counts,
        "missing_percentages": missing_pcts,
        "duplicates": int(df.duplicated().sum()),
        "numerical_stats": df.describe().to_dict() if not df.empty else {},
        "target_stats": {
            "mean": float(df[TARGET_COL].mean()) if TARGET_COL in df and not df[TARGET_COL].dropna().empty else None,
            "median": float(df[TARGET_COL].median()) if TARGET_COL in df and not df[TARGET_COL].dropna().empty else None,
            "min": float(df[TARGET_COL].min()) if TARGET_COL in df and not df[TARGET_COL].dropna().empty else None,
            "max": float(df[TARGET_COL].max()) if TARGET_COL in df and not df[TARGET_COL].dropna().empty else None,
            "std": float(df[TARGET_COL].std()) if TARGET_COL in df and not df[TARGET_COL].dropna().empty else None,
        }
    }
    return summary
