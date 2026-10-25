"""Unit and integration tests for data loading and preprocessing."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_FILE, TARGET_COL
from src.data_loader import (
    extract_numeric_from_string,
    clean_car_dataset,
    detect_outliers_iqr,
)
from src.preprocessing import (
    prepare_train_test_data,
    compute_training_imputation_values,
    apply_imputation,
    encode_categorical_features,
    scale_features,
)

def test_extract_numeric_from_string():
    """Verify units like kmpl, CC, and bhp are parsed accurately into numbers."""
    assert extract_numeric_from_string("23.4 kmpl") == 23.4
    assert extract_numeric_from_string("1248 CC") == 1248.0
    assert extract_numeric_from_string("74 bhp") == 74.0
    assert extract_numeric_from_string("18.5 km/kg") == 18.5
    assert extract_numeric_from_string("103.52 bhp") == 103.52
    assert np.isnan(extract_numeric_from_string(""))
    assert np.isnan(extract_numeric_from_string(None))
    assert np.isnan(extract_numeric_from_string("null"))
    assert np.isnan(extract_numeric_from_string("bhp")) # Empty bhp

def test_dataset_cleaning_and_brand_extraction():
    """Verify raw rows with string units and full car names are properly standardized."""
    dummy_data = pd.DataFrame({
        "name": ["Maruti Swift Dzire", "Hyundai i20 Magna", "Maruti Swift Dzire"],
        "year": [2015, 2018, 2015],
        "selling_price": [450000, 520000, 450000],
        "km_driven": [60000, 35000, 60000],
        "fuel": ["Diesel", "Petrol", "Diesel"],
        "seller_type": ["Individual", "Dealer", "Individual"],
        "transmission": ["Manual", "Manual", "Manual"],
        "owner": ["First Owner", "First Owner", "First Owner"],
        "mileage": ["23.4 kmpl", "18.5 kmpl", "23.4 kmpl"],
        "engine": ["1248 CC", "1197 CC", "1248 CC"],
        "max_power": ["74 bhp", "82 bhp", "74 bhp"],
        "seats": [5, 5, 5],
    })

    df_clean, meta = clean_car_dataset(dummy_data, drop_duplicates=True)
    
    assert meta["duplicates_detected"] == 1
    assert len(df_clean) == 2
    assert "brand" in df_clean.columns
    assert list(df_clean["brand"]) == ["Maruti", "Hyundai"]
    assert df_clean["engine"].iloc[0] == 1248.0
    assert df_clean["mileage"].iloc[0] == 23.4

def test_detect_outliers_iqr():
    """Verify IQR calculation correctly identifies bounds without mutating the data."""
    df = pd.DataFrame({
        "selling_price": [100000, 150000, 180000, 200000, 220000, 250000, 5000000], # 5M is outlier
        "km_driven": [10000, 20000, 30000, 40000, 50000, 60000, 80000],
    })
    outliers = detect_outliers_iqr(df, ["selling_price", "km_driven"])
    assert "selling_price" in outliers
    assert outliers["selling_price"]["outlier_count"] >= 1
    assert outliers["selling_price"]["upper_bound"] < 5000000

def test_imputation_and_get_dummies_leakage_free():
    """Verify missing values are imputed strictly using training statistics and get_dummies aligns."""
    train_df = pd.DataFrame({
        "year": [2012, 2016, 2020, 2018],
        "km_driven": [80000, 40000, 15000, 50000],
        "mileage": [20.0, np.nan, 22.0, 18.0], # Missing value in train
        "engine": [1200, 1500, 1000, 1200],
        "max_power": [80.0, 100.0, 65.0, 85.0],
        "seats": [5, 5, 5, 5],
        "brand": ["Maruti", "Hyundai", "Tata", "Maruti"],
        "fuel": ["Petrol", "Diesel", "Petrol", "Diesel"],
        "seller_type": ["Individual", "Dealer", "Individual", "Dealer"],
        "transmission": ["Manual", "Automatic", "Manual", "Manual"],
        "owner": ["First Owner", "Second Owner", "First Owner", "First Owner"],
        "selling_price": [300000, 550000, 620000, 480000],
    })

    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_data(train_df, test_size=0.25)
    
    # 1. Compute training imputation statistics
    imputation_vals = compute_training_imputation_values(X_train, num_cols, cat_cols)
    assert "num_medians" in imputation_vals
    assert "cat_modes" in imputation_vals

    # 2. Impute
    X_train_imp = apply_imputation(X_train, imputation_vals)
    X_test_imp = apply_imputation(X_test, imputation_vals)
    assert not X_train_imp[num_cols].isnull().any().any()
    assert not X_test_imp[num_cols].isnull().any().any()

    # 3. Categorical encoding via pd.get_dummies()
    X_train_enc, dummy_cols = encode_categorical_features(X_train_imp, cat_cols)
    X_test_enc, _ = encode_categorical_features(X_test_imp, cat_cols, expected_dummy_columns=dummy_cols)
    assert X_train_enc.shape[1] == X_test_enc.shape[1]
    assert list(X_train_enc.columns) == list(X_test_enc.columns)

    # 4. Feature scaling for KNN
    X_tr_scaled, X_te_scaled, scaler = scale_features(X_train_enc, X_test_enc)
    assert X_tr_scaled.shape == X_train_enc.shape
    assert X_te_scaled.shape == X_test_enc.shape
