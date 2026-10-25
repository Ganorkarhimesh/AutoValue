"""Model training pipeline for AutoValue AI.

Supervised regression models:
    1. Experiment A: Simple Linear Regression (single numerical feature, e.g. max_power)
    2. Experiment B: Multiple Linear Regression (all numerical + dummy-encoded categorical)
    3. Experiment C: KNN Regression with k = 3 (scaled features with StandardScaler)
    4. Experiment D: KNN Regression with k = 5 (scaled features with StandardScaler)
- Identical 80:20 train-test split across all four experiments (random_state=42).
- Zero data leakage: Imputers, dummy alignment, and scalers fitted on X_train only.
- Calculates MAE, RMSE, and R² on identical held-out test split using scikit-learn.
- Persists all artifacts to models/ directory with joblib and JSON.
"""

from datetime import datetime
import io
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Any

# Ensure UTF-8 console output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    BEST_MODEL_PATH,
    ALL_MODELS_PATH,
    METRICS_PATH,
    REPORTS_DIR,
    DATA_FILE,
    RANDOM_STATE,
    TEST_SIZE,
    SIMPLE_LR_FEATURE,
    REPORT_METRICS_CSV,
)
from src.data_loader import load_raw_data, clean_car_dataset
from src.preprocessing import (
    prepare_train_test_data,
    compute_training_imputation_values,
    apply_imputation,
    encode_categorical_features,
    scale_features,
    extract_and_save_feature_metadata,
)
from src.evaluate_models import (
    calculate_metrics,
    create_comparison_dataframe,
    export_metrics_to_csv,
)

EXPERIMENT_NAMES = {
    "exp_a": "Experiment A: Simple Linear Regression",
    "exp_b": "Experiment B: Multiple Linear Regression",
    "exp_c": "Experiment C: KNN Regression (k=3)",
    "exp_d": "Experiment D: KNN Regression (k=5)",
}

def train_and_evaluate_all_experiments(
    data_path: Optional[Path] = None,
    save_artifacts: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Executes the complete, rigorous 4-experiment training pipeline.
    
    Returns:
        (metrics_dict, comparison_df)
    """
    print("=" * 65)
    print("AutoValue AI: Starting 4-Experiment Model Training Pipeline")
    print("Regression Suite: Linear Regression & KNN Regression")
    print("=" * 65)

    # 1. Load raw dataset
    df_raw = load_raw_data(data_path)
    print(f"[1/6] Raw dataset loaded successfully: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")

    # 2. Clean dataset & handle duplicates
    df_clean, clean_meta = clean_car_dataset(df_raw, drop_duplicates=True)
    print(
        f"[2/6] Cleaning completed: {clean_meta['duplicates_detected']} duplicate rows detected "
        f"and removed. Retained {clean_meta['rows_after_cleaning']} valid records."
    )

    # 3. Train-test split (80:20, fixed random_state=42)
    X_train_raw, X_test_raw, y_train, y_test, num_cols, cat_cols = prepare_train_test_data(
        df_clean, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[3/6] Split data: Train={len(X_train_raw)} rows (80%), Test={len(X_test_raw)} rows (20%).")

    # Save feature metadata for UI inputs
    extract_and_save_feature_metadata(df_clean, num_cols, cat_cols)

    # 4. Fit Imputers strictly on X_train (Zero Data Leakage)
    imputation_values = compute_training_imputation_values(X_train_raw, num_cols, cat_cols)
    X_train_imputed = apply_imputation(X_train_raw, imputation_values)
    X_test_imputed = apply_imputation(X_test_raw, imputation_values)
    print("[4/6] Missing-value imputation fitted on X_train (median for numeric, mode for categorical).")

    # 5. Categorical Encoding strictly via pd.get_dummies() on X_train
    X_train_encoded, dummy_columns = encode_categorical_features(X_train_imputed, cat_cols)
    X_test_encoded, _ = encode_categorical_features(
        X_test_imputed, cat_cols, expected_dummy_columns=dummy_columns
    )
    print(f"      Categorical encoding generated {len(dummy_columns)} total aligned feature columns.")

    # 6. Feature Scaling with StandardScaler strictly fitted on X_train (for KNN)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_encoded, X_test_encoded)
    print("[5/6] Feature scaling fitted with StandardScaler on X_train for KNN experiments.")

    # --- MODEL EXPERIMENTS ---
    print("\n[6/6] Executing the 4 Regression Model Experiments...")
    trained_models = {}
    metrics_summary = {}

    # -------------------------------------------------------------
    # Experiment A: Simple Linear Regression (Single Numerical Feature)
    # -------------------------------------------------------------
    simple_feat = SIMPLE_LR_FEATURE if SIMPLE_LR_FEATURE in X_train_imputed.columns else num_cols[0]
    X_train_simple = X_train_imputed[[simple_feat]].values
    X_test_simple = X_test_imputed[[simple_feat]].values

    model_exp_a = LinearRegression()
    model_exp_a.fit(X_train_simple, y_train)
    y_pred_exp_a = model_exp_a.predict(X_test_simple)
    metrics_a = calculate_metrics(y_test, y_pred_exp_a)

    exp_a_name = EXPERIMENT_NAMES["exp_a"]
    trained_models[exp_a_name] = model_exp_a
    metrics_summary[exp_a_name] = {
        "model_key": "exp_a",
        "model_type": "Simple Linear Regression",
        "features_used": f"Single Feature: {simple_feat}",
        "feature_count": 1,
        "test_metrics": metrics_a,
        "slope": float(model_exp_a.coef_[0]),
        "intercept": float(model_exp_a.intercept_),
        "formula": f"selling_price = {model_exp_a.coef_[0]:.2f} * {simple_feat} + ({model_exp_a.intercept_:.2f})",
        "description": (
            f"Simple Linear Regression maps a straight line between the single independent variable "
            f"'{simple_feat}' and dependent target 'selling_price' using Ordinary Least Squares (OLS)."
        ),
    }
    print(f"  -> {exp_a_name}: MAE = INR {metrics_a['MAE']:,.2f}, RMSE = INR {metrics_a['RMSE']:,.2f}, R2 = {metrics_a['R2']:.4f}")

    # -------------------------------------------------------------
    # Experiment B: Multiple Linear Regression (All Features)
    # -------------------------------------------------------------
    model_exp_b = LinearRegression()
    model_exp_b.fit(X_train_encoded.values, y_train)
    y_pred_exp_b = model_exp_b.predict(X_test_encoded.values)
    metrics_b = calculate_metrics(y_test, y_pred_exp_b)

    exp_b_name = EXPERIMENT_NAMES["exp_b"]
    trained_models[exp_b_name] = model_exp_b
    metrics_summary[exp_b_name] = {
        "model_key": "exp_b",
        "model_type": "Multiple Linear Regression",
        "features_used": f"All {len(dummy_columns)} Features (Numerical + Dummy Encoded)",
        "feature_count": len(dummy_columns),
        "test_metrics": metrics_b,
        "intercept": float(model_exp_b.intercept_),
        "description": (
            "Multiple Linear Regression fits an n-dimensional hyperplane estimating selling_price "
            "as a weighted linear combination of all vehicle specifications and categorical dummy indicators."
        ),
    }
    print(f"  -> {exp_b_name}: MAE = INR {metrics_b['MAE']:,.2f}, RMSE = INR {metrics_b['RMSE']:,.2f}, R2 = {metrics_b['R2']:.4f}")

    # -------------------------------------------------------------
    # Experiment C: KNN Regression with k = 3
    # -------------------------------------------------------------
    model_exp_c = KNeighborsRegressor(n_neighbors=3, weights="uniform", metric="euclidean")
    model_exp_c.fit(X_train_scaled, y_train)
    y_pred_exp_c = model_exp_c.predict(X_test_scaled)
    metrics_c = calculate_metrics(y_test, y_pred_exp_c)

    exp_c_name = EXPERIMENT_NAMES["exp_c"]
    trained_models[exp_c_name] = model_exp_c
    metrics_summary[exp_c_name] = {
        "model_key": "exp_c",
        "model_type": "KNN Regression (k=3)",
        "features_used": f"All {len(dummy_columns)} Features (StandardScaler Scaled)",
        "feature_count": len(dummy_columns),
        "k_neighbors": 3,
        "distance_metric": "Euclidean",
        "test_metrics": metrics_c,
        "description": (
            "k-Nearest Neighbours regression with k=3 locates the 3 most similar vehicles in normalized "
            "Euclidean space and averages their historical selling prices to predict valuation."
        ),
    }
    print(f"  -> {exp_c_name}: MAE = INR {metrics_c['MAE']:,.2f}, RMSE = INR {metrics_c['RMSE']:,.2f}, R2 = {metrics_c['R2']:.4f}")

    # -------------------------------------------------------------
    # Experiment D: KNN Regression with k = 5
    # -------------------------------------------------------------
    model_exp_d = KNeighborsRegressor(n_neighbors=5, weights="uniform", metric="euclidean")
    model_exp_d.fit(X_train_scaled, y_train)
    y_pred_exp_d = model_exp_d.predict(X_test_scaled)
    metrics_d = calculate_metrics(y_test, y_pred_exp_d)

    exp_d_name = EXPERIMENT_NAMES["exp_d"]
    trained_models[exp_d_name] = model_exp_d
    metrics_summary[exp_d_name] = {
        "model_key": "exp_d",
        "model_type": "KNN Regression (k=5)",
        "features_used": f"All {len(dummy_columns)} Features (StandardScaler Scaled)",
        "feature_count": len(dummy_columns),
        "k_neighbors": 5,
        "distance_metric": "Euclidean",
        "test_metrics": metrics_d,
        "description": (
            "k-Nearest Neighbours regression with k=5 averages the 5 nearest vehicles. Increasing k "
            "reduces sensitivity to individual pricing noise and outliers at the cost of slight smoothing."
        ),
    }
    print(f"  -> {exp_d_name}: MAE = INR {metrics_d['MAE']:,.2f}, RMSE = INR {metrics_d['RMSE']:,.2f}, R2 = {metrics_d['R2']:.4f}")

    # --- SELECT BEST MODEL BASED ON LOWEST TEST MAE ---
    best_exp_name = min(metrics_summary.keys(), key=lambda k: metrics_summary[k]["test_metrics"]["MAE"])
    for name in metrics_summary:
        metrics_summary[name]["is_best"] = (name == best_exp_name)

    print(f"\nWinning Model Selected: '{best_exp_name}' (Lowest Test MAE: INR {metrics_summary[best_exp_name]['test_metrics']['MAE']:,.2f})")

    # Sample actual vs predicted for scatter plot visualization
    sample_size = min(300, len(y_test))
    np.random.seed(RANDOM_STATE)
    sample_indices = np.random.choice(len(y_test), size=sample_size, replace=False)
    
    test_scatter_data = {
        "y_true": [float(y_test.iloc[i]) for i in sample_indices],
        "y_pred_exp_a": [float(y_pred_exp_a[i]) for i in sample_indices],
        "y_pred_exp_b": [float(y_pred_exp_b[i]) for i in sample_indices],
        "y_pred_exp_c": [float(y_pred_exp_c[i]) for i in sample_indices],
        "y_pred_exp_d": [float(y_pred_exp_d[i]) for i in sample_indices],
    }

    # Comparison DataFrame
    comparison_df = create_comparison_dataframe(metrics_summary)

    # 7. Persist Artifacts
    if save_artifacts:
        BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Save all trained models & preprocessing pipeline state
        pipeline_bundle = {
            "models": trained_models,
            "best_model_name": best_exp_name,
            "imputation_values": imputation_values,
            "dummy_columns": dummy_columns,
            "scaler": scaler,
            "simple_lr_feature": simple_feat,
            "active_num_cols": num_cols,
            "active_cat_cols": cat_cols,
        }
        joblib.dump(pipeline_bundle, ALL_MODELS_PATH)
        joblib.dump(trained_models[best_exp_name], BEST_MODEL_PATH)

        # Save JSON metrics
        exportable_metrics = {
            "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "best_model_name": best_exp_name,
            "train_samples": len(X_train_raw),
            "test_samples": len(X_test_raw),
            "total_cleaned_samples": len(df_clean),
            "simple_lr_feature": simple_feat,
            "experiments": metrics_summary,
            "scatter_sample": test_scatter_data,
        }
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(exportable_metrics, f, indent=2)

        # Export CSV report
        export_metrics_to_csv(comparison_df, REPORT_METRICS_CSV)
        print(f"Artifacts successfully saved to {ALL_MODELS_PATH.parent} and {REPORT_METRICS_CSV}")

    print("=" * 65)
    print("Training Pipeline Complete!")
    print("=" * 65)

    return metrics_summary, comparison_df

if __name__ == "__main__":
    train_and_evaluate_all_experiments()
