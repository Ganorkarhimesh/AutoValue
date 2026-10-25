"""Inference and valuation prediction module for AutoValue AI.

Provides vehicle valuation inference functions:
- Models: Simple Linear Regression, Multiple Linear Regression, KNN Regression.
- Validates user input before prediction.
- Applies the EXACT preprocessing transformations fitted during training:
    - Training median/mode imputation
    - Aligned dummy encoding via pd.get_dummies()
    - StandardScaler transformation for KNN models
- Displays predicted price in Indian Rupees (₹).
- Provides the model's test-set MAE and R² separately from the individual predicted price.
- Never accepts selling_price as an input.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import joblib
import numpy as np
import pandas as pd

from config import (
    ALL_MODELS_PATH,
    BEST_MODEL_PATH,
    METRICS_PATH,
    VALIDATION_RANGES,
    DEFAULT_CATEGORIES,
    SIMPLE_LR_FEATURE,
)
from src.preprocessing import apply_imputation, encode_categorical_features
from src.evaluate_models import format_currency_inr

def load_prediction_bundle() -> Dict[str, Any]:
    """Loads the trained models, preprocessing state, and evaluation metrics."""
    if not ALL_MODELS_PATH.exists():
        raise FileNotFoundError(
            "Trained models not found! Please run the training script first: "
            "python src/train_models.py"
        )
    bundle = joblib.load(ALL_MODELS_PATH)
    
    metrics = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}
    bundle["metrics_json"] = metrics
    return bundle

def validate_user_inputs(inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validates user-submitted vehicle specifications against physical domain constraints."""
    errors = []

    # 1. Year validation
    year = inputs.get("year")
    min_year, max_year = VALIDATION_RANGES["year"]
    if year is None:
        errors.append("Manufacturing year is required.")
    else:
        try:
            year_val = int(year)
            if year_val < min_year or year_val > max_year:
                errors.append(f"Manufacturing year must be between {min_year} and {max_year}.")
        except (ValueError, TypeError):
            errors.append("Manufacturing year must be an integer.")

    # 2. Kilometres driven validation
    km = inputs.get("km_driven")
    min_km, max_km = VALIDATION_RANGES["km_driven"]
    if km is None:
        errors.append("Kilometres driven is required.")
    else:
        try:
            km_val = float(km)
            if km_val < min_km or km_val > max_km:
                errors.append(f"Kilometres driven must be between {min_km:,} and {max_km:,} km.")
        except (ValueError, TypeError):
            errors.append("Kilometres driven must be a valid number.")

    # 3. Mileage validation
    mileage = inputs.get("mileage")
    min_m, max_m = VALIDATION_RANGES["mileage"]
    if mileage is not None:
        try:
            m_val = float(mileage)
            if m_val < min_m or m_val > max_m:
                errors.append(f"Mileage ({m_val} kmpl) must be between {min_m} and {max_m} kmpl.")
        except (ValueError, TypeError):
            errors.append("Mileage must be a valid number.")

    # 4. Engine CC validation
    engine = inputs.get("engine")
    min_eng, max_eng = VALIDATION_RANGES["engine"]
    if engine is not None:
        try:
            eng_val = float(engine)
            if eng_val < min_eng or eng_val > max_eng:
                errors.append(f"Engine capacity ({eng_val} CC) must be between {min_eng} and {max_eng} CC.")
        except (ValueError, TypeError):
            errors.append("Engine capacity must be a valid number.")

    # 5. Maximum Power validation
    power = inputs.get("max_power")
    min_p, max_p = VALIDATION_RANGES["max_power"]
    if power is not None:
        try:
            p_val = float(power)
            if p_val < min_p or p_val > max_p:
                errors.append(f"Maximum power ({p_val} bhp) must be between {min_p} and {max_p} bhp.")
        except (ValueError, TypeError):
            errors.append("Maximum power must be a valid number.")

    # 6. Seats validation
    seats = inputs.get("seats")
    min_s, max_s = VALIDATION_RANGES["seats"]
    if seats is not None:
        try:
            s_val = int(seats)
            if s_val < min_s or s_val > max_s:
                errors.append(f"Number of seats ({s_val}) must be between {min_s} and {max_s}.")
        except (ValueError, TypeError):
            errors.append("Number of seats must be an integer.")

    # 7. Check for actual selling_price in inputs (must NEVER be an input)
    if "selling_price" in inputs:
        errors.append("Target feature 'selling_price' must not be provided as a prediction input.")

    return len(errors) == 0, errors

# Backwards compatible alias
validate_car_inputs = validate_user_inputs

def predict_car_price(
    inputs: Dict[str, Any],
    selected_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates an estimated vehicle price using the trained ML model.
    
    Steps:
    1. Validates input values.
    2. Loads pipeline bundle.
    3. Selects requested model (defaults to winning model from training).
    4. Applies identical training-set preprocessing (imputation, get_dummies, scaling).
    5. Predicts selling price and formats in INR (₹).
    6. Extracts test-set MAE and R² for transparency.
    """
    # 1. Validation
    is_valid, errors = validate_user_inputs(inputs)
    if not is_valid:
        return {
            "status": "error",
            "message": "Validation failed for provided vehicle specifications.",
            "errors": errors,
        }

    # 2. Load Pipeline
    try:
        bundle = load_prediction_bundle()
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "errors": [str(e)],
        }

    models_dict = bundle["models"]
    best_model_name = bundle["best_model_name"]
    target_model_name = selected_model_name if (selected_model_name and selected_model_name in models_dict) else best_model_name

    model = models_dict[target_model_name]
    metrics_info = bundle.get("metrics_json", {}).get("experiments", {}).get(target_model_name, {})
    test_metrics = metrics_info.get("test_metrics", {})

    # 3. Prepare Single Record DataFrame
    active_num_cols = bundle["active_num_cols"]
    active_cat_cols = bundle["active_cat_cols"]
    all_active_cols = active_num_cols + active_cat_cols

    # Extract relevant fields only
    clean_input_record = {}
    for col in all_active_cols:
        clean_input_record[col] = inputs.get(col, np.nan)

    df_single = pd.DataFrame([clean_input_record])

    # 4. Apply Imputation using precomputed training medians and modes
    df_imputed = apply_imputation(df_single, bundle["imputation_values"])

    # 5. Model-Specific Feature Preparation and Prediction
    try:
        if "Simple Linear Regression" in target_model_name:
            # Experiment A: Single Feature
            simple_col = bundle.get("simple_lr_feature", SIMPLE_LR_FEATURE)
            X_input = df_imputed[[simple_col]].values
            raw_prediction = float(model.predict(X_input)[0])

        elif "Multiple Linear Regression" in target_model_name:
            # Experiment B: Unscaled dummy-encoded features
            df_encoded, _ = encode_categorical_features(
                df_imputed, active_cat_cols, expected_dummy_columns=bundle["dummy_columns"]
            )
            raw_prediction = float(model.predict(df_encoded.values)[0])

        else:
            # Experiment C or D: KNN Regression with StandardScaler
            df_encoded, _ = encode_categorical_features(
                df_imputed, active_cat_cols, expected_dummy_columns=bundle["dummy_columns"]
            )
            X_scaled = bundle["scaler"].transform(df_encoded)
            raw_prediction = float(model.predict(X_scaled)[0])

        # Clamp physically impossible negative predictions (can occur with OLS linear extrapolation)
        predicted_price = max(15_000.0, raw_prediction)

        # ID3 Decision Tree Deal Recommendation
        id3_verdict = None
        try:
            from src.id3_decision_tree import predict_deal_recommendation
            id3_verdict = predict_deal_recommendation(
                selling_price=predicted_price,
                year=int(inputs.get("year", 2017)),
                km_driven=float(inputs.get("km_driven", 45000)),
                transmission=str(inputs.get("transmission", "Manual")),
                fuel=str(inputs.get("fuel", "Petrol")),
            )
        except Exception:
            pass

        # K-Means Market Cluster Assignment
        kmeans_persona = None
        try:
            from src.kmeans_clustering import assign_car_to_cluster
            kmeans_persona = assign_car_to_cluster(
                selling_price=predicted_price,
                km_driven=float(inputs.get("km_driven", 45000)),
            )
        except Exception:
            pass

        return {
            "status": "success",
            "predicted_price": round(predicted_price, 2),
            "formatted_price": format_currency_inr(predicted_price),
            "formatted_raw_inr": f"₹ {int(round(predicted_price)):,}",
            "model_name": target_model_name,
            "model_type": metrics_info.get("model_type", target_model_name),
            "is_best_model": (target_model_name == best_model_name),
            "test_mae": test_metrics.get("MAE", None),
            "test_rmse": test_metrics.get("RMSE", None),
            "test_r2": test_metrics.get("R2", None),
            "id3_verdict": id3_verdict,
            "kmeans_persona": kmeans_persona,
            "disclaimer": (
                "Educational Estimate Only: This valuation is generated by machine learning regression "
                "models trained on historical Indian used-car sales data. It is an algorithmic estimate "
                "and does not constitute a legally binding appraisal or guaranteed dealership quote."
            ),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Inference pipeline execution error: {str(e)}",
            "errors": [str(e)],
        }
