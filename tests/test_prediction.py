"""Unit and integration tests for validation and inference."""

import sys
from pathlib import Path
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import (
    validate_user_inputs,
    validate_car_inputs,
    predict_car_price,
)
from src.evaluate_models import format_currency_inr

def test_validate_car_inputs_valid():
    """Verify standard valid car specification passes validation."""
    valid_input = {
        "year": 2018,
        "km_driven": 45000,
        "mileage": 19.5,
        "engine": 1197,
        "max_power": 82.0,
        "seats": 5,
        "brand": "Maruti",
        "fuel": "Petrol",
        "transmission": "Manual",
        "seller_type": "Individual",
        "owner": "First Owner",
    }
    is_valid, errors = validate_car_inputs(valid_input)
    assert is_valid is True
    assert len(errors) == 0

def test_validate_car_inputs_invalid_ranges():
    """Verify physically impossible or out-of-range inputs trigger clear validation errors."""
    invalid_input = {
        "year": 1850, # Pre-automobile era
        "km_driven": -5000, # Negative distance
        "mileage": 120.0, # Unrealistic mileage
        "engine": 9500, # Unrealistic engine
        "max_power": -10.0, # Negative power
        "seats": 35, # Impossible car seats
    }
    is_valid, errors = validate_car_inputs(invalid_input)
    assert is_valid is False
    assert len(errors) >= 5

def test_format_currency_inr():
    """Verify Indian currency formatting with Lakhs and Crores."""
    assert "₹ 4.50 Lakh" in format_currency_inr(450000)
    assert "₹ 1.25 Cr" in format_currency_inr(12500000)
    assert "₹ 85,000" in format_currency_inr(85000)

def test_prediction_all_four_models():
    """Verify inference pipeline executes across all 4 regression models."""
    sample_car = {
        "brand": "Maruti",
        "year": 2017,
        "km_driven": 50000,
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "mileage": 20.0,
        "engine": 1197.0,
        "max_power": 82.0,
        "seats": 5,
    }

    models_to_test = [
        "Experiment A: Simple Linear Regression",
        "Experiment B: Multiple Linear Regression",
        "Experiment C: KNN Regression (k=3)",
        "Experiment D: KNN Regression (k=5)",
    ]

    for model_name in models_to_test:
        res = predict_car_price(sample_car, selected_model_name=model_name)
        assert res["status"] == "success", f"Failed for {model_name}: {res.get('message')}"
        assert res["predicted_price"] > 0
        assert "₹" in res["formatted_price"]
        assert res["test_mae"] is not None
        assert res["test_r2"] is not None
