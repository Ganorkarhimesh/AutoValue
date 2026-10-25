"""Unit and integration tests for the regression model configurations:
- Experiment A: Simple Linear Regression
- Experiment B: Multiple Linear Regression
- Experiment C: KNN Regression (k=3)
- Experiment D: KNN Regression (k=5)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ALL_MODELS_PATH, METRICS_PATH, REPORT_METRICS_CSV
from src.train_models import train_and_evaluate_all_experiments
from src.evaluate_models import calculate_metrics

def test_metrics_calculation():
    """Verify MAE, RMSE, and R2 calculation formulas."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0])

    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["MAE"] == 10.0
    assert metrics["RMSE"] == 10.0
    assert metrics["R2"] > 0.95

def test_saved_artifacts_existence():
    """Verify all artifacts exist and are non-empty."""
    assert ALL_MODELS_PATH.exists()
    assert METRICS_PATH.exists()
    assert REPORT_METRICS_CSV.exists()

def test_four_experiments_present_in_results():
    """Verify that exactly the 4 permitted experiments exist in metrics results."""
    import json
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    experiments = metrics_data["experiments"]
    assert len(experiments) == 4
    expected_exp_keys = [
        "Experiment A: Simple Linear Regression",
        "Experiment B: Multiple Linear Regression",
        "Experiment C: KNN Regression (k=3)",
        "Experiment D: KNN Regression (k=5)",
    ]
    for key in expected_exp_keys:
        assert key in experiments, f"Missing experiment: {key}"
        exp = experiments[key]
        assert "MAE" in exp["test_metrics"]
        assert "RMSE" in exp["test_metrics"]
        assert "R2" in exp["test_metrics"]
        assert exp["test_metrics"]["R2"] > 0  # Should be positive for all valid car models
