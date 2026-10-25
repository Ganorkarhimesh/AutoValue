"""Unit and integration tests for ID3 Decision Tree and K-Means Clustering."""

import sys
from pathlib import Path
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.id3_decision_tree import (
    compute_entropy,
    compute_information_gain,
    train_and_evaluate_id3_tree,
    predict_deal_recommendation,
)
from src.kmeans_clustering import (
    run_car_clustering_pipeline,
    assign_car_to_cluster,
)

def test_id3_entropy_and_gain_math():
    """Verify Shannon entropy and information gain math formulas."""
    import pandas as pd
    series = pd.Series(["Yes", "Yes", "No", "No"])
    # 50/50 split -> entropy is exactly 1.0
    entropy = compute_entropy(series)
    assert abs(entropy - 1.0) < 0.01

def test_id3_tree_training_and_metrics():
    """Verify ID3 decision tree trains with entropy criterion and produces valid classification scores."""
    res = train_and_evaluate_id3_tree(max_depth=3)
    assert "metrics" in res
    assert res["metrics"]["accuracy"] > 0.70
    assert "confusion_matrix" in res["metrics"]
    assert "information_gains" in res
    assert len(res["information_gains"]) >= 4

def test_id3_single_prediction():
    """Verify ID3 inference predicts deal recommendation for a car."""
    pred = predict_deal_recommendation(
        selling_price=420000,
        year=2018,
        km_driven=40000,
        transmission="Manual",
        fuel="Petrol"
    )
    assert "recommendation" in pred
    assert "confidence" in pred
    assert pred["confidence"] > 0

def test_kmeans_clustering_pipeline():
    """Verify K-Means runs Elbow WCSS and returns cluster profiles."""
    res = run_car_clustering_pipeline(n_clusters=4)
    assert len(res["wcss"]) == 10
    assert len(res["cluster_profiles"]) == 4
    assert res["wcss"][0] > res["wcss"][-1] # WCSS monotonically decreases with k

def test_kmeans_single_car_assignment():
    """Verify assigning a car to its corresponding K-Means cluster."""
    cluster_res = assign_car_to_cluster(selling_price=450000, km_driven=35000)
    assert "cluster_id" in cluster_res
    assert "persona" in cluster_res
    assert "dealership_offer" in cluster_res
