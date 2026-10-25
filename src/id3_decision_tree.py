"""ID3 Decision Tree Module for AutoValue AI.

Applies ID3 Decision Tree Algorithm for Car Purchase Decision Modeling:
- Discretizes automotive features into categories (Price Bracket, Vehicle Age, Km Bracket, Transmission, Fuel)
- Mathematical Shannon Entropy: H(S) = -sum(p_i * log2(p_i))
- Mathematical Information Gain: Gain(S, A) = H(S) - sum(|S_v|/|S| * H(S_v))
- ID3 Decision Tree Classifier using criterion='entropy'
- Model evaluation: Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (TP, FP, TN, FN) computation and visualization
- Live single-car purchase recommendation inference
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from config import DATA_FILE, RANDOM_STATE
from src.data_loader import load_raw_data, clean_car_dataset

def build_car_purchase_dataset(df_clean: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Prepares customer car buying decision dataset linked to actual vehicle listings."""
    if df_clean is None:
        df_raw = load_raw_data(DATA_FILE)
        df_clean, _ = clean_car_dataset(df_raw, drop_duplicates=True)
    
    df = df_clean.copy()
    n = len(df)
    np.random.seed(RANDOM_STATE)

    # Discretize continuous attributes for ID3 entropy & information gain analysis
    price_brackets = pd.qcut(df["selling_price"], q=3, labels=["Budget", "MidRange", "Premium"])
    age_brackets = pd.cut(df["year"], bins=[1990, 2012, 2017, 2026], labels=["Older", "Moderate", "Recent"])
    km_brackets = pd.cut(df["km_driven"], bins=[-1, 50000, 100000, 1000000], labels=["Low_Km", "Mid_Km", "High_Km"])
    trans = df["transmission"]
    fuel = df["fuel"]

    # Purchase likelihood rule:
    # Buyers strongly favor recent years, lower km, and competitive price brackets
    buy_score = (
        (age_brackets == "Recent").astype(int) * 2.5 +
        (km_brackets == "Low_Km").astype(int) * 2.0 +
        (price_brackets == "Budget").astype(int) * 1.5 +
        (trans == "Automatic").astype(int) * 0.8 +
        np.random.normal(0, 1.0, n)
    )
    purchased = np.where(buy_score >= 3.0, "Recommended Buy", "Not Recommended")

    df_purchase = pd.DataFrame({
        "Price_Bracket": price_brackets,
        "Vehicle_Age": age_brackets,
        "Km_Bracket": km_brackets,
        "Transmission": trans,
        "Fuel": fuel,
        "Recommendation": purchased,
    }).dropna().reset_index(drop=True)

    return df_purchase

def compute_entropy(labels: pd.Series) -> float:
    """Calculates mathematical Shannon Entropy:
    H(S) = -sum(p_i * log2(p_i))
    """
    counts = labels.value_counts()
    probs = counts / len(labels)
    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
    return round(entropy, 4)

def compute_information_gain(df: pd.DataFrame, attribute: str, target: str) -> float:
    """Calculates mathematical Information Gain:
    Gain(S, A) = Entropy(S) - sum(|S_v|/|S| * Entropy(S_v))
    """
    total_entropy = compute_entropy(df[target])
    values = df[attribute].dropna().unique()
    weighted_entropy = 0.0

    for val in values:
        subset = df[df[attribute] == val]
        weight = len(subset) / len(df)
        weighted_entropy += weight * compute_entropy(subset[target])

    gain = total_entropy - weighted_entropy
    return round(float(gain), 4)

def train_and_evaluate_id3_tree(
    max_depth: int = 4,
    min_samples_split: int = 5,
) -> Dict[str, Any]:
    """Trains ID3 Decision Tree on car buying recommendations."""
    df = build_car_purchase_dataset()
    features = ["Price_Bracket", "Vehicle_Age", "Km_Bracket", "Transmission", "Fuel"]
    target_col = "Recommendation"

    # 1. Mathematical Entropy & Information Gain
    dataset_entropy = compute_entropy(df[target_col])
    gains = {feat: compute_information_gain(df, feat, target_col) for feat in features}
    best_root = max(gains, key=gains.get)

    # 2. One-hot encoding for decision tree classifier
    X_encoded = pd.get_dummies(df[features], drop_first=False, dtype=int)
    y_encoded = (df[target_col] == "Recommended Buy").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.25, random_state=RANDOM_STATE, stratify=y_encoded
    )

    # 3. Fit ID3 Decision Tree using 'entropy' criterion
    tree_model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=RANDOM_STATE,
    )
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)

    # 4. Evaluation metrics (Accuracy, Precision, Recall, F1)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "matrix_2x2": [[int(tn), int(fp)], [int(fn), int(tp)]]
        }
    }

    # Feature importances
    feat_imp = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in zip(X_encoded.columns, tree_model.feature_importances_)
    ]
    feat_imp = sorted(feat_imp, key=lambda x: x["importance"], reverse=True)

    return {
        "dataset_sample": df.head(8).to_dict(orient="records"),
        "dataset_entropy": dataset_entropy,
        "information_gains": gains,
        "best_root": best_root,
        "metrics": metrics,
        "feature_importances": feat_imp[:6],
        "model": tree_model,
        "dummy_columns": list(X_encoded.columns),
    }

def predict_deal_recommendation(
    selling_price: float,
    year: int,
    km_driven: float,
    transmission: str,
    fuel: str,
    tree_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Infers deal recommendation for a specific car using the ID3 decision tree."""
    if tree_bundle is None:
        tree_bundle = train_and_evaluate_id3_tree()

    model = tree_bundle["model"]
    dummy_cols = tree_bundle["dummy_columns"]

    # Discretize user car
    price_b = "Budget" if selling_price < 350000 else ("MidRange" if selling_price < 700000 else "Premium")
    age_b = "Older" if year <= 2012 else ("Moderate" if year <= 2017 else "Recent")
    km_b = "Low_Km" if km_driven < 50000 else ("Mid_Km" if km_driven <= 100000 else "High_Km")

    df_single = pd.DataFrame([{
        "Price_Bracket": price_b,
        "Vehicle_Age": age_b,
        "Km_Bracket": km_b,
        "Transmission": transmission,
        "Fuel": fuel,
    }])

    df_encoded = pd.get_dummies(df_single, drop_first=False, dtype=int).reindex(
        columns=dummy_cols, fill_value=0
    )

    pred = int(model.predict(df_encoded.values)[0])
    prob = float(model.predict_proba(df_encoded.values)[0][1])

    return {
        "recommendation": "Recommended Buy (High Value Deal)" if pred == 1 else "Not Recommended (Higher Risk / Overpriced)",
        "confidence": round(prob if pred == 1 else (1.0 - prob), 3),
        "price_bracket": price_b,
        "age_bracket": age_b,
        "km_bracket": km_b,
        "root_attribute": tree_bundle["best_root"],
    }
