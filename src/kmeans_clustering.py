"""K-Means Clustering Module for AutoValue AI.

Applies Unsupervised K-Means Clustering on the Indian used car market:
- Feature scaling via StandardScaler
- Elbow Method: Within-Cluster Sum of Squares (WCSS / Inertia) for k = 1 to 10
- Silhouette Score evaluation for k = 2 to 10
- Segment profiling into meaningful automotive buyer/dealership personas
- Promotional dealership marketing packages
- Real-time car cluster assignment
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import DATA_FILE, RANDOM_STATE
from src.data_loader import load_raw_data, clean_car_dataset
from src.evaluate_models import format_currency_inr

# Dealership Personas by Cluster
DEFAULT_DEAL_PERSONAS = {
    0: ("Budget Commuter Cars", "Affordable Price, Low-to-Moderate Mileage", "Zero-downpayment EMI schemes, 1st year free scheduled oil maintenance"),
    1: ("Value Clearance Segment", "Low Price, High Mileage (>100k km)", "Complimentary 6-month engine warranty, trade-in exchange bonus"),
    2: ("Mainstream Family Fleet", "Mid-Range Price, Moderate Mileage", "Exchange bonus of Rs. 25,000, free comprehensive insurance renewal"),
    3: ("Certified Pre-Owned Luxury", "High Price, Low Mileage (<50k km)", "VIP white-glove home delivery, complimentary ceramic coating, 2-year warranty"),
    4: ("Highway Touring & SUVs", "Upper Mid-Range Price, Higher Mileage", "Free highway roadside assistance for 2 years, tire upgrade discount"),
}

def run_car_clustering_pipeline(
    n_clusters: int = 4,
    df_clean: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Clusters cars by Selling Price and Kilometres Driven."""
    if df_clean is None:
        df_raw = load_raw_data(DATA_FILE)
        df_clean, _ = clean_car_dataset(df_raw, drop_duplicates=True)

    features = ["selling_price", "km_driven"]
    X = df_clean[features].values

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Elbow Method WCSS for k = 1 to 10
    wcss = []
    silhouette_scores = []
    k_range = list(range(1, 11))

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        wcss.append(round(float(km.inertia_), 2))
        if k >= 2:
            sil = silhouette_score(X_scaled, km.labels_)
            silhouette_scores.append(round(float(sil), 4))
        else:
            silhouette_scores.append(0.0)

    # 2. Fit K-Means with specified n_clusters
    kmeans_model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=RANDOM_STATE, n_init=10)
    labels = kmeans_model.fit_predict(X_scaled)
    df_clean["Cluster"] = labels
    centers_original = scaler.inverse_transform(kmeans_model.cluster_centers_)

    # 3. Cluster Profiling
    cluster_profiles = []
    for c in range(n_clusters):
        subset = df_clean[df_clean["Cluster"] == c]
        mean_p = float(subset["selling_price"].mean())
        mean_km = float(subset["km_driven"].mean())
        mean_yr = float(subset["year"].mean()) if "year" in subset.columns else 2015
        count = len(subset)

        p_name, p_desc, p_deal = DEFAULT_DEAL_PERSONAS.get(
            c, (f"Cluster {c}", "Market segment", "Standard dealership warranty discount")
        )

        cluster_profiles.append({
            "cluster_id": c,
            "persona": p_name,
            "count": count,
            "pct": round((count / len(df_clean)) * 100, 1),
            "avg_price_raw": mean_p,
            "avg_price_formatted": format_currency_inr(mean_p),
            "avg_km": f"{int(round(mean_km)):,} km",
            "avg_year": int(round(mean_yr)),
            "profile_desc": p_desc,
            "dealership_offer": p_deal,
        })

    # Sample 150 clustered cars for scatter visualization
    sample_df = df_clean[["name", "brand", "year", "selling_price", "km_driven", "Cluster"]].head(150)

    return {
        "features": features,
        "k_range": k_range,
        "wcss": wcss,
        "silhouette_scores": silhouette_scores,
        "cluster_centers": [
            {
                "Cluster": i,
                "Avg Price": format_currency_inr(c[0]),
                "Avg Km": f"{int(round(c[1])):,} km",
            }
            for i, c in enumerate(centers_original)
        ],
        "cluster_profiles": cluster_profiles,
        "sample_clustered_cars": sample_df.to_dict(orient="records"),
        "model": kmeans_model,
        "scaler": scaler,
    }

def assign_car_to_cluster(
    selling_price: float,
    km_driven: float,
    clustering_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assigns an individual car to its corresponding K-Means market cluster."""
    if clustering_bundle is None:
        clustering_bundle = run_car_clustering_pipeline(n_clusters=4)

    scaler = clustering_bundle["scaler"]
    model = clustering_bundle["model"]
    profiles = clustering_bundle["cluster_profiles"]

    X_single = np.array([[selling_price, km_driven]])
    X_scaled = scaler.transform(X_single)
    cluster_idx = int(model.predict(X_scaled)[0])

    profile = next((p for p in profiles if p["cluster_id"] == cluster_idx), profiles[0])

    return {
        "cluster_id": cluster_idx,
        "persona": profile["persona"],
        "description": profile["profile_desc"],
        "dealership_offer": profile["dealership_offer"],
    }
