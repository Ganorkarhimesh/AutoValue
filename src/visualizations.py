"""Visualization module for AutoValue AI.

Provides visualization routines using Matplotlib and Seaborn:
- Dark navy, electric-blue, and teal theme matching automotive valuation dashboard.
- Renders:
    1. Model evaluation comparison bar charts (MAE, RMSE, R²).
    2. Actual vs. Predicted scatter plot with reference 45-degree diagonal line.
    3. Selling price distribution (Histogram with KDE).
    4. Feature vs. Price relationships (Year vs Price, Km vs Price).
    5. Categorical price distributions (Fuel Type, Transmission).
    6. Numerical correlation heatmap.
    7. Numerical outlier box plots with IQR bounds.
    8. ID3 Confusion Matrix and Information Gain bar charts.
    9. K-Means Elbow Curve and Cluster Scatter Plots.
"""

from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# Color Palette
DARK_NAVY = "#0f172a"
SURFACE_BG = "#1e293b"
CARD_BG = "#182238"
ELECTRIC_BLUE = "#3b82f6"
TEAL_ACCENT = "#14b8a6"
GOLD_ACCENT = "#f59e0b"
TEXT_WHITE = "#f8fafc"
TEXT_MUTED = "#94a3b8"
GRID_COLOR = "#334155"

def setup_matplotlib_theme():
    """Sets a consistent dark navy aesthetic for all Matplotlib/Seaborn figures."""
    plt.rcParams.update({
        "figure.facecolor": DARK_NAVY,
        "axes.facecolor": SURFACE_BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_WHITE,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "text.color": TEXT_WHITE,
        "font.family": "sans-serif",
        "font.size": 10,
    })

setup_matplotlib_theme()

# --- MODEL EVALUATION CHARTS ---

def plot_metric_bar_chart(
    comparison_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    is_currency: bool = False,
) -> plt.Figure:
    """Creates a clean bar chart comparing a metric across all four model experiments."""
    setup_matplotlib_theme()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    # Sort so best model stands out
    df_sorted = comparison_df.sort_values(by=metric_col, ascending=("MAE" in metric_col or "RMSE" in metric_col)).copy()

    # Distinguish best model with teal, others with electric blue
    colors = [TEAL_ACCENT if row.get("Is Best", False) else ELECTRIC_BLUE for _, row in df_sorted.iterrows()]

    # Shorten names for cleaner x-axis
    labels = []
    for m in df_sorted["Experiment"]:
        if "Simple Linear" in m:
            labels.append("Exp A:\nSimple LR")
        elif "Multiple Linear" in m:
            labels.append("Exp B:\nMultiple LR")
        elif "k=3" in m:
            labels.append("Exp C:\nKNN (k=3)")
        elif "k=5" in m:
            labels.append("Exp D:\nKNN (k=5)")
        else:
            labels.append(m)

    bars = ax.bar(labels, df_sorted[metric_col], color=colors, width=0.55, edgecolor="#ffffff", linewidth=0.5)

    # Add data values on top of bars
    for bar in bars:
        height = bar.get_height()
        if is_currency:
            if height >= 100_000:
                label_txt = f"₹{height/100_000:.2f}L"
            else:
                label_txt = f"₹{height:,.0f}"
        elif "R²" in metric_col or "R2" in metric_col:
            label_txt = f"{height:.4f}"
        else:
            label_txt = f"{height:,.2f}"

        ax.annotate(
            label_txt,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=TEXT_WHITE,
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12, color=TEXT_WHITE)
    ax.set_ylabel(ylabel, fontsize=10, color=TEXT_MUTED)
    ax.tick_params(axis="x", labelsize=9)

    # Currency formatting on y-axis
    if is_currency:
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"₹{x/100000:.1f}L" if x >= 100000 else f"₹{x:,.0f}")
        )

    plt.tight_layout()
    return fig

def plot_actual_vs_predicted(
    y_true: List[float],
    y_pred: List[float],
    model_name: str,
) -> plt.Figure:
    """Creates a scatter plot comparing Actual vs. Predicted values with an ideal 45-degree line."""
    setup_matplotlib_theme()
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=100)

    y_true_arr = np.array(y_true) / 100_000.0  # Convert to Lakhs for clean axes
    y_pred_arr = np.array(y_pred) / 100_000.0

    min_val = max(0, min(y_true_arr.min(), y_pred_arr.min()))
    max_val = max(y_true_arr.max(), y_pred_arr.max()) * 1.05

    # Scatter points
    ax.scatter(
        y_true_arr,
        y_pred_arr,
        color=ELECTRIC_BLUE,
        alpha=0.6,
        edgecolor=TEAL_ACCENT,
        linewidth=0.5,
        s=35,
        label="Test Vehicles",
    )

    # Ideal 45-degree diagonal reference line (y = x)
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color=GOLD_ACCENT,
        linestyle="--",
        linewidth=2,
        label="Perfect Fit (y = x)",
    )

    ax.set_title(f"Actual vs. Predicted Price — {model_name}", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Actual Selling Price (in Lakhs ₹)", fontsize=10, color=TEXT_MUTED)
    ax.set_ylabel("Predicted Selling Price (in Lakhs ₹)", fontsize=10, color=TEXT_MUTED)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend(loc="upper left", facecolor=DARK_NAVY, edgecolor=GRID_COLOR, fontsize=9)

    plt.tight_layout()
    return fig

# --- EXPLORATORY DATA ANALYSIS (EDA) CHARTS ---

def plot_selling_price_distribution(df: pd.DataFrame) -> plt.Figure:
    """Plots the distribution of car selling prices with histogram and KDE."""
    setup_matplotlib_theme()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    prices_lakh = df["selling_price"].dropna() / 100_000.0
    # Cap display at 99th percentile for clean readability while avoiding extreme outliers squashing the plot
    p99 = np.percentile(prices_lakh, 99)
    prices_filtered = prices_lakh[prices_lakh <= p99]

    sns.histplot(
        prices_filtered,
        kde=True,
        color=ELECTRIC_BLUE,
        ax=ax,
        bins=35,
        edgecolor="#ffffff",
        linewidth=0.4,
    )

    median_price = prices_filtered.median()
    ax.axvline(
        median_price,
        color=TEAL_ACCENT,
        linestyle="--",
        linewidth=2,
        label=f"Median: ₹{median_price:.2f}L",
    )

    ax.set_title("Selling Price Distribution (Histogram & KDE)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Selling Price (in Lakhs ₹)", fontsize=10, color=TEXT_MUTED)
    ax.set_ylabel("Vehicle Count", fontsize=10, color=TEXT_MUTED)
    ax.legend(facecolor=DARK_NAVY, edgecolor=GRID_COLOR, fontsize=9)

    plt.tight_layout()
    return fig

def plot_price_vs_numerical(df: pd.DataFrame, feature_col: str, feature_label: str) -> plt.Figure:
    """Plots scatter of a numerical feature vs selling price."""
    setup_matplotlib_theme()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    df_plot = df[[feature_col, "selling_price"]].dropna()
    y_lakh = df_plot["selling_price"] / 100_000.0
    x_val = df_plot[feature_col]

    ax.scatter(
        x_val,
        y_lakh,
        color=TEAL_ACCENT,
        alpha=0.45,
        s=25,
        edgecolor="none",
    )

    # Trend line using numpy polyfit (Simple Linear Regression intuition)
    if len(x_val) > 1:
        m, c = np.polyfit(x_val, y_lakh, 1)
        x_trend = np.linspace(x_val.min(), x_val.max(), 100)
        ax.plot(
            x_trend,
            m * x_trend + c,
            color=GOLD_ACCENT,
            linewidth=2,
            label=f"Linear Trend (Slope: {m:.3f})",
        )
        ax.legend(facecolor=DARK_NAVY, edgecolor=GRID_COLOR, fontsize=9)

    ax.set_title(f"{feature_label} vs. Selling Price", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(feature_label, fontsize=10, color=TEXT_MUTED)
    ax.set_ylabel("Selling Price (in Lakhs ₹)", fontsize=10, color=TEXT_MUTED)

    plt.tight_layout()
    return fig

def plot_price_by_categorical(df: pd.DataFrame, cat_col: str, title: str) -> plt.Figure:
    """Box plot of selling prices grouped by a categorical column (e.g., fuel or transmission)."""
    setup_matplotlib_theme()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    df_plot = df[[cat_col, "selling_price"]].dropna().copy()
    df_plot["price_lakh"] = df_plot["selling_price"] / 100_000.0

    # Filter top categories if there are too many
    top_cats = df_plot[cat_col].value_counts().head(6).index
    df_plot = df_plot[df_plot[cat_col].isin(top_cats)]

    sns.boxplot(
        data=df_plot,
        x=cat_col,
        y="price_lakh",
        ax=ax,
        palette="Blues_r",
        showfliers=False,  # Hide extreme dots for clean box visualization
        linewidth=1.2,
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(cat_col.replace("_", " ").title(), fontsize=10, color=TEXT_MUTED)
    ax.set_ylabel("Selling Price (in Lakhs ₹)", fontsize=10, color=TEXT_MUTED)
    ax.tick_params(axis="x", rotation=15 if len(top_cats) > 4 else 0)

    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, num_cols: List[str]) -> plt.Figure:
    """Generates a Seaborn correlation heatmap for numerical features and target."""
    setup_matplotlib_theme()
    cols = [c for c in num_cols if c in df.columns]
    if "selling_price" in df.columns and "selling_price" not in cols:
        cols.append("selling_price")

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=100)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        linecolor=DARK_NAVY,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 9, "weight": "bold"},
    )

    ax.set_title("Numerical Features Correlation Heatmap", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig

def plot_iqr_outliers_boxplot(df: pd.DataFrame, column: str, column_label: str) -> Tuple[plt.Figure, Dict[str, float]]:
    """Plots a boxplot with IQR lower and upper bound threshold lines."""
    setup_matplotlib_theme()
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=100)

    series = df[column].dropna()
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    sns.boxplot(
        x=series,
        ax=ax,
        color=ELECTRIC_BLUE,
        fliersize=3.5,
        flierprops={"markerfacecolor": GOLD_ACCENT, "markeredgecolor": "none"},
    )

    # Upper bound line
    ax.axvline(
        upper_bound,
        color="#ef4444",
        linestyle="--",
        linewidth=1.8,
        label=f"IQR Upper Bound ({upper_bound:,.1f})",
    )

    # Lower bound line (if greater than min)
    if lower_bound > series.min():
        ax.axvline(
            lower_bound,
            color="#ef4444",
            linestyle="--",
            linewidth=1.8,
            label=f"IQR Lower Bound ({lower_bound:,.1f})",
        )

    ax.set_title(f"Outlier Inspection: {column_label}", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(column_label, fontsize=10, color=TEXT_MUTED)
    ax.legend(facecolor=DARK_NAVY, edgecolor=GRID_COLOR, fontsize=8)

    plt.tight_layout()
    bounds = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }
    return fig, bounds

# --- ID3 DECISION TREE VISUALIZATIONS ---

def plot_confusion_matrix_heatmap(matrix_2x2: List[List[int]], title: str = "ID3 Confusion Matrix") -> plt.Figure:
    """Plots a 2x2 confusion matrix heatmap for car recommendation decisions."""
    setup_matplotlib_theme()
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=100)

    cm_arr = np.array(matrix_2x2)
    labels = ["Not Rec.", "Recommended"]

    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold", "color": TEXT_WHITE},
        linewidths=1.5,
        linecolor=DARK_NAVY,
    )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Recommendation", fontsize=9, color=TEXT_MUTED)
    ax.set_ylabel("Actual Recommendation", fontsize=9, color=TEXT_MUTED)
    plt.tight_layout()
    return fig

def plot_information_gain_bars(gains: Dict[str, float]) -> plt.Figure:
    """Bar chart of Information Gain for attributes computed by ID3."""
    setup_matplotlib_theme()
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=100)

    sorted_gains = sorted(gains.items(), key=lambda x: x[1], reverse=True)
    features = [x[0].replace("_", " ") for x in sorted_gains]
    values = [x[1] for x in sorted_gains]

    colors = [TEAL_ACCENT if i == 0 else ELECTRIC_BLUE for i in range(len(values))]
    bars = ax.bar(features, values, color=colors, width=0.55, edgecolor="#ffffff", linewidth=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=TEXT_WHITE,
        )

    ax.set_title("ID3 Attribute Information Gain (Entropy Reduction)", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("Gain(S, A)", fontsize=9, color=TEXT_MUTED)
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    return fig

# --- K-MEANS CLUSTERING VISUALIZATIONS ---

def plot_elbow_wcss_curve(k_range: List[int], wcss: List[float]) -> plt.Figure:
    """Plots Elbow Method WCSS (inertia) curve for optimal K selection."""
    setup_matplotlib_theme()
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=100)

    ax.plot(k_range, wcss, marker="o", color=ELECTRIC_BLUE, linewidth=2.2, markersize=6, markerfacecolor=TEAL_ACCENT)

    # Highlight elbow at k=4
    if len(k_range) >= 4:
        elbow_k = 4
        elbow_wcss = wcss[elbow_k - 1]
        ax.plot(elbow_k, elbow_wcss, marker="*", color=GOLD_ACCENT, markersize=14, label=f"Elbow (k={elbow_k})")
        ax.legend(facecolor=DARK_NAVY, edgecolor=GRID_COLOR, fontsize=8.5)

    ax.set_title("K-Means Elbow Method (Within-Cluster Sum of Squares)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Clusters (k)", fontsize=9, color=TEXT_MUTED)
    ax.set_ylabel("Inertia / WCSS", fontsize=9, color=TEXT_MUTED)
    ax.set_xticks(k_range)
    plt.tight_layout()
    return fig

def plot_kmeans_clusters_scatter(sample_cars: List[Dict[str, Any]]) -> plt.Figure:
    """2D scatter plot of used cars clustered by Price and Km Driven."""
    setup_matplotlib_theme()
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=100)

    df_sample = pd.DataFrame(sample_cars)
    df_sample["price_lakh"] = df_sample["selling_price"] / 100000.0
    df_sample["km_k"] = df_sample["km_driven"] / 1000.0

    cluster_palette = [ELECTRIC_BLUE, TEAL_ACCENT, GOLD_ACCENT, "#ec4899", "#8b5cf6"]

    for cluster_id in sorted(df_sample["Cluster"].unique()):
        subset = df_sample[df_sample["Cluster"] == cluster_id]
        color = cluster_palette[cluster_id % len(cluster_palette)]
        ax.scatter(
            subset["km_k"],
            subset["price_lakh"],
            color=color,
            alpha=0.7,
            s=45,
            edgecolor="#ffffff",
            linewidth=0.5,
            label=f"Cluster {cluster_id}",
        )

    ax.set_title("K-Means Market Segmentation (Price vs. Km Driven)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Kilometres Driven (in Thousands km)", fontsize=9, color=TEXT_MUTED)
    ax.set_ylabel("Selling Price (in Lakhs ₹)", fontsize=9, color=TEXT_MUTED)
    ax.legend(facecolor=DARK_NAVY, edgecolor=GRID_COLOR, fontsize=8.5, loc="upper right")
    plt.tight_layout()
    return fig

