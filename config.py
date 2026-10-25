"""Configuration module for AutoValue AI Used Car Valuation System.

Defines project-wide paths, random seeds, feature definitions, validation boundaries,
and formatting constants adhering strictly to B.Tech ML Lab syllabus (Practicals 1 to 6).
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"

# Primary & Fallback Dataset Paths
DATA_FILE = DATA_DIR / "Car details v3.csv"
FALLBACK_DATA_FILE = DATA_DIR / "car_data.csv"

# Saved Model & Artifact Paths
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
ALL_MODELS_PATH = MODELS_DIR / "all_models.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
FEATURE_INFO_PATH = MODELS_DIR / "feature_info.json"
REPORT_METRICS_CSV = REPORTS_DIR / "model_comparison_results.csv"

# Model Training Parameters (Syllabus: 80:20 split, fixed random seed)
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Target Column
TARGET_COL = "selling_price"

# Features from Dataset
NUMERICAL_FEATURES = [
    "year",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats",
]

CATEGORICAL_FEATURES = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

# Simple Linear Regression (Experiment A) Primary Feature
SIMPLE_LR_FEATURE = "max_power"

# Physical Validation Ranges for Inference Inputs
VALIDATION_RANGES = {
    "year": (1995, 2026),
    "km_driven": (0, 1_000_000),
    "mileage": (5.0, 45.0),       # kmpl or km/kg
    "engine": (600, 5000),         # CC
    "max_power": (30.0, 600.0),    # bhp
    "seats": (2, 10),
    "selling_price": (10_000, 50_000_000),  # INR (₹10k - ₹5 Crore)
}

# Supported Categorical Domains (Fallbacks if dataset missing)
DEFAULT_CATEGORIES = {
    "brand": [
        "Maruti", "Hyundai", "Mahindra", "Tata", "Toyota", "Honda",
        "Ford", "Chevrolet", "Renault", "Volkswagen", "Skoda",
        "Nissan", "Audi", "BMW", "Mercedes-Benz", "Datsun", "Kia"
    ],
    "fuel": ["Petrol", "Diesel", "CNG", "LPG"],
    "seller_type": ["Individual", "Dealer", "Trustmark Dealer"],
    "transmission": ["Manual", "Automatic"],
    "owner": [
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car"
    ]
}

# UI Theme Config
APP_TITLE = "AutoValue AI"
APP_SUBTITLE = "Used Car Price Prediction & Valuation System"
CURRENCY_SYMBOL = "₹"
