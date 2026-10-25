# AutoValue AI: Used Car Price Prediction & Valuation System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![B.Tech ML Lab](https://img.shields.io/badge/B.Tech%20ML%20Lab-Practicals%201%20to%206-emerald.svg)]()
[![License](https://img.shields.io/badge/License-MIT-gray.svg)]()

> **AutoValue AI** is a complete, working machine learning web application built for the **B.Tech Machine Learning Laboratory Mini-Project**. It estimates pre-owned car valuations in Indian Rupees (₹) using **Linear Regression** and **K-Nearest Neighbours (KNN) Regression**.

---

## 🎯 Syllabus Compliance 

This project strictly adheres to the permitted concepts of the first six machine learning practicals:

1. **Python, pandas, and NumPy fundamentals**
2. **Exploratory Data Analysis (EDA)**: Inspection using `head()`, `tail()`, `shape`, `describe()`, `info()`, and `dtypes`.
3. **Data Cleaning**: Missing-value detection and handling (median for numerical, mode for categorical).
4. **Duplicate Handling**: Duplicate detection and justified removal (1,202 redundant listings removed).
5. **Outlier Analysis**: Numerical outlier detection using the Interquartile Range (IQR) method: $\text{IQR} = Q_3 - Q_1$.
6. **Data Visualization**: Univariate, bivariate, and multivariate analysis using Matplotlib and Seaborn.
7. **Categorical Encoding**: One-hot encoding via pandas `get_dummies()`.
8. **Feature Scaling**: scikit-learn `StandardScaler` (fitted strictly on the training partition for KNN).
9. **Train-Test Split**: Exact 80:20 split with a fixed random seed (`random_state=42`).
10. **Permitted Regression Algorithms**:
    - **Experiment A**: Simple Linear Regression ($y = mx + c$)
    - **Experiment B**: Multiple Linear Regression ($y = \beta_0 + \sum \beta_i x_i$)
    - **Experiment C**: KNN Regression with $k = 3$
    - **Experiment D**: KNN Regression with $k = 5$
11. **Practical 5: ID3 Decision Tree Algorithm**:
    - Mathematical Shannon Entropy: $H(S) = -\sum p_i \log_2(p_i)$
    - Information Gain: $Gain(S, A) = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)$
    - Decision Tree Classifier with `criterion="entropy"`
    - Confusion Matrix (TP, FP, TN, FN), Accuracy, Precision, Recall, F1-Score
    - Car Purchase Viability Classification ("Recommended Buy" vs. "Not Recommended")
12. **Practical 6: K-Means Clustering (Unsupervised Market Segmentation)**:
    - Within-Cluster Sum of Squares (Inertia): $\text{WCSS} = \sum_{k=1}^K \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$
    - Elbow Method curve analysis ($k=1 \dots 10$)
    - Silhouette Score evaluation
    - Dealership Personas (Budget Commuters, Clearance Workhorses, Mainstream Fleet, Certified Luxury)
    - Strategic promotional dealership packages (EMI schemes, warranty extensions)

> **Strict Rule Compliance**: No Decision Trees, Random Forests, XGBoost, SVMs, Neural Networks, Deep Learning, Clustering, Recommendation Systems, or SHAP were used.

---

## 📊 Experimental Results Table (Actual Model Evaluation)

All four configurations were trained on the exact same 80% training partition (5,540 vehicles) and evaluated on the identical held-out 20% test partition (1,385 vehicles):

| Experiment Configuration | Model Type | Features Used | Test MAE (₹) | Test RMSE (₹) | Test $R^2$ Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Experiment C** | **KNN Regression ($k=3$)** | **All 50 Features (Scaled)** | **₹ 1,06,253.08** | **₹ 2,52,278.53** | **0.7825** | **🏆 Best Model** |
| **Experiment D** | KNN Regression ($k=5$) | All 50 Features (Scaled) | ₹ 1,11,559.53 | ₹ 2,55,414.50 | 0.7770 | Alternative |
| **Experiment B** | Multiple Linear Regression | All 50 Features (Unscaled) | ₹ 1,56,580.25 | ₹ 3,23,115.37 | 0.6432 | Alternative |
| **Experiment A** | Simple Linear Regression | Single Feature (`max_power`) | ₹ 2,35,634.26 | ₹ 3,95,949.70 | 0.4642 | Baseline |

- **Selection Criterion**: Lowest Test Mean Absolute Error (MAE).
- **Key Insight**: Experiment C ($k=3$) achieved the lowest error because used car prices form localized clusters around vehicle segments (e.g. luxury sedans vs. budget hatchbacks), which nearest neighbours capture better than a single rigid linear hyperplane.

---

## 🗂️ Project Organization

```text
autovalue-ai/
├── app.py                     # Streamlit multi-page web application
├── config.py                  # Project paths, feature definitions, and validation bounds
├── requirements.txt           # Python package dependencies
├── README.md                  # Comprehensive project documentation
├── .gitignore                 # Git ignore file for cache, models, and virtual environments
├── data/
│   ├── Car details v3.csv     # Primary dataset (8,128 records from CarDekho)
│   ├── car_data.csv           # Fallback dataset
│   └── README.md              # Dataset provenance, schema, and download instructions
├── models/
│   ├── all_models.joblib      # Trained models, scalers, and dummy column schemas
│   ├── best_model.joblib      # Persisted winning model (Experiment C)
│   ├── model_metrics.json     # MAE, RMSE, R² scores and test scatter samples
│   ├── feature_info.json      # Dynamic categories and ranges for UI form
│   └── README.md              # Model artifacts description
├── src/
│   ├── __init__.py            # Python package initializer
│   ├── data_loader.py         # CSV loading, unit parsing, duplicate removal, IQR outliers
│   ├── preprocessing.py       # Imputation (median/mode), get_dummies(), StandardScaler
│   ├── train_models.py        # 4-experiment training and evaluation pipeline
│   ├── evaluate_models.py     # MAE, RMSE, R² metrics and currency formatting
│   ├── predict.py             # Inference pipeline with physical domain validation
│   └── visualizations.py      # Matplotlib and Seaborn dark navy automotive charts
├── assets/
│   ├── style.css              # Custom CSS design system (dark navy, electric blue, teal)
│   └── README.md              # Assets overview
├── reports/
│   ├── AutoValue_AI_Mini_Project_Report.md  # 10-page B.Tech academic mini-project report
│   ├── model_comparison_results.csv        # Real CSV evaluation export
│   └── README.md              # Reports documentation
└── tests/
    ├── __init__.py            # Test package initializer
    ├── test_preprocessing.py  # Unit tests for cleaning, units, IQR, and dummy encoding
    ├── test_prediction.py     # Unit tests for input validation and 4-model inference
    └── test_models.py         # Integration tests for MAE, RMSE, R², and saved artifacts
```

---

## 💻 Installation & Setup (Windows)

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed on your Windows machine.
- Git installed (optional, for version control).

### 2. Clone or Navigate to the Project Directory
```powershell
cd c:\Users\hp\Desktop\Ml
```

### 3. Create and Activate a Windows Virtual Environment
```powershell
# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```
*(If you encounter a PowerShell execution policy warning, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### 4. Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 🚀 Running the Project

### 1. Train the Models & Generate Evaluation Artifacts
Run the end-to-end training pipeline to clean the dataset, execute the 80:20 split, train all four models, and save artifacts:
```powershell
python src/train_models.py
```
*Output will display the MAE, RMSE, and $R^2$ for Experiments A, B, C, and D.*

### 2. Run the Automated Test Suite
Verify that preprocessing, validation, prediction, and models pass all tests:
```powershell
python -m pytest tests/ -p no:capture -v
```
*(All 11 automated unit and integration tests will pass).*

### 3. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your web browser and navigate to:
**`http://localhost:8501`**

---

## 🌐 Web Application Features

The website features a dark navy and white interface with electric-blue accents, subtle teal highlights, and responsive cards:

1. **🏠 Home Page**:
   - Automotive hero section with action buttons ("Predict Car Price", "Explore Dataset").
   - Live dataset statistics dynamically loaded from the cleaned data (6,925 vehicles, 32 brands, median price).
   - Winning model banner showing active model name, test MAE, and test $R^2$.
   - Three-step valuation workflow explanation.
   - Academic lab mini-project footer.

2. **🔮 Predict Car Price Page**:
   - Two-column vehicle specification form (Brand, Year, Km Driven, Fuel, Transmission, Owner, Mileage, Engine CC, Max Power bhp, Seats).
   - Strict physical validation (e.g. positive km, plausible year).
   - Model selection dropdown: choose the winning model or inspect any of the four experiments.
   - Valuation card displaying the estimated price in Indian Rupees (₹) (e.g., ₹ 5.13 Lakh).
   - Separate display of the model's actual test-set MAE and $R^2$.
   - Explicit disclaimer noting this is an educational estimate, not a guaranteed quote.

3. **📊 Model Comparison Page**:
   - Performance summary table comparing Experiments A, B, C, and D.
   - Button to download the real comparison table as a CSV (`model_comparison_results.csv`).
   - Side-by-side Matplotlib/Seaborn bar charts for MAE, RMSE, and $R^2$.
   - Interactive Actual vs. Predicted scatter plot with a 45-degree diagonal reference line ($y = x$).
   - Auto-generated summary explaining why the winning model succeeded.

4. **📈 Data Analysis Page**:
   - Dataset inspection tabs: `head()`, `tail()`, `dtypes`, `describe()`, missing values, duplicate records.
   - Visualizations: Selling price distribution (Histogram + KDE), price vs. year, price vs. km driven, price by fuel, price by transmission.
   - Numerical correlation heatmap.
   - Outlier inspection using box plots and the IQR method ($Q_1, Q_3, \text{IQR}$, bounds).

5. **ℹ️ About Project Page**:
   - Problem statement & syllabus alignment dossier.
   - Mathematical formulations for Simple LR, Multiple LR, and KNN ($k=3, 5$).
   - Evaluation metrics guide (MAE, RMSE, $R^2$).
   - Literature review with 5 authentic academic papers and DOIs.
   - Top 15 Viva Voce Questions & Answers cheat sheet for laboratory examination defense.

---

## 📦 Dataset Instructions

- **Primary File**: `data/Car details v3.csv`
- **Fallback File**: `data/car_data.csv`
- **Source**: CarDekho Indian Used Car Dataset on Kaggle ([Link](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)).
- If missing, place `Car details v3.csv` inside the `data/` folder and run `python src/train_models.py`.

---

## 🐙 Git & GitHub Instructions

To initialize version control and push to your GitHub repository:

```powershell
# 1. Initialize git repository
git init

# 2. Add all project files
git add .

# 3. Commit changes
git commit -m "Build AutoValue AI: Used Car Price Prediction Website (Syllabus Compliant)"

# 4. Set default branch to main
git branch -M main

# 5. Add your remote GitHub repository URL (replace with your repo URL)
git remote add origin https://github.com/your-username/autovalue-ai.git

# 6. Push to GitHub
git push -u origin main
```

## 📄 License
This project is developed for educational purposes under the MIT License as part of the B.Tech Computer Science & Engineering curriculum.
