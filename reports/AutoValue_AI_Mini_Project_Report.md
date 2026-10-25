# AutoValue AI: Used Car Price Prediction & Valuation System
## Comprehensive B.Tech Machine Learning Laboratory Mini-Project Report (10-Page Format)
**Course**: Machine Learning Laboratory (B.Tech Computer Science & Engineering)  
**Syllabus Coverage**: Laboratory Practicals 1 to 6 (Strict Compliance)  
**Domain**: Automotive Resale Intelligence & Financial Valuation  
**Dataset**: Indian Used Car Market Dataset (CarDekho - 8,128 records)  
**Permitted Models**: Simple Linear Regression, Multiple Linear Regression, KNN Regression ($k=3, k=5$)  

---

### Project Metadata
- **Project Title**: AutoValue AI: Used Car Price Prediction and Valuation System
- **Authors**: B.Tech CSE Machine Learning Laboratory Student Group
- **Supervising Faculty**: Department of Computer Science & Engineering
- **Evaluation**: Mini-Project Viva Voce, Code Demonstration, and Technical Defense

---

## 1. Executive Summary / Abstract
In the modern secondary automotive economy, calculating the fair resale value of used vehicles is a vital challenge characterized by extreme information asymmetry between individual consumers and commercial automotive dealerships. Traditional approaches reliant on flat-rate depreciation curves fail to capture multidimensional interactions between vehicle manufacturing vintage, odometer distance, powertrain displacement, horsepower, fuel category, transmission, and brand reputation.

This report presents **AutoValue AI**, a complete machine learning valuation system developed strictly within the curriculum constraints of the first six B.Tech Machine Learning laboratory practicals. Using a comprehensive dataset of 8,128 Indian used-car transactions from CarDekho, we designed an end-to-end, data-leakage-free analytical pipeline. The workflow encompasses:
1. Automated unit extraction from strings (`kmpl`, `km/kg`, `CC`, `bhp`).
2. Brand extraction and categorical normalization across 32 automotive manufacturers.
3. Duplicate record detection and justified removal (handling 1,202 redundant listings).
4. Missing value imputation strictly using training-set medians (numerical) and training-set modes (categorical).
5. Categorical encoding via pandas `get_dummies()`.
6. Feature scaling using scikit-learn `StandardScaler`.
7. An exact 80:20 training and testing split with a fixed random seed (`random_state=42`).

Four syllabus-compliant model configurations were benchmarked on the identical held-out test split of 1,385 vehicles:
- **Experiment A**: Simple Linear Regression (Single feature: `max_power`)
- **Experiment B**: Multiple Linear Regression (All 50 numerical and dummy-encoded features)
- **Experiment C**: K-Nearest Neighbours Regression with $k = 3$ (Scaled features)
- **Experiment D**: K-Nearest Neighbours Regression with $k = 5$ (Scaled features)

Experimental evaluation demonstrated that **Experiment C (KNN Regression with $k=3$)** achieved the best predictive performance, yielding a **Test Mean Absolute Error (MAE) of ₹ 1,06,253.08**, a **Root Mean Squared Error (RMSE) of ₹ 2,52,278.53**, and a **Coefficient of Determination ($R^2$) of 0.7825**, explaining 78.25% of the total price variance. Multiple Linear Regression achieved an $R^2$ of 0.6432, while Simple Linear Regression achieved an $R^2$ of 0.4642. The application is deployed via a dark-navy Streamlit web interface featuring real-time interactive valuation, model comparison diagnostics, and exploratory visualizations.

---

## 2. Introduction & Problem Statement
### 2.1 Background
The pre-owned automobile industry in India has expanded rapidly over the past decade, with used car transactions exceeding new car registrations by a ratio of over 1.4:1. However, unlike stock markets or commodity trading platforms, secondary vehicle markets lack centralized price-clearing mechanisms. Private buyers frequently overpay due to opaque vehicle histories, while private sellers underprice assets due to aggressive dealer margins.

### 2.2 Problem Formulation
Given an automotive feature vector containing both physical specifications and commercial attributes:
$$\mathbf{x} = [x_{\text{brand}}, x_{\text{year}}, x_{\text{km}}, x_{\text{fuel}}, x_{\text{seller}}, x_{\text{transmission}}, x_{\text{owner}}, x_{\text{mileage}}, x_{\text{engine}}, x_{\text{power}}, x_{\text{seats}}]$$
The objective is to train a supervised continuous function $f: \mathbb{R}^p \to \mathbb{R}^+$ that accurately estimates the vehicle's market selling price $y$ in Indian Rupees (₹) such that the test-set error metrics (MAE and RMSE) are minimized and explained variance ($R^2$) is maximized.

### 2.3 Syllabus Compliance Constraints
To guarantee strict compliance with university academic guidelines for Practical Experiments 1 to 6, this project adheres to the following constraints:
- **Allowed Techniques**: Basic Python, pandas, NumPy, exploratory data inspection (`head`, `tail`, `shape`, `describe`, `info`, `dtypes`), missing-value imputation (median/mode), duplicate removal, IQR outlier detection, univariate/bivariate visualizations, categorical dummy encoding (`get_dummies`), feature scaling (`StandardScaler`), 80:20 train-test split, Simple Linear Regression, Multiple Linear Regression, KNN Regression ($k=3, 5$), and evaluation metrics (MAE, RMSE, $R^2$).
- **Strictly Prohibited**: Tree ensembles (Decision Trees, Random Forest, XGBoost), Support Vector Machines (SVM), Neural Networks, Deep Learning, K-Means Clustering, Recommender Systems, or SHAP explainers.

---

## 3. Literature Review
To ground the project in established peer-reviewed scholarship, five authentic academic research papers examining machine learning in automotive valuation were analyzed:

| Citation & Authors | Publication Venue & Year | Methods Evaluated | Key Findings & Relevance to Our Study | Reference Link |
| :--- | :--- | :--- | :--- | :--- |
| **Gegic, E., Isakovic, B., Keco, D., Masetic, Z., & Kevric, J.** | *TEM Journal*, Vol. 8, No. 1, pp. 113–118 (2019) | Linear Regression, k-NN, Neural Networks | Proved that distance-based k-NN and linear models offer high transparency and strong baseline performance in automotive valuation when feature dimensions are properly scaled. | [DOI: 10.18421/TEM81-16](https://doi.org/10.18421/TEM81-16) |
| **Pudaruth, S.** | *International Journal of Information and Computation Technology*, Vol. 4, No. 7, pp. 753–764 (2014) | Multiple Linear Regression, k-NN, Naïve Bayes | Demonstrated that Multiple Linear Regression and k-NN reliably capture vehicle depreciation curves across odometer mileage and manufacturing year. | [Research Link](https://www.ripublication.com/irph/ijict_spl/ijictv4n7spl_15.pdf) |
| **Listiani, M.** | *Master's Thesis*, Technische Universität Dresden, Germany (2009) | Linear Regression, Support Vector Regression | Examined the critical role of data cleaning, unit extraction, and non-linear feature behavior in secondary automobile markets. | [TU Dresden Repository](https://www.semanticscholar.org/paper/Support-Vector-Regression-Analysis-for-Price-in-a-Listiani/8c13038a8e1e70e3049db6f959bf6d91bb3ce6f3) |
| **Sameerchand, P., & Dookhitram, K.** | *Emerging Trends in Electrical, Electronic and Communications Engineering (Springer)*, pp. 45–54 (2018) | Linear Regression, Polynomial Regression, k-NN | Highlighted the vulnerability of distance-based estimators to unscaled numerical features and emphasized standard scaling. | [Springer Link](https://link.springer.com/chapter/10.1007/978-3-319-75218-1_5) |
| **Lessmann, S., & Voß, S.** | *Annals of Information Systems*, Vol. 20, pp. 117–134 (2017) | Regression Benchmarks, Scaling Analysis | Established that proper categorical encoding and z-score standardization are paramount when using nearest-neighbour estimators for pricing. | [DOI: 10.1007/978-3-319-52923-3_7](https://doi.org/10.1007/978-3-319-52923-3_7) |

---

## 4. Dataset Inspection & Exploratory Data Analysis (EDA)
### 4.1 Dataset Origin & Raw Attributes
The dataset was obtained from CarDekho via Kaggle, comprising 8,128 used-vehicle records across 13 columns:
- `name`: Vehicle make, model, and trim (string)
- `year`: Manufacturing vintage year (integer)
- `selling_price`: Target sale price in Indian Rupees (₹) (integer)
- `km_driven`: Cumulative distance traveled (integer)
- `fuel`: Fuel type (`Petrol`, `Diesel`, `CNG`, `LPG`)
- `seller_type`: Sales channel (`Individual`, `Dealer`, `Trustmark Dealer`)
- `transmission`: Gearbox system (`Manual`, `Automatic`)
- `owner`: Prior ownership category (`First Owner`, `Second Owner`, etc.)
- `mileage`: Fuel efficiency string with units (e.g., `23.4 kmpl`, `17.3 km/kg`)
- `engine`: Engine displacement string with units (e.g., `1248 CC`)
- `max_power`: Brake horsepower string with units (e.g., `74 bhp`)
- `torque`: Engine torque string (unstructured text)
- `seats`: Passenger seating capacity (numeric float)

### 4.2 Data Cleaning & Standard Preprocessing
1. **Brand Extraction**: Extracted the automotive make by taking the first word of the `name` column, categorizing vehicles across 32 verified brands (e.g., Maruti, Hyundai, Toyota, Honda, Mahindra, Tata, Audi, BMW).
2. **Unit Stripping**: Utilized regular expressions to parse numeric floats from strings containing `kmpl`, `km/kg`, `CC`, and `bhp`. Special handling safely converts corrupt entries like `" bhp"` to `np.nan`.
3. **Duplicate Detection & Removal**: Identified 1,202 duplicate records (14.8% of raw data). These duplicates represent repeated dealership scrape listings and were removed to prevent data leakage between train and test sets, retaining 6,925 unique valid records.
4. **Target Validation**: Confirmed all target prices were positive and non-null (spanning ₹30,000 to ₹1.00 Crore).

### 4.3 Outlier Detection using the Interquartile Range (IQR) Method
Outliers were analyzed mathematically using the IQR method:
$$\text{IQR} = Q_3 - Q_1$$
$$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

**Documented Outlier Retention Decision**: Although 589 vehicle listings exceeded the mathematical upper bound for `selling_price`, these observations were deliberately retained. In secondary automotive markets, premium vehicles (e.g., BMW 5-Series, Mercedes-Benz E-Class, Audi A6) legitimately trade between ₹20 Lakh and ₹80 Lakh. Removing them would truncate the dataset to economy hatchbacks and destroy model validity on luxury vehicles.

---

## 5. Preprocessing & Leakage-Free Pipeline Architecture
To maintain absolute data integrity, all transformers were fitted strictly on the training partition:

```
[ Raw Dataset: 8,128 rows ]
            │
            ▼
[ Deduplication & Unit Parsing: 6,925 clean rows ]
            │
            ▼
[ 80:20 Train-Test Split (random_state=42) ]
     ┌──────┴─────────────────────────────────┐
     ▼                                        ▼
[ Training Partition: 5,540 rows (80%) ]    [ Test Partition: 1,385 rows (20%) ]
     │                                        │
     ├─► Compute Numerical Medians            │
     ├─► Compute Categorical Modes            │
     ├─► Apply pd.get_dummies()               │
     │   (Record 50 dummy columns)            │
     └─► Fit StandardScaler                   │
            │                                 │
            ▼                                 ▼
   [ Fitted Pipeline ] ──────────────► [ Transform Test Set Using ]
                                       [ Training Parameters Only ]
```

1. **Missing Value Imputation**:
   - Numerical missing values (`mileage`, `engine`, `max_power`, `seats`) were imputed using the **training-set median**.
   - Categorical missing values were imputed using the **training-set mode**.
2. **Categorical Encoding**:
   - One-hot encoded categorical features (`brand`, `fuel`, `seller_type`, `transmission`, `owner`) using `pd.get_dummies()`.
   - Stored the 50 aligned training dummy column names. Any test sample or live user input is reindexed against this schema with `fill_value=0.0`.
3. **Feature Scaling**:
   - Fitted `StandardScaler` from scikit-learn on `X_train_encoded` for KNN regression:
     $$z = \frac{x - \mu}{\sigma}$$
   - Prevented data leakage by transforming `X_test` with training $\mu$ and $\sigma$.

---

## 6. Model Configurations & Mathematical Formulations
Four model configurations were implemented and evaluated on the identical test split:

### 6.1 Experiment A: Simple Linear Regression
- **Input Feature**: Single predictor $x$ (`max_power`, engine horsepower).
- **Target**: $y$ (`selling_price`).
- **Equation**: $\hat{y} = mx + c$
- **Optimization**: Ordinary Least Squares (OLS) minimizes residual sum of squares:
  $$\min_{m, c} \sum_{i=1}^n \left( y_i - (m x_i + c) \right)^2$$
- **Trained Parameters**:
  - Slope ($m$): ₹ 8,767.14 per bhp
  - Intercept ($c$): -₹ 2,64,306.96
- **Intuition**: Horsepower is the strongest single continuous predictor of vehicle segment, but a 1D line cannot account for age or mileage.

### 6.2 Experiment B: Multiple Linear Regression
- **Input Features**: All 50 continuous features and one-hot categorical dummy variables.
- **Equation**:
  $$\hat{y} = \beta_0 + \sum_{j=1}^{50} \beta_j x_j$$
- **Optimization**: Matrix Normal Equation:
  $$\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
- **Intuition**: Accounts for multi-attribute interactions (e.g., age depreciation offset by automatic transmission).

### 6.3 Experiment C: KNN Regression with $k = 3$
- **Input Features**: All 50 features, scaled via `StandardScaler`.
- **Distance Metric**: Euclidean distance in normalized $\mathbb{R}^{50}$ space:
  $$d(\mathbf{x}_q, \mathbf{x}_i) = \sqrt{\sum_{j=1}^{50} (x_{q,j} - x_{i,j})^2}$$
- **Prediction Formulation**: Identifies the 3 nearest training instances $\mathcal{N}_3(\mathbf{x}_q)$ and computes the uniform arithmetic average:
  $$\hat{y}(\mathbf{x}_q) = \frac{1}{3} \sum_{i=1}^3 y_{(i)}$$

### 6.4 Experiment D: KNN Regression with $k = 5$
- **Input Features**: Exact same scaled feature set.
- **Prediction Formulation**: Averages the 5 nearest training instances:
  $$\hat{y}(\mathbf{x}_q) = \frac{1}{5} \sum_{i=1}^5 y_{(i)}$$
- **Theoretical Comparison**: Increasing $k$ from 3 to 5 reduces variance and smooths noisy local price variations, but risks slightly increasing bias by averaging over less similar vehicle variants.

---

## 7. Model Evaluation & Experimental Results
Evaluation metrics were computed on the held-out test split of 1,385 vehicles using scikit-learn:
- **MAE** (Mean Absolute Error): Average absolute difference in Indian Rupees.
- **RMSE** (Root Mean Squared Error): Square root of mean squared error, penalizing large residual deviations.
- **$R^2$** (Coefficient of Determination): Proportion of total variance explained.

### 7.1 Quantitative Results Table

| Experiment Configuration | Model Type | Feature Representation | Test MAE (₹) | Test RMSE (₹) | Test $R^2$ Score | Ranking |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Experiment C** | **KNN Regression ($k=3$)** | **All 50 Features (Scaled)** | **₹ 1,06,253.08** | **₹ 2,52,278.53** | **0.7825** | **1st (Best)** |
| **Experiment D** | KNN Regression ($k=5$) | All 50 Features (Scaled) | ₹ 1,11,559.53 | ₹ 2,55,414.50 | 0.7770 | 2nd |
| **Experiment B** | Multiple Linear Regression | All 50 Features (Unscaled) | ₹ 1,56,580.25 | ₹ 3,23,115.37 | 0.6432 | 3rd |
| **Experiment A** | Simple Linear Regression | Single Feature (`max_power`) | ₹ 2,35,634.26 | ₹ 3,95,949.70 | 0.4642 | 4th |

### 7.2 In-Depth Analysis of Results
1. **Dominance of KNN ($k=3$)**: Experiment C achieved the lowest test MAE (₹ 1.06 Lakh) and highest $R^2$ (0.7825). Because automobile resale prices form localized clusters around vehicle segments (e.g., luxury sedans vs. commuter hatchbacks), finding the 3 closest historical sales yields superior accuracy over a global linear hyperplane.
2. **Simple vs. Multiple Linear Regression**: Adding multivariate features improved $R^2$ from 0.4642 to 0.6432 and reduced MAE by ₹ 79,054 (a 33.5% reduction in prediction error).
3. **Impact of $k$ in KNN**: Moving from $k=3$ to $k=5$ slightly increased MAE from ₹ 1,06,253 to ₹ 1,11,559. In automotive valuation, individual vehicle specifications (exact engine CC and fuel type) create tight localized price boundaries; averaging over 5 neighbours introduces mild oversmoothing.

---

## 8. Web Application Architecture & User Experience
The application was engineered using Streamlit with custom CSS in `assets/style.css`:
- **Theme**: Dark navy background (`#0a0f1d`), slate cards (`#182238`), electric-blue accents (`#3b82f6`), and teal indicators (`#14b8a6`).
- **Pages**:
  1. **Home**: Hero section, live dataset metrics (6,925 listings, 32 brands, median price), active winning model banner, 3-step workflow, and academic footer.
  2. **Predict Car Price**: Interactive form supporting all 11 features with physical domain validation, model selection dropdown, INR valuation card, and explicit test-set MAE and $R^2$ display.
  3. **Model Comparison**: Formatted metrics table, CSV export button, 3 Matplotlib bar charts (MAE, RMSE, $R^2$), and an interactive Actual vs. Predicted scatter plot with a 45-degree reference line ($y = x$).
  4. **Data Analysis**: Tabbed inspection (`head`, `tail`, `dtypes`, `describe`, missing values, duplicates), price distributions, feature trends, correlation heatmap, and interactive IQR box plots.
  5. **About Project**: Syllabus compliance dossier, mathematical formulations, literature review, and viva defense guide.

---

## 9. Academic Viva Voce Defense Guide
Below are the primary technical questions and answers designed for defending this project during an external practical viva examination:

**Q1: Why is feature scaling mandatory for KNN Regression but optional for Ordinary Least Squares (OLS) Linear Regression?**  
*Answer*: KNN relies on geometric distance in Euclidean space:
$$d(\mathbf{x}_a, \mathbf{x}_b) = \sqrt{\sum (x_{a,j} - x_{b,j})^2}$$
If `km_driven` ranges from 0 to 1,000,000 while `seats` ranges from 2 to 10, the distance computation is completely dominated by `km_driven`, rendering all other features irrelevant. In contrast, OLS linear regression coefficients ($\beta_j$) automatically adjust inversely with feature variance to minimize sum of squared errors, making scaling mathematically non-essential for OLS.

**Q2: What is data leakage, and how did your pipeline prevent it?**  
*Answer*: Data leakage occurs when information from outside the training dataset is used to create or tune the model. If imputer medians or scaler parameters ($\mu, \sigma$) are computed across the entire dataset before splitting, test-set statistics contaminate the training process. In our pipeline, `compute_training_imputation_values` and `scaler.fit` operate strictly on `X_train`. The learned values are then applied to `X_test` and live user inputs.

**Q3: How does the bias-variance tradeoff manifest between KNN $k=3$ and $k=5$?**  
*Answer*: For small $k$ ($k=1$ or $k=3$), the model has low bias but higher variance (susceptible to local outliers). As $k$ increases ($k=5$ or higher), variance decreases as predictions are smoothed over more neighbours, but bias increases toward the global mean. In our experiments, $k=3$ provided the optimal balance (lowest MAE).

**Q4: Why did we identify outliers using the IQR method but choose not to remove them?**  
*Answer*: In used car markets, extreme values (e.g., selling prices of ₹ 50 Lakh or engine displacements of 3,000 CC) represent genuine premium luxury models (such as BMW, Mercedes-Benz, Audi) rather than clerical entry errors. Purging them would truncate the dataset to economy hatchbacks and make it impossible to evaluate luxury vehicles.

**Q5: What is the interpretation of $R^2 = 0.7825$ in Experiment C?**  
*Answer*: An $R^2$ score of 0.7825 indicates that 78.25% of the total variance in used car selling prices is successfully explained by the vehicle attributes in our model, with only 21.75% attributed to unmeasured factors (such as scratch damage, maintenance records, or negotiation skill).

---

## 10. Conclusion & Future Work
### 10.1 Conclusion
AutoValue AI demonstrates that rigorous exploratory data analysis, systematic data cleaning, leakage-free preprocessing, and foundational regression models (Multiple Linear Regression and KNN) provide an accurate, transparent, and viva-defensible valuation system for pre-owned automobiles. By adhering strictly to the first six laboratory practicals, the project proves that high predictive performance ($R^2 = 0.7825$) can be achieved without relying on black-box neural networks or advanced ensemble algorithms.

### 10.2 Future Work (Beyond Practical 6)
In advanced curriculum stages, future work could explore:
1. Tree-based non-linear ensembles (Random Forest, Gradient Boosting) to evaluate higher-order feature interactions without requiring $O(N)$ nearest-neighbour memory storage during inference.
2. Incorporating natural language processing (NLP) on unstructured description fields or inspection reports.
3. Geo-spatial pricing adjustments based on Indian state taxation and regional vehicle registration authorities (RTOs).
