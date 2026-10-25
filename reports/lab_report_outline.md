# B.Tech Machine Learning Laboratory Manual & Report (15 Marks)

**Course**: Machine Learning Laboratory (B.Tech Computer Science & Engineering)  
**Course Outcomes**: CO1 R, CO2 R  
**Portal**: AutoValue AI — Complete Machine Learning Laboratory Suite  

---

## Index of Practicals

| Practical | Course Outcome | Title | Algorithm / Technique |
| :---: | :---: | :--- | :--- |
| **1** | **CO1 R** | Exploratory Data Analysis & Preprocessing on Titanic Dataset | `head()`, `tail()`, `describe()`, `shape`, Imputation, Duplicate Handling, IQR Outlier Capping, Data Encoding, Scaling, 80:20 Split |
| **2** | **CO1 R** | Find-S Concept Learning Algorithm on EnjoySport Dataset | Maximally Specific Hypothesis $h = \langle h_1, \dots, h_6 \rangle$ with step-by-step trace after every instance |
| **3** | **CO2 R** | Linear Regression for Used Car Price Prediction | OLS Linear Regression, Ridge Hyperparameter Tuning ($\lambda$), MAE, MSE, RMSE, $R^2$ Score, Residual Analysis |
| **4** | **CO2 R** | k-NN Algorithm for Classification & Regression from Scratch | Pure NumPy implementation: Euclidean/Manhattan distance, Plurality Voting, Neighbor Averaging |
| **5** | **CO2 R** | ID3 Decision Tree for Customer Purchase Prediction | Shannon Entropy $H(S)$, Information Gain $Gain(S, A)$, Accuracy, Precision, Recall, F1, Confusion Matrix |
| **6** | **CO2 R** | Unsupervised Customer Segmentation for Shopkeeper Deals | K-Means with Elbow WCSS, Agglomerative Hierarchical Clustering with Dendrogram, Customer Deal Profiles |

---

## Detailed Methodologies & Formulas

### Practical 1: Exploratory Data Analysis & Preprocessing (CO1 R)
- **IQR Outlier Detection**:
  $$\text{IQR} = Q_3 - Q_1$$
  $$\text{Lower Fence} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$$
- **StandardScaler**:
  $$z = \frac{x - \mu}{\sigma}$$
- **MinMaxScaler**:
  $$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

### Practical 2: Find-S Concept Learning (CO1 R)
- **Hypothesis Representation**: $h = \langle h_1, h_2, h_3, h_4, h_5, h_6 \rangle$ where $h_i \in \{ \phi, \text{value}, ? \}$.
- **Update Rule**:
  - Initialize $h_0 = \langle \phi, \phi, \phi, \phi, \phi, \phi \rangle$.
  - For positive instance $x$: If $h_i = \phi$, $h_i \leftarrow x_i$. If $h_i \neq x_i$, $h_i \leftarrow '?'$.
  - For negative instance: Ignore (Find-S only prunes the specific boundary).

### Practical 3: Linear Regression with Ridge Tuning (CO2 R)
- **OLS Objective**:
  $$\min_w \sum_{i=1}^n (y_i - w^T x_i)^2$$
- **Ridge Regularization (L2 Penalty)**:
  $$\min_w \left\{ \sum_{i=1}^n (y_i - w^T x_i)^2 + \alpha \sum_{j=1}^p w_j^2 \right\}$$
- **Evaluation Metrics**:
  - $\text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|$
  - $\text{RMSE} = \sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$
  - $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

### Practical 4: k-NN from Scratch (CO2 R)
- **Euclidean Distance**:
  $$d(p, q) = \sqrt{\sum_{i=1}^d (p_i - q_i)^2}$$
- **Classification Output**: $\hat{y} = \text{mode}(\{ y_{(1)}, \dots, y_{(k)} \})$
- **Regression Output**: $\hat{y} = \frac{1}{k}\sum_{i=1}^k y_{(i)}$

### Practical 5: ID3 Decision Tree Algorithm (CO2 R)
- **Shannon Entropy**:
  $$\text{Entropy}(S) = - \sum_{i=1}^c p_i \log_2(p_i)$$
- **Information Gain**:
  $$\text{Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)$$
- **Metrics**:
  - $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
  - $\text{Precision} = \frac{TP}{TP + FP}$
  - $\text{Recall} = \frac{TP}{TP + FN}$
  - $\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

### Practical 6: Unsupervised Learning & Clustering (CO2 R)
- **K-Means Inertia / WCSS**:
  $$\text{WCSS} = \sum_{k=1}^K \sum_{x \in C_k} ||x - \mu_k||^2$$
- **Agglomerative Hierarchical (Ward's Linkage)**: Minimizes total within-cluster variance when merging clusters.
- **Shopkeeper Targeted Marketing**:
  - Low Income, Low Spending: Budget bundles & discount coupons.
  - Low Income, High Spending: Flash impulse promotions.
  - Moderate Income, Moderate Spending: Loyalty point reward systems.
  - High Income, Low Spending: Quality warranties & premium product demos.
  - High Income, High Spending: Exclusive VIP preview invitations & personalized shopping.
