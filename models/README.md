# Models Directory

This directory stores persisted scikit-learn artifacts generated during training:

1. `best_model.joblib`: The primary fitted pipeline achieving the highest cross-validation score, combining `ColumnTransformer` preprocessing and the tuned regressor.
2. `all_models.joblib`: A dictionary of all four fitted regression pipelines (`Linear Regression`, `Decision Tree`, `Gradient Boosting`, `K-Nearest Neighbours`).
3. `model_metrics.json`: Complete serialized metrics, hyperparameters, cross-validation distributions, sample test predictions, and feature importance mappings.
4. `feature_info.json`: Extracted numerical bounds and categorical option domains used by the UI inference form.

All pipelines handle missing value imputation and one-hot encoding internally.
