### notebooks/README.md
# Notebooks – Exploratory Analysis, Feature Engineering & Modeling

This directory contains **Jupyter notebooks** for step-by-step analysis, prototyping, and generating visualizations/report assets. Notebooks are organized to follow the project workflow sequentially.

| Notebook                        | Purpose                                                                 |
|---------------------------------|-------------------------------------------------------------------------|
| **`eda-fraud-data.ipynb`**      | **Exploratory Data Analysis** for the e-commerce dataset (`Fraud_Data.csv`): distributions, fraud rates by source/browser/device/country, class imbalance visualization. |
| **`eda-creditcard.ipynb`**      | **Exploratory Data Analysis** for the bank credit card dataset (`creditcard.csv`): PCA features (V1-V28), amount/time patterns, extreme imbalance quantification. |
| **`feature-engineering.ipynb`** | **Full preprocessing pipeline**: cleaning, IP-to-country merge, time-based features (hour/day, time_since_signup), velocity calculations, encoding/scaling, and class imbalance handling (SMOTE or undersampling). Saves processed data. |
| **`modeling.ipynb`**            | **Model development**: Logistic Regression baseline, ensemble models (Random Forest/XGBoost/LightGBM), hyperparameter tuning, stratified CV, evaluation with AUC-PR/F1/Confusion Matrix, and final model selection. |
| **`shap-explainability.ipynb`** | **SHAP analysis** on the best model: global summary plots, force plots for individual predictions (true positive, false positive, false negative), feature importance comparison, and business recommendations. |

**Tip:** Run notebooks in the listed order for a smooth end-to-end experience.