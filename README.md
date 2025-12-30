# Improved Detection of Fraud Cases for E-Commerce and Bank Transactions
**Adey Innovations Inc. – Fraud Detection Project**  
*December 2025*

An end-to-end machine learning project to develop accurate and robust fraud detection models for e-commerce and bank credit card transactions. The models incorporate geolocation analysis, transaction velocity patterns, and advanced ensemble techniques while addressing severe class imbalance and providing explainable insights via SHAP.

---
## Business Goal
Adey Innovations Inc., a leader in financial technology, seeks to deliver cutting-edge fraud detection solutions that:

- **Accurately identify fraudulent transactions** in real-time to minimize financial losses for e-commerce and banking clients.  
- **Reduce false positives** to maintain excellent customer experience and avoid alienating legitimate users.  
- **Leverage geolocation and behavioral signals** (e.g., time between signup and purchase, transaction frequency) for stronger fraud detection.  
- **Provide transparent, interpretable models** that build trust with customers, partners, and regulators.

A successful system will enable faster risk mitigation, lower fraud-related losses, and stronger confidence in digital transactions.

## Project Overview
This project analyzes two highly imbalanced datasets:  
1. E-commerce transactions (`Fraud_Data.csv`) with rich user, device, IP, and timing details.  
2. Anonymized bank credit card transactions (`creditcard.csv`) with PCA-transformed features.

We perform thorough EDA, geolocation merging, feature engineering, imbalance handling (SMOTE and SMOTETomek), model training, and SHAP-based explainability to derive actionable business rules.

### Key Deliverables
- [x] Comprehensive EDA with visualizations of fraud patterns and class imbalance  
- [x] Geolocation integration (IP → Country) and rich feature engineering (velocity, time-since-signup, etc.)  
- [x] Resampling strategy for severe class imbalance (SMOTE for e-commerce, SMOTETomek for credit card)  
- [ ] Baseline Logistic Regression + advanced ensemble (Random Forest/XGBoost/LightGBM)  
- [ ] Evaluation using AUC-PR, F1-Score, Precision-Recall curves, and Confusion Matrix  
- [ ] SHAP global/local explanations with business recommendations  
- [x] Clean, organized, and reproducible repository

---
## Fraud Detection Business Understanding
### Key Challenges in Fraud Detection
1. **Extreme Class Imbalance**  
   Fraudulent transactions typically represent <<1% of total volume. Standard accuracy is misleading — we prioritize **Precision-Recall AUC** and **F1-score**.

2. **Security vs. Customer Experience Trade-off**  
   - High recall → catches more fraud but risks more false positives (customer friction).  
   - High precision → fewer false alarms but may miss real fraud (financial loss).  
   **Strategy:** Tune models and thresholds to optimize business cost (false negative cost >> false positive cost).

3. **Need for Explainability**  
   Financial institutions require interpretable decisions for regulatory compliance, audit trails, and deriving actionable rules (e.g., "flag transactions from high-risk countries within 1 hour of signup").

4. **Real-Time Requirements**  
   Models must be fast and deployable for live scoring while incorporating contextual signals like device reuse and geolocation anomalies.

---
## Project Structure
```text
fraud-detection/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/                           # Add this folder to .gitignore
│   ├── raw/                        # Original datasets
│   └── processed/                  # Cleaned and feature-engineered data
├── notebooks/
│   ├── __init__.py
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   ├── shap-explainability.ipynb
│   └── README.md
├── src/
│   ├── __init__.py
├── tests/
│   ├── __init__.py
├── models/                         # Saved model artifacts
├── scripts/
│   ├── __init__.py
│   └── README.md
├── requirements.txt
├── README.md
└── .gitignore
```
---
## Tech Stack
- Core: Python 3.12, pandas, numpy
- Visualization: matplotlib, seaborn, plotly
- Machine Learning: scikit-learn, imbalanced-learn, xgboost, lightgbm
- Explainability: shap
- Utilities: joblib, jupyter
---
## Quick Start
```bash
# Clone the repository
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```
Place the raw datasets in `data/raw/`:

* `Fraud_Data.csv`
* `IpAddress_to_Country.csv`
* `creditcard.csv`

Run the preprocessing pipeline:
```bash
   python scripts/preprocess.py
```

Explore and build the project using the Jupyter notebooks in order:
1. `eda-fraud-data.ipynb`
2. `eda-creditcard.ipynb`
3. `feature-engineering.ipynb`
4. `modeling.ipynb`
5. `shap-explainability.ipynb`
---
## Current Progress (as of December 21, 2025)
| Task | Status | Notes |
| :--- | :--- | :--- |
| **Data Loading & Initial Cleaning** | ✅ Completed | Raw datasets loaded and inspected |
| **EDA (Both Datasets)** | ✅ Completed | Visualizations and imbalance analysis ongoing |
| **Geolocation Merging & Feature Engineering** | ✅ Completed | IP-to-country + velocity features next |
| **Data Transformation & Imbalance Handling** | ✅ Completed | Scaling/encoding pipeline + SMOTE/SMOTETomek on training only |
| **Model Building & Evaluation** | 🔄 In Progress | Baseline + ensemble models |
| **SHAP Explainability & Recommendations** | ⏳ Planned | Final insights and business rules |
---
Challenge completed – Dec 2025  
Built by Elias Wakgari
## Task-2: Model Building and Evaluation

**Branch:** `task-2`  
**Notebook:** `notebooks/modeling.ipynb`

This task contains:

- Train-test split of Fraud_Data.csv
- Baseline Logistic Regression model
- Random Forest / XGBoost ensemble models
- Handling class imbalance with SMOTE
- Model evaluation: AUC-PR, F1-score, confusion matrix
- Stratified K-Fold cross-validation
- Task 3: Model Explainability (Fraud Detection)
🎯 Objective

The objective of Task 3 is to interpret the predictions of the best-performing fraud detection model using model explainability techniques. By applying SHAP (SHapley Additive exPlanations), we identify the key drivers behind fraud predictions and translate these insights into actionable business recommendations.

🧠 Model Used

Model: Random Forest Classifier

Reason for Selection:

Strong performance on imbalanced fraud data

Handles non-linear relationships well

Provides built-in feature importance for baseline comparison

The model was trained on preprocessed and resampled data produced in Task 1 (EDA & Feature Engineering) and Task 2 (Modeling).

📊 Feature Importance (Baseline)

Extracted built-in feature importance from the Random Forest model

Visualized the top 10 most important features

Used as a baseline to compare against SHAP explanations

🔍 SHAP Analysis
Global Explainability

Generated SHAP Summary Plot

Shows:

Overall feature importance

Direction of impact on fraud prediction

Distribution of feature effects across all samples

Local Explainability (Individual Predictions)

SHAP force plots were generated for:

True Positive (Fraud correctly detected)

False Positive (Legitimate transaction flagged as fraud)

False Negative (Fraud transaction missed by the model)

These plots explain why the model made each specific decision.

🔎 Interpretation & Insights
Top Fraud Drivers Identified (SHAP)

Time since signup

Transaction frequency

Purchase amount

Hour of day

Device / browser related features

Key Observations

Very short time between signup and purchase strongly increases fraud risk

High transaction velocity is a major fraud indicator

Some legitimate users are flagged due to unusually high purchase amounts

A small number of fraud cases are missed when fraud behavior closely resembles normal activity

SHAP explanations align well with Random Forest feature importance, increasing trust in the model.

💡 Business Recommendations

Add extra verification for new accounts

Transactions occurring shortly after signup should require additional checks
(Driven by high SHAP impact of time_since_signup)

Monitor high-velocity transactions

Multiple transactions within short time windows should trigger alerts
(Driven by SHAP importance of transaction frequency)

Adaptive thresholds for high-value purchases

Large transactions should be evaluated in context (user history, time, device)
(Driven by SHAP insights on purchase amount)

These recommendations directly map model explanations to real-world fraud prevention strategies.

✅ Conclusion

SHAP provides transparency into the fraud detection model by explaining both global trends and individual predictions. This interpretability enables better trust, regulatory compliance, and data-driven business decisions.
