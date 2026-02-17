# tests/test_src.py  (or test_task2.py)
import pytest
import pandas as pd
import numpy as np
import os
from src.data import load_processed_data, prepare_split
from src.models import train_baseline_logreg, train_xgboost
from src.evaluation import compute_metrics
import joblib


@pytest.fixture(scope="module")
def fraud_data():
    """Load small subset of fraud data once for all tests"""
    df = load_processed_data("fraud")
    return df.head(1500)  # enough rows for realistic testing


def test_load_processed_data():
    df = load_processed_data("fraud")
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 100000
    assert "class" in df.columns.str.lower()


def test_prepare_split_shapes(fraud_data):
    X_train, X_test, y_train, y_test = prepare_split(fraud_data.head(600))
    assert X_train.shape[0] + X_test.shape[0] == 600
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]


def test_prepare_split_stratified(fraud_data):
    X_train, X_test, y_train, y_test = prepare_split(fraud_data.head(600))
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.02, f"Fraud rates differ too much: {train_rate:.4f} vs {test_rate:.4f}"


def test_train_baseline_logreg(fraud_data):
    X_train, _, y_train, _ = prepare_split(fraud_data.head(400))
    model, info = train_baseline_logreg(X_train.iloc[:150], y_train.iloc[:150])
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert info["model_type"] == "LogisticRegression"


def test_train_xgboost(fraud_data):
    X_train, _, y_train, _ = prepare_split(fraud_data.head(400))
    model, info = train_xgboost(X_train.iloc[:150], y_train.iloc[:150], n_estimators=50)
    assert hasattr(model, "predict")
    assert hasattr(model, "feature_importances_")
    assert info["model_type"] == "XGBoost"


def test_compute_metrics():
    y_true = np.array([0]*90 + [1]*10)
    y_pred = np.array([0]*92 + [1]*8)
    y_prob = np.array([0.1]*90 + [0.9]*10)
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert "auc_pr" in metrics
    assert "f1" in metrics
    assert 0 < metrics["auc_pr"] <= 1
    assert 0 <= metrics["f1"] <= 1


# ────────────────────────────────────────────────
# NEW TESTS ADDED NOW (3 more)
# ────────────────────────────────────────────────

def test_xgboost_feature_importances(fraud_data):
    """Check that XGBoost learns feature importances."""
    X_train, _, y_train, _ = prepare_split(fraud_data.head(500))
    model, _ = train_xgboost(X_train.iloc[:200], y_train.iloc[:200], n_estimators=30)
    importances = model.feature_importances_
    assert len(importances) == X_train.shape[1]
    assert np.any(importances > 0), "All feature importances are zero — model didn't learn"


def test_compute_metrics_edge_cases():
    """Test metrics on extreme cases (all positive, all negative, perfect prediction)"""
    # All legitimate
    y_true_all_legit = np.zeros(100)
    y_pred_all_legit = np.zeros(100)
    y_prob_all_legit = np.zeros(100)
    metrics_legit = compute_metrics(y_true_all_legit, y_pred_all_legit, y_prob_all_legit)
    assert metrics_legit["auc_pr"] == 0.0
    assert metrics_legit["f1"] == 0.0

    # Perfect prediction
    y_true_perfect = np.array([0]*50 + [1]*50)
    y_pred_perfect = y_true_perfect.copy()
    y_prob_perfect = y_true_perfect.astype(float)
    metrics_perf = compute_metrics(y_true_perfect, y_pred_perfect, y_prob_perfect)
    assert metrics_perf["auc_pr"] == 1.0
    assert metrics_perf["f1"] == 1.0


def test_model_save_file_exists(fraud_data, tmp_path):
    """Test that model is actually saved to disk"""
    X_train, _, y_train, _ = prepare_split(fraud_data.head(400))
    model, _ = train_baseline_logreg(X_train.iloc[:100], y_train.iloc[:100])
    
    # Use tmp_path for test isolation
    save_dir = tmp_path / "models_test"
    save_dir.mkdir()
    save_path = save_dir / "test_logreg.joblib"
    
    joblib.dump(model, save_path)
    assert save_path.exists(), "Model file was not saved"
    assert save_path.stat().st_size > 1000, "Saved model file is too small — probably empty"
    
def test_preprocess_fraud_full_is_model_ready():
    """
    Task 1: Verify that preprocess_fraud_full produces clean, numeric data ready for modeling.
    - No missing values
    - Only numeric columns (except target 'class')
    - Expected shape after processing
    - Key engineered columns exist
    - Target column exists (case-insensitive check)
    """
    from src.preprocessing import preprocess_fraud_full

    # Run the full pipeline (save=False to avoid overwriting)
    df_processed = preprocess_fraud_full(save=False)

    # Basic shape check (should be close to original after dedup)
    assert df_processed.shape[0] > 140000, "Too few rows after preprocessing"
    assert df_processed.shape[1] > 100, "Too few columns after encoding"

    # No missing values
    assert df_processed.isna().sum().sum() == 0, "Missing values remain after preprocessing"

    # Only numeric columns + target (case-insensitive check for 'class')
    non_numeric = df_processed.select_dtypes(exclude=['number', 'bool', 'uint8']).columns.tolist()
    lower_cols = [col.lower() for col in df_processed.columns]
    assert 'class' in lower_cols, "Target 'class' (case-insensitive) missing"

    # If non-numeric exist, they should only be the target
    non_target_non_numeric = [col for col in non_numeric if col.lower() != 'class']
    assert len(non_target_non_numeric) == 0, \
        f"Non-numeric columns still present (excluding target): {non_target_non_numeric}"

    # Key Task 1 engineered features must exist
    required_cols = [
        'time_since_signup_hours', 'hour_of_day', 'day_of_week',
        'tx_per_user', 'tx_per_device',
        'purchase_value', 'age'  # scaled numerics
    ]
    missing_required = [c for c in required_cols if c not in df_processed.columns]
    assert not missing_required, f"Missing engineered features: {missing_required}"

    print("Task 1 preprocessing test passed: data is clean and model-ready ✓")