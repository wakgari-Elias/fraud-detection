"""
Tests for src/explainability.py – focused on Task 3 functionality

Run with:
    pytest tests/test_explainability_task3.py -v
    or
    pytest tests/test_explainability_task3.py::test_name -v
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import shap

from src.explainability import (
    load_model,
    get_feature_importance,
    plot_top_features,
    explain_with_shap,
    plot_shap_force,
    plot_shap_dependence
)

# ────────────────────────────────────────────────
# Fixtures – shared setup
# ────────────────────────────────────────────────

@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).parent.parent  # assumes tests/ is at root level


@pytest.fixture(scope="session")
def model(project_root):
    model_path = project_root / "models" / "best_model_xgboost.joblib"
    if not model_path.exists():
        pytest.skip("Model file not found – skipping all model-dependent tests")
    return load_model(str(model_path))


@pytest.fixture(scope="session")
def sample_data(project_root, model):
    data_path = project_root / "data" / "processed" / "fraud_processed.csv"
    if not data_path.exists():
        pytest.skip("Processed data file not found")

    df = pd.read_csv(data_path)
    expected_features = model.feature_names_in_
    
    # Use only features the model was trained on
    X = df[expected_features].copy()
    y = df["class"].copy()
    
    # Small sample for fast tests
    X_small = X.sample(n=min(50, len(X)), random_state=42)
    y_small = y.loc[X_small.index]
    
    return X_small, y_small


# ────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────

def test_load_model_returns_xgboost(model):
    """Model should load and be an XGBoost classifier"""
    assert model is not None
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "feature_importances_")


def test_get_feature_importance_returns_dataframe(model, sample_data):
    X_small, _ = sample_data
    importance_df = get_feature_importance(model, X_small.columns.tolist())
    
    assert isinstance(importance_df, pd.DataFrame)
    assert len(importance_df) == X_small.shape[1]
    assert set(importance_df.columns) == {"feature", "importance"}
    assert importance_df["importance"].sum() > 0.99  # normalized ≈ 1
    assert importance_df["importance"].is_monotonic_decreasing


def test_plot_top_features_runs_without_error(model, sample_data, tmp_path):
    X_small, _ = sample_data
    importance_df = get_feature_importance(model, X_small.columns.tolist())
    
    save_path = tmp_path / "test_top_features.png"
    plot_top_features(importance_df, top_n=5, save_path=str(save_path))
    
    assert save_path.exists()
    assert save_path.stat().st_size > 10000  # rough check that something was saved


def test_explain_with_shap_returns_valid_objects(model, sample_data, tmp_path):
    X_small, _ = sample_data
    
    explanation, X_sample = explain_with_shap(
        model,
        X_small,
        sample_size=len(X_small),      # use all for test
        max_display=5,
        save_dir=str(tmp_path / "shap_test"),
        use_kmeans_background=True,
        kmeans_clusters=5
    )
    
    assert isinstance(explanation, shap.Explanation)
    assert explanation.values.shape == X_sample.shape
    assert len(explanation.feature_names) == X_small.shape[1]
    
    # Check that files were saved
    assert (tmp_path / "shap_test" / "shap_summary_bar.png").exists()
    assert (tmp_path / "shap_test" / "shap_summary_beeswarm.png").exists()


def test_plot_shap_force_runs_without_crash(model, sample_data, tmp_path):
    X_small, _ = sample_data
    
    explanation, X_sample = explain_with_shap(
        model, X_small, sample_size=len(X_small), kmeans_clusters=5
    )
    
    save_path = tmp_path / "test_force.png"
    plot_shap_force(
        explanation,
        index=0,
        save_path=str(save_path),
        title_suffix=" - Test Instance"
    )
    
    assert save_path.exists()


def test_plot_shap_dependence_runs_without_crash(model, sample_data, tmp_path):
    X_small, _ = sample_data
    
    explanation, X_sample = explain_with_shap(
        model, X_small, sample_size=len(X_small), kmeans_clusters=5
    )
    
    top_feature = explanation.feature_names[0]  # usually time_since_signup_hours
    
    save_path = tmp_path / "test_dependence.png"
    plot_shap_dependence(
        explanation.values,
        X_sample,
        feature=top_feature,
        interact_feature=None,
        save_path=str(save_path)
    )
    
    assert save_path.exists()


def test_shap_summary_plots_are_created(model, sample_data, tmp_path):
    X_small, _ = sample_data
    
    _, _ = explain_with_shap(
        model,
        X_small,
        sample_size=30,               # small for speed
        save_dir=str(tmp_path / "shap"),
        kmeans_clusters=5
    )
    
    assert (tmp_path / "shap" / "shap_summary_bar.png").exists()
    assert (tmp_path / "shap" / "shap_summary_beeswarm.png").exists()


# Optional: basic sanity check on predictions (not strictly Task 3, but useful)
def test_model_can_predict(model, sample_data):
    X_small, _ = sample_data
    probs = model.predict_proba(X_small)[:, 1]
    assert len(probs) == len(X_small)
    assert all(0 <= p <= 1 for p in probs)