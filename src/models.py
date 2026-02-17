# src/models.py
from __future__ import annotations

import joblib
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from typing import Tuple, Dict, Any
from pathlib import Path
from .evaluation import compute_metrics


def train_baseline_logreg(
    X_train, y_train,
    random_state: int = 42
) -> Tuple[Any, Dict[str, float]]:
    """Train simple Logistic Regression baseline."""
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=random_state,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    return model, {'model_type': 'LogisticRegression'}


def train_xgboost(
    X_train, y_train,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    scale_pos_weight: float | None = None
) -> Tuple[Any, Dict[str, float]]:
    """Train XGBoost with basic tuning parameters."""
    if scale_pos_weight is None:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric='aucpr',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, {
        'model_type': 'XGBoost',
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'scale_pos_weight': scale_pos_weight
    }


def evaluate_and_save_model(
    model: Any,
    X_test, y_test,
    model_name: str,
    save_dir: str = "models"
) -> Dict[str, Any]:
    """Evaluate model and save it if good."""
    Path(save_dir).mkdir(exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)

    # Save model
    save_path = Path(save_dir) / f"{model_name}.joblib"
    joblib.dump(model, save_path)
    print(f"Model saved: {save_path}")

    return metrics