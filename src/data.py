# src/data.py
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Tuple, Dict, Any
import os
from pathlib import Path

PROCESSED_FRAUD_PATH = "data/processed/fraud_processed.csv"
PROCESSED_CC_PATH = "data/processed/creditcard_processed.csv"


def load_processed_data(dataset: str = "fraud") -> pd.DataFrame:
    """
    Load preprocessed dataset from disk.
    
    Args:
        dataset: 'fraud' or 'creditcard'
    
    Returns:
        pandas DataFrame
    
    Raises:
        FileNotFoundError, ValueError
    """
    if dataset == "fraud":
        path = PROCESSED_FRAUD_PATH
    elif dataset == "creditcard":
        path = PROCESSED_CC_PATH
    else:
        raise ValueError("dataset must be 'fraud' or 'creditcard'")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed file not found: {path}")

    df = pd.read_csv(path)
    print(f"Loaded {dataset} data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def prepare_split(
    df: pd.DataFrame,
    target_col: str = "class",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train-test split.
    
    Args:
        df: processed DataFrame
        target_col: target column name ('class' or 'Class')
        test_size: fraction for test set
        random_state: seed
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"Train set: {X_train.shape[0]:,} samples | Fraud rate: {y_train.mean():.4f}")
    print(f"Test set:  {X_test.shape[0]:,} samples  | Fraud rate: {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test