from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import Tuple
import pandas as pd
from .config import ModelConfig

def split_data(X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> Tuple:
    return train_test_split(
        X, y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state
    )

def train_random_forest(X_train, y_train, config: ModelConfig):
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state
    )
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model
