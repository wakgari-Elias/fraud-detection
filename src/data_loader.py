import pandas as pd

def load_fraud_data(path: str) -> pd.DataFrame:
    """Load Fraud_Data.csv"""
    return pd.read_csv(path)

def load_credit_data(path: str) -> pd.DataFrame:
    """Load creditcard.csv"""
    return pd.read_csv(path)
