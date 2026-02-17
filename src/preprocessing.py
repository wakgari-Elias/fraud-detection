from __future__ import annotations

import pandas as pd
import bisect
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import os

# Constants for paths (capstone maintainability)
RAW_FRAUD_PATH = "data/raw/Fraud_Data.csv"
RAW_IP_PATH = "data/raw/IpAddress_to_Country.csv"
RAW_CC_PATH = "data/raw/creditcard.csv"
PROCESSED_FRAUD_PATH = "data/processed/fraud_processed.csv"
PROCESSED_CC_PATH = "data/processed/creditcard_processed.csv"

def add_geolocation(df: pd.DataFrame, ip_path: str = RAW_IP_PATH) -> pd.DataFrame:
    """Add country via efficient range lookup. Handles float IP in dataset."""
    ip_df = pd.read_csv(ip_path)
    df = df.copy()
    
    # Critical fix: ip_address is float in this dataset — cast directly to int
    df["ip_int"] = df["ip_address"].astype('int64')
    
    ip_df["lower"] = ip_df["lower_bound_ip_address"].astype('int64')
    ip_df["upper"] = ip_df["upper_bound_ip_address"].astype('int64')
    
    lowers = ip_df["lower"].values
    uppers = ip_df["upper"].values
    countries = ip_df["country"].values
    
    def map_country(ip_int: int) -> str:
        idx = bisect.bisect_right(lowers, ip_int) - 1
        if idx >= 0 and ip_int <= uppers[idx]:
            return countries[idx]
        return "Unknown"
    
    df["country"] = df["ip_int"].apply(map_country)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time, velocity, and derived features."""
    df = df.copy()
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    
    df["time_since_signup_hours"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds() / 3600
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.weekday
    
    df["tx_per_user"] = df.groupby("user_id")["user_id"].transform("count")
    df["tx_per_device"] = df.groupby("device_id")["device_id"].transform("count")
    
    return df

def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numerical, one-hot categorical, aggressively clean non-modeling columns."""
    df = df.copy()
    
    cat_cols = ["source", "browser", "sex", "country"]
    num_cols = ["purchase_value", "age", "time_since_signup_hours", "hour_of_day", 
                "day_of_week", "tx_per_user", "tx_per_device"]
    
    # One-hot encoding
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Scaling
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Remove all known useless columns
    useless = ['user_id', 'device_id', 'ip_address', 'ip_int',
               'signup_time', 'purchase_time']
    to_drop = [c for c in useless if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
    
    # Last resort: keep ONLY numeric columns + target
    numeric_cols = df.select_dtypes(include=['number', 'bool', 'uint8']).columns
    if 'class' in df.columns:
        numeric_cols = numeric_cols.union(['class'])
    df = df[numeric_cols]
    
    print(f"Cleaned shape for modeling: {df.shape}")
    print(f"Final columns: {df.columns.tolist()}")
    
    return df

def preprocess_fraud_full(save: bool = True) -> pd.DataFrame:
    """Full Task 1 pipeline for e-commerce data."""
    df = pd.read_csv(RAW_FRAUD_PATH)
    
    # Cleaning
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_shape[0] - df.shape[0]} duplicates")
    print(f"Missing values: {df.isna().sum().sum()} (none expected)")
    
    # Pipeline
    df = add_geolocation(df)
    df = engineer_features(df)
    df = transform_features(df)  # Scaled + encoded
    
    # Imbalance note (SMOTE later in modeling)
    fraud_rate = df["class"].mean()
    print(f"Fraud rate: {fraud_rate:.3%} (highly imbalanced — will use SMOTE on train only)")
    
    unknown_rate = (df.filter(like='country_Unknown').sum(axis=1) > 0).mean() if 'country_Unknown' in df.columns else 0
    print(f"Unknown countries rate: {unknown_rate:.2%}")
    
    if save:
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(PROCESSED_FRAUD_PATH, index=False)
        print(f"Saved processed data to {PROCESSED_FRAUD_PATH}")
    
    return df