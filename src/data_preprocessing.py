# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import logging


# Initialize logger for this module
logger = logging.getLogger(__name__)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering operations to the cleaned Fraud_Data dataframe.
    
    Creates:
        - hour_of_day: hour of purchase (0-23)
        - day_of_week: weekday of purchase (0=Monday, 6=Sunday)
        - time_since_signup: hours between signup and purchase
        - user_txn_count: total transactions per user
        - user_total_spent: total purchase value per user (additional velocity signal)
    
    Args:
        df (pd.DataFrame): Cleaned dataframe with datetime columns.
    
    Returns:
        pd.DataFrame: DataFrame with new engineered features.
    """
    df = df.copy()

    # Ensure datetime conversion (safe if already converted)
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    # Time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.weekday  # Monday=0, Sunday=6

    # Time since signup in hours (more granular than days)
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600

    # User-level velocity features
    df['user_txn_count'] = df.groupby('user_id')['user_id'].transform('count')  # Cleaner than grouping on purchase_value
    df['user_total_spent'] = df.groupby('user_id')['purchase_value'].transform('sum')

    # Optional: Average purchase per user
    df['user_avg_purchase'] = df['user_total_spent'] / df['user_txn_count']
    cols_to_drop = ['user_id', 'device_id', 'signup_time', 'purchase_time']
    
    # Check if they exist before dropping to avoid errors
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    logger.info("✅ Feature engineering completed.")
    logger.info("Sample of new features:\n%s", 
                df[['hour_of_day', 'day_of_week', 'time_since_signup', 
                    'user_txn_count', 'user_total_spent', 'user_avg_purchase']].head())

    return df



def build_preprocessor(df: pd.DataFrame):
    """
    Builds a sklearn ColumnTransformer for scaling numeric + one-hot encoding categorical features.
    Excludes target column ('class' or 'Class').
    """
    # Auto-detect feature types (exclude target)
    target_candidates = ['class', 'Class']
    target = next((col for col in target_candidates if col in df.columns), None)

    feature_df = df.drop(columns=[target]) if target else df

    numeric_features = feature_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = feature_df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor

