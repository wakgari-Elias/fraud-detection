import pandas as pd

def calculate_time_since_signup(df: pd.DataFrame) -> pd.DataFrame:
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['time_since_signup'] = (
        df['purchase_time'] - df['signup_time']
    ).dt.total_seconds()
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    return df
